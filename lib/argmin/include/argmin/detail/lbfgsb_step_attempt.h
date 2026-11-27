#ifndef HPP_GUARD_ARGMIN_DETAIL_LBFGSB_STEP_ATTEMPT_H
#define HPP_GUARD_ARGMIN_DETAIL_LBFGSB_STEP_ATTEMPT_H

#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/bound_projection.h"
#include "argmin/detail/evaluation_count.h"
#include "argmin/detail/lbfgsb_direction.h"

#include "argmin/formulation/concepts.h"

#include "argmin/line_search/armijo.h"
#include "argmin/line_search/result.h"
#include "argmin/line_search/options.h"
#include "argmin/line_search/strong_wolfe.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <cstdint>
#include <optional>
#include <algorithm>

namespace argmin::detail
{

template <typename Scalar, int N>
struct lbfgsb_attempt
{
    explicit lbfgsb_attempt(Eigen::Index n)
        : value(0)
        , accepted(false)
        , non_finite(false)
        , cached_alpha(-1)
        , evaluations(0)
        , x(Eigen::Vector<Scalar, N>::Zero(n))
        , g(Eigen::Vector<Scalar, N>::Zero(n))
        , ls()
    {}

    Scalar value;
    bool accepted;
    bool non_finite;
    Scalar cached_alpha;
    std::uint32_t evaluations;
    Eigen::Vector<Scalar, N> x;
    Eigen::Vector<Scalar, N> g;
    line_search_result<Scalar> ls;
};

// A failed search whose capped step still satisfies sufficient decrease is kept, as
// L-BFGS-B 3.0 keeps a search stopped at the maximum step (Morales and Nocedal 2011).
template <typename Scalar, int N>
bool accepts_lbfgsb_step(const lbfgsb_attempt<Scalar, N>& a, Scalar f_old, Scalar c1, Scalar dphi0)
{
    if(!a.x.allFinite() || !std::isfinite(a.value) || !(a.ls.alpha > Scalar(0)))
        return false;
    return a.ls.success || a.value <= f_old + c1 * a.ls.alpha * dphi0;
}

template <typename P, typename Scalar, int N, typename State>
Eigen::Vector<Scalar, N> lbfgsb_trial_point(const State& s, const Eigen::Vector<Scalar, N>& d, Scalar alpha)
{
    if constexpr(bound_constrained<P, Scalar>)
        return detail::project((s.x + alpha * d).eval(), s.lower, s.upper);
    else
        return (s.x + alpha * d).eval();
}

// A step re-evaluates the gradient at its iterate; a non-finite value there keeps the finite one the commit evaluated.
template <typename Scalar, int N, typename State>
void refresh_lbfgsb_gradient(State& s)
{
    Eigen::Vector<Scalar, N> g(s.g.size());
    s.problem->gradient(s.x, g);
    if(g.allFinite())
        s.g = g;
}

// The gradient at an accepted trial is evaluated before the commit, so a non-finite one fails the trial.
template <typename P, typename Scalar, int N, typename State>
bool lbfgsb_trial_gradient(const State& s, lbfgsb_attempt<Scalar, N>& a, const Eigen::Vector<Scalar, N>& d)
{
    bool cached = a.cached_alpha == a.ls.alpha;
    if constexpr(bound_constrained<P, Scalar>)
        cached = cached && a.x.isApprox((s.x + a.ls.alpha * d).eval(), Scalar(0.0));
    if(!cached)
        s.problem->gradient(a.x, a.g);
    return a.g.allFinite();
}

template <typename P, typename Scalar, int N, typename State, typename Options>
bool attempt_lbfgsb_step(State& s, const Eigen::Vector<Scalar, N>& d, Scalar alpha_max,
                         const Options& options, lbfgsb_attempt<Scalar, N>& a)
{
    using line_search_kind = decltype(options.line_search_type);
    a.cached_alpha = Scalar(-1.0);
    auto phi = [&](Scalar t) { return counted_value<Scalar>(s.evaluations, *s.problem, (s.x + t * d).eval()); };
    auto dphi = [&](Scalar t) {
        s.problem->gradient((s.x + t * d).eval(), a.g);
        a.cached_alpha = t;
        return a.g.dot(d);
    };
    line_search_options<Scalar> ls_opts = options.line_search;
    ls_opts.max_alpha = std::min(ls_opts.max_alpha, alpha_max);
    Scalar dphi0 = s.g.dot(d);
    a.ls = (options.line_search_type == line_search_kind::armijo)
        ? armijo(phi, s.objective_value, dphi0, ls_opts)
        : strong_wolfe(phi, dphi, s.objective_value, dphi0, ls_opts);
    a.x = lbfgsb_trial_point<P>(s, d, a.ls.alpha);
    a.value = counted_value<Scalar>(s.evaluations, *s.problem, a.x);
    a.evaluations += static_cast<std::uint32_t>(a.ls.evaluations) + 1u;
    a.accepted = accepts_lbfgsb_step(a, s.objective_value, ls_opts.c1, dphi0) && lbfgsb_trial_gradient<P>(s, a, d);
    a.non_finite = a.ls.diagnostics.nan_eval_count > 0 || !std::isfinite(a.value) || !a.g.allFinite();
    return a.accepted;
}

template <typename Scalar, int N, typename State>
step_result<Scalar> lbfgsb_accepted_result(const State& s, Scalar step_norm, Scalar old_f)
{
    Scalar x_norm = s.x.norm();
    std::optional<solver_status> policy_status{};
    constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();
    if(step_norm < eps * std::max(x_norm, Scalar(1.0)) * Scalar(10.0))
        policy_status = solver_status::roundoff_limited;
    return step_result<Scalar>{
        .objective_value = s.objective_value,
        .gradient_norm = s.g.norm(),
        .step_size = step_norm,
        .objective_change = s.objective_value - old_f,
        .improved = s.objective_value < old_f,
        .x_norm = x_norm,
        .kkt_residual = kkt_residual_bound<Scalar, N>(s.x, s.g, s.lower, s.upper),
        .policy_status = policy_status,
    };
}

template <typename Scalar, int N, typename State>
step_result<Scalar> commit_lbfgsb_step(State& s, const lbfgsb_attempt<Scalar, N>& a)
{
    Eigen::Vector<Scalar, N> sk = (a.x - s.x).eval();
    Eigen::Vector<Scalar, N> yk = (a.g - s.g).eval();
    s.B.push(sk, yk);
    Scalar old_f = s.objective_value;
    s.x = a.x;
    s.objective_value = a.value;
    s.g = a.g;
    ++s.iteration;
    return lbfgsb_accepted_result<Scalar, N>(s, sk.norm(), old_f);
}

// Both memories failed. When the last search met a non-finite value or the attempt still holds a non-finite gradient
// (after an Armijo retry it can be the first attempt's), the search's own limit is where it ended.
template <typename Scalar, int N, typename State>
step_result<Scalar> lbfgsb_line_search_failure(const State& s, const lbfgsb_attempt<Scalar, N>& a)
{
    return step_result<Scalar>{
        .objective_value = s.objective_value,
        .gradient_norm = s.g.norm(),
        .step_size = Scalar(0.0),
        .objective_change = Scalar(0.0),
        .improved = false,
        .is_null_step = true,
        .x_norm = s.x.norm(),
        .kkt_residual = kkt_residual_bound<Scalar, N>(s.x, s.g, s.lower, s.upper),
        .policy_status = a.non_finite ? solver_status::non_finite_evaluation : solver_status::line_search_failed,
    };
}

// Returns nullopt when the reset memory yields no direction; the caller reports its own null step.
template <typename P, typename Scalar, int N, typename State, typename Options>
std::optional<step_result<Scalar>> retry_lbfgsb_step(State& s, const Options& options,
                                                     lbfgsb_attempt<Scalar, N>& a)
{
    if(s.B.size() == 0)
        return lbfgsb_line_search_failure<Scalar, N>(s, a);
    s.B.reset();
    std::optional<direction_result<Scalar, N>> dir = compute_direction<P, Scalar, N>(
        s.x, s.g, s.lower, s.upper, s.B, s.gcp_solver, s.ssm_solver);
    if(!dir)
        return std::nullopt;
    step_result<Scalar> r = attempt_lbfgsb_step<P>(s, dir->d, dir->alpha_max, options, a)
        ? commit_lbfgsb_step<Scalar, N>(s, a)
        : lbfgsb_line_search_failure<Scalar, N>(s, a);
    r.diagnostics.bfgs_reset_count = 1;
    return r;
}

}

#endif
