#ifndef HPP_GUARD_ARGMIN_SOLVER_LBFGSB_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_LBFGSB_POLICY_H

// L-BFGS-B solver policy for basic_solver.
//
// Implements the Limited-memory BFGS algorithm for Bound-constrained
// optimization. Each step computes: (1) Generalized Cauchy Point via
// breakpoint search along the projected gradient path, (2) subspace
// minimization over free variables using the compact L-BFGS reduced
// Hessian, (3) strong Wolfe line search capped at the maximum feasible
// step length.
//
// When the problem satisfies only differentiable (no bounds), all bounds
// default to +/-infinity and the algorithm reduces to standard L-BFGS
// with projected steps (D-04).
//
// Reference: Byrd, Lu, Nocedal, Zhu (1995) "A Limited Memory Algorithm
//            for Bound Constrained Optimization", SIAM J. Sci. Comput.
//            16(5), pp. 1190-1208.
//            N&W Sections 9.2 (compact representation), 16.6 (GCP +
//            subspace minimization).
//            K&W Section 6.5 (quasi-Newton methods).

#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/compact_lbfgs.h"
#include "argmin/detail/bound_projection.h"
#include "argmin/detail/lbfgsb_direction.h"
#include "argmin/line_search/armijo.h"
#include "argmin/line_search/strong_wolfe.h"
#include "argmin/line_search/options.h"
#include "argmin/result/step_result.h"
#include "argmin/solver/options.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace argmin
{

// Line search strategy for L-BFGS-B policies.
// Reference: Byrd et al. 1995 (original uses Armijo), N&W Section 3.1.
enum class lbfgsb_line_search : std::uint8_t
{
    strong_wolfe,
    armijo
};

template <int N = dynamic_dimension>
struct lbfgsb_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = lbfgsb_policy<M>;

    struct options_type
    {
        line_search_options line_search{};             // Embedded line search params
        lbfgsb_line_search line_search_type{lbfgsb_line_search::strong_wolfe};
        std::uint16_t stall_window{50};
    };

    options_type options{};

    // The Problem object passed to init() MUST outlive the solver.
    // This is the same lifetime model as all argmin policies (init takes
    // const Problem&). The problem pointer enables direct inlinable calls
    // without std::function type-erasure overhead.
    template <typename P = void>
    struct state_type
    {
        const P* problem{nullptr};
        Eigen::Vector<double, N> x;
        Eigen::Vector<double, N> g;
        Eigen::Vector<double, N> lower;
        Eigen::Vector<double, N> upper;
        double objective_value{};
        detail::compact_lbfgs<double, N, 10> B;
        detail::cauchy_point_solver<double, N> gcp_solver;
        detail::subspace_minimizer<double, N> ssm_solver;
        std::uint32_t iteration{0};
    };

    template <typename Problem, typename Convergence>
    state_type<Problem> init(const Problem& problem,
                    const Eigen::Vector<double, N>& x0,
                    const solver_options<Convergence>& opts, const options_type& policy_opts)
    {
        options = policy_opts;
        return init(problem, x0, opts);
    }

    template <typename Problem, typename Convergence = default_convergence>
    state_type<Problem> init(const Problem& problem,
                    const Eigen::Vector<double, N>& x0,
                    const solver_options<Convergence>& opts)
    {
        const int n = problem.dimension();
        state_type<Problem> s;
        s.problem = &problem;

        s.x = x0;
        s.g.setZero(n);
        s.objective_value = problem.value(x0);
        problem.gradient(x0, s.g);

        if constexpr(bound_constrained<Problem>)
        {
            s.lower = problem.lower_bounds();
            s.upper = problem.upper_bounds();
        }
        else
        {
            constexpr double inf = std::numeric_limits<double>::infinity();
            s.lower = Eigen::Vector<double, N>::Constant(n, -inf);
            s.upper = Eigen::Vector<double, N>::Constant(n, inf);
        }

        s.B = detail::compact_lbfgs<double, N, 10>{};
        s.gcp_solver = detail::cauchy_point_solver<double, N>{n};
        s.ssm_solver = detail::subspace_minimizer<double, N>{n};
        s.iteration = 0;

        return s;
    }

    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        // Evaluate gradient (skip on first iteration -- init already computed it)
        if(s.iteration != 0)
            s.problem->gradient(s.x, s.g);

        // Direction computation: compile-time unconstrained fast path,
        // runtime all-free fast path, or full GCP + subspace minimization.
        auto dir = detail::compute_direction<P, double, N>(
            s.x, s.g, s.lower, s.upper, s.B, s.gcp_solver, s.ssm_solver);
        if(!dir)
        {
            // KKT residual for bound-constrained optimality: projected-gradient
            // infinity-norm. For unconstrained problems, s.lower/s.upper are
            // +/-inf and the expression collapses to ||s.g||_inf.
            // Reference: N&W 2e Section 16.7 (projected gradient optimality).
            auto kkt = detail::kkt_residual_bound<double, N>(
                s.x, s.g, s.lower, s.upper);
            return step_result<double>{
                .objective_value = s.objective_value,
                .gradient_norm = s.g.norm(),
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
                .x_norm = s.x.norm(),
                .kkt_residual = kkt,
                .evaluations = 0,  // no line search ran on this null step
            };
        }
        auto& result = *dir;
        auto& d = result.d;
        auto& alpha_max = result.alpha_max;

        // Line search functors with gradient caching.
        // The dphi callback computes the gradient at the trial point, which
        // is the same point accepted after line search. Cache it to avoid
        // a redundant gradient evaluation after step acceptance.
        Eigen::Vector<double, N> cached_g(s.x.size());
        double cached_alpha = -1.0;

        auto phi = [&](double a) {
            return s.problem->value((s.x + a * d).eval());
        };
        auto dphi = [&](double a) {
            s.problem->gradient((s.x + a * d).eval(), cached_g);
            cached_alpha = a;
            return cached_g.dot(d);
        };

        // Line search with feasible step cap, using embedded line search options.
        // Dispatch based on configured strategy (Strong Wolfe default, Armijo optional).
        line_search_options ls_opts = options.line_search;
        ls_opts.max_alpha = std::min(ls_opts.max_alpha, alpha_max);
        double dphi0 = s.g.dot(d);
        auto ls = (options.line_search_type == lbfgsb_line_search::armijo)
            ? armijo(phi, s.objective_value, dphi0, ls_opts)
            : strong_wolfe(phi, dphi, s.objective_value, dphi0, ls_opts);

        // Update iterate
        Eigen::Vector<double, N> x_old = s.x;
        double old_f = s.objective_value;
        if constexpr(bound_constrained<P>)
            s.x = detail::project((s.x + ls.alpha * d).eval(), s.lower, s.upper);
        else
            s.x = (s.x + ls.alpha * d).eval();

        // Evaluate objective at projected point (projection may differ from
        // line search trial point due to bound clamping)
        s.objective_value = s.problem->value(s.x);

        // Reuse cached gradient from line search dphi callback if it was
        // evaluated at the accepted alpha and projection didn't change x.
        // Unconstrained: no projection, so cached gradient is always valid.
        Eigen::Vector<double, N> new_g(s.x.size());
        if constexpr(bound_constrained<P>)
        {
            if(cached_alpha == ls.alpha &&
               s.x.isApprox((x_old + ls.alpha * d).eval(), 0.0))
                new_g = cached_g;
            else
                s.problem->gradient(s.x, new_g);
        }
        else
        {
            if(cached_alpha == ls.alpha)
                new_g = cached_g;
            else
                s.problem->gradient(s.x, new_g);
        }

        // Update L-BFGS curvature pairs
        Eigen::Vector<double, N> sk = (s.x - x_old).eval();
        Eigen::Vector<double, N> yk = (new_g - s.g).eval();
        s.B.push(sk, yk);

        // Store new gradient
        s.g = new_g;

        // Increment iteration
        ++s.iteration;

        double step_norm = sk.norm();
        double x_norm = s.x.norm();

        // Roundoff detection (N&W Section 3.4):
        // step too small relative to iterate magnitude
        std::optional<solver_status> policy_status{};
        constexpr double eps = std::numeric_limits<double>::epsilon();
        if(step_norm < eps * std::max(x_norm, 1.0) * 10.0)
            policy_status = solver_status::roundoff_limited;

        // KKT residual for bound-constrained optimality: projected-gradient
        // infinity-norm at the newly-accepted iterate using the freshly-
        // evaluated gradient new_g. For unconstrained problems
        // s.lower/s.upper are +/-inf and this collapses to ||new_g||_inf.
        // Reference: N&W 2e Section 16.7 (projected gradient optimality).
        auto kkt = detail::kkt_residual_bound<double, N>(
            s.x, new_g, s.lower, s.upper);

        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = new_g.norm(),
            .step_size = step_norm,
            .objective_change = s.objective_value - old_f,
            .improved = s.objective_value < old_f,
            .x_norm = x_norm,
            .kkt_residual = kkt,
            .policy_status = policy_status,
            // Line-search trial evaluations plus the objective evaluation at
            // the projected accepted iterate -- the genuine per-step count.
            .evaluations = static_cast<std::uint32_t>(ls.evaluations) + 1u,
        };
    }

    // Hot start -- preserves curvature pairs (D-05).
    template <typename P>
    void reset(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        s.x = x0;
        s.objective_value = s.problem->value(x0);
        s.problem->gradient(x0, s.g);
        s.iteration = 0;
        // B (compact_lbfgs) is NOT reset -- preserves curvature pairs
    }

    // Cold restart -- clears curvature history (D-05).
    template <typename P>
    void reset_clear(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        reset(s, x0);
        s.B.reset();
    }
};

}

#endif
