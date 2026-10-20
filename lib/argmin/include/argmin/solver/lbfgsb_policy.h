#ifndef HPP_GUARD_ARGMIN_SOLVER_LBFGSB_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_LBFGSB_POLICY_H

// L-BFGS-B solver policy for step_budget_solver.
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
// with projected steps.
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
#include "argmin/detail/lbfgsb_step_attempt.h"
#include "argmin/line_search/armijo.h"
#include "argmin/line_search/strong_wolfe.h"
#include "argmin/line_search/options.h"
#include "argmin/result/step_result.h"
#include "argmin/solver/options.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
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

template <scalar Scalar = double, int N = dynamic_dimension>
struct lbfgsb_policy
{
    using scalar_type = Scalar;

    template <int M>
    using rebind = lbfgsb_policy<Scalar, M>;

    struct options_type
    {
        line_search_options<Scalar> line_search{};  // Embedded line search params
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
        Eigen::Vector<Scalar, N> x;
        Eigen::Vector<Scalar, N> g;
        Eigen::Vector<Scalar, N> lower;
        Eigen::Vector<Scalar, N> upper;
        Scalar objective_value{};
        detail::compact_lbfgs<Scalar, N, 10> B;
        detail::cauchy_point_solver<Scalar, N> gcp_solver;
        detail::subspace_minimizer<Scalar, N> ssm_solver;
        std::uint32_t iteration{0};
    };

    template <typename Problem, typename Convergence>
    state_type<Problem> init(const Problem& problem,
                    const Eigen::Vector<Scalar, N>& x0,
                    const solver_options<Convergence>& opts, const options_type& policy_opts)
    {
        options = policy_opts;
        return init(problem, x0, opts);
    }

    template <typename Problem, typename Convergence = default_convergence<scalar_type>>
    state_type<Problem> init(const Problem& problem,
                    const Eigen::Vector<Scalar, N>& x0,
                    const solver_options<Convergence>& /*opts*/)
    {
        const int n = problem.dimension();
        state_type<Problem> s;
        s.problem = &problem;

        s.x = x0;
        s.g.setZero(n);
        s.objective_value = problem.value(x0);
        problem.gradient(x0, s.g);

        if constexpr(bound_constrained<Problem, Scalar>)
        {
            s.lower = problem.lower_bounds();
            s.upper = problem.upper_bounds();
        }
        else
        {
            constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
            s.lower = Eigen::Vector<Scalar, N>::Constant(n, -inf);
            s.upper = Eigen::Vector<Scalar, N>::Constant(n, inf);
        }

        s.B = detail::compact_lbfgs<Scalar, N, 10>{};
        s.gcp_solver = detail::cauchy_point_solver<Scalar, N>{n};
        s.ssm_solver = detail::subspace_minimizer<Scalar, N>{n};
        s.iteration = 0;

        return s;
    }

    template <typename P>
    step_result<Scalar> step(state_type<P>& s)
    {
        // Evaluate gradient (skip on first iteration -- init already computed it)
        if(s.iteration != 0)
            s.problem->gradient(s.x, s.g);

        auto dir = detail::compute_direction<P, Scalar, N>(
            s.x, s.g, s.lower, s.upper, s.B, s.gcp_solver, s.ssm_solver);
        if(!dir)
            return null_direction_step(s, 0);
        detail::lbfgsb_attempt<Scalar, N> a(s.x.size());
        std::optional<step_result<Scalar>> r;
        if(detail::attempt_lbfgsb_step<P>(s, dir->d, dir->alpha_max, options, a))
            r = detail::commit_lbfgsb_step<P>(s, a, dir->d);
        else
            r = detail::retry_lbfgsb_step<P>(s, options, a);
        step_result<Scalar> result = r ? *r : null_direction_step(s, 1);
        // Every line-search trial plus the objective evaluation at each attempted iterate.
        result.evaluations = a.evaluations;
        return result;
    }

    // Hot start -- preserves curvature pairs.
    template <typename P>
    void reset(state_type<P>& s, Eigen::Ref<const Eigen::Vector<Scalar, N>> x0)
    {
        s.x = x0;
        s.objective_value = s.problem->value(x0);
        s.problem->gradient(x0, s.g);
        s.iteration = 0;
        // B (compact_lbfgs) is NOT reset -- preserves curvature pairs
    }

    // Cold restart -- clears curvature history.
    template <typename P>
    void reset_clear(state_type<P>& s, Eigen::Ref<const Eigen::Vector<Scalar, N>> x0)
    {
        reset(s, x0);
        s.B.reset();
    }

private:
    template <typename P>
    step_result<Scalar> null_direction_step(const state_type<P>& s, std::size_t resets) const
    {
        // KKT residual for bound-constrained optimality: projected-gradient
        // infinity-norm. For unconstrained problems, s.lower/s.upper are
        // +/-inf and the expression collapses to ||s.g||_inf.
        // Reference: N&W 2e Section 16.7 (projected gradient optimality).
        auto kkt = detail::kkt_residual_bound<Scalar, N>(
            s.x, s.g, s.lower, s.upper);
        return step_result<Scalar>{
            .objective_value = s.objective_value,
            .gradient_norm = s.g.norm(),
            .step_size = Scalar(0.0),
            .objective_change = Scalar(0.0),
            .improved = false,
            .x_norm = s.x.norm(),
            .kkt_residual = kkt,
            .evaluations = 0,
            .diagnostics = {.bfgs_reset_count = resets},
        };
    }
};

}

#endif
