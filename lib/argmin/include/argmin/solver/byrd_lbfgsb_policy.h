#ifndef HPP_GUARD_ARGMIN_SOLVER_BYRD_LBFGSB_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_BYRD_LBFGSB_POLICY_H

// Byrd 1995 variant L-BFGS-B solver policy for step_budget_solver.
//
// Structurally identical to lbfgsb_policy but with different defaults
// motivated by non-convex landscapes (cartan IK benchmarks):
//   - Armijo backtracking line search (instead of Strong Wolfe)
//   - 5-pair curvature history (instead of 10)
//
// The Armijo line search avoids the curvature condition that can
// over-constrain step acceptance on non-smooth or non-convex objectives.
// Shorter history (m=5) reduces the cost of each two-loop recursion
// and limits stale curvature influence.
//
// Reference: Byrd, Lu, Nocedal, Zhu (1995) "A Limited Memory Algorithm
//            for Bound Constrained Optimization", SIAM J. Sci. Comput.
//            16(5), pp. 1190-1208.
//            N&W Sections 9.2 (compact representation), 16.6 (GCP +
//            subspace minimization).

#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/lbfgsb_direction.h"

#include "argmin/solver/lbfgsb_policy.h"

namespace argmin
{

template <scalar Scalar = double, int N = dynamic_dimension>
struct byrd_lbfgsb_policy
{
    using scalar_type = Scalar;

    template <int M>
    using rebind = byrd_lbfgsb_policy<Scalar, M>;

    struct options_type
    {
        line_search_options<Scalar> line_search{};
        lbfgsb_line_search line_search_type{lbfgsb_line_search::armijo};
        std::uint16_t stall_window{50};
    };

    options_type options{};

    template <typename P = void>
    struct state_type
    {
        const P* problem{nullptr};
        Eigen::Vector<Scalar, N> x;
        Eigen::Vector<Scalar, N> g;
        Eigen::Vector<Scalar, N> lower;
        Eigen::Vector<Scalar, N> upper;
        Scalar objective_value{};
        detail::compact_lbfgs<Scalar, N, 5> B;
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

        s.B = detail::compact_lbfgs<Scalar, N, 5>{};
        s.gcp_solver = detail::cauchy_point_solver<Scalar, N>{n};
        s.ssm_solver = detail::subspace_minimizer<Scalar, N>{n};
        s.iteration = 0;

        return s;
    }

    template <typename P>
    step_result<Scalar> step(state_type<P>& s)
    {
        if(s.iteration != 0)
            s.problem->gradient(s.x, s.g);

        // Direction computation: compile-time unconstrained fast path,
        // runtime all-free fast path, or full GCP + subspace minimization.
        auto dir = detail::compute_direction<P, Scalar, N>(
            s.x, s.g, s.lower, s.upper, s.B, s.gcp_solver, s.ssm_solver);
        if(!dir)
        {
            // Null step: compute_direction returned nullopt because
            // GCP breakpoint exhaustion or subspace solve degeneracy
            // left no usable direction. This is the canonical
            // roundoff-floor signal on badly-scaled problems. Report
            // roundoff_limited rather than running to max_iterations
            // silently; the accepted-step path at lines 200-203 uses
            // the same status for the sibling "step too small relative
            // to iterate magnitude" roundoff case. is_null_step stays
            // true so step_tolerance_criterion still exempts this
            // iterate from stall detection; kkt_residual is populated
            // via the projected-gradient infinity-norm so
            // objective_tolerance_criterion can still declare
            // convergence when the iterate is a genuine KKT point (a
            // zero direction at a KKT point is the correct answer).
            //
            // Reference: Byrd, Lu, Nocedal, Zhu 1995 Algorithm CP (GCP
            //            breakpoint exhaustion); N&W 2e Section 3.5
            //            (roundoff limitation in line search); N&W 2e
            //            Section 10.7 (problem scaling is the user's
            //            responsibility, so the honest response is an
            //            explicit termination status rather than a
            //            rescaled metric); N&W 2e Section 16.7
            //            (projected gradient optimality for
            //            bound-constrained first-order conditions).
            auto kkt = detail::kkt_residual_bound<Scalar, N>(
                s.x, s.g, s.lower, s.upper);
            return step_result<Scalar>{
                .objective_value = s.objective_value,
                .gradient_norm = s.g.norm(),
                .step_size = Scalar(0.0),
                .objective_change = Scalar(0.0),
                .improved = false,
                .is_null_step = true,
                .x_norm = s.x.norm(),
                .kkt_residual = kkt,
                .policy_status = solver_status::roundoff_limited,
            };
        }
        auto& [d, alpha_max] = *dir;

        Eigen::Vector<Scalar, N> cached_g = Eigen::Vector<Scalar, N>::Zero(s.x.size());
        Scalar cached_alpha = Scalar(-1.0);

        auto phi = [&](Scalar a) {
            return s.problem->value((s.x + a * d).eval());
        };
        auto dphi = [&](Scalar a) {
            s.problem->gradient((s.x + a * d).eval(), cached_g);
            cached_alpha = a;
            return cached_g.dot(d);
        };

        // Line search dispatch (Armijo default for Byrd variant).
        line_search_options<Scalar> ls_opts = options.line_search;
        ls_opts.max_alpha = std::min(ls_opts.max_alpha, alpha_max);
        Scalar dphi0 = s.g.dot(d);
        auto ls = (options.line_search_type == lbfgsb_line_search::armijo)
            ? armijo(phi, s.objective_value, dphi0, ls_opts)
            : strong_wolfe(phi, dphi, s.objective_value, dphi0, ls_opts);

        Eigen::Vector<Scalar, N> x_old = s.x;
        Scalar old_f = s.objective_value;
        if constexpr(bound_constrained<P, Scalar>)
            s.x = detail::project((s.x + ls.alpha * d).eval(), s.lower, s.upper);
        else
            s.x = (s.x + ls.alpha * d).eval();

        s.objective_value = s.problem->value(s.x);

        Eigen::Vector<Scalar, N> new_g = Eigen::Vector<Scalar, N>::Zero(s.x.size());
        if constexpr(bound_constrained<P, Scalar>)
        {
            if(cached_alpha == ls.alpha &&
               s.x.isApprox((x_old + ls.alpha * d).eval(), Scalar(0.0)))
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

        Eigen::Vector<Scalar, N> sk = (s.x - x_old).eval();
        Eigen::Vector<Scalar, N> yk = (new_g - s.g).eval();
        s.B.push(sk, yk);

        s.g = new_g;
        ++s.iteration;

        Scalar step_norm = sk.norm();
        Scalar x_norm = s.x.norm();

        std::optional<solver_status> policy_status{};
        constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();
        if(step_norm < eps * std::max(x_norm, Scalar(1.0)) * Scalar(10.0))
            policy_status = solver_status::roundoff_limited;

        // KKT residual for bound-constrained optimality: projected-gradient
        // infinity-norm at the newly-accepted iterate using the freshly-
        // evaluated gradient new_g. For unconstrained problems
        // s.lower/s.upper are +/-inf and this collapses to ||new_g||_inf.
        // Reference: N&W 2e Section 16.7 (projected gradient optimality).
        auto kkt = detail::kkt_residual_bound<Scalar, N>(
            s.x, new_g, s.lower, s.upper);

        return step_result<Scalar>{
            .objective_value = s.objective_value,
            .gradient_norm = new_g.norm(),
            .step_size = step_norm,
            .objective_change = s.objective_value - old_f,
            .improved = s.objective_value < old_f,
            .x_norm = x_norm,
            .kkt_residual = kkt,
            .policy_status = policy_status,
        };
    }

    template <typename P>
    void reset(state_type<P>& s, Eigen::Ref<const Eigen::Vector<Scalar, N>> x0)
    {
        s.x = x0;
        s.objective_value = s.problem->value(x0);
        s.problem->gradient(x0, s.g);
        s.iteration = 0;
    }

    template <typename P>
    void reset_clear(state_type<P>& s, Eigen::Ref<const Eigen::Vector<Scalar, N>> x0)
    {
        reset(s, x0);
        s.B.reset();
    }
};

}

#endif
