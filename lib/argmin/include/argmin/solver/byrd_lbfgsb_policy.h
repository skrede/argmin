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

template <int N = dynamic_dimension>
struct byrd_lbfgsb_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = byrd_lbfgsb_policy<M>;

    struct options_type
    {
        line_search_options line_search{};
        lbfgsb_line_search line_search_type{lbfgsb_line_search::armijo};
        std::uint16_t stall_window{50};
    };

    options_type options{};

    template <typename P = void>
    struct state_type
    {
        const P* problem{nullptr};
        Eigen::Vector<double, N> x;
        Eigen::Vector<double, N> g;
        Eigen::Vector<double, N> lower;
        Eigen::Vector<double, N> upper;
        double objective_value{};
        detail::compact_lbfgs<double, N, 5> B;
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
                    const solver_options<Convergence>& /*opts*/)
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

        s.B = detail::compact_lbfgs<double, N, 5>{};
        s.gcp_solver = detail::cauchy_point_solver<double, N>{n};
        s.ssm_solver = detail::subspace_minimizer<double, N>{n};
        s.iteration = 0;

        return s;
    }

    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        if(s.iteration != 0)
            s.problem->gradient(s.x, s.g);

        // Direction computation: compile-time unconstrained fast path,
        // runtime all-free fast path, or full GCP + subspace minimization.
        auto dir = detail::compute_direction<P, double, N>(
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
            auto kkt = detail::kkt_residual_bound<double, N>(
                s.x, s.g, s.lower, s.upper);
            return step_result<double>{
                .objective_value = s.objective_value,
                .gradient_norm = s.g.norm(),
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
                .is_null_step = true,
                .x_norm = s.x.norm(),
                .kkt_residual = kkt,
                .policy_status = solver_status::roundoff_limited,
            };
        }
        auto& [d, alpha_max] = *dir;

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

        // Line search dispatch (Armijo default for Byrd variant).
        line_search_options ls_opts = options.line_search;
        ls_opts.max_alpha = std::min(ls_opts.max_alpha, alpha_max);
        double dphi0 = s.g.dot(d);
        auto ls = (options.line_search_type == lbfgsb_line_search::armijo)
            ? armijo(phi, s.objective_value, dphi0, ls_opts)
            : strong_wolfe(phi, dphi, s.objective_value, dphi0, ls_opts);

        Eigen::Vector<double, N> x_old = s.x;
        double old_f = s.objective_value;
        if constexpr(bound_constrained<P>)
            s.x = detail::project((s.x + ls.alpha * d).eval(), s.lower, s.upper);
        else
            s.x = (s.x + ls.alpha * d).eval();

        s.objective_value = s.problem->value(s.x);

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

        Eigen::Vector<double, N> sk = (s.x - x_old).eval();
        Eigen::Vector<double, N> yk = (new_g - s.g).eval();
        s.B.push(sk, yk);

        s.g = new_g;
        ++s.iteration;

        double step_norm = sk.norm();
        double x_norm = s.x.norm();

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
        };
    }

    template <typename P>
    void reset(state_type<P>& s, Eigen::Ref<const Eigen::Vector<double, N>> x0)
    {
        s.x = x0;
        s.objective_value = s.problem->value(x0);
        s.problem->gradient(x0, s.g);
        s.iteration = 0;
    }

    template <typename P>
    void reset_clear(state_type<P>& s, Eigen::Ref<const Eigen::Vector<double, N>> x0)
    {
        reset(s, x0);
        s.B.reset();
    }
};

}

#endif
