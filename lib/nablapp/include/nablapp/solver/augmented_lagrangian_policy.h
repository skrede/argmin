#ifndef HPP_GUARD_NABLAPP_SOLVER_AUGMENTED_LAGRANGIAN_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_AUGMENTED_LAGRANGIAN_POLICY_H

// Augmented Lagrangian (method of multipliers) wrapper policy.
//
// Converts any unconstrained or bound-constrained inner policy (e.g.,
// lbfgsb_policy, bobyqa_policy) into a constrained solver. Each outer
// step() performs a full inner solve on the augmented Lagrangian
// subproblem followed by a multiplier/penalty update. The inner
// subproblem satisfies differentiable + bound_constrained so that
// gradient-based and derivative-free inner solvers both work.
//
// Reference: K&W Section 10.9, Algorithm 10.2;
//            N&W Section 17.4, Algorithm 17.4.

#include "nablapp/detail/augmented_lagrangian.h"
#include "nablapp/detail/lagrangian.h"
#include "nablapp/detail/bound_projection.h"
#include "nablapp/solver/lbfgsb_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/solver/options.h"
#include "nablapp/result/step_result.h"
#include "nablapp/types.h"

#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace nablapp
{

template <typename InnerPolicy = lbfgsb_policy<>, int N = dynamic_dimension>
struct augmented_lagrangian_policy
{
    using scalar_type = typename InnerPolicy::scalar_type;

    template <int M>
    using rebind = augmented_lagrangian_policy<InnerPolicy, M>;

    struct options_type
    {
        std::optional<scalar_type> mu_init{};                      // default: 0.1 (Conn-Gould-Toint)
        std::optional<scalar_type> mu_decrease{};                  // default: 0.25 (Conn-Gould-Toint)
        std::optional<scalar_type> mu_min{};                       // default: 1e-6 (Conn-Gould-Toint)
        std::optional<std::uint32_t> inner_max_iterations{};       // default: 200
        std::optional<std::uint32_t> max_outer_iterations{};       // default: 50 (K&W 10.9)
        std::optional<scalar_type> constraint_tolerance{};         // default: 1e-6 (K&W 10.9)
        std::optional<scalar_type> inner_gradient_tolerance{};     // default: 1e-6 (K&W 10.9)
        std::optional<scalar_type> feasibility_progress{};         // default: 0.25 (N&W 17.4)
        typename InnerPolicy::options_type inner_opts = {};
    };

    options_type options{};

    template <typename P = void>
    struct state_type
    {
        const P* problem{nullptr};
        Eigen::Vector<scalar_type, N> x;
        scalar_type f{};
        Eigen::VectorX<scalar_type> c_eq;
        Eigen::VectorX<scalar_type> c_ineq;

        Eigen::VectorX<scalar_type> lambda_eq;
        Eigen::VectorX<scalar_type> lambda_ineq;

        scalar_type mu{};
        scalar_type prev_viol{};

        Eigen::Vector<scalar_type, N> lower;
        Eigen::Vector<scalar_type, N> upper;

        int n_eq = 0;
        int n_ineq = 0;
        std::uint32_t outer_iter = 0;

        options_type opts;
    };

    template <typename Problem, typename Convergence>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<scalar_type, N>& x0,
                             const solver_options<Convergence>& solver_opts,
                             const options_type& policy_opts)
    {
        options = policy_opts;
        auto s = init(problem, x0, solver_opts);
        s.opts = policy_opts;
        s.mu = policy_opts.mu_init.value_or(scalar_type(0.1));
        s.prev_viol = detail::constraint_violation(s.c_eq, s.c_ineq);
        return s;
    }

    template <typename Problem, typename Convergence = default_convergence>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<scalar_type, N>& x0,
                             const solver_options<Convergence>& /*solver_opts*/)
    {
        static_assert(constrained<Problem, scalar_type>,
                      "augmented_lagrangian_policy requires constrained<Problem>");

        const int n = problem.dimension();
        state_type<Problem> s;
        s.problem = &problem;
        s.opts = options;

        s.n_eq = problem.num_equality();
        s.n_ineq = problem.num_inequality();

        s.x = x0;
        s.f = problem.value(x0);

        // Evaluate constraints
        const int m = s.n_eq + s.n_ineq;
        Eigen::VectorX<scalar_type> c_all(m);
        if(m > 0)
            problem.constraints(x0, c_all);
        s.c_eq = c_all.head(s.n_eq);
        s.c_ineq = c_all.tail(s.n_ineq);

        // Initialize multipliers to zero
        s.lambda_eq = Eigen::VectorX<scalar_type>::Zero(s.n_eq);
        s.lambda_ineq = Eigen::VectorX<scalar_type>::Zero(s.n_ineq);

        // Penalty
        s.mu = s.opts.mu_init.value_or(scalar_type(0.1));
        s.prev_viol = detail::constraint_violation(s.c_eq, s.c_ineq);
        s.outer_iter = 0;

        // Box bounds
        if constexpr(bound_constrained<Problem, scalar_type>)
        {
            s.lower = problem.lower_bounds();
            s.upper = problem.upper_bounds();
        }
        else
        {
            constexpr scalar_type inf = std::numeric_limits<scalar_type>::infinity();
            s.lower = Eigen::Vector<scalar_type, N>::Constant(n, -inf);
            s.upper = Eigen::Vector<scalar_type, N>::Constant(n, inf);
        }

        return s;
    }

    template <typename P>
    step_result<scalar_type> step(state_type<P>& s)
    {
        const int n = static_cast<int>(s.x.size());

        const scalar_type inner_grad_tol = s.opts.inner_gradient_tolerance.value_or(scalar_type(1e-6));
        const std::uint32_t inner_max = s.opts.inner_max_iterations.value_or(200);
        const scalar_type mu_dec = s.opts.mu_decrease.value_or(scalar_type(0.25));
        const scalar_type mu_floor = s.opts.mu_min.value_or(scalar_type(1e-6));
        const scalar_type feas_progress = s.opts.feasibility_progress.value_or(scalar_type(0.25));
        const scalar_type con_tol = s.opts.constraint_tolerance.value_or(scalar_type(1e-6));

        // Evaluate constraints at current x
        {
            const int m = s.n_eq + s.n_ineq;
            Eigen::VectorX<scalar_type> c_all(m);
            if(m > 0)
                s.problem->constraints(s.x, c_all);
            s.c_eq = c_all.head(s.n_eq);
            s.c_ineq = c_all.tail(s.n_ineq);
        }

        // Synthetic subproblem for the inner solver.
        // Stores const P* and penalty state; satisfies differentiable + bound_constrained.
        struct subproblem
        {
            enum : int { problem_dimension = N };
            const P* outer;
            int dim;
            const Eigen::Vector<scalar_type, N>* lo;
            const Eigen::Vector<scalar_type, N>* hi;
            const Eigen::VectorX<scalar_type>* lam_eq;
            const Eigen::VectorX<scalar_type>* lam_ineq;
            scalar_type pen;
            int neq, nineq;

            [[nodiscard]] int dimension() const { return dim; }

            [[nodiscard]] scalar_type value(const Eigen::Vector<scalar_type, N>& x) const
            {
                scalar_type fval = outer->value(x);
                const int m = neq + nineq;
                Eigen::VectorX<scalar_type> c_all(m);
                if(m > 0)
                    outer->constraints(x, c_all);
                Eigen::VectorX<scalar_type> ceq = c_all.head(neq);
                Eigen::VectorX<scalar_type> cineq = c_all.tail(nineq);
                return detail::augmented_lagrangian_value(
                    fval, ceq, cineq, *lam_eq, *lam_ineq, pen);
            }

            void gradient(const Eigen::Vector<scalar_type, N>& x,
                          Eigen::Vector<scalar_type, N>& g) const
            {
                constexpr int PD = P::problem_dimension;
                if constexpr(N == PD)
                    outer->gradient(x, g);
                else
                {
                    Eigen::Vector<scalar_type, PD> g_tmp(g.size());
                    outer->gradient(Eigen::VectorX<scalar_type>(x), g_tmp);
                    g = g_tmp;
                }
                const int m = neq + nineq;
                Eigen::VectorX<scalar_type> c_all(m);
                Eigen::MatrixX<scalar_type> J_all(m, dim);
                if(m > 0)
                {
                    outer->constraints(x, c_all);
                    outer->constraint_jacobian(x, J_all);
                }
                Eigen::Matrix<scalar_type, Eigen::Dynamic, N> Jeq = J_all.topRows(neq);
                Eigen::Matrix<scalar_type, Eigen::Dynamic, N> Jineq = J_all.bottomRows(nineq);
                Eigen::VectorX<scalar_type> ceq = c_all.head(neq);
                Eigen::VectorX<scalar_type> cineq = c_all.tail(nineq);
                g = detail::augmented_lagrangian_gradient(
                    g, Jeq, Jineq, ceq, cineq, *lam_eq, *lam_ineq, pen);
            }

            [[nodiscard]] Eigen::Vector<scalar_type, N> lower_bounds() const { return *lo; }
            [[nodiscard]] Eigen::Vector<scalar_type, N> upper_bounds() const { return *hi; }
        };

        subproblem sub{
            .outer = s.problem,
            .dim = n,
            .lo = &s.lower,
            .hi = &s.upper,
            .lam_eq = &s.lambda_eq,
            .lam_ineq = &s.lambda_ineq,
            .pen = s.mu,
            .neq = s.n_eq,
            .nineq = s.n_ineq,
        };

        // Inner solver: solve the augmented Lagrangian subproblem
        solver_options<> inner_opts;
        inner_opts.max_iterations = inner_max;
        std::get<gradient_tolerance_criterion>(inner_opts.convergence.criteria)
            .threshold = inner_grad_tol;
        std::get<objective_tolerance_criterion>(inner_opts.convergence.criteria)
            .threshold = scalar_type(1e-15);
        std::get<step_tolerance_criterion>(inner_opts.convergence.criteria)
            .threshold = scalar_type(1e-15);

        using rebound_inner = typename InnerPolicy::template rebind<N>;
        basic_solver<rebound_inner, N, subproblem> inner_solver{sub, s.x, inner_opts};
        auto inner_result = inner_solver.solve(inner_opts);

        // Extract solution from inner solver
        scalar_type f_old = s.f;
        Eigen::Vector<scalar_type, N> x_old = s.x;
        s.x = inner_result.x;
        s.f = s.problem->value(s.x);
        {
            const int m = s.n_eq + s.n_ineq;
            Eigen::VectorX<scalar_type> c_all(m);
            if(m > 0)
                s.problem->constraints(s.x, c_all);
            s.c_eq = c_all.head(s.n_eq);
            s.c_ineq = c_all.tail(s.n_ineq);
        }

        // Constraint violation
        scalar_type viol = detail::constraint_violation(s.c_eq, s.c_ineq);

        scalar_type step_norm = (s.x - x_old).norm();

        // Actual augmented Lagrangian gradient norm at current iterate
        scalar_type reported_grad_norm;
        if constexpr(differentiable<P, scalar_type>)
        {
            constexpr int PD = P::problem_dimension;
            Eigen::Vector<scalar_type, N> grad_f(n);
            if constexpr(N == PD)
                s.problem->gradient(s.x, grad_f);
            else
            {
                Eigen::Vector<scalar_type, PD> g_tmp(n);
                s.problem->gradient(Eigen::VectorX<scalar_type>(s.x), g_tmp);
                grad_f = g_tmp;
            }

            const int m = s.n_eq + s.n_ineq;
            Eigen::MatrixX<scalar_type> J_all(m, n);
            if(m > 0)
                s.problem->constraint_jacobian(s.x, J_all);
            Eigen::Matrix<scalar_type, Eigen::Dynamic, N> J_eq_new = J_all.topRows(s.n_eq);
            Eigen::Matrix<scalar_type, Eigen::Dynamic, N> J_ineq_new = J_all.bottomRows(s.n_ineq);

            auto aug_grad = detail::augmented_lagrangian_gradient(
                grad_f, J_eq_new, J_ineq_new, s.c_eq, s.c_ineq,
                s.lambda_eq, s.lambda_ineq, s.mu);
            reported_grad_norm = aug_grad.norm();
        }
        else
        {
            reported_grad_norm = step_norm;
        }

        // K&W Algorithm 10.2 / N&W Algorithm 17.4:
        // Conn, Gould & Toint (1991) recommend updating multipliers only
        // when sufficient feasibility progress is observed, and decreasing
        // the penalty parameter otherwise.
        if(viol < feas_progress * s.prev_viol || s.outer_iter == 0)
        {
            detail::update_multipliers(
                s.lambda_eq, s.lambda_ineq, s.c_eq, s.c_ineq, s.mu);
        }
        else
        {
            s.mu = std::max(s.mu * mu_dec, mu_floor);
        }
        s.prev_viol = viol;

        ++s.outer_iter;

        return step_result<scalar_type>{
            .objective_value = s.f,
            .gradient_norm = reported_grad_norm,
            .step_size = step_norm,
            .objective_change = s.f - f_old,
            .improved = s.f < f_old || viol < con_tol,
            .constraint_violation = viol,
            .x_norm = s.x.norm(),
        };
    }

    template <typename P>
    void reset(state_type<P>& s, const Eigen::Vector<scalar_type, N>& x0)
    {
        s.x = x0;
        s.f = s.problem->value(x0);
        {
            const int m = s.n_eq + s.n_ineq;
            Eigen::VectorX<scalar_type> c_all(m);
            if(m > 0)
                s.problem->constraints(x0, c_all);
            s.c_eq = c_all.head(s.n_eq);
            s.c_ineq = c_all.tail(s.n_ineq);
        }
        s.outer_iter = 0;
    }

    template <typename P>
    void reset_clear(state_type<P>& s,
                     const Eigen::Vector<scalar_type, N>& x0)
    {
        reset(s, x0);
        s.lambda_eq.setZero();
        s.lambda_ineq.setZero();
        s.mu = s.opts.mu_init.value_or(scalar_type(0.1));
        s.prev_viol = detail::constraint_violation(s.c_eq, s.c_ineq);
    }
};

}

#endif
