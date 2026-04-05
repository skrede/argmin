#ifndef HPP_GUARD_NABLAPP_SOLVER_MMA_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_MMA_POLICY_H

// Method of Moving Asymptotes (MMA) policy.
//
// Solves inequality-constrained optimization problems by iteratively
// building convex separable reciprocal approximations around moving
// asymptotes and solving the resulting dual subproblem. Equality
// constraints are rejected (D-06).
//
// Reference: Svanberg 1987, "The method of moving asymptotes --
//            a new method for structural optimization".

#include "nablapp/detail/asymptote_update.h"
#include "nablapp/detail/mma_subproblem.h"
#include "nablapp/detail/lagrangian.h"
#include "nablapp/options/asymptote_options.h"
#include "nablapp/options/mma_subproblem_options.h"
#include "nablapp/result/step_result.h"
#include "nablapp/solver/options.h"
#include "nablapp/types.h"

#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>

namespace nablapp
{

template <int N = dynamic_dimension>
struct mma_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = mma_policy<M>;

    struct options_type
    {
        std::optional<double> move_limit{};              // default: 0.5 (Svanberg 1987)
        std::optional<double> asymptote_init{};          // default: 0.5 (Svanberg 1987)
        std::optional<double> asymptote_decrease{};      // default: 0.7 (Svanberg 1987)
        std::optional<double> asymptote_increase{};      // default: 1.2 (Svanberg 1987)
        std::optional<double> kkt_tolerance{};           // default: 1e-6 (Svanberg 1987)
        std::optional<double> effective_bounds_scale{};  // default: 10.0
        asymptote_options asymptote{};                   // Embedded asymptote update params
        mma_subproblem_options subproblem{};              // Embedded subproblem params
    };

    struct state_type
    {
        Eigen::Vector<double, N> x, g;
        double f{};
        Eigen::VectorXd c_eq, c_ineq;
        Eigen::MatrixXd J_ineq;

        Eigen::Vector<double, N> L, U;
        Eigen::Vector<double, N> x_old1, x_old2;
        Eigen::Vector<double, N> lower, upper;

        std::optional<detail::mma_subproblem_solver<double, N>> subproblem;
        std::uint32_t iteration{0};
        options_type opts;

        std::function<double(const Eigen::Vector<double, N>&)> eval_value;
        std::function<void(const Eigen::Vector<double, N>&,
                           Eigen::Vector<double, N>&)> eval_gradient;
        std::function<void(const Eigen::Vector<double, N>&, Eigen::VectorXd&,
                           Eigen::VectorXd&)> eval_constraints;
        std::function<void(const Eigen::Vector<double, N>&, Eigen::MatrixXd&,
                           Eigen::MatrixXd&)> eval_jacobian;
    };

    template <typename Problem, typename Convergence>
    state_type init(const Problem& problem,
                    const Eigen::Vector<double, N>& x0,
                    const solver_options<Convergence>& opts,
                    const options_type& policy_opts)
    {
        auto s = init(problem, x0, opts);
        s.opts = policy_opts;
        // Re-initialize asymptotes with new opts
        const int n = problem.dimension();
        double asym_init = policy_opts.asymptote_init.value_or(0.5);
        constexpr double inf_val = std::numeric_limits<double>::infinity();
        for(int j = 0; j < n; ++j)
        {
            double half;
            if(s.lower[j] > -inf_val && s.upper[j] < inf_val)
                half = asym_init * (s.upper[j] - s.lower[j]);
            else
                half = 1.0;
            s.L[j] = x0[j] - half;
            s.U[j] = x0[j] + half;
        }
        return s;
    }

    template <typename Problem, typename Convergence = default_convergence>
    state_type init(const Problem& problem,
                    const Eigen::Vector<double, N>& x0,
                    const solver_options<Convergence>& /*opts*/)
    {
        static_assert(differentiable<Problem>,
                      "mma_policy requires differentiable<Problem>");
        static_assert(constrained<Problem>,
                      "mma_policy requires constrained<Problem>");

        // D-06: MMA handles inequality constraints only.
        // num_equality() is not constexpr, so checked at runtime.
        assert(problem.num_equality() == 0
               && "MMA handles inequality constraints only. "
                  "Use SQP or augmented Lagrangian for equality constraints.");

        const int n = problem.dimension();
        const int m_ineq = problem.num_inequality();
        state_type s;

        s.x = x0;
        s.g.resize(n);
        s.c_eq = Eigen::VectorXd{};
        s.c_ineq.resize(m_ineq);
        s.J_ineq.resize(m_ineq, n);

        // Evaluate at x0
        s.f = problem.value(x0);
        problem.gradient(x0, s.g);

        Eigen::VectorXd c_all(m_ineq);
        if(m_ineq > 0)
            problem.constraints(x0, c_all);
        s.c_ineq = c_all;

        Eigen::MatrixXd J_all(m_ineq, n);
        if(m_ineq > 0)
            problem.constraint_jacobian(x0, J_all);
        s.J_ineq = J_all;

        // Box bounds
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

        // Initialize asymptotes
        s.L.resize(n);
        s.U.resize(n);
        s.x_old1 = x0;
        s.x_old2 = x0;

        double asym_init = s.opts.asymptote_init.value_or(0.5);
        constexpr double inf = std::numeric_limits<double>::infinity();
        for(int j = 0; j < n; ++j)
        {
            double half;
            if(s.lower[j] > -inf && s.upper[j] < inf)
                half = asym_init * (s.upper[j] - s.lower[j]);
            else
                half = 1.0;
            s.L[j] = x0[j] - half;
            s.U[j] = x0[j] + half;
        }

        s.iteration = 0;
        s.subproblem.emplace(n, m_ineq);

        // Closures capturing problem by const reference
        s.eval_value = [&problem](const Eigen::Vector<double, N>& v) {
            return problem.value(v);
        };
        s.eval_gradient = [&problem](const Eigen::Vector<double, N>& v,
                                     Eigen::Vector<double, N>& grad) {
            problem.gradient(v, grad);
        };

        const int n_ineq_cap = m_ineq;
        s.eval_constraints = [&problem, n_ineq_cap](
            const Eigen::Vector<double, N>& v,
            Eigen::VectorXd& ceq, Eigen::VectorXd& cineq)
        {
            ceq = Eigen::VectorXd{};
            cineq.resize(n_ineq_cap);
            if(n_ineq_cap > 0)
                problem.constraints(v, cineq);
        };

        s.eval_jacobian = [&problem, n_ineq_cap](
            const Eigen::Vector<double, N>& v,
            Eigen::MatrixXd& Jeq, Eigen::MatrixXd& Jineq)
        {
            Jeq = Eigen::MatrixXd{};
            Jineq.resize(n_ineq_cap, v.size());
            if(n_ineq_cap > 0)
                problem.constraint_jacobian(v, Jineq);
        };

        return s;
    }

    step_result<double> step(state_type& s)
    {
        const int n = static_cast<int>(s.x.size());
        const int m = static_cast<int>(s.c_ineq.size());

        double move_lim = s.opts.move_limit.value_or(0.5);
        double asym_init = s.opts.asymptote_init.value_or(0.5);
        double asym_dec = s.opts.asymptote_decrease.value_or(0.7);
        double asym_inc = s.opts.asymptote_increase.value_or(1.2);
        double eff_scale = s.opts.effective_bounds_scale.value_or(2.0);

        // State (f, g, c_ineq, J_ineq) is current: init() evaluates at x0,
        // and each step evaluates at the accepted iterate before returning.

        // 1. Update asymptotes, passing embedded options
        Eigen::Vector<double, N> x_min_eff = effective_bounds(s.lower, s.x, n, false, eff_scale);
        Eigen::Vector<double, N> x_max_eff = effective_bounds(s.upper, s.x, n, true, eff_scale);

        detail::update_asymptotes(
            s.L, s.U, s.x, s.x_old1, s.x_old2,
            x_min_eff, x_max_eff,
            s.iteration, asym_init, asym_dec, asym_inc,
            s.opts.asymptote);

        // 2. Compute subproblem variable bounds per Svanberg 1987.
        // Incorporates move limits, asymptote safety, and variable bounds
        // into a single pair (alpha, beta) BEFORE the dual solve, so the
        // KKT conditions of the subproblem are consistent with the bounds.
        Eigen::Vector<double, N> alpha(n), beta(n);
        for(int j = 0; j < n; ++j)
        {
            // Asymptote safety: stay away from L and U to keep the
            // reciprocal approximation well-conditioned.
            double L_safe = s.L[j] + 0.01 * (s.x[j] - s.L[j]);
            double U_safe = s.U[j] - 0.01 * (s.U[j] - s.x[j]);

            alpha[j] = std::max(L_safe, s.lower[j]);
            beta[j] = std::min(U_safe, s.upper[j]);

            // Move limit relative to current asymptote half-width.
            // The half-width adapts via oscillation detection: contracts
            // on oscillation (0.7x), expands on monotone progress (1.2x).
            // Reference: Svanberg 1987, Section 5.
            double half_L = s.x[j] - s.L[j];
            double half_U = s.U[j] - s.x[j];
            alpha[j] = std::max(alpha[j], s.x[j] - move_lim * half_L);
            beta[j] = std::min(beta[j], s.x[j] + move_lim * half_U);

            if(alpha[j] >= beta[j])
            {
                double mid = 0.5 * (alpha[j] + beta[j]);
                alpha[j] = mid - 1e-10;
                beta[j] = mid + 1e-10;
            }
        }

        // 3. Compute MMA coefficients
        // For MMA, constraint signs: we negate c_ineq to get g_i <= 0 form
        // (MMA convention: constraints are g_i(x) <= 0, nablapp uses c >= 0)
        // So g_i = -c_ineq_i, dg_i = -J_ineq_i
        Eigen::VectorXd g_mma = -s.c_ineq;
        Eigen::Matrix<double, Eigen::Dynamic, N> dg_mma = -s.J_ineq;

        s.subproblem->compute_coefficients(
            s.x, s.f, s.g, g_mma, dg_mma, s.L, s.U,
            s.opts.subproblem);

        // 4. Solve dual subproblem with proper bounds
        Eigen::Vector<double, N> x_new = s.subproblem->dual_solve(
            s.L, s.U, alpha, beta,
            s.opts.subproblem);

        // 5. L1 merit function line search.
        // Prevents oscillation between feasible and infeasible points by
        // rejecting steps that worsen the merit function phi(x) = f(x) +
        // mu * cv(x). Backtrack along the MMA direction until merit
        // improves; accept null step if all backtracks fail.
        // Reference: Nocedal & Wright, Section 18.3.
        double f_old = s.f;
        double cv_old = detail::constraint_violation(s.c_eq, s.c_ineq);
        constexpr double mu = 10.0;
        double merit_old = s.f + mu * cv_old;

        Eigen::Vector<double, N> dx = x_new - s.x;

        // Evaluate full step
        double f_trial = s.eval_value(x_new);
        Eigen::VectorXd c_eq_trial, c_ineq_trial;
        s.eval_constraints(x_new, c_eq_trial, c_ineq_trial);
        double cv_trial = detail::constraint_violation(c_eq_trial, c_ineq_trial);
        double merit_trial = f_trial + mu * cv_trial;

        for(int bt = 0; bt < 4 && merit_trial > merit_old; ++bt)
        {
            double step_alpha = 1.0 / (1 << (bt + 1));
            x_new = s.x + step_alpha * dx;
            f_trial = s.eval_value(x_new);
            s.eval_constraints(x_new, c_eq_trial, c_ineq_trial);
            cv_trial = detail::constraint_violation(c_eq_trial, c_ineq_trial);
            merit_trial = f_trial + mu * cv_trial;
        }

        // Null step with asymptote contraction: if all backtracks failed,
        // stay at current iterate but tighten asymptotes. This contracts
        // the trust region so the next iteration proposes a smaller step,
        // breaking the reject-repeat cycle.
        if(merit_trial > merit_old)
        {
            for(int j = 0; j < n; ++j)
            {
                s.L[j] = s.x[j] - asym_dec * (s.x[j] - s.L[j]);
                s.U[j] = s.x[j] + asym_dec * (s.U[j] - s.x[j]);
            }
            x_new = s.x;
            f_trial = s.f;
            c_eq_trial = s.c_eq;
            c_ineq_trial = s.c_ineq;
            cv_trial = cv_old;
        }

        // 6. Shift history
        s.x_old2 = s.x_old1;
        s.x_old1 = s.x;

        // 7. Accept step
        double step_size = (x_new - s.x).norm();
        s.x = x_new;
        s.f = f_trial;
        s.eval_gradient(s.x, s.g);
        s.c_eq = c_eq_trial;
        s.c_ineq = c_ineq_trial;
        Eigen::MatrixXd Jeq_dummy;
        s.eval_jacobian(s.x, Jeq_dummy, s.J_ineq);

        // 8. Iteration++
        ++s.iteration;

        double violation = cv_trial;
        double grad_norm = std::max(s.g.norm(), violation);

        return step_result<double>{
            .objective_value = s.f,
            .gradient_norm = grad_norm,
            .step_size = step_size,
            .objective_change = s.f - f_old,
            .improved = s.f < f_old,
            .constraint_violation = violation,
            .x_norm = s.x.norm(),
        };
    }

    void reset(state_type& s, const Eigen::Vector<double, N>& x0)
    {
        s.x = x0;
        s.f = s.eval_value(x0);
        s.eval_gradient(x0, s.g);
        s.eval_constraints(x0, s.c_eq, s.c_ineq);
        Eigen::MatrixXd Jeq_dummy;
        s.eval_jacobian(x0, Jeq_dummy, s.J_ineq);
        s.iteration = 0;
        s.x_old1 = x0;
        s.x_old2 = x0;
    }

    void reset_clear(state_type& s,
                     const Eigen::Vector<double, N>& x0)
    {
        reset(s, x0);
        const int n = static_cast<int>(x0.size());
        double asym_init = s.opts.asymptote_init.value_or(0.5);
        constexpr double inf = std::numeric_limits<double>::infinity();
        for(int j = 0; j < n; ++j)
        {
            double half;
            if(s.lower[j] > -inf && s.upper[j] < inf)
                half = asym_init * (s.upper[j] - s.lower[j]);
            else
                half = 1.0;
            s.L[j] = x0[j] - half;
            s.U[j] = x0[j] + half;
        }
    }

private:
    static double finite_range(double lo, double hi)
    {
        constexpr double inf = std::numeric_limits<double>::infinity();
        if(lo <= -inf && hi >= inf)
            return 1.0;
        if(lo <= -inf)
            return std::max(std::abs(hi), 1.0);
        if(hi >= inf)
            return std::max(std::abs(lo), 1.0);
        return std::max(hi - lo, 1e-10);
    }

    static Eigen::Vector<double, N> effective_bounds(
        const Eigen::Vector<double, N>& bounds, const Eigen::Vector<double, N>& x,
        int n, bool is_upper, double scale = 10.0)
    {
        constexpr double inf = std::numeric_limits<double>::infinity();
        Eigen::Vector<double, N> result(n);
        for(int j = 0; j < n; ++j)
        {
            if(is_upper && bounds[j] >= inf)
                result[j] = x[j] + std::max(std::abs(x[j]), 1.0) * scale;
            else if(!is_upper && bounds[j] <= -inf)
                result[j] = x[j] - std::max(std::abs(x[j]), 1.0) * scale;
            else
                result[j] = bounds[j];
        }
        return result;
    }
};

}

#endif
