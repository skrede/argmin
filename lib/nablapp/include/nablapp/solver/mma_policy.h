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
#include "nablapp/result/step_result.h"
#include "nablapp/solver/options.h"

#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <limits>

namespace nablapp
{

struct mma_policy
{
    using scalar_type = double;

    struct options_type
    {
        double move_limit = 0.2;
        double asyminit = 0.5;
        double asymdec = 0.7;
        double asyminc = 1.2;
        double kkttol = 1e-6;
    };

    struct state_type
    {
        Eigen::VectorXd x, g;
        double f{};
        Eigen::VectorXd c_eq, c_ineq;
        Eigen::MatrixXd J_ineq;

        Eigen::VectorXd L, U;
        Eigen::VectorXd x_old1, x_old2;
        Eigen::VectorXd lower, upper;

        int iteration{0};
        options_type opts;

        std::function<double(const Eigen::VectorXd&)> eval_value;
        std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> eval_gradient;
        std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&,
                           Eigen::VectorXd&)> eval_constraints;
        std::function<void(const Eigen::VectorXd&, Eigen::MatrixXd&,
                           Eigen::MatrixXd&)> eval_jacobian;
    };

    template <typename Problem>
    state_type init(this auto&&, const Problem& problem,
                    const Eigen::VectorXd& x0,
                    const solver_options<double>& /*opts*/)
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
            s.lower = Eigen::VectorXd::Constant(n, -inf);
            s.upper = Eigen::VectorXd::Constant(n, inf);
        }

        // Initialize asymptotes
        s.L.resize(n);
        s.U.resize(n);
        s.x_old1 = x0;
        s.x_old2 = x0;

        for(int j = 0; j < n; ++j)
        {
            double range = finite_range(s.lower[j], s.upper[j]);
            s.L[j] = x0[j] - s.opts.asyminit * range;
            s.U[j] = x0[j] + s.opts.asyminit * range;
        }

        s.iteration = 0;

        // Closures capturing problem by const reference
        s.eval_value = [&problem](const Eigen::VectorXd& v) {
            return problem.value(v);
        };
        s.eval_gradient = [&problem](const Eigen::VectorXd& v,
                                     Eigen::VectorXd& grad) {
            problem.gradient(v, grad);
        };

        const int n_ineq_cap = m_ineq;
        s.eval_constraints = [&problem, n_ineq_cap](
            const Eigen::VectorXd& v,
            Eigen::VectorXd& ceq, Eigen::VectorXd& cineq)
        {
            ceq = Eigen::VectorXd{};
            cineq.resize(n_ineq_cap);
            if(n_ineq_cap > 0)
                problem.constraints(v, cineq);
        };

        s.eval_jacobian = [&problem, n_ineq_cap](
            const Eigen::VectorXd& v,
            Eigen::MatrixXd& Jeq, Eigen::MatrixXd& Jineq)
        {
            Jeq = Eigen::MatrixXd{};
            Jineq.resize(n_ineq_cap, v.size());
            if(n_ineq_cap > 0)
                problem.constraint_jacobian(v, Jineq);
        };

        return s;
    }

    step_result<double> step(this auto&&, state_type& s)
    {
        const int n = static_cast<int>(s.x.size());
        const int m = static_cast<int>(s.c_ineq.size());

        // Re-evaluate at current x (skip on first iteration -- init did it)
        if(s.iteration != 0)
        {
            s.f = s.eval_value(s.x);
            s.eval_gradient(s.x, s.g);
            s.eval_constraints(s.x, s.c_eq, s.c_ineq);
            Eigen::MatrixXd Jeq_dummy;
            s.eval_jacobian(s.x, Jeq_dummy, s.J_ineq);
        }

        // 1. Update asymptotes
        Eigen::VectorXd x_min_eff = effective_bounds(s.lower, s.x, n, false);
        Eigen::VectorXd x_max_eff = effective_bounds(s.upper, s.x, n, true);

        detail::update_asymptotes(
            s.L, s.U, s.x, s.x_old1, s.x_old2,
            x_min_eff, x_max_eff,
            s.iteration, s.opts.asyminit, s.opts.asymdec, s.opts.asyminc);

        // 2. Compute MMA coefficients
        // For MMA, constraint signs: we negate c_ineq to get g_i <= 0 form
        // (MMA convention: constraints are g_i(x) <= 0, nablapp uses c >= 0)
        // So g_i = -c_ineq_i, dg_i = -J_ineq_i
        Eigen::VectorXd g_mma = -s.c_ineq;
        Eigen::MatrixXd dg_mma = -s.J_ineq;

        auto coeffs = detail::mma_coefficients(
            s.x, s.f, s.g, g_mma, dg_mma, s.L, s.U);

        // 3. Solve dual subproblem
        Eigen::VectorXd x_new = detail::mma_dual_solve(
            coeffs, s.L, s.U, x_min_eff, x_max_eff);

        // 4. Apply move limits
        for(int j = 0; j < n; ++j)
        {
            double range = finite_range(s.lower[j], s.upper[j]);
            double delta = s.opts.move_limit * range;
            x_new[j] = std::clamp(x_new[j],
                std::max(s.x[j] - delta, s.lower[j]),
                std::min(s.x[j] + delta, s.upper[j]));
        }

        // 5. Shift history
        double f_old = s.f;
        s.x_old2 = s.x_old1;
        s.x_old1 = s.x;

        // 6. Update iterate
        double step_size = (x_new - s.x).norm();
        s.x = x_new;
        s.f = s.eval_value(s.x);
        s.eval_gradient(s.x, s.g);
        s.eval_constraints(s.x, s.c_eq, s.c_ineq);
        Eigen::MatrixXd Jeq_dummy;
        s.eval_jacobian(s.x, Jeq_dummy, s.J_ineq);

        // 7. Iteration++
        ++s.iteration;

        // KKT violation: max(grad_norm, constraint_violation)
        double violation = detail::constraint_violation(s.c_eq, s.c_ineq);
        double grad_norm = std::max(s.g.norm(), violation);

        return step_result<double>{
            .objective_value = s.f,
            .gradient_norm = grad_norm,
            .step_size = step_size,
            .objective_change = s.f - f_old,
            .improved = s.f < f_old,
        };
    }

    void reset(this auto&&, state_type& s, const Eigen::VectorXd& x0)
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

    void reset_clear(this auto&& self, state_type& s,
                     const Eigen::VectorXd& x0)
    {
        self.reset(s, x0);
        const int n = static_cast<int>(x0.size());
        for(int j = 0; j < n; ++j)
        {
            double range = finite_range(s.lower[j], s.upper[j]);
            s.L[j] = x0[j] - s.opts.asyminit * range;
            s.U[j] = x0[j] + s.opts.asyminit * range;
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

    static Eigen::VectorXd effective_bounds(
        const Eigen::VectorXd& bounds, const Eigen::VectorXd& x,
        int n, bool is_upper)
    {
        constexpr double inf = std::numeric_limits<double>::infinity();
        Eigen::VectorXd result(n);
        for(int j = 0; j < n; ++j)
        {
            if(is_upper && bounds[j] >= inf)
                result[j] = x[j] + std::max(std::abs(x[j]), 1.0) * 10.0;
            else if(!is_upper && bounds[j] <= -inf)
                result[j] = x[j] - std::max(std::abs(x[j]), 1.0) * 10.0;
            else
                result[j] = bounds[j];
        }
        return result;
    }
};

}

#endif
