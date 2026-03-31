#ifndef HPP_GUARD_NABLAPP_SOLVER_GCMMA_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_GCMMA_POLICY_H

// Globally Convergent MMA (GCMMA) policy.
//
// Wraps MMA with a conservatism loop that tightens asymptotes
// until the actual function/constraint values are bounded by their
// convex separable approximations (sufficient decrease condition).
// This guarantees global convergence under standard assumptions.
//
// Reference: Svanberg 2002, "A class of globally convergent
//            optimization methods based on conservative convex
//            separable approximations".

#include "nablapp/solver/mma_policy.h"
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

struct gcmma_policy
{
    using scalar_type = double;

    struct options_type
    {
        mma_policy::options_type mma_opts{};
        int max_inner_iter = 15;
        double raa0 = 1e-5;
    };

    struct state_type
    {
        mma_policy::state_type mma_state;
        options_type opts;

        // Proxy members for basic_solver compatibility (it accesses state_.x,
        // state_.c_eq, state_.c_ineq directly).
        Eigen::VectorXd& x = mma_state.x;
        Eigen::VectorXd& c_eq = mma_state.c_eq;
        Eigen::VectorXd& c_ineq = mma_state.c_ineq;

        state_type() = default;
        state_type(const state_type& o)
            : mma_state{o.mma_state}, opts{o.opts}
            , x{mma_state.x}, c_eq{mma_state.c_eq}, c_ineq{mma_state.c_ineq}
        {}
        state_type(state_type&& o) noexcept
            : mma_state{std::move(o.mma_state)}, opts{std::move(o.opts)}
            , x{mma_state.x}, c_eq{mma_state.c_eq}, c_ineq{mma_state.c_ineq}
        {}
        state_type& operator=(const state_type& o)
        {
            if(this != &o)
            {
                mma_state = o.mma_state;
                opts = o.opts;
            }
            return *this;
        }
        state_type& operator=(state_type&& o) noexcept
        {
            if(this != &o)
            {
                mma_state = std::move(o.mma_state);
                opts = std::move(o.opts);
            }
            return *this;
        }
    };

    template <typename Problem>
    state_type init(this auto&& self, const Problem& problem,
                    const Eigen::VectorXd& x0,
                    const solver_options<double>& sopts,
                    const options_type& policy_opts)
    {
        auto s = self.init(problem, x0, sopts);
        s.opts = policy_opts;
        s.mma_state.opts = policy_opts.mma_opts;
        return s;
    }

    template <typename Problem>
    state_type init(this auto&&, const Problem& problem,
                    const Eigen::VectorXd& x0,
                    const solver_options<double>& sopts)
    {
        state_type s;
        s.mma_state = mma_policy{}.init(problem, x0, sopts);
        return s;
    }

    step_result<double> step(this auto&&, state_type& s)
    {
        auto& ms = s.mma_state;
        const int n = static_cast<int>(ms.x.size());
        const int m = static_cast<int>(ms.c_ineq.size());

        // Re-evaluate at current x (skip on first iteration)
        if(ms.iteration != 0)
        {
            ms.f = ms.eval_value(ms.x);
            ms.eval_gradient(ms.x, ms.g);
            ms.eval_constraints(ms.x, ms.c_eq, ms.c_ineq);
            Eigen::MatrixXd Jeq_dummy;
            ms.eval_jacobian(ms.x, Jeq_dummy, ms.J_ineq);
        }

        // Effective bounds for asymptote update
        Eigen::VectorXd x_min_eff = effective_bounds(ms.lower, ms.x, n, false);
        Eigen::VectorXd x_max_eff = effective_bounds(ms.upper, ms.x, n, true);

        // Update asymptotes
        detail::update_asymptotes(
            ms.L, ms.U, ms.x, ms.x_old1, ms.x_old2,
            x_min_eff, x_max_eff,
            ms.iteration, ms.opts.asyminit, ms.opts.asymdec, ms.opts.asyminc);

        // Work with copies of L, U for inner tightening
        Eigen::VectorXd L_inner = ms.L;
        Eigen::VectorXd U_inner = ms.U;

        // MMA convention: g_i <= 0 form
        Eigen::VectorXd g_mma = -ms.c_ineq;
        Eigen::MatrixXd dg_mma = -ms.J_ineq;

        Eigen::VectorXd x_trial(n);
        double f_trial{};
        Eigen::VectorXd c_ineq_trial(m);

        for(int inner = 0; inner < s.opts.max_inner_iter; ++inner)
        {
            // Compute coefficients with current (possibly tightened) asymptotes
            auto coeffs = detail::mma_coefficients(
                ms.x, ms.f, ms.g, g_mma, dg_mma, L_inner, U_inner);

            // Solve dual subproblem
            x_trial = detail::mma_dual_solve(
                coeffs, L_inner, U_inner, x_min_eff, x_max_eff);

            // Apply move limits
            for(int j = 0; j < n; ++j)
            {
                double range = finite_range(ms.lower[j], ms.upper[j]);
                double delta = ms.opts.move_limit * range;
                x_trial[j] = std::clamp(x_trial[j],
                    std::max(ms.x[j] - delta, ms.lower[j]),
                    std::min(ms.x[j] + delta, ms.upper[j]));
            }

            // Evaluate actual values at trial point
            f_trial = ms.eval_value(x_trial);
            Eigen::VectorXd ceq_trial;
            ms.eval_constraints(x_trial, ceq_trial, c_ineq_trial);

            // Check conservatism: actual <= approximation for objective
            // and for each constraint (in g <= 0 form)
            double approx_f = detail::mma_subproblem_value(
                coeffs, x_trial, L_inner, U_inner);

            bool conservative = (f_trial <= approx_f + s.opts.raa0);

            if(conservative && m > 0)
            {
                for(int i = 0; i < m; ++i)
                {
                    double g_actual = -c_ineq_trial[i];
                    double g_approx = detail::mma_subproblem_constraint(
                        coeffs, i, x_trial, L_inner, U_inner);
                    if(g_actual > g_approx + s.opts.raa0)
                    {
                        conservative = false;
                        break;
                    }
                }
            }

            if(conservative)
                break;

            // Tighten asymptotes: move L, U closer by factor 0.7
            constexpr double tighten = 0.7;
            for(int j = 0; j < n; ++j)
            {
                double midL = (ms.x[j] + L_inner[j]) * 0.5;
                double midU = (ms.x[j] + U_inner[j]) * 0.5;
                L_inner[j] = ms.x[j] - tighten * (ms.x[j] - L_inner[j]);
                U_inner[j] = ms.x[j] + tighten * (U_inner[j] - ms.x[j]);

                // Safety: keep minimum distance
                double range = finite_range(ms.lower[j], ms.upper[j]);
                double min_dist = 0.001 * range;
                L_inner[j] = std::min(L_inner[j], ms.x[j] - min_dist);
                U_inner[j] = std::max(U_inner[j], ms.x[j] + min_dist);
            }
        }

        // Accept trial point
        double f_old = ms.f;
        double step_size = (x_trial - ms.x).norm();

        ms.x_old2 = ms.x_old1;
        ms.x_old1 = ms.x;
        ms.x = x_trial;
        ms.f = f_trial;
        ms.c_ineq = c_ineq_trial;

        // Re-evaluate gradient and Jacobian at new point
        ms.eval_gradient(ms.x, ms.g);
        Eigen::MatrixXd Jeq_dummy;
        ms.eval_jacobian(ms.x, Jeq_dummy, ms.J_ineq);

        ++ms.iteration;

        double violation = detail::constraint_violation(ms.c_eq, ms.c_ineq);
        double grad_norm = std::max(ms.g.norm(), violation);

        return step_result<double>{
            .objective_value = ms.f,
            .gradient_norm = grad_norm,
            .step_size = step_size,
            .objective_change = ms.f - f_old,
            .improved = ms.f < f_old,
        };
    }

    void reset(this auto&&, state_type& s, const Eigen::VectorXd& x0)
    {
        mma_policy{}.reset(s.mma_state, x0);
    }

    void reset_clear(this auto&&, state_type& s, const Eigen::VectorXd& x0)
    {
        mma_policy{}.reset_clear(s.mma_state, x0);
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
