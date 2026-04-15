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
#include <limits>
#include <optional>

namespace nablapp
{

template <int N = dynamic_dimension>
struct gcmma_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = gcmma_policy<M>;

    struct options_type
    {
        typename mma_policy<N>::options_type mma_opts{};
        std::optional<std::uint16_t> max_inner_iterations{};    // default: 15 (Svanberg 2002)
        std::optional<double> raa0{};                           // default: 1e-5 (Svanberg 2002)
        std::optional<double> tighten_factor{};                 // default: 0.7 (Svanberg 2002)
        std::optional<double> minimum_distance_fraction{};      // default: 0.001 (Svanberg 2002)
        asymptote_options asymptote{};                          // Embedded asymptote params
        mma_subproblem_options subproblem{};                     // Embedded subproblem params
        std::uint16_t stall_window{50};
        double feasibility_gate{1e-4};
    };

    options_type options{};

    template <typename P = void>
    struct state_type
    {
        static constexpr int M = [] {
            if constexpr(has_constraint_count<P>) return constraint_count_v<P>;
            else return dynamic_dimension;
        }();

        typename mma_policy<N>::template state_type<P> mma_state;
        options_type opts;

        // Proxy members for basic_solver compatibility (it accesses state_.x,
        // state_.c_eq, state_.c_ineq directly).
        Eigen::Vector<double, N>& x = mma_state.x;
        Eigen::VectorXd& c_eq = mma_state.c_eq;
        Eigen::Vector<double, M>& c_ineq = mma_state.c_ineq;

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

    template <typename Problem, typename Convergence>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<double, N>& x0,
                             const solver_options<Convergence>& sopts,
                             const options_type& policy_opts)
    {
        options = policy_opts;
        auto s = init(problem, x0, sopts);
        s.opts = policy_opts;
        s.mma_state.opts = policy_opts.mma_opts;
        return s;
    }

    template <typename Problem, typename Convergence = default_convergence>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<double, N>& x0,
                             const solver_options<Convergence>& sopts)
    {
        state_type<Problem> s;
        s.mma_state = mma_policy<N>{}.init(problem, x0, sopts);
        return s;
    }

    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        constexpr int MC = state_type<P>::M;

        auto& ms = s.mma_state;
        const int n = static_cast<int>(ms.x.size());
        const int m = static_cast<int>(ms.c_ineq.size());

        double asym_init = ms.opts.asymptote_init.value_or(0.5);
        double asym_dec = ms.opts.asymptote_decrease.value_or(0.7);
        double asym_inc = ms.opts.asymptote_increase.value_or(1.2);
        double move_lim = ms.opts.move_limit.value_or(0.2);
        double eff_scale = ms.opts.effective_bounds_scale.value_or(10.0);
        std::uint16_t max_inner = s.opts.max_inner_iterations.value_or(15);
        double raa0_val = s.opts.raa0.value_or(1e-5);
        double tighten = s.opts.tighten_factor.value_or(0.7);
        double min_dist_frac = s.opts.minimum_distance_fraction.value_or(0.001);

        // Re-evaluate at current x (skip on first iteration)
        if(ms.iteration != 0)
        {
            ms.f = ms.problem->value(ms.x);
            ms.problem->gradient(ms.x, ms.g);
            if(m > 0)
            {
                Eigen::VectorXd c_tmp(m);
                ms.problem->constraints(ms.x, c_tmp);
                ms.c_ineq = c_tmp;
            }
            if(m > 0)
            {
                Eigen::MatrixXd J_tmp(m, n);
                ms.problem->constraint_jacobian(ms.x, J_tmp);
                ms.J_ineq = J_tmp;
            }
        }

        // Effective bounds for asymptote update
        Eigen::Vector<double, N> x_min_eff = effective_bounds(ms.lower, ms.x, n, false, eff_scale);
        Eigen::Vector<double, N> x_max_eff = effective_bounds(ms.upper, ms.x, n, true, eff_scale);

        // Update asymptotes, passing embedded options
        detail::update_asymptotes(
            ms.L, ms.U, ms.x, ms.x_old1, ms.x_old2,
            x_min_eff, x_max_eff,
            ms.iteration, asym_init, asym_dec, asym_inc,
            s.opts.asymptote);

        // Work with copies of L, U for inner tightening
        Eigen::Vector<double, N> L_inner = ms.L;
        Eigen::Vector<double, N> U_inner = ms.U;

        // MMA convention: g_i <= 0 form
        Eigen::Vector<double, MC> g_mma = -ms.c_ineq;
        Eigen::Matrix<double, MC, N> dg_mma = -ms.J_ineq;

        Eigen::Vector<double, N> x_trial(n);
        double f_trial{};
        Eigen::VectorXd c_ineq_trial(m);

        for(std::uint16_t inner = 0; inner < max_inner; ++inner)
        {
            // Compute coefficients with current (possibly tightened) asymptotes
            ms.subproblem->compute_coefficients(
                ms.x, ms.f, ms.g, g_mma, dg_mma, L_inner, U_inner,
                s.opts.subproblem);

            // Solve dual subproblem using pre-allocated solver.
            // Feasible region = tightened asymptote interval, not effective bounds.
            // Svanberg (2002): subproblem solution must lie within (L_inner, U_inner)
            // for the conservatism check to succeed without post-hoc distortion.
            x_trial = ms.subproblem->dual_solve(
                L_inner, U_inner, L_inner, U_inner,
                s.opts.subproblem);

            // Apply move limits
            for(int j = 0; j < n; ++j)
            {
                double range = finite_range(ms.lower[j], ms.upper[j]);
                double delta = move_lim * range;
                x_trial[j] = std::clamp(x_trial[j],
                    std::max(ms.x[j] - delta, ms.lower[j]),
                    std::min(ms.x[j] + delta, ms.upper[j]));
            }

            // Evaluate actual values at trial point
            f_trial = ms.problem->value(x_trial);
            if(m > 0)
                ms.problem->constraints(x_trial, c_ineq_trial);

            // Check conservatism: actual <= approximation for objective
            // and for each constraint (in g <= 0 form)
            double approx_f = ms.subproblem->subproblem_value(
                x_trial, L_inner, U_inner);

            bool conservative = (f_trial <= approx_f + raa0_val);

            if(conservative && m > 0)
            {
                for(int i = 0; i < m; ++i)
                {
                    double g_actual = -c_ineq_trial[i];
                    double g_approx = ms.subproblem->subproblem_constraint(
                        i, x_trial, L_inner, U_inner);
                    if(g_actual > g_approx + raa0_val)
                    {
                        conservative = false;
                        break;
                    }
                }
            }

            if(conservative)
                break;

            // Tighten asymptotes: move L, U closer by tighten_factor
            for(int j = 0; j < n; ++j)
            {
                L_inner[j] = ms.x[j] - tighten * (ms.x[j] - L_inner[j]);
                U_inner[j] = ms.x[j] + tighten * (U_inner[j] - ms.x[j]);

                // Safety: keep minimum distance
                double range = finite_range(ms.lower[j], ms.upper[j]);
                double min_dist = min_dist_frac * range;
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
        ms.problem->gradient(ms.x, ms.g);
        if(m > 0)
        {
            Eigen::MatrixXd J_tmp(m, n);
            ms.problem->constraint_jacobian(ms.x, J_tmp);
            ms.J_ineq = J_tmp;
        }

        ++ms.iteration;

        double violation = detail::constraint_violation(ms.c_eq, ms.c_ineq);
        double grad_norm = std::max(ms.g.norm(), violation);

        return step_result<double>{
            .objective_value = ms.f,
            .gradient_norm = grad_norm,
            .step_size = step_size,
            .objective_change = ms.f - f_old,
            .improved = ms.f < f_old,
            .x_norm = ms.x.norm(),
        };
    }

    template <typename P>
    void reset(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        mma_policy<N>{}.template reset<P>(s.mma_state, x0);
    }

    template <typename P>
    void reset_clear(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        mma_policy<N>{}.template reset_clear<P>(s.mma_state, x0);
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
