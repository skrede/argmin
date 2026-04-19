#ifndef HPP_GUARD_NABLAPP_SOLVER_GCMMA_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_GCMMA_POLICY_H

// Globally Convergent MMA (GCMMA) policy.
//
// Wraps MMA with a conservativity loop per Svanberg 2002 Section 4.2.
// On each outer iteration, the asymptotes (L, U) are updated once and
// held fixed; an inner loop then repeatedly builds the convex separable
// approximation, solves the dual subproblem, and checks whether the
// approximation is conservative at the trial point:
//     f_trial    <= approx_f    + epsimin
//     g_i_trial  <= approx_g_i  + epsimin   for all i.
// If non-conservative, the per-component conservativity regularizers
// raa_0 / raa_i grow (objective grows iff the objective was
// non-conservative; each raa_i grows iff constraint i was violating)
// via the arjendeetman/GCMMA-MMA-Python raacof step-geometry weight
// (in-tree analog of NLopt mma.c's wval-based growth formula), and
// the inner loop retries. On max-inner exhaustion a null step is
// returned so the paper's global convergence proof is preserved.
// Per-outer raa decay (NLopt mma.c:385-389 mirror, MMA_RHOMIN-floored)
// is applied at both the conservative-accept exit and the null-step
// exit so the regularizer relaxes between outer iters instead of
// saturating at the per-outer growth cap.
//
// References:
//   Svanberg 2002, "A class of globally convergent optimization
//     methods based on conservative convex separable approximations",
//     SIAM J. Optim. 12(2):555-573.
//   arjendeetman/GCMMA-MMA-Python (Svanberg MATLAB port): raacof
//     step-geometry-scaled growth factor.
//   NLopt LD_MMA reference (Steven G. Johnson 2008-2012 implementation
//     of Svanberg 2002 CCSA): src/algs/mma/mma.c lines 265-389
//     (inner conservativity loop + inter-outer rho/raa decay).

#include "nablapp/solver/mma_policy.h"
#include "nablapp/detail/asymptote_update.h"
#include "nablapp/detail/mma_subproblem.h"
#include "nablapp/detail/kkt_residual.h"
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

        // Svanberg 2002 Section 4.2 initial regularization floor.
        // raa_0 and raa_i are state members that adapt (grow) on
        // non-conservative trials; raa0 is both the floor they cannot
        // drop below and the initial value they start from at outer
        // iteration 0 (scaled by the asymp formula).
        std::optional<double> raa0{};                           // default: 1e-5 (floor + initial)

        // Svanberg 2002 Section 4.2 per-component conservativity growth
        // controls.
        //
        // raa_growth: multiplicative factor applied to raa on
        //             non-conservative trials. Svanberg 2002 uses 1.1
        //             (paper default); nablapp default is 2.0 (approximately
        //             4 inner iterations to reach the 10x ceiling, versus
        //             7-8 at 1.1x).
        //
        // raa_max_factor: 10x ceiling on raa growth within a single outer
        //                 iteration (Svanberg 2002 cap).
        //
        // conservativity_slack: epsimin in the paper; a fixed positive
        //                       slack on the conservativity inequality,
        //                       independent of the adaptive raa.
        //
        // Reference: Svanberg 2002, Section 4.2.
        std::optional<double> raa_growth{};                     // default: 2.0
        std::optional<double> raa_max_factor{};                 // default: 10.0
        std::optional<double> conservativity_slack{};           // default: 1e-7 (epsimin)

        // Svanberg 2002 Section 4.2 inter-outer raa decay (NLopt
        // mma.c:385-389 MMA_RHOMIN-floored decay).  raa_decay is the
        // per-outer multiplicative factor applied AFTER the inner loop
        // exits; raa_min is the floor below which raa cannot decay
        // (analogous to NLopt's MMA_RHOMIN constant).  Without this
        // decay, raa monotonically saturates at raa_max_factor *
        // initial_raa for the rest of the run, which prevents the
        // regularizer from relaxing between outer iterations on
        // well-behaved trajectories.
        //
        // Reference: NLopt mma.c:385-389; Svanberg 2002 Section 4.2
        //            (per-outer regularizer decay).
        std::optional<double> raa_decay{};                       // default: 0.1   (NLopt mma.c:385)
        std::optional<double> raa_min{};                         // default: 1e-5  (NLopt MMA_RHOMIN)

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

        // Svanberg 2002 Section 4.2 per-component conservativity
        // coefficients. Persistent across outer iterations (global
        // algorithm state; not reset per outer). raa_0 scales the
        // objective regularization; raa[i] scales the constraint-i
        // regularization. Both grow on non-conservative inner iterations
        // within a single outer iter.
        //
        // Reference: Svanberg 2002, Section 3 (regularization role);
        //            Section 4.2 (growth rule).
        double raa_0{0.0};
        Eigen::Vector<double, M> raa;

        // Proxy members for basic_solver compatibility (it accesses state_.x,
        // state_.c_eq, state_.c_ineq directly).
        Eigen::Vector<double, N>& x = mma_state.x;
        Eigen::VectorXd& c_eq = mma_state.c_eq;
        Eigen::Vector<double, M>& c_ineq = mma_state.c_ineq;

        state_type() = default;
        state_type(const state_type& o)
            : mma_state{o.mma_state}, opts{o.opts}
            , raa_0{o.raa_0}, raa{o.raa}
            , x{mma_state.x}, c_eq{mma_state.c_eq}, c_ineq{mma_state.c_ineq}
        {}
        state_type(state_type&& o) noexcept
            : mma_state{std::move(o.mma_state)}, opts{std::move(o.opts)}
            , raa_0{o.raa_0}, raa{std::move(o.raa)}
            , x{mma_state.x}, c_eq{mma_state.c_eq}, c_ineq{mma_state.c_ineq}
        {}
        state_type& operator=(const state_type& o)
        {
            if(this != &o)
            {
                mma_state = o.mma_state;
                opts = o.opts;
                raa_0 = o.raa_0;
                raa = o.raa;
            }
            return *this;
        }
        state_type& operator=(state_type&& o) noexcept
        {
            if(this != &o)
            {
                mma_state = std::move(o.mma_state);
                opts = std::move(o.opts);
                raa_0 = o.raa_0;
                raa = std::move(o.raa);
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

        // raa_0 / raa are zero-initialized; the first step() detects
        // the zero-state at iteration 0 and seeds from the asymp formula.
        const int m = static_cast<int>(s.mma_state.c_ineq.size());
        s.raa.resize(m);
        s.raa.setZero();
        return s;
    }

    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        constexpr int MC = state_type<P>::M;

        auto& ms = s.mma_state;
        const int n = static_cast<int>(ms.x.size());
        const int m = static_cast<int>(ms.c_ineq.size());

        const double asym_init = ms.opts.asymptote_init.value_or(0.5);
        const double asym_dec = ms.opts.asymptote_decrease.value_or(0.7);
        const double asym_inc = ms.opts.asymptote_increase.value_or(1.2);
        const double move_lim = ms.opts.move_limit.value_or(0.2);
        const double eff_scale = ms.opts.effective_bounds_scale.value_or(10.0);
        const std::uint16_t max_inner = s.opts.max_inner_iterations.value_or(15);
        const double raa0_floor = s.opts.raa0.value_or(1e-5);
        const double raa_growth = s.opts.raa_growth.value_or(2.0);
        const double raa_max_factor = s.opts.raa_max_factor.value_or(10.0);
        const double epsimin = s.opts.conservativity_slack.value_or(1e-7);

        // Effective bounds for asymptote update (open-box handling).
        Eigen::Vector<double, N> x_min_eff =
            effective_bounds(ms.lower, ms.x, n, false, eff_scale);
        Eigen::Vector<double, N> x_max_eff =
            effective_bounds(ms.upper, ms.x, n, true, eff_scale);

        // Update asymptotes ONCE per outer iteration. Svanberg 2002
        // Section 4.2 keeps (L, U) fixed across the inner loop; only
        // raa_0 / raa grow.
        detail::update_asymptotes(
            ms.L, ms.U, ms.x, ms.x_old1, ms.x_old2,
            x_min_eff, x_max_eff,
            ms.iteration, asym_init, asym_dec, asym_inc,
            s.opts.asymptote);

        // Svanberg 2002 Section 4.2 initial raa per problem scale
        // (computed once at outer iteration 0 when state is fresh):
        //     raa_0_init = max( raa0_floor, (0.1 / n) * |grad_f| dot (U-L) )
        //     raa_i_init = max( raa0_floor, (0.1 / n) * |grad_g_i| dot (U-L) )
        // Subsequent outer iterations carry the grown value forward.
        //
        // Reference: Svanberg 2002 Section 4.2; arjendeetman/GCMMA-MMA-Python
        //            src/mmapy/mma.py asymp initializer.
        if(ms.iteration == 0 && s.raa_0 == 0.0)
        {
            const Eigen::Vector<double, N> UL_range =
                (ms.U - ms.L).cwiseMax(1e-12);
            const double inv_n = 1.0 / static_cast<double>(n);
            const double grad_scale = ms.g.cwiseAbs().dot(UL_range);
            s.raa_0 = std::max(raa0_floor, 0.1 * grad_scale * inv_n);

            if(static_cast<int>(s.raa.size()) != m)
                s.raa.resize(m);
            for(int i = 0; i < m; ++i)
            {
                const double dg_row_scale =
                    ms.J_ineq.row(i).cwiseAbs().dot(UL_range);
                s.raa[i] = std::max(raa0_floor, 0.1 * dg_row_scale * inv_n);
            }
        }

        // Svanberg 1987 Section 3 eq. (3.3)-(3.5) subproblem bounds:
        // asymptote-safety-shift the current iterate inward within (L, U),
        // intersect with the true variable bounds, then apply a
        // sigma-based move limit (scale-adaptive via the asymptote
        // half-width). Computed ONCE per outer iter because the inner
        // conservativity loop keeps asymptotes fixed.
        //
        // Reference: Svanberg 1987, Section 3 eq. (3.3)-(3.5);
        //            NLopt ccsa_quadratic sigma-based move limit pattern.
        Eigen::Vector<double, N> alpha(n);
        Eigen::Vector<double, N> beta(n);
        for(int j = 0; j < n; ++j)
        {
            const double L_safe = ms.L[j] + 0.01 * (ms.x[j] - ms.L[j]);
            const double U_safe = ms.U[j] - 0.01 * (ms.U[j] - ms.x[j]);

            alpha[j] = std::max(L_safe, ms.lower[j]);
            beta[j]  = std::min(U_safe, ms.upper[j]);

            const double sigma_j = 0.5 * (ms.U[j] - ms.L[j]);
            const double delta_j = move_lim * sigma_j;
            alpha[j] = std::max(alpha[j], ms.x[j] - delta_j);
            beta[j]  = std::min(beta[j],  ms.x[j] + delta_j);

            if(alpha[j] >= beta[j])
            {
                const double mid = 0.5 * (alpha[j] + beta[j]);
                alpha[j] = mid - 1e-10;
                beta[j]  = mid + 1e-10;
            }
        }

        // MMA convention: g_i <= 0 form (nablapp uses c_ineq >= 0).
        Eigen::Vector<double, MC> g_mma = -ms.c_ineq;
        Eigen::Matrix<double, MC, N> dg_mma = -ms.J_ineq;

        Eigen::Vector<double, N> x_trial(n);
        double f_trial{};
        Eigen::Vector<double, MC> c_ineq_trial(m);

        // Svanberg 2002 Section 4.2 inner conservativity loop.
        // On non-conservative trials, per-component raa grows (objective
        // iff objective was non-conservative, constraint i iff that
        // constraint was violating). On max-inner exhaustion a null step
        // is returned so the global convergence proof is preserved.
        for(std::uint16_t inner = 0; inner < max_inner; ++inner)
        {
            ms.subproblem->compute_coefficients(
                ms.x, ms.f, ms.g, g_mma, dg_mma,
                ms.L, ms.U,
                s.raa_0, s.raa,
                s.opts.subproblem);

            x_trial = ms.subproblem->dual_solve(
                ms.L, ms.U, alpha, beta,
                s.opts.subproblem);

            f_trial = ms.problem->value(x_trial);
            if(m > 0)
            {
                Eigen::VectorXd c_tmp(m);
                ms.problem->constraints(x_trial, c_tmp);
                c_ineq_trial = c_tmp;
            }

            // Conservativity test per Svanberg 2002 Section 3.
            const double approx_f = ms.subproblem->subproblem_value(
                x_trial, ms.L, ms.U);

            bool conservative_obj = (f_trial <= approx_f + epsimin);
            bool conservative_ineq = true;
            for(int i = 0; i < m && conservative_ineq; ++i)
            {
                const double g_actual = -c_ineq_trial[i];
                const double g_approx = ms.subproblem->subproblem_constraint(
                    i, x_trial, ms.L, ms.U);
                if(g_actual > g_approx + epsimin)
                    conservative_ineq = false;
            }

            if(conservative_obj && conservative_ineq)
                break;

            // Svanberg 2002 Section 4.2 raacof: step-geometry-scaled
            // growth factor computed from the actual trial displacement.
            //
            // Reference: arjendeetman/GCMMA-MMA-Python raaupdate
            //            (Svanberg MATLAB port).
            double raacof = 0.0;
            for(int j = 0; j < n; ++j)
            {
                const double dx = x_trial[j] - ms.x[j];
                const double denom_ux =
                    std::max(ms.U[j] - x_trial[j], 1e-12);
                const double denom_xl =
                    std::max(x_trial[j] - ms.L[j], 1e-12);
                const double xxux = dx / denom_ux;
                const double xxxl = dx / denom_xl;
                const double scale_range =
                    std::max(ms.upper[j] - ms.lower[j], 1.0);
                const double ulxx = (ms.U[j] - ms.L[j]) / scale_range;
                raacof += (xxux * xxxl) * ulxx;
            }
            raacof = std::max(raacof, 1e-12);

            // Objective grows iff non-conservative; the 10x cap is
            // Svanberg 2002's paper-faithful ceiling on per-outer-iter
            // growth.
            if(!conservative_obj)
            {
                const double delta = (f_trial - approx_f) / raacof;
                s.raa_0 = std::min(
                    raa_growth * (s.raa_0 + delta),
                    raa_max_factor * s.raa_0);
            }

            // Per-constraint growth on violating components only.
            for(int i = 0; i < m; ++i)
            {
                const double g_actual = -c_ineq_trial[i];
                const double g_approx =
                    ms.subproblem->subproblem_constraint(
                        i, x_trial, ms.L, ms.U);
                if(g_actual > g_approx + epsimin)
                {
                    const double delta = (g_actual - g_approx) / raacof;
                    s.raa[i] = std::min(
                        raa_growth * (s.raa[i] + delta),
                        raa_max_factor * s.raa[i]);
                }
            }

            // Max-inner exhaustion: return null step so the Svanberg 2002
            // global convergence proof holds (no non-conservative accept).
            // Null steps are exempt from step_tolerance per the framework
            // contract; raa will be tighter on the next outer iter.
            //
            // Reference: Svanberg 2002 Section 4.2 (closure condition of
            //            the convergence proof).
            if(inner + 1 == max_inner)
            {
                const Eigen::Matrix<double, 0, N> J_eq_empty_ex(0, n);
                const Eigen::Vector<double, 0> lambda_eq_empty_ex;
                const Eigen::Vector<double, 0> c_eq_empty_ex;
                const auto& mu_ineq_ex = ms.subproblem->multipliers();
                const double kkt_ex =
                    detail::kkt_residual<double, N, 0, MC>(
                        ms.g,
                        J_eq_empty_ex,
                        ms.J_ineq,
                        lambda_eq_empty_ex,
                        mu_ineq_ex,
                        c_eq_empty_ex,
                        ms.c_ineq);

                // Svanberg 2002 Section 4.2 inter-outer raa decay
                // (NLopt mma.c:385-389 mirror, applied here at the
                // null-step exit so the next outer iter starts from a
                // relaxed regularizer rather than the saturated cap).
                {
                    const double raa_decay_val = s.opts.raa_decay.value_or(0.1);
                    const double raa_min_val = s.opts.raa_min.value_or(1e-5);
                    s.raa_0 = std::max(raa_decay_val * s.raa_0, raa_min_val);
                    for(int i = 0; i < m; ++i)
                        s.raa[i] = std::max(raa_decay_val * s.raa[i], raa_min_val);
                }

                return step_result<double>{
                    .objective_value = ms.f,
                    .gradient_norm = ms.g.norm(),
                    .step_size = 0.0,
                    .objective_change = 0.0,
                    .improved = false,
                    .is_null_step = true,
                    .constraint_violation =
                        detail::primal_feasibility_inf(
                            ms.c_eq, ms.c_ineq),
                    .x_norm = ms.x.norm(),
                    .kkt_residual = kkt_ex,
                };
            }
        }

        // Accept the conservative trial point.
        const double f_old = ms.f;
        const double step_size = (x_trial - ms.x).norm();

        ms.x_old2 = ms.x_old1;
        ms.x_old1 = ms.x;
        ms.x = x_trial;
        ms.f = f_trial;
        if(m > 0)
            ms.c_ineq = c_ineq_trial;

        // Re-evaluate gradient and Jacobian at the accepted iterate.
        ms.problem->gradient(ms.x, ms.g);
        if(m > 0)
        {
            Eigen::MatrixXd J_tmp(m, n);
            ms.problem->constraint_jacobian(ms.x, J_tmp);
            ms.J_ineq = J_tmp;
        }

        // Svanberg 2002 Section 4.2 inter-outer raa decay
        // (NLopt mma.c:385-389 mirror, applied at the conservative-accept
        // exit before the next outer iter; symmetric with the null-step
        // exit decay above).
        {
            const double raa_decay_val = s.opts.raa_decay.value_or(0.1);
            const double raa_min_val = s.opts.raa_min.value_or(1e-5);
            s.raa_0 = std::max(raa_decay_val * s.raa_0, raa_min_val);
            for(int i = 0; i < m; ++i)
                s.raa[i] = std::max(raa_decay_val * s.raa[i], raa_min_val);
        }

        ++ms.iteration;

        // E-measure KKT residual via the subproblem dual multipliers
        // (mirrors the MMA accept-step wire). MMA / GCMMA share the same
        // subproblem form; sign convention is identical (y_i = mu_ineq_i
        // with no flip; see mma_subproblem.h multipliers() comment).
        //
        // Reference: Nocedal and Wright, "Numerical Optimization" 2e,
        //            Definition 12.1 + eq. 12.34 (E-measure composition);
        //            Svanberg 1987 Section 5; Svanberg 2002 Section 5.
        const Eigen::Matrix<double, 0, N> J_eq_empty_acc(0, n);
        const Eigen::Vector<double, 0> lambda_eq_empty_acc;
        const Eigen::Vector<double, 0> c_eq_empty_acc;
        const auto& mu_ineq_acc = ms.subproblem->multipliers();
        const double kkt_acc = detail::kkt_residual<double, N, 0, MC>(
            ms.g,
            J_eq_empty_acc,
            ms.J_ineq,
            lambda_eq_empty_acc,
            mu_ineq_acc,
            c_eq_empty_acc,
            ms.c_ineq);

        return step_result<double>{
            .objective_value = ms.f,
            .gradient_norm = ms.g.norm(),
            .step_size = step_size,
            .objective_change = ms.f - f_old,
            .improved = ms.f < f_old,
            .constraint_violation = detail::primal_feasibility_inf(
                ms.c_eq, ms.c_ineq),
            .x_norm = ms.x.norm(),
            .kkt_residual = kkt_acc,
        };
    }

    template <typename P>
    void reset(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        mma_policy<N>{}.template reset<P>(s.mma_state, x0);
        // Reset adaptive conservativity state so the next step() re-seeds
        // the asymp formula from the gradient at x0.
        s.raa_0 = 0.0;
        s.raa.setZero();
    }

    template <typename P>
    void reset_clear(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        mma_policy<N>{}.template reset_clear<P>(s.mma_state, x0);
        s.raa_0 = 0.0;
        s.raa.setZero();
    }

private:
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
