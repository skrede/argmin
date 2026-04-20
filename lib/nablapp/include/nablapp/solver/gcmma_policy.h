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

        // Svanberg 1987 Theorem 5.1 Cauchy-sequence convergence
        // precondition: finite-iteration termination detection is the
        // mechanical realization of the theorem's implicit terminal
        // behavior.  These three thresholds parameterize the policy-
        // internal stall / divergence signal cascade surfaced via
        // step_result.policy_status.  All three are std::optional with
        // value_or() defaults at the usage sites so users can override
        // per-problem without reaching into the cascade logic.
        //
        // asymptote_floor_stall_consecutive_count: number of consecutive
        //     outer iterations in which the asymptote schedule sits at
        //     its safety floor before firing the stall signal.
        // raa_saturated_stall_consecutive_count: number of consecutive
        //     outer iterations in which the raa_0 regularizer saturates
        //     at raa_max_factor before firing the stall signal
        //     (symmetric with mma's rho_saturated signal, renamed for
        //     the per-component raa machinery).
        // kkt_jump_threshold_factor: single-event multiplicative ratio
        //     of kkt_residual between two consecutive outer iterations
        //     that fires the destabilization signal.
        //
        // Reference: Svanberg 1987, "The method of moving asymptotes",
        //            IJNME 24:359-373, Theorem 5.1; Svanberg 2002,
        //            Section 4.2 Proposition 1 (conservativity
        //            precondition).
        std::optional<std::uint16_t> asymptote_floor_stall_consecutive_count{}; // default: 5
        std::optional<std::uint16_t> raa_saturated_stall_consecutive_count{};   // default: 5
        std::optional<double>        kkt_jump_threshold_factor{};               // default: 1000.0

        // Svanberg 1987 Theorem 5.1 Cauchy-sequence convergence precondition:
        // the stall_tolerance_criterion framework-side mechanism requires a
        // threshold to be non-nullopt to activate (the criterion's
        // `if(!threshold) return std::nullopt;` short-circuit disables it
        // when the option is unset).  The framework-side wire at
        // basic_solver::forward_policy_hints consumes this option via
        // value_or and also auto-enables the best_seen_feasible metric for
        // the MMA / GCMMA policy family, which is robust against the
        // objective-oscillation mode documented in in-tree diagnosis traces.
        // Unspecified => wire applies default 1e-6 per the forward-hint
        // rule; the criterion then fires on a stalled best-seen-feasible
        // metric inside the rolling window.
        //
        // Reference: Svanberg 1987, "The method of moving asymptotes",
        //            IJNME 24:359-373, Theorem 5.1.
        std::optional<double> stall_tolerance_threshold{};  // default: 1e-6 (wired at forward_policy_hints)

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

        // Svanberg 1987 Theorem 5.1 Cauchy-sequence precondition:
        // finite-iteration termination detection is mathematically well-
        // defined.  raa_saturated_consecutive_count and kkt_previous
        // are gcmma-owned counters alongside raa_0 / raa; the third
        // counter (asymptote_floor_consecutive_count) lives on the
        // inner mma_state because the asymptote machinery is mma-
        // subproblem-owned.  The rolling counter increments on
        // consecutive signal fires and resets to zero on non-fire;
        // kkt_previous stores the prior outer iter's kkt_residual for
        // the single-event ratio comparison.
        //
        // Reference: Svanberg 1987, "The method of moving asymptotes",
        //            IJNME 24:359-373, Theorem 5.1; Svanberg 2002,
        //            Section 4.2 Proposition 1.
        std::uint16_t raa_saturated_consecutive_count{0};
        double        kkt_previous{0.0};

        // Per-step flag set inside the inner conservativity loop whenever
        // the raa_0 growth rule hits its per-iter cap (raa_max_factor *
        // raa_0 term winning the min over raa_growth * (raa_0 + delta)).
        // Reset at step() entry and evaluated at Signal 2 on the accept
        // path.  This is what "raa saturated" actually means
        // algorithmically -- a fixed absolute threshold on raa_0 is
        // scale-dependent and fires spuriously on problems whose
        // initial raa_0 (max(raa0_floor, 0.1 * |grad| / n)) already
        // sits above the fixed threshold.
        //
        // Reference: Svanberg 2002 Section 4.2 Proposition 1 (GCMMA
        //            per-constraint conservativity regularization).
        bool raa_cap_hit_in_inner{false};

        // Proxy members for basic_solver compatibility (it accesses state_.x,
        // state_.c_eq, state_.c_ineq directly).
        Eigen::Vector<double, N>& x = mma_state.x;
        Eigen::VectorXd& c_eq = mma_state.c_eq;
        Eigen::Vector<double, M>& c_ineq = mma_state.c_ineq;

        state_type() = default;
        state_type(const state_type& o)
            : mma_state{o.mma_state}, opts{o.opts}
            , raa_0{o.raa_0}, raa{o.raa}
            , raa_saturated_consecutive_count{o.raa_saturated_consecutive_count}
            , kkt_previous{o.kkt_previous}
            , raa_cap_hit_in_inner{o.raa_cap_hit_in_inner}
            , x{mma_state.x}, c_eq{mma_state.c_eq}, c_ineq{mma_state.c_ineq}
        {}
        state_type(state_type&& o) noexcept
            : mma_state{std::move(o.mma_state)}, opts{std::move(o.opts)}
            , raa_0{o.raa_0}, raa{std::move(o.raa)}
            , raa_saturated_consecutive_count{o.raa_saturated_consecutive_count}
            , kkt_previous{o.kkt_previous}
            , raa_cap_hit_in_inner{o.raa_cap_hit_in_inner}
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
                raa_saturated_consecutive_count = o.raa_saturated_consecutive_count;
                kkt_previous = o.kkt_previous;
                raa_cap_hit_in_inner = o.raa_cap_hit_in_inner;
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
                raa_saturated_consecutive_count = o.raa_saturated_consecutive_count;
                kkt_previous = o.kkt_previous;
                raa_cap_hit_in_inner = o.raa_cap_hit_in_inner;
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

        // Reset the per-step saturation flag before the inner loop;
        // Signal 2 reads this after the loop exits conservatively.
        s.raa_cap_hit_in_inner = false;

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
                const double grown = raa_growth * (s.raa_0 + delta);
                const double capped = raa_max_factor * s.raa_0;
                if(capped < grown)
                    s.raa_cap_hit_in_inner = true;
                s.raa_0 = std::min(grown, capped);
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

                // Null-step path: only the kkt-jump signal applies here.
                // The asymptote-floor and raa-saturation counters are
                // tied to the outer-iter accept-commit which the null
                // step does NOT reach; incrementing them on null steps
                // would double-count.  s.kkt_previous is NOT updated
                // here because the null step does not commit a new
                // outer iter; it stays anchored to the last committed
                // outer iter.
                //
                // Reference: Svanberg 1987 Theorem 5.1 (Cauchy-sequence
                //            convergence); Svanberg 2002 Section 4.2
                //            Proposition 1 (conservativity precondition).
                std::optional<solver_status> policy_status_ex{};
                const double kkt_jump_factor_ex =
                    s.opts.kkt_jump_threshold_factor.value_or(1000.0);
                if(ms.iteration >= 2 && s.kkt_previous > 0.0
                   && kkt_ex / s.kkt_previous > kkt_jump_factor_ex)
                    policy_status_ex = solver_status::stalled;

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
                    .policy_status = policy_status_ex,
                };
            }
        }

        // Svanberg 2002 Section 4.2 Proposition 1 second precondition
        // (descent-monotone outer iterate sequence) with cv-aware
        // relaxation per the filter-method acceptance test (Fletcher
        // and Leyffer 2002 Section 2): the inner conservativity loop
        // above enforces approximation-validity (first precondition);
        // this block enforces a descent-or-feasibility-restoration
        // gate on the accepted trial.  Reject the conservative trial
        // ONLY when BOTH (a) the objective worsens and (b) primal
        // feasibility does not improve.  Any strict cv decrease gives
        // the trial a pass regardless of f rise -- the feasibility-
        // restoration trial is what pure-f rejection could not admit,
        // and the cv-aware form is the minimal extension that makes
        // the descent-monotone precondition correct across constraint-
        // boundary-crossing trajectories.  This block is symmetric
        // with the mma_policy::step descent rejection; both policies
        // share the conservativity-accept-without-descent structure.
        //
        // Rejection path returns BEFORE the accept-commit below AND
        // BEFORE the inter-outer raa decay that follows the commit.
        // raa stays at its current-grown value so the next outer iter
        // begins with the regularizer already grown, forcing
        // asymptote-schedule contraction on the next step.
        // policy_status is deliberately left unset -- a rejection is
        // a retry request, not a stall, convergence, or divergence.
        //
        // Reference: Svanberg 2002, SIAM J. Optim. 12(2):555-573,
        //            Section 4.2 Proposition 1 (descent-monotone
        //            precondition + closure condition).
        //            Fletcher and Leyffer 2002, SIAM J. Optim.
        //            13(1):44-59, Section 2 (filter acceptance:
        //            non-dominated in (f, cv)).  The in-tree
        //            filter_set implementation (used by
        //            filter_slsqp_policy and filter_nw_sqp_policy) is
        //            the fuller form; this block is the minimal cv-
        //            axis specialization that keeps the gcmma_policy
        //            step shape simple.
        const double descent_slack = 1e-12;
        const double cv_improvement_slack = 0.0;
        const double feasibility_tolerance = 1e-9;
        const double cv_current = detail::primal_feasibility_inf(
            ms.c_eq, ms.c_ineq);
        const Eigen::Vector<double, 0> c_eq_empty_dr;
        const double cv_trial = detail::primal_feasibility_inf(
            c_eq_empty_dr, c_ineq_trial);
        // Already-feasible case: when both iterates satisfy feasibility
        // within tolerance, the cv axis is not a discriminator (both
        // effectively cv = 0 and strict cv-decrease is unmeasurable at
        // floating-point roundoff).  Admit the trial unconditionally in
        // this regime -- the descent-monotone precondition applies
        // cleanly only when the cv axis can distinguish trials.  The
        // Fletcher-Leyffer 2002 filter reduces to the same behavior
        // when all entries lie on the cv = 0 contour.
        const bool both_already_feasible =
            cv_current <= feasibility_tolerance
            && cv_trial <= feasibility_tolerance;
        const bool f_worsened = f_trial > ms.f + descent_slack;
        const bool cv_improved =
            cv_trial < cv_current - cv_improvement_slack;
        if(!both_already_feasible && f_worsened && !cv_improved)
        {
            const Eigen::Matrix<double, 0, N> J_eq_empty_dr(0, n);
            const Eigen::Vector<double, 0> lambda_eq_empty_dr;
            const auto& mu_ineq_dr = ms.subproblem->multipliers();
            const double kkt_dr = detail::kkt_residual<double, N, 0, MC>(
                ms.g,
                J_eq_empty_dr,
                ms.J_ineq,
                lambda_eq_empty_dr,
                mu_ineq_dr,
                c_eq_empty_dr,
                ms.c_ineq);

            return step_result<double>{
                .objective_value = ms.f,
                .gradient_norm = ms.g.norm(),
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
                .is_null_step = true,
                .constraint_violation = cv_current,
                .x_norm = ms.x.norm(),
                .kkt_residual = kkt_dr,
            };
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

        // Svanberg 1987 Theorem 5.1 Cauchy-sequence convergence
        // precondition: finite-iteration termination detection is the
        // mechanical realization of the theorem's implicit terminal
        // behavior.  The three signals below mirror the mma-side
        // cascade with gcmma-specific adjustments: the asymptote-
        // floor counter lives on the inner mma_state (asymptote
        // machinery is mma-subproblem-owned), while raa-saturation and
        // kkt-previous are outer-state gcmma-owned.
        //   1. asymptote-floor consecutive: asymptote schedule at
        //      safety floor for K consecutive outer iters.
        //   2. raa-saturated consecutive: raa_0 has hit
        //      raa_max_factor * raa0_floor for K consecutive outer
        //      iters (the symmetric analog of mma's rho-saturated
        //      signal for the per-component raa machinery).
        //   3. kkt-jump single-event: kkt_residual jumps by more than
        //      kkt_jump_threshold_factor across two consecutive outer
        //      iters (destabilization signature).  Guarded with
        //      ms.iteration >= 2 and s.kkt_previous > 0.0 per the
        //      first-iter guard convention.
        //
        // Reference: Svanberg 1987 Theorem 5.1 (Cauchy-sequence
        //            convergence); Svanberg 2002 Section 4.2
        //            Proposition 1 (GCMMA per-constraint conservativity
        //            regularization + monotone-objective precondition).
        std::optional<solver_status> policy_status{};

        // Signal 1: asymptote-floor consecutive.  Fires when x has been
        // pushed to within 1% of one end of the asymptote interval
        // (U_j - L_j) for K consecutive outer iters.  The counter lives
        // on the inner mma_state because the asymptote machinery is
        // mma-subproblem-owned; ms.x / ms.L / ms.U are the mma-owned
        // fields.
        //
        // The 0.01 threshold matches the L_safe / U_safe safety-shift
        // convention used at the inner alpha / beta clipping step; if
        // the accepted iterate itself has reached this same 1%
        // boundary consistently, the asymptote schedule cannot
        // contract further productively.
        //
        // The ratio formulation (half-distance / interval width) uses
        // only asymptote state (ms.L, ms.U) and the current iterate
        // (ms.x); it is well-defined regardless of problem box bounds,
        // including the unconstrained case where ms.upper / ms.lower
        // hold +/- infinity.  The previous floor-width formulation was
        // relative to (ms.upper - ms.lower) and degenerated to infinity
        // on unbounded dimensions, which made the comparison trivially
        // true and caused the counter to increment every iter.
        //
        // Reference: Svanberg 1987 Theorem 5.1 (Cauchy-sequence
        //            convergence); Svanberg 2002 Section 4.2
        //            Proposition 1 (GCMMA per-constraint conservativity
        //            regularization).  The 0.01 constant matches the
        //            in-file L_safe / U_safe asymptote-interior safety
        //            convention.
        const std::uint16_t K_asymptote =
            s.opts.asymptote_floor_stall_consecutive_count.value_or(5);
        constexpr double asymptote_floor_ratio = 0.01;
        constexpr double asymptote_interval_floor = 1e-300;
        double min_ratio = std::numeric_limits<double>::infinity();
        for(Eigen::Index j = 0; j < n; ++j)
        {
            const double interval = ms.U[j] - ms.L[j];
            if(interval < asymptote_interval_floor)
            {
                min_ratio = 0.0;
                break;
            }
            const double half_width = std::min(ms.x[j] - ms.L[j],
                                               ms.U[j] - ms.x[j]);
            const double ratio = half_width / interval;
            min_ratio = std::min(min_ratio, ratio);
        }
        if(min_ratio <= asymptote_floor_ratio)
            ++ms.asymptote_floor_consecutive_count;
        else
            ms.asymptote_floor_consecutive_count = 0;
        if(ms.asymptote_floor_consecutive_count >= K_asymptote)
            policy_status = solver_status::stalled;

        // Signal 2: raa saturated consecutive.  Fires when the inner
        // conservativity loop hit the per-iter raa_0 growth cap
        // (raa_max_factor * raa_0 term winning the min over
        // raa_growth * (raa_0 + delta)) at least once within the inner
        // loop, for K consecutive outer iters.  This measures "inner
        // loop is exhausting growth" directly via the growth rule's
        // own saturation signal, rather than comparing raa_0 to a
        // fixed absolute threshold (raa_max_factor * raa0_floor).  The
        // fixed-threshold form is scale-dependent and fires
        // spuriously on problems whose initial raa_0 is already above
        // the threshold; recall that raa_0 is initialized to
        // max(raa0_floor, 0.1 * |grad| / n), which on non-trivial
        // problems sits orders of magnitude above raa0_floor.
        //
        // The s.raa_cap_hit_in_inner flag is reset at step() entry
        // and set inside the inner loop on any inner iter where the
        // cap branch of the growth min wins; it reflects this step's
        // inner loop only.
        //
        // Reference: Svanberg 2002 Section 4.2 Proposition 1 (GCMMA
        //            per-constraint conservativity regularization).
        const std::uint16_t K_saturation =
            s.opts.raa_saturated_stall_consecutive_count.value_or(5);
        if(s.raa_cap_hit_in_inner)
            ++s.raa_saturated_consecutive_count;
        else
            s.raa_saturated_consecutive_count = 0;
        if(!policy_status.has_value()
           && s.raa_saturated_consecutive_count >= K_saturation)
            policy_status = solver_status::stalled;

        // Signal 3: kkt jump single-event.  Guarded with
        // ms.iteration >= 2 and s.kkt_previous > 0.0 so iter 0 / iter 1
        // kkt values (not reliably comparable across outer iters) do
        // not spuriously fire the signal.
        const double kkt_jump_factor =
            s.opts.kkt_jump_threshold_factor.value_or(1000.0);
        if(!policy_status.has_value()
           && ms.iteration >= 2 && s.kkt_previous > 0.0
           && kkt_acc / s.kkt_previous > kkt_jump_factor)
            policy_status = solver_status::stalled;
        s.kkt_previous = kkt_acc;

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
            .policy_status = policy_status,
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
