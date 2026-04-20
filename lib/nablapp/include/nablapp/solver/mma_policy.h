#ifndef HPP_GUARD_NABLAPP_SOLVER_MMA_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_MMA_POLICY_H

// Method of Moving Asymptotes (MMA) policy.
//
// Solves inequality-constrained optimization problems by iteratively
// building convex separable reciprocal approximations around moving
// asymptotes and solving the resulting dual subproblem under a
// Svanberg 2002 inner conservativity loop.  Asymptote schedule per
// Svanberg 1987 Section 3; structured regularization, per-iter
// rho-growth on non-conservative trials, and inter-outer rho decay
// per Svanberg 2002 Sections 3 and 4.2 (NLopt LD_MMA reference port
// at src/algs/mma/mma.c lines 265-389).  Equality constraints are
// rejected (use SQP or augmented Lagrangian for equality-constrained
// NLPs).
//
// References:
//   Svanberg 1987, "The method of moving asymptotes -- a new method
//     for structural optimization", IJNME 24:359-373.
//   Svanberg 2002, "A class of globally convergent optimization
//     methods based on conservative convex separable approximations",
//     SIAM J. Optim. 12(2):555-573.
//   NLopt LD_MMA reference (Steven G. Johnson 2008-2012 implementation
//     of Svanberg 2002 CCSA): src/algs/mma/mma.c lines 265-389.

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
struct mma_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = mma_policy<M>;

    struct options_type
    {
        std::optional<double> move_limit{};              // default: 0.9 (Svanberg 1987 Section 5)
        std::optional<double> asymptote_init{};          // default: 0.5 (Svanberg 1987)
        std::optional<double> asymptote_decrease{};      // default: 0.7 (Svanberg 1987)
        std::optional<double> asymptote_increase{};      // default: 1.2 (Svanberg 1987)
        std::optional<double> kkt_tolerance{};           // default: 1e-6 (Svanberg 1987)
        std::optional<double> effective_bounds_scale{};  // default: 10.0

        // Svanberg 2002 Section 4.2 inner conservativity loop controls
        // (NLopt LD_MMA reference at src/algs/mma/mma.c:265-389).  These
        // knobs parameterize the rho / rhoc apparatus -- a structured
        // regularizer that grows on non-conservative inner-loop trials
        // within an outer iter and decays between outer iters.  Distinct
        // from the GCMMA per-component raa machinery (gcmma_policy uses
        // the raa_* family); MMA uses scalar rho + per-constraint rhoc
        // per Svanberg 2002 paper notation.
        //
        // Reference: Svanberg 2002, "A class of globally convergent
        //            optimization methods based on conservative convex
        //            separable approximations", SIAM J. Optim. 12(2),
        //            Sections 3 and 4.2; NLopt mma.c:265-389.
        // rho_init default is 0.1 rather than NLopt mma.c's 1.0 because the
        // in-tree mma_subproblem coefficient form scales rho as
        // `rho / (U - L)` per dimension (Svanberg 2002 GCMMA structured
        // form at detail/mma_subproblem.h:275-280), whereas NLopt's mma.c
        // uses `0.5 * rho` additively in its quadratic-equation kernel
        // (mma.c:102-105).  Against the in-tree kernel, NLopt's `rho_init
        // = 1.0` makes the structured term swamp small near-x0 gradients
        // on cubic problems (HS024 traps the iterate at the x[1] = 0
        // boundary corner where the cubic factor zeros the objective).
        // 0.1 was empirically chosen as the value that satisfies both the
        // HS024 cubic-descent guard and the HS076 reciprocal-approximation
        // margin guard simultaneously while preserving the rho-growth /
        // rho-decay machinery for the cases that need damping.
        std::optional<double>        rho_init{};               // default: 0.1   (in-tree kernel scaling; NLopt mma.c uses 1.0)
        std::optional<double>        rho_min{};                // default: 1e-5  (NLopt MMA_RHOMIN floor)
        std::optional<double>        rho_growth{};             // default: 1.1   (Svanberg 2002 paper-faithful)
        std::optional<double>        rho_growth_cap{};         // default: 10.0  (NLopt mma.c growth cap)
        std::optional<double>        rho_decay{};              // default: 0.1   (NLopt mma.c:385 inter-outer)
        std::optional<std::uint16_t> max_inner_iterations{};   // default: 15    (Svanberg 2002 Section 4.2)
        std::optional<double>        conservativity_slack{};   // default: 1e-7  (matches gcmma epsimin)

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
        // rho_saturated_stall_consecutive_count: number of consecutive
        //     outer iterations in which the conservativity regularizer
        //     saturates at rho_growth_cap before firing the stall
        //     signal.
        // kkt_jump_threshold_factor: single-event multiplicative ratio
        //     of kkt_residual between two consecutive outer iterations
        //     that fires the destabilization signal.
        //
        // Reference: Svanberg 1987, "The method of moving asymptotes",
        //            IJNME 24:359-373, Theorem 5.1.
        std::optional<std::uint16_t> asymptote_floor_stall_consecutive_count{}; // default: 5
        std::optional<std::uint16_t> rho_saturated_stall_consecutive_count{};   // default: 5
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

        asymptote_options asymptote{};                   // Embedded asymptote update params
        mma_subproblem_options subproblem{};              // Embedded subproblem params
        std::uint16_t stall_window{50};
        double feasibility_gate{1e-4};
    };

    template <typename P = void>
    struct state_type
    {
        static constexpr int M = [] {
            if constexpr(has_constraint_count<P>) return constraint_count_v<P>;
            else return dynamic_dimension;
        }();

        const P* problem{nullptr};
        Eigen::Vector<double, N> x, g;
        double f{};
        Eigen::VectorXd c_eq;
        Eigen::Vector<double, M> c_ineq;
        Eigen::Matrix<double, M, N> J_ineq;

        Eigen::Vector<double, N> L, U;
        Eigen::Vector<double, N> x_old1, x_old2;
        Eigen::Vector<double, N> lower, upper;

        std::optional<detail::mma_subproblem_solver<double, N, M>> subproblem;
        std::uint32_t iteration{0};

        // Svanberg 2002 Section 4.2 conservativity regularizer state.
        // rho is a scalar; rhoc is per-constraint.  Both initialized to
        // rho_init in init(), grown on non-conservative inner-loop trials
        // per the NLopt mma.c:363-370 formula (mirror of gcmma's
        // raacof-based growth at gcmma_policy.h:339-353), decayed by
        // rho_decay between outer iters per NLopt mma.c:385-389.
        //
        // Reference: Svanberg 2002 Section 3 (regularization role);
        //            Section 4.2 (growth + decay rules);
        //            NLopt mma.c:209-214 (init), :363-370 (growth),
        //            :385-389 (inter-outer decay).
        double rho{0.0};
        Eigen::Vector<double, M> rhoc;

        // Svanberg 1987 Theorem 5.1 Cauchy-sequence precondition:
        // finite-iteration termination detection is mathematically well-
        // defined.  The three policy_status signals are the mechanical
        // realization of that termination.  All three counters
        // initialized to zero; the two rolling counters increment on
        // consecutive signal fires and reset to zero on non-fire;
        // kkt_previous stores the prior outer iter's kkt_residual for
        // the single-event ratio comparison.
        //
        // Reference: Svanberg 1987, "The method of moving asymptotes",
        //            IJNME 24:359-373, Theorem 5.1 (Cauchy-sequence
        //            convergence).
        std::uint16_t asymptote_floor_consecutive_count{0};
        std::uint16_t rho_saturated_consecutive_count{0};
        double        kkt_previous{0.0};

        options_type opts;
    };

    template <typename Problem, typename Convergence>
    state_type<Problem> init(const Problem& problem,
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

        // Re-initialize rho/rhoc from the freshly-overridden opts so the
        // first overload's user-supplied policy_opts.rho_init takes effect
        // (mirrors the asymptote_init re-init pattern in the loop above).
        const double rho_init_val_o = s.opts.rho_init.value_or(0.1);
        s.rho = rho_init_val_o;
        s.rhoc.setConstant(rho_init_val_o);

        return s;
    }

    template <typename Problem, typename Convergence = default_convergence>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<double, N>& x0,
                             const solver_options<Convergence>& /*opts*/)
    {
        static_assert(differentiable<Problem>,
                      "mma_policy requires differentiable<Problem>");
        static_assert(constrained<Problem>,
                      "mma_policy requires constrained<Problem>");

        // MMA handles inequality constraints only;
        // num_equality() is not constexpr, so checked at runtime.
        assert(problem.num_equality() == 0
               && "MMA handles inequality constraints only. "
                  "Use SQP or augmented Lagrangian for equality constraints.");

        constexpr int MC = state_type<Problem>::M;

        const int n = problem.dimension();
        const int m_ineq = problem.num_inequality();
        state_type<Problem> s;
        s.problem = &problem;

        s.x = x0;
        s.g.resize(n);
        // c_eq stays zero-size: MMA handles inequality constraints only.
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

        // Svanberg 2002 Section 4.2 regularizer init: rho and per-
        // constraint rhoc both start at rho_init (NLopt mma.c:209-214).
        // rhoc.resize(0) is a no-op safe call when m_ineq == 0
        // (the policy rejects equality-constrained problems but is
        // valid for unconstrained-but-bounded inputs in principle).
        const double rho_init_val = s.opts.rho_init.value_or(0.1);
        s.rho = rho_init_val;
        s.rhoc.resize(m_ineq);
        s.rhoc.setConstant(rho_init_val);

        s.subproblem.emplace(n, m_ineq);

        return s;
    }

    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        constexpr int MC = state_type<P>::M;

        const int n = static_cast<int>(s.x.size());
        const int m = static_cast<int>(s.c_ineq.size());

        double move_lim = s.opts.move_limit.value_or(0.9);
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
        Eigen::Vector<double, MC> g_mma = -s.c_ineq;
        Eigen::Matrix<double, MC, N> dg_mma = -s.J_ineq;

        // 4. Svanberg 2002 Section 4.2 inner conservativity loop.
        //
        // Per-iteration: compute coefficients with the current regularizer
        // (s.rho, s.rhoc); solve the dual subproblem; evaluate trial
        // f / c_ineq; test the conservativity inequality
        //     f_approx(x_trial) + epsimin >= f(x_trial) AND
        //     g_approx_i(x_trial) + epsimin >= g_actual_i  for all i.
        // If conservative -> break and accept the trial.  Otherwise grow
        // rho / rhoc per the NLopt mma.c:363-370 formula and retry.  On
        // max_inner_iterations exhaustion, return a null step
        // (step_size = 0, is_null_step = true) so the Svanberg 2002
        // global convergence proof's closure condition holds.  The next
        // outer iter starts with the grown regularizer.
        //
        // The per-step weight uses an in-tree raacof step-geometry
        // formula (mirror of gcmma_policy.h raacof) instead of NLopt's
        // wval-based formula at mma.c:363-370.  wval is not currently
        // exposed by mma_subproblem_solver; raacof has the same
        // positive-definite step-weight regularization shape.
        //
        // Reference: Svanberg 2002 Section 4.2 (inner conservativity
        //            loop + rho-growth schedule);
        //            NLopt mma.c:265-376 (Steven G. Johnson 2008-2012
        //            implementation of LD_MMA = Svanberg 2002 CCSA).
        const std::uint16_t max_inner = s.opts.max_inner_iterations.value_or(15);
        const double epsimin = s.opts.conservativity_slack.value_or(1e-7);
        const double rho_growth = s.opts.rho_growth.value_or(1.1);
        const double rho_growth_cap = s.opts.rho_growth_cap.value_or(10.0);

        Eigen::Vector<double, N> x_trial(n);
        double f_trial{};
        // Use VectorXd (dynamic) instead of Vector<double, MC> here so the
        // m == 0 unconstrained-but-bounded edge case constructs cleanly even
        // when MC is fixed-size.  Eigen handles the dynamic-to-fixed
        // assignment to s.c_ineq below (which is Eigen::Vector<double, M>)
        // at runtime via its built-in conversion.
        Eigen::VectorXd c_ineq_trial(m);

        for(std::uint16_t inner = 0; inner < max_inner; ++inner)
        {
            s.subproblem->compute_coefficients(
                s.x, s.f, s.g, g_mma, dg_mma, s.L, s.U,
                s.rho, s.rhoc,
                s.opts.subproblem);

            x_trial = s.subproblem->dual_solve(
                s.L, s.U, alpha, beta,
                s.opts.subproblem);

            f_trial = s.problem->value(x_trial);
            if(m > 0)
                s.problem->constraints(x_trial, c_ineq_trial);

            // Conservativity test per Svanberg 2002 Section 3 + Section
            // 4.2; NLopt parity at mma.c:302 (objective) and mma.c:314-318
            // (per-constraint).  approx_f and g_approx_i are evaluated
            // from the subproblem's coefficient state populated by the
            // most recent compute_coefficients call.
            const double approx_f = s.subproblem->subproblem_value(
                x_trial, s.L, s.U);

            bool conservative_obj = (f_trial <= approx_f + epsimin);
            bool conservative_ineq = true;
            for(int i = 0; i < m && conservative_ineq; ++i)
            {
                const double g_actual = -c_ineq_trial[i];
                const double g_approx = s.subproblem->subproblem_constraint(
                    i, x_trial, s.L, s.U);
                if(g_actual > g_approx + epsimin)
                    conservative_ineq = false;
            }

            if(conservative_obj && conservative_ineq)
                break;

            // rho / rhoc growth on non-conservative trial.  The raacof
            // step-geometry weight (mirror of gcmma_policy.h) plays the
            // role of NLopt's wval at mma.c:363-370.  raacof is a
            // positive-definite step weight: sum over j of
            // (dx_j / (U_j - x_trial_j)) * (dx_j / (x_trial_j - L_j))
            // scaled by (U_j - L_j) / max(upper_j - lower_j, 1).
            //
            // Reference: NLopt mma.c:363-370 (canonical growth shape);
            //            gcmma_policy.h raacof block (in-tree analog).
            double raacof = 0.0;
            for(int j = 0; j < n; ++j)
            {
                const double dx = x_trial[j] - s.x[j];
                const double denom_ux = std::max(s.U[j] - x_trial[j], 1e-12);
                const double denom_xl = std::max(x_trial[j] - s.L[j], 1e-12);
                const double xxux = dx / denom_ux;
                const double xxxl = dx / denom_xl;
                const double scale_range = std::max(s.upper[j] - s.lower[j], 1.0);
                const double ulxx = (s.U[j] - s.L[j]) / scale_range;
                raacof += (xxux * xxxl) * ulxx;
            }
            raacof = std::max(raacof, 1e-12);

            if(!conservative_obj)
            {
                const double delta = (f_trial - approx_f) / raacof;
                s.rho = std::min(rho_growth * (s.rho + delta),
                                 rho_growth_cap * s.rho);
            }

            for(int i = 0; i < m; ++i)
            {
                const double g_actual = -c_ineq_trial[i];
                const double g_approx = s.subproblem->subproblem_constraint(
                    i, x_trial, s.L, s.U);
                if(g_actual > g_approx + epsimin)
                {
                    const double delta = (g_actual - g_approx) / raacof;
                    s.rhoc[i] = std::min(rho_growth * (s.rhoc[i] + delta),
                                         rho_growth_cap * s.rhoc[i]);
                }
            }

            // Null-step return on max_inner exhaustion.  Byte-for-byte
            // mirror of gcmma_policy.h null-step exit modulo state-name
            // swap (ms.X -> s.X) and the empty c_eq local (mma is
            // inequality-only).  Preserves Svanberg 2002 Section 4.2
            // closure condition for the global convergence proof.
            //
            // Reference: Svanberg 2002 Section 4.2; gcmma_policy.h
            //            null-step exit (in-tree byte template).
            if(inner + 1 == max_inner)
            {
                const Eigen::Matrix<double, 0, N> J_eq_empty_ex(0, n);
                const Eigen::Vector<double, 0> lambda_eq_empty_ex;
                const Eigen::Vector<double, 0> c_eq_empty_ex;
                const auto& mu_ineq_ex = s.subproblem->multipliers();
                const double kkt_ex = detail::kkt_residual<double, N, 0, MC>(
                    s.g,
                    J_eq_empty_ex,
                    s.J_ineq,
                    lambda_eq_empty_ex,
                    mu_ineq_ex,
                    c_eq_empty_ex,
                    s.c_ineq);

                // Null-step path: only the kkt-jump signal applies here.
                // The asymptote-floor and rho-saturation counters are
                // tied to the outer-iter accept-commit which the null
                // step does NOT reach; incrementing them on null steps
                // would double-count.  The null step itself already
                // signals an iteration that could not produce a
                // conservative trial; layering the kkt-jump comparison
                // on top catches the destabilization-within-exhaustion
                // case.  s.kkt_previous is NOT updated here because the
                // null step does not commit a new outer iter; it stays
                // anchored to the last committed outer iter.
                //
                // Reference: Svanberg 1987 Theorem 5.1 (Cauchy-sequence
                //            convergence); Svanberg 2002 Section 4.2
                //            Proposition 1 (conservativity precondition).
                std::optional<solver_status> policy_status_ex{};
                const double kkt_jump_factor_ex =
                    s.opts.kkt_jump_threshold_factor.value_or(1000.0);
                if(s.iteration >= 2 && s.kkt_previous > 0.0
                   && kkt_ex / s.kkt_previous > kkt_jump_factor_ex)
                    policy_status_ex = solver_status::stalled;

                return step_result<double>{
                    .objective_value = s.f,
                    .gradient_norm = s.g.norm(),
                    .step_size = 0.0,
                    .objective_change = 0.0,
                    .improved = false,
                    .is_null_step = true,
                    .constraint_violation = detail::primal_feasibility_inf(
                        c_eq_empty_ex, s.c_ineq),
                    .x_norm = s.x.norm(),
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
        // boundary-crossing trajectories.
        //
        // The rejection path returns BEFORE the inter-outer rho decay
        // below: s.x / s.f stay anchored and rho stays at its current-
        // grown value, forcing asymptote schedule contraction on the
        // next outer iter.  policy_status is deliberately left unset
        // -- a rejection is a retry request, not a stall, convergence,
        // or divergence.
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
        //            axis specialization that keeps the mma_policy
        //            step shape simple.
        const double descent_slack = 1e-12;
        const double cv_improvement_slack = 0.0;
        const double feasibility_tolerance = 1e-9;
        const Eigen::Matrix<double, 0, N> J_eq_empty_dr(0, n);
        const Eigen::Vector<double, 0> lambda_eq_empty_dr;
        const Eigen::Vector<double, 0> c_eq_empty_dr;
        const double cv_current = detail::primal_feasibility_inf(
            c_eq_empty_dr, s.c_ineq);
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
        const bool f_worsened = f_trial > s.f + descent_slack;
        const bool cv_improved =
            cv_trial < cv_current - cv_improvement_slack;
        if(!both_already_feasible && f_worsened && !cv_improved)
        {
            const auto& mu_ineq_dr = s.subproblem->multipliers();
            const double kkt_dr = detail::kkt_residual<double, N, 0, MC>(
                s.g,
                J_eq_empty_dr,
                s.J_ineq,
                lambda_eq_empty_dr,
                mu_ineq_dr,
                c_eq_empty_dr,
                s.c_ineq);

            return step_result<double>{
                .objective_value = s.f,
                .gradient_norm = s.g.norm(),
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
                .is_null_step = true,
                .constraint_violation = cv_current,
                .x_norm = s.x.norm(),
                .kkt_residual = kkt_dr,
            };
        }

        // Inter-outer rho decay per NLopt mma.c:385-389.  Applied at the
        // conservative-accept exit (the only path that reaches here; the
        // null-step exit returned above).  Symmetric with the gcmma path
        // decay; both decay the regularizer between outer iters so it
        // can RELAX when the trajectory is well-behaved instead of
        // monotonically saturating at the growth cap.
        //
        // Reference: NLopt mma.c:385-389 (MMA_RHOMIN-floored decay).
        const double rho_decay = s.opts.rho_decay.value_or(0.1);
        const double rho_min = s.opts.rho_min.value_or(1e-5);
        s.rho = std::max(rho_decay * s.rho, rho_min);
        for(int i = 0; i < m; ++i)
            s.rhoc[i] = std::max(rho_decay * s.rhoc[i], rho_min);

        // 5. Accept the conservative trial point.
        const double f_old = s.f;

        // 6. Shift history
        s.x_old2 = s.x_old1;
        s.x_old1 = s.x;

        // 7. Accept step
        const double step_size = (x_trial - s.x).norm();
        s.x = x_trial;
        s.f = f_trial;
        s.problem->gradient(s.x, s.g);
        if(m > 0)
            s.c_ineq = c_ineq_trial;
        if(m > 0)
        {
            Eigen::MatrixXd J_tmp(m, n);
            s.problem->constraint_jacobian(s.x, J_tmp);
            s.J_ineq = J_tmp;
        }

        // 8. Iteration++
        ++s.iteration;

        // 9. Composite first-order optimality (N&W 2e Definition 12.1 +
        //    eq. 12.34). MMA is inequality-only (equalities rejected in
        //    init()); the primal-equality leg is zero by construction.
        //
        // Sign convention: the subproblem dual y_ maps directly to the
        // nablapp mu_ineq multiplier without a flip. MMA's KKT reads
        //     grad_L_MMA = grad_f + sum_i y_i * grad(g_i)
        //                = grad_f - sum_i y_i * grad(c_ineq_i)    // g = -c_ineq
        // and nablapp's Lagrangian reads
        //     grad_L     = grad_f - sum_i mu_i * grad(c_ineq_i).
        // Setting mu_i = y_i gives the identical expression (verified
        // detail/lagrangian.h lines 42-60 and Svanberg 1987 Section 5).
        //
        // Reference: Nocedal and Wright, "Numerical Optimization" 2e,
        //            Definition 12.1 + eq. 12.34 (E-measure composition);
        //            Svanberg 1987, Section 5 (MMA dual KKT);
        //            Svanberg 2002, Section 5.
        const Eigen::Matrix<double, 0, N> J_eq_empty(0, n);
        const Eigen::Vector<double, 0> lambda_eq_empty;
        const Eigen::Vector<double, 0> c_eq_empty;

        const auto& mu_ineq = s.subproblem->multipliers();
        const double kkt = detail::kkt_residual<double, N, 0, MC>(
            s.g,
            J_eq_empty,
            s.J_ineq,
            lambda_eq_empty,
            mu_ineq,
            c_eq_empty,
            s.c_ineq);

        // Svanberg 1987 Theorem 5.1 Cauchy-sequence convergence
        // precondition: finite-iteration termination detection is the
        // mechanical realization of the theorem's implicit terminal
        // behavior.  The three signals below are derived from the in-
        // tree diagnosis traces of the early-convergence-then-
        // destabilization mechanism observed on the HS043 hot-path:
        //   1. asymptote-floor consecutive: schedule is at its safety
        //      floor for K consecutive outer iters (structurally cannot
        //      contract further).
        //   2. rho-saturated consecutive: rho has hit rho_growth_cap
        //      for K consecutive outer iters with no conservative
        //      trial -- inner conservativity loop is exhausting growth.
        //   3. kkt-jump single-event: kkt_residual jumps by more than
        //      kkt_jump_threshold_factor across two consecutive outer
        //      iters (destabilization signature).  Guarded with
        //      s.iteration >= 2 and s.kkt_previous > 0.0 per the same
        //      first-iter guard convention used by
        //      objective_tolerance_criterion::check (iter 0 / iter 1
        //      kkt values are not reliably comparable across outer
        //      iters).
        //
        // Reference: Svanberg 1987 Theorem 5.1 (Cauchy-sequence
        //            convergence); Svanberg 2002 Section 4.2
        //            Proposition 1 (conservativity precondition).  The
        //            NLopt LD_MMA reference (mma.c:253-389) relies on
        //            wrapper-level ftol_rel / xtol_rel
        //            (nlopt_stop_ftol at mma.c:378 and nlopt_stop_x at
        //            mma.c:380) instead of policy-internal detection.
        std::optional<solver_status> policy_status{};

        // Signal 1: asymptote-floor consecutive.  Fires when x has been
        // pushed to within 1% of one end of the asymptote interval
        // (U_j - L_j) for K consecutive outer iters.  The 0.01
        // threshold matches the L_safe / U_safe safety-shift convention
        // at the alpha / beta clipping step above (alpha / beta keep
        // trial x at least 1% inside the asymptote interval; if the
        // accepted iterate itself has reached this same 1% boundary
        // consistently, the asymptote schedule cannot contract further
        // productively).
        //
        // The ratio formulation (half-distance / interval width) uses
        // only asymptote state (s.L, s.U) and the current iterate
        // (s.x); it is well-defined regardless of problem box bounds,
        // including the unconstrained case where s.upper / s.lower hold
        // +/- infinity.  The previous floor-width formulation was
        // relative to (s.upper - s.lower) and degenerated to infinity
        // on unbounded dimensions, which made the comparison trivially
        // true and caused the counter to increment every iter.
        //
        // Reference: Svanberg 1987 Theorem 5.1 (Cauchy-sequence
        //            convergence).  The 0.01 constant matches the
        //            in-file L_safe / U_safe asymptote-interior safety
        //            convention.
        const std::uint16_t K_asymptote =
            s.opts.asymptote_floor_stall_consecutive_count.value_or(5);
        constexpr double asymptote_floor_ratio = 0.01;
        constexpr double asymptote_interval_floor = 1e-300;
        double min_ratio = std::numeric_limits<double>::infinity();
        for(Eigen::Index j = 0; j < n; ++j)
        {
            const double interval = s.U[j] - s.L[j];
            if(interval < asymptote_interval_floor)
            {
                min_ratio = 0.0;
                break;
            }
            const double half_width = std::min(s.x[j] - s.L[j],
                                               s.U[j] - s.x[j]);
            const double ratio = half_width / interval;
            min_ratio = std::min(min_ratio, ratio);
        }
        if(min_ratio <= asymptote_floor_ratio)
            ++s.asymptote_floor_consecutive_count;
        else
            s.asymptote_floor_consecutive_count = 0;
        if(s.asymptote_floor_consecutive_count >= K_asymptote)
            policy_status = solver_status::stalled;

        // Signal 2: rho saturated consecutive.  "rho at cap" means
        // s.rho >= rho_growth_cap * rho_init (the algebraic ceiling of
        // the inner-loop growth rule) within a small epsilon.  The
        // inner loop may run to max_inner_iterations (null-step return)
        // without ever finding a conservative trial; this signal fires
        // when that pathology repeats across consecutive outer iters.
        const std::uint16_t K_saturation =
            s.opts.rho_saturated_stall_consecutive_count.value_or(5);
        const double rho_init_for_cap = s.opts.rho_init.value_or(0.1);
        const double rho_saturation_threshold =
            rho_growth_cap * rho_init_for_cap * (1.0 - 1e-9);
        if(s.rho >= rho_saturation_threshold)
            ++s.rho_saturated_consecutive_count;
        else
            s.rho_saturated_consecutive_count = 0;
        if(!policy_status.has_value()
           && s.rho_saturated_consecutive_count >= K_saturation)
            policy_status = solver_status::stalled;

        // Signal 3: kkt jump single-event.  Guarded with
        // s.iteration >= 2 and s.kkt_previous > 0.0 so that iter 0 /
        // iter 1 kkt values (which are not reliably comparable across
        // outer iters) do not spuriously fire the signal; matches the
        // first-iter guard convention used elsewhere in the convergence
        // machinery.
        const double kkt_jump_factor =
            s.opts.kkt_jump_threshold_factor.value_or(1000.0);
        if(!policy_status.has_value()
           && s.iteration >= 2 && s.kkt_previous > 0.0
           && kkt / s.kkt_previous > kkt_jump_factor)
            policy_status = solver_status::stalled;
        s.kkt_previous = kkt;

        // Primal feasibility reported as L-infinity (matches the
        // dimensional form used by the E-measure legs 2 and 3). Other
        // constrained policies migrated to L-infinity in an earlier
        // convergence-polish pass.
        //
        // Reference: Nocedal and Wright, "Numerical Optimization" 2e,
        //            Definition 12.1 (primal feasibility).
        return step_result<double>{
            .objective_value = s.f,
            .gradient_norm = s.g.norm(),
            .step_size = step_size,
            .objective_change = s.f - f_old,
            .improved = s.f < f_old,
            .constraint_violation = detail::primal_feasibility_inf(
                c_eq_empty, s.c_ineq),
            .x_norm = s.x.norm(),
            .kkt_residual = kkt,
            .policy_status = policy_status,
        };
    }

    template <typename P>
    void reset(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        s.x = x0;
        s.f = s.problem->value(x0);
        s.problem->gradient(x0, s.g);
        if(static_cast<int>(s.c_ineq.size()) > 0)
        {
            Eigen::VectorXd c_tmp(s.c_ineq.size());
            s.problem->constraints(x0, c_tmp);
            s.c_ineq = c_tmp;
        }
        if(s.J_ineq.rows() > 0)
        {
            Eigen::MatrixXd J_tmp(s.J_ineq.rows(), s.J_ineq.cols());
            s.problem->constraint_jacobian(x0, J_tmp);
            s.J_ineq = J_tmp;
        }
        s.iteration = 0;
        s.x_old1 = x0;
        s.x_old2 = x0;

        // Re-initialize rho/rhoc from rho_init so a re-run starts with
        // a clean regularizer state (stale rho/rhoc across reset() would
        // give non-reproducible behavior).
        const double rho_init_val_r = s.opts.rho_init.value_or(0.1);
        s.rho = rho_init_val_r;
        s.rhoc.setConstant(rho_init_val_r);
    }

    template <typename P>
    void reset_clear(state_type<P>& s,
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
