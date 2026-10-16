#ifndef HPP_GUARD_ARGMIN_SOLVER_ALTERNATIVE_GCMMA_RHO_WVAL_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_ALTERNATIVE_GCMMA_RHO_WVAL_POLICY_H

// GCMMA variant: NLopt-mma.c-style rho-augmentation with wval-based
// growth.
//
// The MMA reciprocal approximation is augmented with a separable
// quadratic penalty term whose weight (rho) grows on non-conservative
// trials. The growth formula follows NLopt mma.c lines 388-391:
//
//   if (f_trial > g_tilde_0_at_trial)
//       rho_obj := min(10*rho_obj, 1.1*(rho_obj + (f_trial - g_tilde_0)/wval))
//   for each constraint i:
//     if (-c_trial[i] > g_tilde_i_at_trial)   // MMA convention
//         rho_con[i] := min(10*rho_con[i],
//                           1.1*(rho_con[i] + (-c_trial[i] - g_tilde_i)/wval))
//
// where wval = sum_j 0.5 * w_j * (x_trial_j - x_kj)^2 is the
// quadratic-penalty mass per unit rho at the trial point (the
// rho-linear coefficient of the augmented approximation, and the
// denominator of the minimal conservative rho increment).
//
// Trade-off vs. the move-limit-shrink variant:
//   + Adapts the approximation itself rather than wasting dual solves
//     on shrunk trust regions; on non-separable problems (HS076) it
//     should reach the optimum where shrinkage stalls.
//   - Requires a numerical per-component Newton solve in the dual
//     primal (the augmented FOC is no longer a closed-form sqrt
//     formula); each dual evaluation costs ~5-10 Newton iters per j.
//
// References:
//   Svanberg 2002, SIAM J. Optim. 12(2):555-573.
//   NLopt mma.c (Steven G. Johnson 2008-2012), lines 388-391.

#include "argmin/detail/lagrangian.h"
#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/mma_subproblem.h"
#include "argmin/detail/mma_augmented_dual_problem.h"
#include "argmin/result/step_result.h"
#include "argmin/solver/options.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/types.h"

#include "argmin/options/mma_defaults.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace argmin::alternative::gcmma
{

template <scalar Scalar = double, int N = dynamic_dimension,
          template <scalar, int> typename DualPolicy = lbfgsb_policy>
struct rho_wval_policy
{
    using scalar_type = Scalar;

    template <int M>
    using rebind = rho_wval_policy<Scalar, M, DualPolicy>;

    struct options_type
    {
        std::optional<Scalar> asymptote_contract{};       // 0.7
        std::optional<Scalar> asymptote_expand{};         // 1.2
        std::optional<Scalar> asymptote_init{};           // 0.5
        std::optional<Scalar> asymptote_init_unbounded{}; // 1.0
        std::optional<Scalar> asymptote_min_fraction{};   // 1e-4
        std::optional<Scalar> asymptote_max_fraction{};   // 10.0
        std::optional<Scalar> move_limit_fraction{};      // 0.1
        Scalar move_bound_fraction{mma_defaults<Scalar>::move_bound_fraction};  // Svanberg XXMOVE
        Scalar raai{mma_defaults<Scalar>::raai};  // p/q regularizer

        // Bounded-dual elastics: dual upper bound c_i = dual_bound_scale *
        // max(|g_i(x0)|, 1). The per-constraint scale reference keeps the
        // exact-l1-penalty condition c_i > |mu_i*| scale-relative; the
        // scale multiplier is empirical (swept). See detail/mma_subproblem.h
        // recover_elastic_slacks(). Direct value.
        Scalar dual_bound_scale{mma_defaults<Scalar>::dual_bound_scale};

        // Conservativity loop and rho parameters.
        std::optional<std::uint16_t> max_inner_iterations{};  // 15
        std::optional<Scalar> rho_init{};                     // 1.0
        std::optional<Scalar> rho_min{};                      // 1e-5
        std::optional<Scalar> rho_decay{};                    // 0.1 (between outer iters)

        std::optional<Scalar> stall_tolerance_threshold{};
        std::uint16_t stall_window{50};
    };

    template <typename P = void>
    struct state_type
    {
        static constexpr int M = [] {
            if constexpr(has_constraint_count<P>) return constraint_count_v<P>;
            else return dynamic_dimension;
        }();

        const P* problem{nullptr};
        Eigen::Vector<Scalar, N> x, g;
        Scalar f{};
        Eigen::Vector<Scalar, M> c_ineq;
        Eigen::Matrix<Scalar, M, N> J_ineq;

        Eigen::Vector<Scalar, N> L, U;
        Eigen::Vector<Scalar, N> alpha, beta;
        Eigen::Vector<Scalar, N> w;          // separable quadratic weights

        Eigen::Vector<Scalar, N> p_obj, q_obj;
        Eigen::Matrix<Scalar, M, N> p_con, q_con;
        Scalar r_obj{0};
        Eigen::Vector<Scalar, M> r_con;

        // Penalty parameters (mutate inside conservativity loop).
        Scalar rho_obj{mma_defaults<Scalar>::rho_init};
        Eigen::Vector<Scalar, M> rho_con;

        Eigen::Vector<Scalar, N> x_old1, x_old2;
        Eigen::Vector<Scalar, N> lower, upper;
        Eigen::Vector<Scalar, M> y_dual;

        // Bounded-dual elastics. c_dual_ref[i] = max(|g_i(x0)|, 1) is the
        // fixed per-constraint scale reference; c_dual[i] =
        // dual_bound_scale * c_dual_ref[i] is the working dual upper bound.
        Eigen::Vector<Scalar, M> c_dual_ref;
        Eigen::Vector<Scalar, M> c_dual;

        std::uint32_t iteration{0};

        // Set during init() when the problem carries equality constraints.
        // GCMMA sizes its constraint buffers for inequalities only; an
        // equality-constrained problem is rejected at the top of step()
        // before any constraint evaluation. See init() for the rationale.
        bool invalid_problem{false};

        options_type opts;
    };

    template <typename Problem, typename Convergence>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<Scalar, N>& x0,
                             const solver_options<Convergence>& opts,
                             const options_type& policy_opts)
    {
        auto s = init(problem, x0, opts);
        s.opts = policy_opts;
        const Scalar ri = s.opts.rho_init.value_or(mma_defaults<Scalar>::rho_init);
        s.rho_obj = ri;
        s.rho_con.setConstant(ri);
        init_asymptotes(s);
        return s;
    }

    template <typename Problem, typename Convergence = default_convergence<scalar_type>>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<Scalar, N>& x0,
                             const solver_options<Convergence>&)
    {
        static_assert(differentiable<Problem, Scalar>);
        static_assert(constrained<Problem, Scalar>);

        const int n = problem.dimension();
        const int n_eq = problem.num_equality();
        const int m = problem.num_inequality();
        // Full constraint count. problem.constraints()/constraint_jacobian()
        // write n_eq + n_ineq rows with the equalities first, so any probe
        // buffer must be sized on the total or an equality-constrained
        // problem overflows it (a heap out-of-bounds write in release,
        // where the debug precondition assert is compiled out). GCMMA
        // handles inequality constraints only; equality-constrained
        // problems are flagged here and rejected at the top of step(). Use
        // SQP or an augmented Lagrangian for equality constraints.
        const int n_all = n_eq + m;
        state_type<Problem> s;
        s.problem = &problem;
        s.invalid_problem = (n_eq > 0);

        s.x = x0;
        s.g.resize(n);
        s.c_ineq.resize(m);
        s.J_ineq.resize(m, n);
        s.L.resize(n);
        s.U.resize(n);
        s.alpha.resize(n);
        s.beta.resize(n);
        s.w.resize(n);
        s.p_obj.resize(n);
        s.q_obj.resize(n);
        s.p_con.resize(m, n);
        s.q_con.resize(m, n);
        s.r_con.resize(m);
        s.rho_con.resize(m);
        s.rho_con.setConstant(mma_defaults<Scalar>::rho_init);
        s.rho_obj = mma_defaults<Scalar>::rho_init;
        s.y_dual.resize(m);
        s.y_dual.setZero();
        s.c_dual_ref.resize(m);
        s.c_dual_ref.setOnes();
        s.c_dual.resize(m);

        s.f = problem.value(x0);
        problem.gradient(x0, s.g);
        if(n_all > 0)
        {
            // Size the probe on the full constraint count (equalities
            // first) so the evaluation cannot overflow even when the
            // problem is about to be rejected for carrying equalities.
            Eigen::VectorX<Scalar> c_tmp(n_all);
            problem.constraints(x0, c_tmp);
            Eigen::MatrixX<Scalar> J_tmp(n_all, n);
            problem.constraint_jacobian(x0, J_tmp);
            if(!s.invalid_problem)
            {
                s.c_ineq = c_tmp;
                s.J_ineq = J_tmp;
                // Elastic dual-bound scale reference max(|g_i(x0)|, 1),
                // g_i = -c_ineq_i (fixed for the run).
                for(int i = 0; i < m; ++i)
                    s.c_dual_ref[i] = std::max(std::abs(s.c_ineq[i]), Scalar(1.0));
            }
        }

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

        init_asymptotes(s);
        s.x_old1 = x0;
        s.x_old2 = x0;
        s.iteration = 0;
        return s;
    }

    template <typename P>
    step_result<Scalar> step(state_type<P>& s)
    {
        constexpr int MC = state_type<P>::M;
        const int n = static_cast<int>(s.x.size());
        const int m = static_cast<int>(s.c_ineq.size());

        // Reject equality-constrained problems before touching any
        // constraint buffer (they are sized for inequalities only). This
        // is a hard, exception-free runtime precondition failure surfaced
        // as a terminal status; the caller must route equalities through
        // SQP or an augmented Lagrangian.
        if(s.invalid_problem)
        {
            return step_result<Scalar>{
                .objective_value = s.f,
                .gradient_norm = s.g.norm(),
                .step_size = Scalar(0.0),
                .objective_change = Scalar(0.0),
                .improved = false,
                .is_null_step = true,
                .x_norm = s.x.norm(),
                .policy_status = solver_status::invalid_problem,
            };
        }

        const Scalar gam_dec = s.opts.asymptote_contract.value_or(mma_defaults<Scalar>::asymptote_contract);
        const Scalar gam_inc = s.opts.asymptote_expand.value_or(mma_defaults<Scalar>::asymptote_expand);
        const Scalar s_min_f = s.opts.asymptote_min_fraction.value_or(mma_defaults<Scalar>::asymptote_min_fraction);
        const Scalar s_max_f = s.opts.asymptote_max_fraction.value_or(mma_defaults<Scalar>::asymptote_max_fraction);
        const Scalar move_lim = s.opts.move_limit_fraction.value_or(mma_defaults<Scalar>::move_limit_fraction);
        const Scalar raai = s.opts.raai;
        const std::uint16_t max_inner =
            s.opts.max_inner_iterations.value_or(15);
        const Scalar rho_min = s.opts.rho_min.value_or(mma_defaults<Scalar>::rho_min);
        const Scalar rho_decay = s.opts.rho_decay.value_or(mma_defaults<Scalar>::rho_decay);

        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();

        // 1. Asymptote oscillation update.
        if(s.iteration >= 2)
        {
            for(int j = 0; j < n; ++j)
            {
                const Scalar dx2 =
                    (s.x[j] - s.x_old1[j]) * (s.x_old1[j] - s.x_old2[j]);
                const Scalar gam = dx2 < Scalar(0.0) ? gam_dec
                                 : dx2 > Scalar(0.0) ? gam_inc : Scalar(1.0);
                s.L[j] = s.x[j] - gam * (s.x_old1[j] - s.L[j]);
                s.U[j] = s.x[j] + gam * (s.U[j] - s.x_old1[j]);

                Scalar range;
                if(s.lower[j] > -inf && s.upper[j] < inf)
                    range = s.upper[j] - s.lower[j];
                else
                    range = std::max(std::abs(s.x[j]), Scalar(1.0));

                const Scalar min_d = std::max(s_min_f * range, mma_defaults<Scalar>::min_asymptote_distance);
                const Scalar max_d = s_max_f * range;
                if(s.x[j] - s.L[j] < min_d) s.L[j] = s.x[j] - min_d;
                if(s.x[j] - s.L[j] > max_d) s.L[j] = s.x[j] - max_d;
                if(s.U[j] - s.x[j] < min_d) s.U[j] = s.x[j] + min_d;
                if(s.U[j] - s.x[j] > max_d) s.U[j] = s.x[j] + max_d;
            }
        }

        // 2. Move limits + separable quadratic weights. Move limits
        //    (Svanberg 2007 notes eqs. 2.8-2.9): the box, the 0.1 asymptote
        //    buffer, and the XXMOVE=0.5 move bound x_k +/- move_bnd*range_j
        //    with the box-width analog 2*max(|x_kj|, 1) for an unbounded
        //    variable (engages only once the asymptote inflates).
        const Scalar move_bnd = s.opts.move_bound_fraction;
        for(int j = 0; j < n; ++j)
        {
            Scalar range;
            if(s.lower[j] > -inf && s.upper[j] < inf)
                range = s.upper[j] - s.lower[j];
            else
                range = Scalar(2.0) * std::max(std::abs(s.x[j]), Scalar(1.0));
            const Scalar mb = move_bnd * range;

            s.alpha[j] = std::max({s.lower[j],
                                   s.L[j] + move_lim * (s.x[j] - s.L[j]),
                                   s.x[j] - mb});
            s.beta[j]  = std::min({s.upper[j],
                                   s.U[j] - move_lim * (s.U[j] - s.x[j]),
                                   s.x[j] + mb});
            // Asymmetric-asymptote generalization of NLopt's
            // sigma^-2 weight. In the symmetric case U-x_k = x_k-L =
            // sigma this weight (U-L)/(dxU*dxL) = 2*sigma/sigma^2 = 2/sigma;
            // the factor-2 is absorbed into the 0.5 penalty-mass scaling so
            // the augmented quadratic matches NLopt's 0.5*(dx/sigma)^2.
            const Scalar dxU = s.U[j] - s.x[j];
            const Scalar dxL = s.x[j] - s.L[j];
            s.w[j] = (s.U[j] - s.L[j]) / (dxU * dxL);
        }

        // 3. Approximation params + r_i constants.
        s.r_obj = s.f;
        for(int i = 0; i < m; ++i)
            s.r_con[i] = -s.c_ineq[i];

        for(int j = 0; j < n; ++j)
        {
            const Scalar dxU = s.U[j] - s.x[j];
            const Scalar dxL = s.x[j] - s.L[j];

            // Distance-scaled regularizer raai/range_j (Svanberg 2007
            // notes eqs. 2.3-2.4) with a scale proxy for unbounded
            // variables, plus the 1.001/0.001 gradient mixing so a
            // zero-gradient component keeps its minimizer at x_k.
            Scalar range;
            if(s.lower[j] > -inf && s.upper[j] < inf)
                range = s.upper[j] - s.lower[j];
            else
                range = std::max(std::abs(s.x[j]), Scalar(1.0));
            const Scalar raa_reg = raai / range;

            const Scalar gp = std::max(s.g[j], Scalar(0.0));
            const Scalar gm = std::max(-s.g[j], Scalar(0.0));
            s.p_obj[j] = dxU * dxU * (Scalar(1.001) * gp + Scalar(0.001) * gm + raa_reg);
            s.q_obj[j] = dxL * dxL * (Scalar(0.001) * gp + Scalar(1.001) * gm + raa_reg);
            s.r_obj -= s.p_obj[j] / dxU + s.q_obj[j] / dxL;

            for(int i = 0; i < m; ++i)
            {
                const Scalar gij = -s.J_ineq(i, j);
                const Scalar gpij = std::max(gij, Scalar(0.0));
                const Scalar gmij = std::max(-gij, Scalar(0.0));
                s.p_con(i, j) =
                    dxU * dxU * (Scalar(1.001) * gpij + Scalar(0.001) * gmij + raa_reg);
                s.q_con(i, j) =
                    dxL * dxL * (Scalar(0.001) * gpij + Scalar(1.001) * gmij + raa_reg);
                s.r_con[i] -= s.p_con(i, j) / dxU + s.q_con(i, j) / dxL;
            }
        }

        // 4. Conservativity loop with rho-growth on non-conservative trial.
        detail::mma_augmented_dual_problem<Scalar, N, MC> dual_prob;
        dual_prob.L_out = &s.L;
        dual_prob.U_out = &s.U;
        dual_prob.x_k_out = &s.x;
        dual_prob.alpha_out = &s.alpha;
        dual_prob.beta_out = &s.beta;
        dual_prob.w_out = &s.w;
        dual_prob.p_obj_out = &s.p_obj;
        dual_prob.q_obj_out = &s.q_obj;
        dual_prob.p_con_out = &s.p_con;
        dual_prob.q_con_out = &s.q_con;
        dual_prob.r_obj = s.r_obj;
        dual_prob.r_con_out = &s.r_con;
        dual_prob.rho_con_out = &s.rho_con;
        dual_prob.n_primal = n;
        dual_prob.m_dual = m;
        dual_prob.x_primal.resize(n);
        dual_prob.gcval.resize(m);

        // Bounded-dual elastics: box the constraint multipliers at
        // c_i = dual_bound_scale * max(|g_i(x0)|, 1) so an inequality-
        // infeasible iterate cannot make the subproblem infeasible and
        // drive the dual unbounded (Svanberg 2002 relaxed subproblem,
        // a_i = 0 instance).
        if(m > 0)
            s.c_dual = s.opts.dual_bound_scale * s.c_dual_ref;
        dual_prob.c_dual_out = &s.c_dual;

        Eigen::Vector<Scalar, N> x_trial = s.x;
        Scalar f_trial = s.f;
        Eigen::VectorX<Scalar> c_trial = s.c_ineq;

        // Persistent inner dual solver: construct the box-constrained solver
        // state once per outer iteration and cold-restart it (reset_clear
        // clears the L-BFGS curvature so each inner solve starts fresh, as a
        // per-inner re-construction would, while reusing the allocated
        // buffers) rather than re-allocating it on every conservativity
        // iteration.
        DualPolicy<scalar_type, MC> dp;
        solver_options<default_convergence<scalar_type>> dopts;
        dopts.max_iterations = 100;
        dopts.set_gradient_threshold(mma_defaults<Scalar>::dual_gradient_threshold);
        dopts.set_step_threshold(mma_defaults<Scalar>::dual_step_threshold);
        using dual_state_t = decltype(dp.init(dual_prob, s.y_dual, dopts));
        std::optional<dual_state_t> ds;

        for(std::uint16_t inner = 0; inner < max_inner; ++inner)
        {
            dual_prob.rho_obj = s.rho_obj;
            // rho_obj/rho_con changed since the previous inner iteration:
            // drop the single-evaluation cache before re-solving the dual.
            dual_prob.invalidate_cache();

            if(m > 0)
            {
                if(!ds)
                    ds.emplace(dp.init(dual_prob, s.y_dual, dopts));
                else
                    dp.reset_clear(*ds, s.y_dual);
                for(int k = 0; k < 100; ++k)
                {
                    auto dsr = dp.step(*ds);
                    // Terminate on the projected KKT residual (the box-
                    // constrained dual's stationarity measure), not the raw
                    // gradient norm: at a dual solution with an inactive
                    // constraint (y_i = 0) the raw component stays bounded
                    // away from zero and the loop would run to its cap.
                    if(dsr.kkt_residual.value_or(dsr.gradient_norm) < mma_defaults<Scalar>::dual_gradient_threshold
                       || dsr.step_size < mma_defaults<Scalar>::dual_step_threshold)
                        break;
                }
                s.y_dual = ds->x;
                (void)dual_prob.value(s.y_dual);
                x_trial = dual_prob.x_primal;
            }
            else
            {
                const Eigen::Vector<Scalar, MC> y_empty =
                    Eigen::Vector<Scalar, MC>::Zero(0);
                (void)dual_prob.value(y_empty);
                x_trial = dual_prob.x_primal;
            }

            f_trial = s.problem->value(x_trial);
            if(m > 0)
            {
                Eigen::VectorX<Scalar> c_tmp(m);
                s.problem->constraints(x_trial, c_tmp);
                c_trial = c_tmp;
            }

            // Conservativity test (Svanberg 2002 §4.2 concheck) against the
            // raw approximation values g_tilde_i(x*): conservative when
            // g_tilde_i(x*) >= f_i(x*) for every constraint. The artificial
            // slacks that keep the subproblem feasible do not enter the
            // comparison.
            bool conservative = (dual_prob.gval >= f_trial);
            for(int i = 0; i < m && conservative; ++i)
                conservative = (dual_prob.gcval[i] >= -c_trial[i]);

            if(conservative) break;

            // NLopt mma.c rho-growth (lines 388-391), on the raw
            // approximation shortfall.
            const Scalar wval = std::max(dual_prob.wval, Scalar(1e-20));
            if(f_trial > dual_prob.gval)
                s.rho_obj = std::min(
                    Scalar(10.0) * s.rho_obj,
                    Scalar(1.1) * (s.rho_obj + (f_trial - dual_prob.gval) / wval));
            for(int i = 0; i < m; ++i)
            {
                const Scalar gi_trial = -c_trial[i];
                if(gi_trial > dual_prob.gcval[i])
                    s.rho_con[i] = std::min(
                        Scalar(10.0) * s.rho_con[i],
                        Scalar(1.1) * (s.rho_con[i]
                               + (gi_trial - dual_prob.gcval[i]) / wval));
            }

            // Null-step-and-retry on inner exhaustion (mirrors the CCSA
            // quadratic policy). rho has just been grown but the last trial
            // is still non-conservative; committing it can move to a worse
            // or more-infeasible point (the approximation under-predicted
            // there). Commit only a merit-improving trial; otherwise take a
            // null step that keeps x fixed and retains the grown rho (no
            // inter-outer decay) so the next outer iteration re-solves a
            // strictly more conservative approximation.
            if(inner + 1 == max_inner)
            {
                Scalar viol_old = Scalar(0.0);
                Scalar viol_new = Scalar(0.0);
                for(int i = 0; i < m; ++i)
                {
                    viol_old = std::max(viol_old, -s.c_ineq[i]);
                    viol_new = std::max(viol_new, -c_trial[i]);
                }
                // A trial at the current iterate is a fixed point: conservativity there fails only by
                // roundoff, so null-stepping would grow rho forever. Committing it lets the step
                // tolerance stop the run.
                const bool at_iterate = (x_trial - s.x).norm()
                    <= std::numeric_limits<Scalar>::epsilon() * (Scalar(1.0) + s.x.norm());
                const bool acceptable =
                    ((f_trial < s.f || at_iterate) && viol_new <= viol_old + mma_defaults<Scalar>::violation_slack)
                    || (viol_new < viol_old - mma_defaults<Scalar>::violation_slack);

                if(!acceptable)
                {
                    const Eigen::Matrix<Scalar, 0, N> J_eq_empty(0, n);
                    const Eigen::Vector<Scalar, 0> lambda_eq_empty;
                    const Eigen::Vector<Scalar, 0> c_eq_empty;
                    const Scalar kkt = detail::kkt_residual<Scalar, N, 0, MC>(
                        s.g, J_eq_empty, s.J_ineq,
                        lambda_eq_empty, s.y_dual, c_eq_empty, s.c_ineq);

                    return step_result<Scalar>{
                        .objective_value = s.f,
                        .gradient_norm = s.g.norm(),
                        .step_size = Scalar(0.0),
                        .objective_change = Scalar(0.0),
                        .improved = false,
                        .is_null_step = true,
                        .constraint_violation =
                            detail::primal_feasibility_inf(
                                c_eq_empty, s.c_ineq),
                        .x_norm = s.x.norm(),
                        .kkt_residual = kkt,
                    };
                }
            }
        }

        // 5. Inter-outer rho decay.
        s.rho_obj = std::max(rho_decay * s.rho_obj, rho_min);
        for(int i = 0; i < m; ++i)
            s.rho_con[i] = std::max(rho_decay * s.rho_con[i], rho_min);

        // 6. Commit trial.
        const Scalar f_old = s.f;
        const Scalar step_size = (x_trial - s.x).norm();

        s.x_old2 = s.x_old1;
        s.x_old1 = s.x;
        s.x = x_trial;

        s.f = f_trial;
        s.problem->gradient(s.x, s.g);
        if(m > 0)
        {
            s.c_ineq = c_trial;
            Eigen::MatrixX<Scalar> J_tmp(m, n);
            s.problem->constraint_jacobian(s.x, J_tmp);
            s.J_ineq = J_tmp;
        }

        ++s.iteration;

        const Eigen::Matrix<Scalar, 0, N> J_eq_empty(0, n);
        const Eigen::Vector<Scalar, 0> lambda_eq_empty;
        const Eigen::Vector<Scalar, 0> c_eq_empty;
        const Scalar kkt = detail::kkt_residual<Scalar, N, 0, MC>(
            s.g, J_eq_empty, s.J_ineq,
            lambda_eq_empty, s.y_dual, c_eq_empty, s.c_ineq);

        return step_result<Scalar>{
            .objective_value = s.f,
            .gradient_norm = s.g.norm(),
            .step_size = step_size,
            .objective_change = s.f - f_old,
            .improved = s.f < f_old,
            .constraint_violation = detail::primal_feasibility_inf(
                c_eq_empty, s.c_ineq),
            .x_norm = s.x.norm(),
            .kkt_residual = kkt,
        };
    }

    template <typename P>
    void reset(state_type<P>& s, Eigen::Ref<const Eigen::Vector<Scalar, N>> x0)
    {
        s.x = x0;
        s.f = s.problem->value(x0);
        s.problem->gradient(x0, s.g);
        const int m = static_cast<int>(s.c_ineq.size());
        if(m > 0)
        {
            Eigen::VectorX<Scalar> c_tmp(m);
            s.problem->constraints(x0, c_tmp);
            s.c_ineq = c_tmp;
            Eigen::MatrixX<Scalar> J_tmp(m, static_cast<int>(s.x.size()));
            s.problem->constraint_jacobian(x0, J_tmp);
            s.J_ineq = J_tmp;
            for(int i = 0; i < m; ++i)
                s.c_dual_ref[i] = std::max(std::abs(s.c_ineq[i]), Scalar(1.0));
        }
        s.iteration = 0;
        s.x_old1 = x0;
        s.x_old2 = x0;
        s.y_dual.setZero();
        const Scalar ri = s.opts.rho_init.value_or(mma_defaults<Scalar>::rho_init);
        s.rho_obj = ri;
        s.rho_con.setConstant(ri);
    }

    template <typename P>
    void reset_clear(state_type<P>& s, Eigen::Ref<const Eigen::Vector<Scalar, N>> x0)
    {
        reset(s, x0);
        init_asymptotes(s);
    }

private:
    template <typename P>
    void init_asymptotes(state_type<P>& s)
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        const int n = static_cast<int>(s.x.size());
        const Scalar s_init = s.opts.asymptote_init.value_or(mma_defaults<Scalar>::asymptote_init);
        const Scalar s_init_un = s.opts.asymptote_init_unbounded.value_or(mma_defaults<Scalar>::asymptote_init_unbounded);

        for(int j = 0; j < n; ++j)
        {
            Scalar half;
            if(s.lower[j] > -inf && s.upper[j] < inf)
                half = s_init * (s.upper[j] - s.lower[j]);
            else
                half = s_init_un * std::max(std::abs(s.x[j]), Scalar(1.0));
            s.L[j] = s.x[j] - half;
            s.U[j] = s.x[j] + half;
        }
    }
};

}

#endif
