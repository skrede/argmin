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

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace argmin::alternative::gcmma
{

template <int N = dynamic_dimension,
          template<int> typename DualPolicy = lbfgsb_policy>
struct rho_wval_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = rho_wval_policy<M, DualPolicy>;

    struct options_type
    {
        std::optional<double> asymptote_contract{};       // 0.7
        std::optional<double> asymptote_expand{};         // 1.2
        std::optional<double> asymptote_init{};           // 0.5
        std::optional<double> asymptote_init_unbounded{}; // 1.0
        std::optional<double> asymptote_min_fraction{};   // 1e-4
        std::optional<double> asymptote_max_fraction{};   // 10.0
        std::optional<double> move_limit_fraction{};      // 0.1
        double move_bound_fraction{0.5};                  // Svanberg XXMOVE
        double raai{1e-5};                                // p/q regularizer

        // Bounded-dual elastics: dual upper bound c_i = dual_bound_scale *
        // max(|g_i(x0)|, 1). The per-constraint scale reference keeps the
        // exact-l1-penalty condition c_i > |mu_i*| scale-relative; the
        // scale multiplier is empirical (swept). See detail/mma_subproblem.h
        // recover_elastic_slacks(). Direct value.
        double dual_bound_scale{1000.0};

        // Conservativity loop and rho parameters.
        std::optional<std::uint16_t> max_inner_iterations{};  // 15
        std::optional<double> rho_init{};                     // 1.0
        std::optional<double> rho_min{};                      // 1e-5
        std::optional<double> rho_decay{};                    // 0.1 (between outer iters)

        std::optional<double> stall_tolerance_threshold{};
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
        Eigen::Vector<double, N> x, g;
        double f{};
        Eigen::Vector<double, M> c_ineq;
        Eigen::Matrix<double, M, N> J_ineq;

        Eigen::Vector<double, N> L, U;
        Eigen::Vector<double, N> alpha, beta;
        Eigen::Vector<double, N> w;          // separable quadratic weights

        Eigen::Vector<double, N> p_obj, q_obj;
        Eigen::Matrix<double, M, N> p_con, q_con;
        double r_obj{0};
        Eigen::Vector<double, M> r_con;

        // Penalty parameters (mutate inside conservativity loop).
        double rho_obj{1.0};
        Eigen::Vector<double, M> rho_con;

        Eigen::Vector<double, N> x_old1, x_old2;
        Eigen::Vector<double, N> lower, upper;
        Eigen::Vector<double, M> y_dual;

        // Bounded-dual elastics. c_dual_ref[i] = max(|g_i(x0)|, 1) is the
        // fixed per-constraint scale reference; c_dual[i] =
        // dual_bound_scale * c_dual_ref[i] is the working dual upper bound.
        Eigen::Vector<double, M> c_dual_ref;
        Eigen::Vector<double, M> c_dual;

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
                             const Eigen::Vector<double, N>& x0,
                             const solver_options<Convergence>& opts,
                             const options_type& policy_opts)
    {
        auto s = init(problem, x0, opts);
        s.opts = policy_opts;
        const double ri = s.opts.rho_init.value_or(1.0);
        s.rho_obj = ri;
        s.rho_con.setConstant(ri);
        init_asymptotes(s);
        return s;
    }

    template <typename Problem, typename Convergence = default_convergence>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<double, N>& x0,
                             const solver_options<Convergence>&)
    {
        static_assert(differentiable<Problem>);
        static_assert(constrained<Problem>);

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
        s.rho_con.setConstant(1.0);
        s.rho_obj = 1.0;
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
            Eigen::VectorXd c_tmp(n_all);
            problem.constraints(x0, c_tmp);
            Eigen::MatrixXd J_tmp(n_all, n);
            problem.constraint_jacobian(x0, J_tmp);
            if(!s.invalid_problem)
            {
                s.c_ineq = c_tmp;
                s.J_ineq = J_tmp;
                // Elastic dual-bound scale reference max(|g_i(x0)|, 1),
                // g_i = -c_ineq_i (fixed for the run).
                for(int i = 0; i < m; ++i)
                    s.c_dual_ref[i] = std::max(std::abs(s.c_ineq[i]), 1.0);
            }
        }

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

        init_asymptotes(s);
        s.x_old1 = x0;
        s.x_old2 = x0;
        s.iteration = 0;
        return s;
    }

    template <typename P>
    step_result<double> step(state_type<P>& s)
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
            return step_result<double>{
                .objective_value = s.f,
                .gradient_norm = s.g.norm(),
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
                .is_null_step = true,
                .x_norm = s.x.norm(),
                .policy_status = solver_status::invalid_problem,
            };
        }

        const double gam_dec = s.opts.asymptote_contract.value_or(0.7);
        const double gam_inc = s.opts.asymptote_expand.value_or(1.2);
        const double s_min_f = s.opts.asymptote_min_fraction.value_or(1e-4);
        const double s_max_f = s.opts.asymptote_max_fraction.value_or(10.0);
        const double move_lim = s.opts.move_limit_fraction.value_or(0.1);
        const double raai = s.opts.raai;
        const std::uint16_t max_inner =
            s.opts.max_inner_iterations.value_or(15);
        const double rho_min = s.opts.rho_min.value_or(1e-5);
        const double rho_decay = s.opts.rho_decay.value_or(0.1);

        constexpr double inf = std::numeric_limits<double>::infinity();

        // 1. Asymptote oscillation update.
        if(s.iteration >= 2)
        {
            for(int j = 0; j < n; ++j)
            {
                const double dx2 =
                    (s.x[j] - s.x_old1[j]) * (s.x_old1[j] - s.x_old2[j]);
                const double gam = dx2 < 0.0 ? gam_dec
                                 : dx2 > 0.0 ? gam_inc : 1.0;
                s.L[j] = s.x[j] - gam * (s.x_old1[j] - s.L[j]);
                s.U[j] = s.x[j] + gam * (s.U[j] - s.x_old1[j]);

                double range;
                if(s.lower[j] > -inf && s.upper[j] < inf)
                    range = s.upper[j] - s.lower[j];
                else
                    range = std::max(std::abs(s.x[j]), 1.0);

                const double min_d = std::max(s_min_f * range, 1e-12);
                const double max_d = s_max_f * range;
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
        const double move_bnd = s.opts.move_bound_fraction;
        for(int j = 0; j < n; ++j)
        {
            double range;
            if(s.lower[j] > -inf && s.upper[j] < inf)
                range = s.upper[j] - s.lower[j];
            else
                range = 2.0 * std::max(std::abs(s.x[j]), 1.0);
            const double mb = move_bnd * range;

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
            const double dxU = s.U[j] - s.x[j];
            const double dxL = s.x[j] - s.L[j];
            s.w[j] = (s.U[j] - s.L[j]) / (dxU * dxL);
        }

        // 3. Approximation params + r_i constants.
        s.r_obj = s.f;
        for(int i = 0; i < m; ++i)
            s.r_con[i] = -s.c_ineq[i];

        for(int j = 0; j < n; ++j)
        {
            const double dxU = s.U[j] - s.x[j];
            const double dxL = s.x[j] - s.L[j];

            // Distance-scaled regularizer raai/range_j (Svanberg 2007
            // notes eqs. 2.3-2.4) with a scale proxy for unbounded
            // variables, plus the 1.001/0.001 gradient mixing so a
            // zero-gradient component keeps its minimizer at x_k.
            double range;
            if(s.lower[j] > -inf && s.upper[j] < inf)
                range = s.upper[j] - s.lower[j];
            else
                range = std::max(std::abs(s.x[j]), 1.0);
            const double raa_reg = raai / range;

            const double gp = std::max(s.g[j], 0.0);
            const double gm = std::max(-s.g[j], 0.0);
            s.p_obj[j] = dxU * dxU * (1.001 * gp + 0.001 * gm + raa_reg);
            s.q_obj[j] = dxL * dxL * (0.001 * gp + 1.001 * gm + raa_reg);
            s.r_obj -= s.p_obj[j] / dxU + s.q_obj[j] / dxL;

            for(int i = 0; i < m; ++i)
            {
                const double gij = -s.J_ineq(i, j);
                const double gpij = std::max(gij, 0.0);
                const double gmij = std::max(-gij, 0.0);
                s.p_con(i, j) =
                    dxU * dxU * (1.001 * gpij + 0.001 * gmij + raa_reg);
                s.q_con(i, j) =
                    dxL * dxL * (0.001 * gpij + 1.001 * gmij + raa_reg);
                s.r_con[i] -= s.p_con(i, j) / dxU + s.q_con(i, j) / dxL;
            }
        }

        // 4. Conservativity loop with rho-growth on non-conservative trial.
        detail::mma_augmented_dual_problem<double, N, MC> dual_prob;
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

        Eigen::Vector<double, N> x_trial(n);
        double f_trial = s.f;
        Eigen::VectorXd c_trial = s.c_ineq;

        // Persistent inner dual solver: construct the box-constrained solver
        // state once per outer iteration and cold-restart it (reset_clear
        // clears the L-BFGS curvature so each inner solve starts fresh, as a
        // per-inner re-construction would, while reusing the allocated
        // buffers) rather than re-allocating it on every conservativity
        // iteration.
        DualPolicy<MC> dp;
        solver_options<default_convergence> dopts;
        dopts.max_iterations = 100;
        dopts.set_gradient_threshold(1e-9);
        dopts.set_step_threshold(1e-15);
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
                    if(dsr.kkt_residual.value_or(dsr.gradient_norm) < 1e-9
                       || dsr.step_size < 1e-15)
                        break;
                }
                s.y_dual = ds->x;
                (void)dual_prob.value(s.y_dual);
                x_trial = dual_prob.x_primal;
            }
            else
            {
                const Eigen::Vector<double, MC> y_empty =
                    Eigen::Vector<double, MC>::Zero(0);
                (void)dual_prob.value(y_empty);
                x_trial = dual_prob.x_primal;
            }

            f_trial = s.problem->value(x_trial);
            if(m > 0)
            {
                Eigen::VectorXd c_tmp(m);
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
            const double wval = std::max(dual_prob.wval, 1e-20);
            if(f_trial > dual_prob.gval)
                s.rho_obj = std::min(
                    10.0 * s.rho_obj,
                    1.1 * (s.rho_obj + (f_trial - dual_prob.gval) / wval));
            for(int i = 0; i < m; ++i)
            {
                const double gi_trial = -c_trial[i];
                if(gi_trial > dual_prob.gcval[i])
                    s.rho_con[i] = std::min(
                        10.0 * s.rho_con[i],
                        1.1 * (s.rho_con[i]
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
                double viol_old = 0.0;
                double viol_new = 0.0;
                for(int i = 0; i < m; ++i)
                {
                    viol_old = std::max(viol_old, -s.c_ineq[i]);
                    viol_new = std::max(viol_new, -c_trial[i]);
                }
                const bool acceptable =
                    (f_trial < s.f && viol_new <= viol_old + 1e-12)
                    || (viol_new < viol_old - 1e-12);

                if(!acceptable)
                {
                    const Eigen::Matrix<double, 0, N> J_eq_empty(0, n);
                    const Eigen::Vector<double, 0> lambda_eq_empty;
                    const Eigen::Vector<double, 0> c_eq_empty;
                    const double kkt = detail::kkt_residual<double, N, 0, MC>(
                        s.g, J_eq_empty, s.J_ineq,
                        lambda_eq_empty, s.y_dual, c_eq_empty, s.c_ineq);

                    return step_result<double>{
                        .objective_value = s.f,
                        .gradient_norm = s.g.norm(),
                        .step_size = 0.0,
                        .objective_change = 0.0,
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
        const double f_old = s.f;
        const double step_size = (x_trial - s.x).norm();

        s.x_old2 = s.x_old1;
        s.x_old1 = s.x;
        s.x = x_trial;

        s.f = f_trial;
        s.problem->gradient(s.x, s.g);
        if(m > 0)
        {
            s.c_ineq = c_trial;
            Eigen::MatrixXd J_tmp(m, n);
            s.problem->constraint_jacobian(s.x, J_tmp);
            s.J_ineq = J_tmp;
        }

        ++s.iteration;

        const Eigen::Matrix<double, 0, N> J_eq_empty(0, n);
        const Eigen::Vector<double, 0> lambda_eq_empty;
        const Eigen::Vector<double, 0> c_eq_empty;
        const double kkt = detail::kkt_residual<double, N, 0, MC>(
            s.g, J_eq_empty, s.J_ineq,
            lambda_eq_empty, s.y_dual, c_eq_empty, s.c_ineq);

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
        };
    }

    template <typename P>
    void reset(state_type<P>& s, Eigen::Ref<const Eigen::Vector<double, N>> x0)
    {
        s.x = x0;
        s.f = s.problem->value(x0);
        s.problem->gradient(x0, s.g);
        const int m = static_cast<int>(s.c_ineq.size());
        if(m > 0)
        {
            Eigen::VectorXd c_tmp(m);
            s.problem->constraints(x0, c_tmp);
            s.c_ineq = c_tmp;
            Eigen::MatrixXd J_tmp(m, static_cast<int>(s.x.size()));
            s.problem->constraint_jacobian(x0, J_tmp);
            s.J_ineq = J_tmp;
            for(int i = 0; i < m; ++i)
                s.c_dual_ref[i] = std::max(std::abs(s.c_ineq[i]), 1.0);
        }
        s.iteration = 0;
        s.x_old1 = x0;
        s.x_old2 = x0;
        s.y_dual.setZero();
        const double ri = s.opts.rho_init.value_or(1.0);
        s.rho_obj = ri;
        s.rho_con.setConstant(ri);
    }

    template <typename P>
    void reset_clear(state_type<P>& s, Eigen::Ref<const Eigen::Vector<double, N>> x0)
    {
        reset(s, x0);
        init_asymptotes(s);
    }

private:
    template <typename P>
    void init_asymptotes(state_type<P>& s)
    {
        constexpr double inf = std::numeric_limits<double>::infinity();
        const int n = static_cast<int>(s.x.size());
        const double s_init = s.opts.asymptote_init.value_or(0.5);
        const double s_init_un = s.opts.asymptote_init_unbounded.value_or(1.0);

        for(int j = 0; j < n; ++j)
        {
            double half;
            if(s.lower[j] > -inf && s.upper[j] < inf)
                half = s_init * (s.upper[j] - s.lower[j]);
            else
                half = s_init_un * std::max(std::abs(s.x[j]), 1.0);
            s.L[j] = s.x[j] - half;
            s.U[j] = s.x[j] + half;
        }
    }
};

}

#endif
