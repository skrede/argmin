#ifndef HPP_GUARD_ARGMIN_SOLVER_ALTERNATIVE_GCMMA_MOVE_LIMIT_SHRINK_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_ALTERNATIVE_GCMMA_MOVE_LIMIT_SHRINK_POLICY_H

// GCMMA variant: move-limit shrinkage globalization.
//
// Plain Svanberg 1987 MMA reciprocal approximation with a trust-region-style
// inner loop: solve the standard MMA subproblem, test conservativity, and
// on failure shrink the move-limit window [alpha, beta] toward x_k. Repeat
// until conservativity holds or max_inner_iterations is reached.
//
// Conservativity test (Svanberg 2002 Section 4.2 form, applied to the
// MMA reciprocal approximation rather than the CCSA-quadratic one):
//   g_tilde_i(x_trial) >= g_i(x_trial)  for all i = 0..m
// where g_0 is the objective. Since g_tilde matches g exactly at x = x_k,
// shrinking [alpha, beta] toward x_k drives x_trial toward x_k and the
// inequality becomes tight; the loop is guaranteed to terminate.
//
// Trade-off vs. the canonical Svanberg 2002 raa-augmented GCMMA:
//   + Preserves the analytic per-component primal x_j(y) of plain MMA
//     (Svanberg 1987 closed form) -- no per-iteration cubic solve.
//   + Simple to implement (~70 lines of conservativity logic on top of
//     plain mma_policy).
//   - Not the canonical GCMMA: trust-region shrinkage rather than
//     raa-penalty growth. The global-convergence proof from Svanberg
//     2002 does not directly transfer (it relies on the raa structure).
//   - Repeated dual solves on shrinking [alpha, beta] are wasted work
//     compared to the rho-grow variant which adapts the approximation
//     itself rather than the trust region.
//
// This variant is preserved as a research artifact for the empirical
// comparison reported in the paper. The production GCMMA policy is in
// argmin::gcmma_policy (chosen post-benchmark).
//
// References:
//   Svanberg 1987, IJNME 24:359-373 (reciprocal approximation, Section 3
//     asymptote oscillation rule).
//   Svanberg 2002, SIAM J. Optim. 12(2):555-573 (conservativity
//     condition; raa-augmentation for the canonical globalization).

#include "argmin/detail/lagrangian.h"
#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/mma_subproblem.h"
#include "argmin/detail/mma_reciprocal_dual_problem.h"
#include "argmin/result/step_result.h"
#include "argmin/solver/options.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/types.h"

#include "argmin/options/mma_defaults.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace argmin::alternative::gcmma
{

template <scalar Scalar = double, int N = dynamic_dimension,
          template <scalar, int> typename DualPolicy = lbfgsb_policy>
struct move_limit_shrink_policy
{
    using scalar_type = Scalar;

    template <int M>
    using rebind = move_limit_shrink_policy<Scalar, M, DualPolicy>;

    struct options_type
    {
        // Asymptote oscillation factors (Svanberg 1987 §3 defaults).
        std::optional<Scalar> asymptote_contract{};       // 0.7
        std::optional<Scalar> asymptote_expand{};         // 1.2

        // Initial asymptote distance fraction.
        std::optional<Scalar> asymptote_init{};           // 0.5
        std::optional<Scalar> asymptote_init_unbounded{}; // 1.0

        // Asymptote-distance numerical safeguards.
        std::optional<Scalar> asymptote_min_fraction{};   // 1e-4
        std::optional<Scalar> asymptote_max_fraction{};   // 10.0

        // Plain MMA move-limit fraction (Svanberg 1987 eq. 19).
        std::optional<Scalar> move_limit_fraction{};      // 0.1
        Scalar move_bound_fraction{mma_defaults<Scalar>::move_bound_fraction};  // Svanberg XXMOVE

        // Distance-scaled p/q regularizer (Svanberg 2007 notes 2.3-2.4).
        Scalar raai{mma_defaults<Scalar>::raai};

        // Bounded-dual elastics: dual upper bound c_i = dual_bound_scale *
        // max(|g_i(x0)|, 1). Scale multiplier is empirical (swept). See
        // detail/mma_subproblem.h recover_elastic_slacks(). Direct value.
        Scalar dual_bound_scale{mma_defaults<Scalar>::dual_bound_scale};

        // Conservativity loop: max inner iterations and shrink factor.
        // shrink_factor = 0.5 contracts [alpha, beta] toward x_k by half
        // each non-conservative trial.
        std::optional<std::uint16_t> max_inner_iterations{};  // 15
        std::optional<Scalar> shrink_factor{};               // 0.5

        // Stall detection (forwarded by step_budget_solver).
        std::optional<Scalar> stall_tolerance_threshold{};   // 1e-6
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

        Eigen::Vector<Scalar, N> p_obj, q_obj;
        Eigen::Matrix<Scalar, M, N> p_con, q_con;
        Scalar r_obj{0};
        Eigen::Vector<Scalar, M> r_con;

        Eigen::Vector<Scalar, N> x_old1, x_old2;
        Eigen::Vector<Scalar, N> lower, upper;
        Eigen::Vector<Scalar, M> y_dual;

        // Bounded-dual elastics. c_dual_ref[i] = max(|g_i(x0)|, 1) is the
        // fixed per-constraint scale reference; c_dual[i] =
        // dual_bound_scale * c_dual_ref[i] is the working dual upper bound.
        Eigen::Vector<Scalar, M> c_dual_ref;
        Eigen::Vector<Scalar, M> c_dual;

        std::uint32_t iteration{0};
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
        assert(problem.num_equality() == 0
               && "GCMMA handles inequality constraints only.");

        const int n = problem.dimension();
        const int m = problem.num_inequality();
        state_type<Problem> s;
        s.problem = &problem;

        s.x = x0;
        s.g.resize(n);
        s.c_ineq.resize(m);
        s.J_ineq.resize(m, n);
        s.L.resize(n);
        s.U.resize(n);
        s.alpha.resize(n);
        s.beta.resize(n);
        s.p_obj.resize(n);
        s.q_obj.resize(n);
        s.p_con.resize(m, n);
        s.q_con.resize(m, n);
        s.r_con.resize(m);
        s.y_dual.resize(m);
        s.y_dual.setZero();
        s.c_dual_ref.resize(m);
        s.c_dual_ref.setOnes();
        s.c_dual.resize(m);

        s.f = problem.value(x0);
        problem.gradient(x0, s.g);
        if(m > 0)
        {
            Eigen::VectorX<Scalar> c_tmp(m);
            problem.constraints(x0, c_tmp);
            s.c_ineq = c_tmp;
            Eigen::MatrixX<Scalar> J_tmp(m, n);
            problem.constraint_jacobian(x0, J_tmp);
            s.J_ineq = J_tmp;
            for(int i = 0; i < m; ++i)
                s.c_dual_ref[i] = std::max(std::abs(s.c_ineq[i]), Scalar(1.0));
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

        const Scalar gam_dec = s.opts.asymptote_contract.value_or(mma_defaults<Scalar>::asymptote_contract);
        const Scalar gam_inc = s.opts.asymptote_expand.value_or(mma_defaults<Scalar>::asymptote_expand);
        const Scalar s_min_f = s.opts.asymptote_min_fraction.value_or(mma_defaults<Scalar>::asymptote_min_fraction);
        const Scalar s_max_f = s.opts.asymptote_max_fraction.value_or(mma_defaults<Scalar>::asymptote_max_fraction);
        const Scalar move_lim = s.opts.move_limit_fraction.value_or(mma_defaults<Scalar>::move_limit_fraction);
        const Scalar raai = s.opts.raai;
        const std::uint16_t max_inner =
            s.opts.max_inner_iterations.value_or(15);
        const Scalar shrink = s.opts.shrink_factor.value_or(mma_defaults<Scalar>::shrink_factor);

        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();

        // 1. Asymptote oscillation update (Svanberg 1987 §3).
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

        // 2. Initial move limits (Svanberg 2007 notes eqs. 2.8-2.9): the
        //    box, the 0.1 asymptote buffer, and the XXMOVE=0.5 move bound
        //    x_k +/- move_bnd * range_j with the box-width analog
        //    2*max(|x_kj|, 1) for an unbounded variable. The conservativity
        //    loop below shrinks this window further toward x_k.
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

        // 4. Conservativity loop. Solve the dual under the current
        //    move-limit window; on non-conservative trial, shrink the
        //    window toward x_k and re-solve. Loop terminates when
        //    conservativity holds or max_inner is reached.
        detail::mma_reciprocal_dual_problem<Scalar, N, MC> dual_prob;
        dual_prob.L_out = &s.L;
        dual_prob.U_out = &s.U;
        dual_prob.alpha_out = &s.alpha;
        dual_prob.beta_out = &s.beta;
        dual_prob.p_obj_out = &s.p_obj;
        dual_prob.q_obj_out = &s.q_obj;
        dual_prob.p_con_out = &s.p_con;
        dual_prob.q_con_out = &s.q_con;
        dual_prob.r_obj = s.r_obj;
        dual_prob.r_con_out = &s.r_con;
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
        // buffers) rather than re-allocating it on every shrink iteration.
        DualPolicy<scalar_type, MC> dp;
        // Near the dual optimum the decrease left falls below the roundoff of the dual value, so the line
        // search certifies those steps from the exact dual gradient instead.
        dp.options.line_search.relative_value_noise = mma_defaults<Scalar>::dual_value_noise;
        solver_options<default_convergence<scalar_type>> dopts;
        dopts.max_iterations = 100;
        dopts.set_gradient_threshold(mma_defaults<Scalar>::dual_gradient_threshold);
        dopts.set_step_threshold(mma_defaults<Scalar>::dual_step_threshold);
        using dual_state_t = decltype(dp.init(dual_prob, s.y_dual, dopts));
        std::optional<dual_state_t> ds;

        for(std::uint16_t inner = 0; inner < max_inner; ++inner)
        {
            // The move-limit window [alpha, beta] shrinks between inner
            // iterations, so the per-component primal changes at a fixed y:
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

            // True f, c at trial.
            f_trial = s.problem->value(x_trial);
            if(m > 0)
            {
                Eigen::VectorX<Scalar> c_tmp(m);
                s.problem->constraints(x_trial, c_tmp);
                c_trial = c_tmp;
            }

            // Conservativity test (Svanberg 2002 §4.2 concheck) against the
            // raw approximation values g_tilde_i(x*):
            //   Objective: gval >= f_trial.
            //   Constraints (MMA convention g_i = -c_i):
            //     gcval[i] >= -c_trial[i].
            // The artificial slacks that keep the subproblem feasible do
            // not enter the comparison.
            bool conservative = (dual_prob.gval >= f_trial);
            for(int i = 0; i < m && conservative; ++i)
                conservative = (dual_prob.gcval[i] >= -c_trial[i]);

            if(conservative) break;

            // Shrink alpha, beta toward x_k.
            for(int j = 0; j < n; ++j)
            {
                s.alpha[j] = shrink * s.alpha[j] + (Scalar(1.0) - shrink) * s.x[j];
                s.beta[j]  = shrink * s.beta[j]  + (Scalar(1.0) - shrink) * s.x[j];
            }
        }

        // 5. Commit trial.
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

        // 6. KKT residual.
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
