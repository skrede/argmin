#ifndef HPP_GUARD_ARGMIN_SOLVER_ALTERNATIVE_GCMMA_RAA_AUGMENTED_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_ALTERNATIVE_GCMMA_RAA_AUGMENTED_POLICY_H

// GCMMA variant: strict Svanberg 2002 raa-augmented globalization.
//
// Augments the MMA reciprocal approximation with the asymptote-divergent
// d_j penalty term from Svanberg 2002 §3:
//   d_j(x, x_k) = (U_j - L_j) * (x_j - x_kj)^2
//                 / ( (U_j - x_j) * (x_j - L_j) )
//
// Unlike rho_wval's separable quadratic (with x_k-only denominator),
// d_j keeps the asymptotic divergence of the underlying reciprocal
// approximation -- the penalty grows without bound as x approaches
// L or U. This is the canonical CCSA conservativity mechanism with
// the global-convergence proof of Svanberg 2002.
//
// raa-growth on non-conservative trial (Svanberg 2002 eq. 3.7-3.8):
//   delta_i = (g_i_at_trial - g_tilde_i_at_trial) / dval_sum
//   raa_i_new = max(1.1 * raa_i, raa_i + delta_i / 0.5)
//   (capped at 10x raa_i_old to avoid explosive growth).
//
// References:
//   Svanberg 2002, SIAM J. Optim. 12(2):555-573, eq. 3.4-3.8.

#include "argmin/detail/lagrangian.h"
#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/mma_subproblem.h"
#include "argmin/detail/mma_raa_augmented_dual_problem.h"
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
struct raa_augmented_policy
{
    using scalar_type = Scalar;

    template <int M>
    using rebind = raa_augmented_policy<Scalar, M, DualPolicy>;

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
        // max(|g_i(x0)|, 1). Scale multiplier is empirical (swept). See
        // detail/mma_subproblem.h recover_elastic_slacks(). Direct value.
        Scalar dual_bound_scale{mma_defaults<Scalar>::dual_bound_scale};

        std::optional<std::uint16_t> max_inner_iterations{};  // 15
        std::optional<Scalar> raa_init{};                     // 1e-5
        std::optional<Scalar> raa_min{};                      // 1e-7
        std::optional<Scalar> raa_decay{};                    // 0.5

        // raa penalty schedule (Svanberg 2002 §3, eqs. 3.6-3.9). When true,
        // the raa floor is recomputed each outer iteration from the current
        // gradient as (0.1/n) * sum_j |dg_j| * range_j (3.6), so the penalty
        // scales with the objective/constraint magnitude instead of the
        // fixed raa_min (which wastes inner budget on large-scale
        // objectives), and the non-conservative growth is 1.1*(raa + delta)
        // (3.9) rather than raa + 2*delta. When false, the pre-fix schedule
        // (fixed floor + raa + 2*delta growth). Default set by the empirical
        // scale-robustness sweep.
        bool raa_svanberg_schedule{true};

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

        Eigen::Vector<Scalar, N> p_obj, q_obj;
        Eigen::Matrix<Scalar, M, N> p_con, q_con;
        Scalar r_obj{0};
        Eigen::Vector<Scalar, M> r_con;

        Scalar raa_obj{mma_defaults<Scalar>::raa_init};
        Eigen::Vector<Scalar, M> raa_con;

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
        const Scalar ri = s.opts.raa_init.value_or(mma_defaults<Scalar>::raa_init);
        s.raa_obj = ri;
        s.raa_con.setConstant(ri);
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
        assert(problem.num_equality() == 0);

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
        s.raa_con.resize(m);
        s.raa_con.setConstant(mma_defaults<Scalar>::raa_init);
        s.raa_obj = mma_defaults<Scalar>::raa_init;
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
        const Scalar raa_min = s.opts.raa_min.value_or(mma_defaults<Scalar>::raa_min);
        const Scalar raa_decay = s.opts.raa_decay.value_or(mma_defaults<Scalar>::raa_decay);
        const bool svanberg = s.opts.raa_svanberg_schedule;

        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();

        // Svanberg 2002 eq. 3.6 gradient-scaled raa floor, recomputed each
        // outer iteration so the penalty tracks the objective/constraint
        // magnitude: raa_floor_i = (0.1/n) * sum_j |dg_ij| * range_j. This
        // replaces the fixed raa_min as the decay floor and as a lower bound
        // on the starting raa (the fixed 1e-5/1e-7 wastes inner conservativity
        // budget on large-scale objectives). range_j is the box width for a
        // bounded variable, the scale proxy max(|x_j|, 1) for an unbounded one.
        auto range_of = [&](int j) {
            if(s.lower[j] > -inf && s.upper[j] < inf)
                return s.upper[j] - s.lower[j];
            return std::max(std::abs(s.x[j]), Scalar(1.0));
        };
        Scalar raa_floor_obj = raa_min;
        Eigen::Vector<Scalar, MC> raa_floor_con =
            Eigen::Vector<Scalar, MC>::Constant(m, raa_min);
        if(svanberg)
        {
            const Scalar inv_n = Scalar(0.1) / static_cast<Scalar>(std::max(n, 1));
            Scalar sum_obj = Scalar(0.0);
            for(int j = 0; j < n; ++j)
                sum_obj += std::abs(s.g[j]) * range_of(j);
            raa_floor_obj = std::max(inv_n * sum_obj, raa_min);
            for(int i = 0; i < m; ++i)
            {
                Scalar sum_con = Scalar(0.0);
                for(int j = 0; j < n; ++j)
                    sum_con += std::abs(s.J_ineq(i, j)) * range_of(j);
                raa_floor_con[i] = std::max(inv_n * sum_con, raa_min);
            }
            // Start each outer at least at the gradient-scaled floor.
            s.raa_obj = std::max(s.raa_obj, raa_floor_obj);
            for(int i = 0; i < m; ++i)
                s.raa_con[i] = std::max(s.raa_con[i], raa_floor_con[i]);
        }

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

        // 2. Move limits (Svanberg 2007 notes eqs. 2.8-2.9): the box, the
        //    0.1 asymptote buffer, and the XXMOVE=0.5 move bound
        //    x_k +/- move_bnd * range_j with the box-width analog
        //    2*max(|x_kj|, 1) for an unbounded variable.
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

        // 4. Conservativity loop with raa-growth on non-conservative trial.
        detail::mma_raa_augmented_dual_problem<Scalar, N, MC> dual_prob;
        dual_prob.L_out = &s.L;
        dual_prob.U_out = &s.U;
        dual_prob.x_k_out = &s.x;
        dual_prob.alpha_out = &s.alpha;
        dual_prob.beta_out = &s.beta;
        dual_prob.p_obj_out = &s.p_obj;
        dual_prob.q_obj_out = &s.q_obj;
        dual_prob.p_con_out = &s.p_con;
        dual_prob.q_con_out = &s.q_con;
        dual_prob.r_obj = s.r_obj;
        dual_prob.r_con_out = &s.r_con;
        dual_prob.raa_con_out = &s.raa_con;
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
            dual_prob.raa_obj = s.raa_obj;
            // raa_obj/raa_con changed since the previous inner iteration:
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

            // Conservativity test (Svanberg 2002 concheck) against the raw
            // approximation values g_tilde_i(x*): conservative when
            // g_tilde_i(x*) >= f_i(x*) for every constraint. The artificial
            // slacks that keep the subproblem feasible do not enter the
            // comparison.
            bool conservative = (dual_prob.gval >= f_trial);
            for(int i = 0; i < m && conservative; ++i)
                conservative = (dual_prob.gcval[i] >= -c_trial[i]);

            if(conservative) break;

            // raa-growth: Svanberg 2002 eq. 3.7-3.8 spirit. dval_sum
            // captures total d_j mass at the trial; growth ensures the
            // augmented approximation crosses the true value for the
            // failing constraints (evaluated on the raw approximation).
            // raa-growth. Svanberg 2002 eq. 3.9 raa' = 1.1*(raa + delta)
            // (svanberg schedule) vs the pre-fix raa + 2*delta; delta is the
            // conservativity shortfall per unit d_j mass. Both capped at 10x.
            const Scalar dsum = std::max(dual_prob.dval_sum, Scalar(1e-20));
            auto grow = [&](Scalar raa, Scalar delta) {
                const Scalar target = svanberg
                    ? Scalar(1.1) * (raa + delta)
                    : std::max(Scalar(1.1) * raa, raa + delta / Scalar(0.5));
                return std::min(Scalar(10.0) * raa, target);
            };
            if(f_trial > dual_prob.gval)
            {
                const Scalar delta = (f_trial - dual_prob.gval) / dsum;
                s.raa_obj = grow(s.raa_obj, delta);
            }
            for(int i = 0; i < m; ++i)
            {
                const Scalar gi_trial = -c_trial[i];
                if(gi_trial > dual_prob.gcval[i])
                {
                    const Scalar delta = (gi_trial - dual_prob.gcval[i]) / dsum;
                    s.raa_con[i] = grow(s.raa_con[i], delta);
                }
            }
        }

        // 5. Inter-outer raa decay toward the scale-adaptive floor (the
        //    gradient-scaled raa_floor under the Svanberg schedule, else the
        //    fixed raa_min).
        s.raa_obj = std::max(raa_decay * s.raa_obj, raa_floor_obj);
        for(int i = 0; i < m; ++i)
            s.raa_con[i] = std::max(raa_decay * s.raa_con[i], raa_floor_con[i]);

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
        const Scalar ri = s.opts.raa_init.value_or(mma_defaults<Scalar>::raa_init);
        s.raa_obj = ri;
        s.raa_con.setConstant(ri);
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
