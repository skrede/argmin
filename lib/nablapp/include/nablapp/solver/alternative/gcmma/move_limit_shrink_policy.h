#ifndef HPP_GUARD_NABLAPP_SOLVER_ALTERNATIVE_GCMMA_MOVE_LIMIT_SHRINK_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_ALTERNATIVE_GCMMA_MOVE_LIMIT_SHRINK_POLICY_H

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
// nablapp::gcmma_policy (chosen post-benchmark).
//
// References:
//   Svanberg 1987, IJNME 24:359-373 (reciprocal approximation, Section 3
//     asymptote oscillation rule).
//   Svanberg 2002, SIAM J. Optim. 12(2):555-573 (conservativity
//     condition; raa-augmentation for the canonical globalization).

#include "nablapp/detail/lagrangian.h"
#include "nablapp/detail/kkt_residual.h"
#include "nablapp/detail/mma_reciprocal_dual_problem.h"
#include "nablapp/result/step_result.h"
#include "nablapp/solver/options.h"
#include "nablapp/solver/lbfgsb_policy.h"
#include "nablapp/types.h"

#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace nablapp::alternative::gcmma
{

template <int N = dynamic_dimension,
          template<int> typename DualPolicy = lbfgsb_policy>
struct move_limit_shrink_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = move_limit_shrink_policy<M, DualPolicy>;

    struct options_type
    {
        // Asymptote oscillation factors (Svanberg 1987 §3 defaults).
        std::optional<double> asymptote_contract{};       // 0.7
        std::optional<double> asymptote_expand{};         // 1.2

        // Initial asymptote distance fraction.
        std::optional<double> asymptote_init{};           // 0.5
        std::optional<double> asymptote_init_unbounded{}; // 1.0

        // Asymptote-distance numerical safeguards.
        std::optional<double> asymptote_min_fraction{};   // 1e-4
        std::optional<double> asymptote_max_fraction{};   // 10.0

        // Plain MMA move-limit fraction (Svanberg 1987 eq. 19).
        std::optional<double> move_limit_fraction{};      // 0.1

        // Approximation stabilization epsilon.
        std::optional<double> approximation_epsilon{};    // 1e-10

        // Conservativity loop: max inner iterations and shrink factor.
        // shrink_factor = 0.5 contracts [alpha, beta] toward x_k by half
        // each non-conservative trial.
        std::optional<std::uint16_t> max_inner_iterations{};  // 15
        std::optional<double> shrink_factor{};               // 0.5

        // Stall detection (forwarded by basic_solver).
        std::optional<double> stall_tolerance_threshold{};   // 1e-6
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

        Eigen::Vector<double, N> p_obj, q_obj;
        Eigen::Matrix<double, M, N> p_con, q_con;
        double r_obj{0};
        Eigen::Vector<double, M> r_con;

        Eigen::Vector<double, N> x_old1, x_old2;
        Eigen::Vector<double, N> lower, upper;
        Eigen::Vector<double, M> y_dual;

        std::uint32_t iteration{0};
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

        s.f = problem.value(x0);
        problem.gradient(x0, s.g);
        if(m > 0)
        {
            Eigen::VectorXd c_tmp(m);
            problem.constraints(x0, c_tmp);
            s.c_ineq = c_tmp;
            Eigen::MatrixXd J_tmp(m, n);
            problem.constraint_jacobian(x0, J_tmp);
            s.J_ineq = J_tmp;
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

        const double gam_dec = s.opts.asymptote_contract.value_or(0.7);
        const double gam_inc = s.opts.asymptote_expand.value_or(1.2);
        const double s_min_f = s.opts.asymptote_min_fraction.value_or(1e-4);
        const double s_max_f = s.opts.asymptote_max_fraction.value_or(10.0);
        const double move_lim = s.opts.move_limit_fraction.value_or(0.1);
        const double eps_pq = s.opts.approximation_epsilon.value_or(1e-10);
        const std::uint16_t max_inner =
            s.opts.max_inner_iterations.value_or(15);
        const double shrink = s.opts.shrink_factor.value_or(0.5);

        constexpr double inf = std::numeric_limits<double>::infinity();

        // 1. Asymptote oscillation update (Svanberg 1987 §3).
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

        // 2. Initial move limits (Svanberg 1987 eq. 19).
        for(int j = 0; j < n; ++j)
        {
            s.alpha[j] = std::max(s.L[j] + move_lim * (s.x[j] - s.L[j]),
                                  s.lower[j]);
            s.beta[j]  = std::min(s.U[j] - move_lim * (s.U[j] - s.x[j]),
                                  s.upper[j]);
        }

        // 3. Approximation params + r_i constants.
        s.r_obj = s.f;
        for(int i = 0; i < m; ++i)
            s.r_con[i] = -s.c_ineq[i];

        for(int j = 0; j < n; ++j)
        {
            const double dxU = s.U[j] - s.x[j];
            const double dxL = s.x[j] - s.L[j];
            const double gp = std::max(s.g[j], 0.0);
            const double gm = std::max(-s.g[j], 0.0);
            s.p_obj[j] = gp * dxU * dxU + eps_pq;
            s.q_obj[j] = gm * dxL * dxL + eps_pq;
            s.r_obj -= s.p_obj[j] / dxU + s.q_obj[j] / dxL;

            for(int i = 0; i < m; ++i)
            {
                const double gij = -s.J_ineq(i, j);
                s.p_con(i, j) = std::max(gij, 0.0) * dxU * dxU + eps_pq;
                s.q_con(i, j) = std::max(-gij, 0.0) * dxL * dxL + eps_pq;
                s.r_con[i] -= s.p_con(i, j) / dxU + s.q_con(i, j) / dxL;
            }
        }

        // 4. Conservativity loop. Solve the dual under the current
        //    move-limit window; on non-conservative trial, shrink the
        //    window toward x_k and re-solve. Loop terminates when
        //    conservativity holds or max_inner is reached.
        detail::mma_reciprocal_dual_problem<double, N, MC> dual_prob;
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

        Eigen::Vector<double, N> x_trial(n);
        double f_trial = s.f;
        Eigen::VectorXd c_trial = s.c_ineq;

        for(std::uint16_t inner = 0; inner < max_inner; ++inner)
        {
            if(m > 0)
            {
                DualPolicy<MC> dp;
                solver_options<default_convergence> dopts;
                dopts.max_iterations = 100;
                dopts.set_gradient_threshold(1e-9);
                dopts.set_step_threshold(1e-15);
                auto ds = dp.init(dual_prob, s.y_dual, dopts);
                for(int k = 0; k < 100; ++k)
                {
                    auto dsr = dp.step(ds);
                    if(dsr.gradient_norm < 1e-9 || dsr.step_size < 1e-15)
                        break;
                }
                s.y_dual = ds.x;
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

            // True f, c at trial.
            f_trial = s.problem->value(x_trial);
            if(m > 0)
            {
                Eigen::VectorXd c_tmp(m);
                s.problem->constraints(x_trial, c_tmp);
                c_trial = c_tmp;
            }

            // Conservativity test (Svanberg 2002 §4.2 form):
            //   g_tilde_i(x_trial) >= g_i(x_trial) for all i.
            //   Objective: gval >= f_trial.
            //   Constraints (MMA convention g_i = -c_i):
            //     gcval[i] >= -c_trial[i].
            bool conservative = (dual_prob.gval >= f_trial);
            for(int i = 0; i < m && conservative; ++i)
                conservative = (dual_prob.gcval[i] >= -c_trial[i]);

            if(conservative) break;

            // Shrink alpha, beta toward x_k.
            for(int j = 0; j < n; ++j)
            {
                s.alpha[j] = shrink * s.alpha[j] + (1.0 - shrink) * s.x[j];
                s.beta[j]  = shrink * s.beta[j]  + (1.0 - shrink) * s.x[j];
            }
        }

        // 5. Commit trial.
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

        // 6. KKT residual.
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
    void reset(state_type<P>& s, const Eigen::Vector<double, N>& x0)
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
        }
        s.iteration = 0;
        s.x_old1 = x0;
        s.x_old2 = x0;
        s.y_dual.setZero();
    }

    template <typename P>
    void reset_clear(state_type<P>& s, const Eigen::Vector<double, N>& x0)
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
