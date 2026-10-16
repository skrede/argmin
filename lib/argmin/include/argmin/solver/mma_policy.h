#ifndef HPP_GUARD_ARGMIN_SOLVER_MMA_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_MMA_POLICY_H

// Method of Moving Asymptotes (MMA) — Svanberg 1987 reciprocal form.
//
// Plain MMA (no conservativity loop): each outer iteration unconditionally
// accepts the trial computed from the reciprocal separable approximation
// at the current iterate. The conservativity-protected variant is GCMMA
// (gcmma_policy) and the quadratic-penalty CCSA variant is
// ccsa_quadratic_policy.
//
// Approximation (Svanberg 1987 eq. 6-7):
//   f_tilde(x) = const + sum_j [ p_0j / (U_j - x_j) + q_0j / (x_j - L_j) ]
//   p_ij = max(d g_i / d x_j, 0) * (U_j - x_kj)^2 + epsilon
//   q_ij = max(-d g_i / d x_j, 0) * (x_kj - L_j)^2 + epsilon
//
// Asymptote update (Svanberg 1987 Section 3):
//   k = 0:    L_j = x_j - s_init * range,  U_j = x_j + s_init * range
//   k >= 2:   per-component oscillation (3-history rule)
//     sign = (x_j^k - x_j^{k-1}) * (x_j^{k-1} - x_j^{k-2})
//     gam  = 0.7 if sign < 0, 1.2 if sign > 0, 1.0 otherwise
//     L_j  = x_j^k - gam * (x_j^{k-1} - L_j^{k-1})
//     U_j  = x_j^k + gam * (U_j^{k-1} - x_j^{k-1})
//
// Move limits (Svanberg 1987 eq. 19) keep x_j strictly off the asymptotes:
//   alpha_j = max(L_j + 0.1 * (x_j - L_j), x_min_j)
//   beta_j  = min(U_j - 0.1 * (U_j - x_j), x_max_j)
//
// Subproblem solved via Lagrange dual decomposition (m-dimensional concave
// dual; closed-form per-component primal x_j(y); see
// detail/mma_reciprocal_dual_problem.h). The dual is maximized via
// DualPolicy (default lbfgsb_policy), warm-started across outer iterations.
//
// References:
//   Svanberg 1987, "The method of moving asymptotes",
//     Int. J. Numer. Methods Engng 24:359-373.

#include "argmin/detail/lagrangian.h"
#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/mma_reciprocal_dual_problem.h"
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

namespace argmin
{

template <scalar Scalar = double, int N = dynamic_dimension,
          template <scalar, int> typename DualPolicy = lbfgsb_policy>
struct mma_policy
{
    using scalar_type = Scalar;

    template <int M>
    using rebind = mma_policy<Scalar, M, DualPolicy>;

    struct options_type
    {
        // Asymptote contraction / expansion factors (Svanberg 1987 §3
        // defaults).
        std::optional<Scalar> asymptote_contract{};       // default: 0.7
        std::optional<Scalar> asymptote_expand{};         // default: 1.2

        // Initial asymptote distance as a fraction of the bound range
        // (Svanberg 1987 §3 default: 0.5). For unbounded variables the
        // initial distance is asymptote_init_unbounded.
        std::optional<Scalar> asymptote_init{};           // default: 0.5
        std::optional<Scalar> asymptote_init_unbounded{}; // default: 1.0

        // Numerical safeguards on the asymptote distance (fraction of
        // bound range). Not in the original paper.
        std::optional<Scalar> asymptote_min_fraction{};   // default: 1e-4
        std::optional<Scalar> asymptote_max_fraction{};   // default: 10.0

        // Move-limit buffer (Svanberg 1987 eq. 19): fraction of the
        // asymptote distance reserved off each side of x_k.
        std::optional<Scalar> move_limit_fraction{};      // default: 0.1

        // Move bound (Svanberg 2007 notes eqs. 2.8-2.9, "XXMOVE"): the
        // trial step is additionally capped to x_k +/- move_bound_fraction
        // * range_j, where range_j is the box width for a bounded variable
        // and, for an unbounded one, the scale proxy 2*max(|x_kj|, 1)
        // (a box-width analog). Without this cap an unbounded variable's
        // asymptote can inflate without bound. Literature default 0.5, so a
        // direct value.
        Scalar move_bound_fraction{mma_defaults<Scalar>::move_bound_fraction};

        // Distance-scaled regularizer constant for p_ij, q_ij (Svanberg
        // 2007 notes eqs. 2.3-2.4, "raai"). Each coefficient carries a
        // raai/range_j term that keeps P_j(y), Q_j(y) strictly positive
        // (they never vanish when a gradient component is zero) while
        // scaling with the problem range, unlike a flat additive epsilon
        // which collapses the reciprocal approximation onto its window
        // midpoint. Literature default 1e-5; provisional pending the
        // empirical sweep. Direct value.
        Scalar raai{mma_defaults<Scalar>::raai};

        // Bounded-dual elastics: dual upper bound c_i = dual_bound_scale *
        // max(|g_i(x0)|, 1). Boxing the constraint multipliers keeps the
        // subproblem dual bounded when an inequality-infeasible iterate
        // makes the subproblem infeasible (Svanberg 2002 relaxed
        // subproblem, a_i = 0 instance). Scale multiplier is empirical
        // (swept). Direct value.
        Scalar dual_bound_scale{mma_defaults<Scalar>::dual_bound_scale};

        // Stall detection (forwarded by step_budget_solver into a
        // stall_tolerance_criterion).
        std::optional<Scalar> stall_tolerance_threshold{}; // default: 1e-6
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

        // Asymptotes per component.
        Eigen::Vector<Scalar, N> L, U;

        // Move-limit window for the current step (recomputed each iter).
        Eigen::Vector<Scalar, N> alpha, beta;

        // Approximation params at x_k (recomputed each iter). The
        // reciprocal approximation has a constant term r_i so that
        // g_tilde_i(x_k) = g_i(x_k); see
        // detail/mma_reciprocal_dual_problem.h header.
        Eigen::Vector<Scalar, N> p_obj, q_obj;
        Eigen::Matrix<Scalar, M, N> p_con, q_con;
        Scalar r_obj{0};
        Eigen::Vector<Scalar, M> r_con;

        // History for the asymptote oscillation rule.
        Eigen::Vector<Scalar, N> x_old1, x_old2;

        // Box bounds (or +/-inf for unbounded variables).
        Eigen::Vector<Scalar, N> lower, upper;

        // Warm-started dual variables (m-dim, y >= 0).
        Eigen::Vector<Scalar, M> y_dual;

        // Bounded-dual elastics. c_dual_ref[i] = max(|g_i(x0)|, 1) is the
        // fixed per-constraint scale reference; c_dual[i] =
        // dual_bound_scale * c_dual_ref[i] is the working dual upper bound.
        Eigen::Vector<Scalar, M> c_dual_ref;
        Eigen::Vector<Scalar, M> c_dual;

        std::uint32_t iteration{0};

        // Set during init() when the problem carries equality constraints.
        // MMA sizes its constraint buffers for inequalities only; an
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
        init_asymptotes(s);
        return s;
    }

    template <typename Problem, typename Convergence = default_convergence<scalar_type>>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<Scalar, N>& x0,
                             const solver_options<Convergence>&)
    {
        static_assert(differentiable<Problem, Scalar>,
                      "mma_policy requires differentiable<Problem, Scalar>");
        static_assert(constrained<Problem, Scalar>,
                      "mma_policy requires constrained<Problem, Scalar>");

        const int n = problem.dimension();
        const int n_eq = problem.num_equality();
        const int m = problem.num_inequality();
        // Full constraint count. problem.constraints()/constraint_jacobian()
        // write n_eq + n_ineq rows with the equalities first, so any probe
        // buffer must be sized on the total or an equality-constrained
        // problem overflows it (a heap out-of-bounds write in release,
        // where the debug precondition assert is compiled out). MMA handles
        // inequality constraints only; equality-constrained problems are
        // flagged here and rejected at the top of step(). Use SQP or an
        // augmented Lagrangian for equality constraints.
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

        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();

        // 1. Asymptote update via oscillation detection (Svanberg 1987 §3).
        //    Iter 0 and 1 keep the initial asymptotes (no 3-history yet).
        if(s.iteration >= 2)
        {
            for(int j = 0; j < n; ++j)
            {
                const Scalar dx2 =
                    (s.x[j] - s.x_old1[j]) * (s.x_old1[j] - s.x_old2[j]);
                const Scalar gam = dx2 < Scalar(0.0) ? gam_dec
                                 : dx2 > Scalar(0.0) ? gam_inc : Scalar(1.0);
                const Scalar dist_L = s.x_old1[j] - s.L[j];
                const Scalar dist_U = s.U[j] - s.x_old1[j];
                s.L[j] = s.x[j] - gam * dist_L;
                s.U[j] = s.x[j] + gam * dist_U;

                // Numerical safeguard: clamp asymptote distance to a
                // [s_min_f, s_max_f] band of the bound range. For
                // unbounded variables, only enforce s_min_f * |x_j|
                // (with floor 1e-12) so the dual always sees positive
                // U - x and x - L.
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
        //    x_k +/- move_bnd * range_j. range_j is the box width for a
        //    bounded variable; for an unbounded one it is the scale proxy
        //    2*max(|x_kj|, 1) -- the factor 2 gives a box-width analog (twice
        //    the 0.5-fraction initial asymptote half-width), so the bound
        //    coincides with the initial asymptote and engages only once the
        //    asymptote inflates, rather than over-constraining every step.
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

        // 3. Approximation params (Svanberg 1987 eq. 7) and constants
        //    r_i so that g_tilde_i(x_k) = g_i(x_k):
        //      r_i = g_i(x_k) - sum_j [p_ij/(U-x_kj) + q_ij/(x_kj-L_j)]
        s.r_obj = s.f;
        for(int i = 0; i < m; ++i)
            s.r_con[i] = -s.c_ineq[i];  // MMA convention: g_i = -c_i

        for(int j = 0; j < n; ++j)
        {
            const Scalar dxU = s.U[j] - s.x[j];
            const Scalar dxL = s.x[j] - s.L[j];

            // Distance-scaled regularizer raai/range_j (Svanberg 2007
            // notes eqs. 2.3-2.4). range_j is the box width for a bounded
            // variable, a scale proxy max(|x_kj|, 1) for an unbounded one.
            Scalar range;
            if(s.lower[j] > -inf && s.upper[j] < inf)
                range = s.upper[j] - s.lower[j];
            else
                range = std::max(std::abs(s.x[j]), Scalar(1.0));
            const Scalar raa_reg = raai / range;

            // Svanberg's gradient mixing: the uphill direction carries
            // 1.001 * |g|, the downhill direction 0.001 * |g|, so a
            // component with vanishing gradient still has a positive,
            // range-scaled coefficient and the zero-gradient minimizer
            // stays at x_k rather than the window midpoint.
            const Scalar gp = std::max(s.g[j], Scalar(0.0));
            const Scalar gm = std::max(-s.g[j], Scalar(0.0));
            s.p_obj[j] = dxU * dxU * (Scalar(1.001) * gp + Scalar(0.001) * gm + raa_reg);
            s.q_obj[j] = dxL * dxL * (Scalar(0.001) * gp + Scalar(1.001) * gm + raa_reg);
            s.r_obj -= s.p_obj[j] / dxU + s.q_obj[j] / dxL;

            for(int i = 0; i < m; ++i)
            {
                // argmin convention: c_i >= 0 feasible.
                // MMA convention:    g_i = -c_i <= 0 feasible.
                // Constraint gradient in MMA convention: -J_ineq(i, j).
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

        // 4. Solve the dual subproblem via DualPolicy.
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
        // drive the dual unbounded. Plain MMA has no conservativity loop,
        // so only the dual box (not the primal y-recovery) applies here.
        if(m > 0)
            s.c_dual = s.opts.dual_bound_scale * s.c_dual_ref;
        dual_prob.c_dual_out = &s.c_dual;

        Eigen::Vector<Scalar, N> x_trial(n);

        if(m > 0)
        {
            DualPolicy<scalar_type, MC> dp;
            // Near the dual optimum the decrease left falls below the roundoff of the dual value, so the line
            // search certifies those steps from the exact dual gradient instead.
            dp.options.line_search.relative_value_noise = mma_defaults<Scalar>::dual_value_noise;
            solver_options<default_convergence<scalar_type>> dopts;
            dopts.max_iterations = 100;
            dopts.set_gradient_threshold(mma_defaults<Scalar>::dual_gradient_threshold);
            dopts.set_step_threshold(mma_defaults<Scalar>::dual_step_threshold);
            auto ds = dp.init(dual_prob, s.y_dual, dopts);
            for(int k = 0; k < 100; ++k)
            {
                auto dsr = dp.step(ds);
                // Terminate on the projected KKT residual (the box-
                // constrained dual's stationarity measure), not the raw
                // gradient norm: at a dual solution with an inactive
                // constraint (y_i = 0) the raw component stays bounded
                // away from zero and the loop would run to its cap.
                if(dsr.kkt_residual.value_or(dsr.gradient_norm) < mma_defaults<Scalar>::dual_gradient_threshold
                   || dsr.step_size < mma_defaults<Scalar>::dual_step_threshold)
                    break;
            }
            s.y_dual = ds.x;
            (void)dual_prob.value(s.y_dual);
            x_trial = dual_prob.x_primal;
        }
        else
        {
            // m = 0 unconstrained: dual is 0-dim, skip the dual solve and
            // evaluate the primal at an empty y. eval_primal then runs
            // the analytic per-j minimizer with P_j = p_obj[j],
            // Q_j = q_obj[j].
            const Eigen::Vector<Scalar, MC> y_empty =
                Eigen::Vector<Scalar, MC>::Zero(0);
            (void)dual_prob.value(y_empty);
            x_trial = dual_prob.x_primal;
        }

        // 5. Commit the trial unconditionally (plain MMA: no conservativity).
        const Scalar f_old = s.f;
        const Scalar step_size = (x_trial - s.x).norm();

        s.x_old2 = s.x_old1;
        s.x_old1 = s.x;
        s.x = x_trial;

        s.f = s.problem->value(s.x);
        s.problem->gradient(s.x, s.g);
        if(m > 0)
        {
            Eigen::VectorX<Scalar> c_tmp(m);
            s.problem->constraints(s.x, c_tmp);
            s.c_ineq = c_tmp;
            Eigen::MatrixX<Scalar> J_tmp(m, n);
            s.problem->constraint_jacobian(s.x, J_tmp);
            s.J_ineq = J_tmp;
        }

        ++s.iteration;

        // 6. KKT residual (N&W 2e Definition 12.1 + eq. 12.34).
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
