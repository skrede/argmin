#ifndef HPP_GUARD_ARGMIN_SOLVER_CCSA_QUADRATIC_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_CCSA_QUADRATIC_POLICY_H

// CCSA quadratic-penalty policy (Svanberg 2002 §4.2 / NLopt LD_CCSAQ).
//
// Solves inequality-constrained optimization problems using the CCSA
// (Conservative Convex Separable Approximations) framework of Svanberg
// 2002 with the quadratic penalty approximation:
//
//   g_0(x) = f + grad_f . dx + 0.5 rho  ||dx/sigma||^2
//   g_i(x) = fc_i + dfc_i . dx + 0.5 rhoc_i  ||dx/sigma||^2
//
// Trust region: sigma-based (per-dimension radius), updated via
// oscillation detection (Svanberg 1987 Section 3 heuristic) with
// min/max clamping per NLopt ccsa_quadratic.c.
//
// Inner loop: conservativity test (Svanberg 2002 Section 4.2) with
// wval-based rho growth (NLopt mma.c:502-503) and inter-outer 0.1x
// decay (NLopt mma.c:524). Acceptance follows NLopt's logic: accept
// improving or feasibility-restoring steps even before conservativity
// is established. Null step on max_inner exhaustion.
//
// Dual solve: warm-started projected Newton across inner and outer
// iterations (fixes cold-start-at-y=1 overhead).
//
// References:
//   Svanberg 1987, "The method of moving asymptotes", IJNME 24:359-373.
//   Svanberg 2002, "A class of globally convergent optimization methods
//     based on conservative convex separable approximations", SIAM J.
//     Optim. 12(2):555-573.
//   NLopt ccsa_quadratic.c (Steven G. Johnson 2008-2012).

#include "argmin/detail/ccsa_dual_problem.h"
#include "argmin/detail/mma_subproblem.h"
#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/lagrangian.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/options/mma_subproblem_options.h"
#include "argmin/result/step_result.h"
#include "argmin/solver/options.h"
#include "argmin/types.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace argmin
{

template <int N = dynamic_dimension,
          template<int> typename DualPolicy = lbfgsb_policy>
struct ccsa_quadratic_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = ccsa_quadratic_policy<M, DualPolicy>;

    struct options_type
    {
        // Sigma trust-region controls (replaces asymptote schedule).
        // sigma_j starts at sigma_init * range_j, contracts by
        // sigma_decrease on oscillation, expands by sigma_increase on
        // monotone progress. Clamped to [sigma_min_frac, sigma_max_frac]
        // times range_j.
        //
        // Reference: NLopt ccsa_quadratic.c lines 322-327 (init) and
        //            531-546 (oscillation update).
        std::optional<double> sigma_init{};           // default: 0.5
        std::optional<double> sigma_decrease{};       // default: 0.7
        std::optional<double> sigma_increase{};       // default: 1.2
        std::optional<double> sigma_min_fraction{};   // default: 1e-8
        std::optional<double> sigma_max_fraction{};   // default: 10.0

        // CCSA regularizer controls. rho and per-constraint rhoc start
        // at rho_init, grow on non-conservative trials via the wval-
        // based formula (NLopt mma.c:502-503), decay by rho_decay
        // between outer iterations (NLopt mma.c:524), floored at
        // rho_min (CCSA_RHOMIN).
        //
        // Reference: Svanberg 2002 Section 4.2; NLopt ccsa_quadratic.c.
        std::optional<double> rho_init{};             // default: 1.0
        std::optional<double> rho_min{};              // default: 1e-5
        std::optional<double> rho_decay{};            // default: 0.1

        std::optional<std::uint16_t> max_inner_iterations{};  // default: 15

        // Bounded-dual elastics: dual upper bound c_i = dual_bound_scale *
        // max(|g_i(x0)|, 1). The per-constraint scale reference keeps the
        // exact-l1-penalty condition c_i > |mu_i*| scale-relative; the
        // scale multiplier is empirical (swept). See detail/mma_subproblem.h
        // recover_elastic_slacks(). Direct value.
        double dual_bound_scale{1000.0};

        // Framework-side stall detection threshold. Wired by
        // basic_solver::forward_policy_hints into
        // stall_tolerance_criterion with best_seen_feasible tracking.
        std::optional<double> stall_tolerance_threshold{};  // default: 1e-6
        std::uint16_t stall_window{50};

        mma_subproblem_options subproblem{};
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

        Eigen::Vector<double, N> sigma;          // trust-region radii
        Eigen::Vector<double, N> x_old1, x_old2; // history for oscillation
        Eigen::Vector<double, N> lower, upper;   // box bounds

        // Warm-started dual variables (m-dimensional). Persists across
        // inner and outer iterations. Initialized to zero.
        Eigen::Vector<double, M> y_dual;

        // Bounded-dual elastics. c_dual_ref[i] = max(|g_i(x0)|, 1) is the
        // fixed per-constraint scale reference; c_dual[i] =
        // dual_bound_scale * c_dual_ref[i] is the working dual upper bound.
        Eigen::Vector<double, M> c_dual_ref;
        Eigen::Vector<double, M> c_dual;

        // Fallback subproblem solver for the m=0 case (no constraints,
        // analytic primal without dual solve).
        std::optional<detail::mma_subproblem_solver<double, N, M>> subproblem_m0;

        std::uint32_t iteration{0};

        double rho{1.0};
        Eigen::Vector<double, M> rhoc;

        // Best feasible tracking (NLopt acceptance logic).
        bool feasible{false};
        double infeasibility{0.0};

        // Set during init() when the problem carries equality constraints.
        // CCSA sizes its constraint buffers for inequalities only; an
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
        s.rho = ri;
        s.rhoc.setConstant(ri);

        const int n = problem.dimension();
        const double si = s.opts.sigma_init.value_or(0.5);
        constexpr double inf = std::numeric_limits<double>::infinity();
        for(int j = 0; j < n; ++j)
        {
            if(s.lower[j] > -inf && s.upper[j] < inf)
                s.sigma[j] = si * (s.upper[j] - s.lower[j]);
            else
                s.sigma[j] = 1.0;
        }

        return s;
    }

    template <typename Problem, typename Convergence = default_convergence>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<double, N>& x0,
                             const solver_options<Convergence>&)
    {
        static_assert(differentiable<Problem>,
                      "ccsa_quadratic_policy requires differentiable<Problem>");
        static_assert(constrained<Problem>,
                      "ccsa_quadratic_policy requires constrained<Problem>");

        const int n = problem.dimension();
        const int n_eq = problem.num_equality();
        const int m_ineq = problem.num_inequality();
        // Full constraint count. problem.constraints()/constraint_jacobian()
        // write n_eq + n_ineq rows with the equalities first, so any probe
        // buffer must be sized on the total or an equality-constrained
        // problem overflows it (a heap out-of-bounds write in release,
        // where the debug precondition assert is compiled out). CCSA
        // handles inequality constraints only; equality-constrained
        // problems are flagged here and rejected at the top of step(). Use
        // SQP or an augmented Lagrangian for equality constraints.
        const int n_all = n_eq + m_ineq;
        state_type<Problem> s;
        s.problem = &problem;
        s.invalid_problem = (n_eq > 0);

        s.x = x0;
        s.g.resize(n);
        s.c_ineq.resize(m_ineq);
        s.J_ineq.resize(m_ineq, n);

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
            }
        }

        s.c_dual_ref.resize(m_ineq);
        s.c_dual_ref.setOnes();
        s.c_dual.resize(m_ineq);
        if(!s.invalid_problem)
            for(int i = 0; i < m_ineq; ++i)
                s.c_dual_ref[i] = std::max(std::abs(s.c_ineq[i]), 1.0);

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

        s.sigma.resize(n);
        constexpr double inf = std::numeric_limits<double>::infinity();
        const double si = s.opts.sigma_init.value_or(0.5);
        for(int j = 0; j < n; ++j)
        {
            if(s.lower[j] > -inf && s.upper[j] < inf)
                s.sigma[j] = si * (s.upper[j] - s.lower[j]);
            else
                s.sigma[j] = 1.0;
        }

        s.x_old1 = x0;
        s.x_old2 = x0;
        s.iteration = 0;

        const double ri = s.opts.rho_init.value_or(1.0);
        s.rho = ri;
        s.rhoc.resize(m_ineq);
        s.rhoc.setConstant(ri);

        // Initial feasibility assessment. Skipped for a rejected problem
        // whose inequality buffer was intentionally left unpopulated.
        s.feasible = true;
        s.infeasibility = 0.0;
        for(int i = 0; i < m_ineq && !s.invalid_problem; ++i)
        {
            double gi = -s.c_ineq[i]; // g_i <= 0 form
            if(gi > 0.0)
            {
                s.feasible = false;
                s.infeasibility = std::max(s.infeasibility, gi);
            }
        }

        s.y_dual.resize(m_ineq);
        s.y_dual.setZero();

        // Fallback for the m=0 unconstrained case (analytic primal).
        if(m_ineq == 0)
            s.subproblem_m0.emplace(n, 0);

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

        const std::uint16_t max_inner =
            s.opts.max_inner_iterations.value_or(15);
        const double rho_min = s.opts.rho_min.value_or(1e-5);
        const double rho_decay = s.opts.rho_decay.value_or(0.1);
        const double sig_dec = s.opts.sigma_decrease.value_or(0.7);
        const double sig_inc = s.opts.sigma_increase.value_or(1.2);
        const double sig_min_f = s.opts.sigma_min_fraction.value_or(1e-8);
        const double sig_max_f = s.opts.sigma_max_fraction.value_or(10.0);
        constexpr double inf = std::numeric_limits<double>::infinity();

        // 1. Update sigma via oscillation detection (NLopt ccsa_quadratic.c
        //    lines 531-546). Skipped for the first two iterations where
        //    x_old2 == x_old1.
        //
        // Reference: Svanberg 1987 Section 3; NLopt ccsa_quadratic.c.
        if(s.iteration > 1)
        {
            for(int j = 0; j < n; ++j)
            {
                double dx2 = (s.x[j] - s.x_old1[j])
                           * (s.x_old1[j] - s.x_old2[j]);
                double gam = dx2 < 0.0 ? sig_dec
                           : dx2 > 0.0 ? sig_inc : 1.0;
                s.sigma[j] *= gam;
                if(s.lower[j] > -inf && s.upper[j] < inf)
                {
                    double range = s.upper[j] - s.lower[j];
                    s.sigma[j] = std::min(s.sigma[j], sig_max_f * range);
                    s.sigma[j] = std::max(s.sigma[j], sig_min_f * range);
                }
            }
        }

        // Constraint sign convention: argmin c_ineq >= 0 feasible,
        // MMA/CCSA g_i <= 0 feasible. g_i = -c_ineq_i, dg_i = -J_ineq_i.
        Eigen::Vector<double, MC> g_mma = -s.c_ineq;
        Eigen::Matrix<double, MC, N> dg_mma = -s.J_ineq;

        // Effective bounds: clamp sigma-based move limits into box bounds.
        Eigen::Vector<double, N> eff_lb(n), eff_ub(n);
        for(int j = 0; j < n; ++j)
        {
            eff_lb[j] = std::max(s.lower[j], s.x[j] - s.sigma[j]);
            eff_ub[j] = std::min(s.upper[j], s.x[j] + s.sigma[j]);
        }

        // 2. Inner conservativity loop (Svanberg 2002 Section 4.2).
        //
        // The m-dimensional Lagrange dual is solved by an intra-argmin
        // solver (DualPolicy, default lbfgsb_policy). This mirrors
        // NLopt's architecture where LD_MMA solves its own dual via a
        // recursive LD_MMA call. The dual variables y are warm-started
        // across inner and outer iterations.
        //
        // Reference: NLopt ccsa_quadratic.c ccsa_quadratic_minimize()
        //            inner loop; optimize.c line 642 (dual_opt creation).
        detail::ccsa_dual_problem<double, N, MC> dual_prob;
        dual_prob.x_out = &s.x;
        dual_prob.f_out = s.f;
        dual_prob.grad_f_out = &s.g;
        dual_prob.fc_out = &g_mma;
        dual_prob.dfc_out = &dg_mma;
        dual_prob.sigma_out = &s.sigma;
        dual_prob.lb_out = &eff_lb;
        dual_prob.ub_out = &eff_ub;
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
        Eigen::VectorXd c_ineq_trial = s.c_ineq;
        // Relaxed constraint values g_tilde_i(x*) - y_i* and recovered
        // elastic slacks (populated by the dual solve).
        Eigen::Vector<double, MC> gcval_relaxed = dual_prob.gcval;
        Eigen::Vector<double, MC> y_elastic = dual_prob.gcval;

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
            // Update mutable rho/rhoc on the dual problem (these grow
            // on non-conservative trials within the inner loop) and drop
            // the single-evaluation cache before re-solving the dual.
            dual_prob.rho_out = s.rho;
            dual_prob.rhoc_out = &s.rhoc;
            dual_prob.invalidate_cache();

            // Solve the dual via DualPolicy. The dual problem is
            // m-dimensional box-constrained (y >= 0). The solver
            // warm-starts from s.y_dual.
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

                // Evaluate primal at the converged y to populate
                // dual_prob.x_primal, gval, wval, gcval.
                (void)dual_prob.value(s.y_dual);
                x_trial = dual_prob.x_primal;
                // Recover the primal elastic slacks and the relaxed
                // constraint values (raw g_tilde_i on an inactive
                // constraint, 0 on one the elastic absorbed).
                detail::recover_elastic_slacks(
                    s.y_dual, s.c_dual, dual_prob.gcval,
                    y_elastic, gcval_relaxed);
            }
            else
            {
                // m=0 unconstrained: analytic primal, no dual solve.
                x_trial = s.subproblem_m0->solve(
                    s.x, s.f, s.g, g_mma, dg_mma,
                    s.sigma, s.rho, s.rhoc, eff_lb, eff_ub,
                    s.opts.subproblem);
                dual_prob.gval = s.subproblem_m0->gval();
                dual_prob.wval = s.subproblem_m0->wval();
            }

            f_trial = s.problem->value(x_trial);
            if(m > 0)
            {
                Eigen::VectorXd c_tmp(m);
                s.problem->constraints(x_trial, c_tmp);
                c_ineq_trial = c_tmp;
            }

            // Conservativity test (NLopt ccsa_quadratic.c lines 448, 461)
            // against the relaxed constraint values g_tilde_i - y_i.
            bool inner_done = (dual_prob.gval >= f_trial);
            bool feasible_cur = true;
            double infeasibility_cur = 0.0;
            for(int i = 0; i < m; ++i)
            {
                double gi_trial = -c_ineq_trial[i];
                inner_done = inner_done
                    && (gcval_relaxed[i] >= gi_trial);
                if(gi_trial > 0.0)
                {
                    feasible_cur = false;
                    infeasibility_cur =
                        std::max(infeasibility_cur, gi_trial);
                }
            }

            // NLopt acceptance logic (ccsa_quadratic.c lines 468-492).
            if((f_trial < s.f
                && (inner_done || feasible_cur || !s.feasible))
               || (!s.feasible && infeasibility_cur < s.infeasibility))
            {
                s.infeasibility = infeasibility_cur;
                if(infeasibility_cur == 0.0)
                    s.feasible = true;
            }

            if(inner_done)
                break;

            // Rho growth on non-conservative trial (NLopt mma.c:502-503),
            // on the relaxed constraint approximation.
            double wval = std::max(dual_prob.wval, 1e-20);
            if(f_trial > dual_prob.gval)
                s.rho = std::min(10.0 * s.rho,
                    1.1 * (s.rho + (f_trial - dual_prob.gval) / wval));
            for(int i = 0; i < m; ++i)
            {
                double gi_trial = -c_ineq_trial[i];
                if(gi_trial > gcval_relaxed[i])
                    s.rhoc[i] = std::min(10.0 * s.rhoc[i],
                        1.1 * (s.rhoc[i]
                               + (gi_trial - gcval_relaxed[i])
                                 / wval));
            }

            // Null step on max_inner exhaustion.
            if(inner + 1 == max_inner)
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
                    .constraint_violation = detail::primal_feasibility_inf(
                        c_eq_empty, s.c_ineq),
                    .x_norm = s.x.norm(),
                    .kkt_residual = kkt,
                };
            }
        }

        // 3. Inter-outer rho decay (NLopt ccsa_quadratic.c line 524).
        s.rho = std::max(rho_decay * s.rho, rho_min);
        for(int i = 0; i < m; ++i)
            s.rhoc[i] = std::max(rho_decay * s.rhoc[i], rho_min);

        // 4. Always commit the trial as the new base point. NLopt
        //    rebuilds the approximation at x which is the best accepted
        //    point, but sigma uses the trial trajectory (xprev = xcur
        //    at line 381). By always committing, sigma adapts properly
        //    and the approximation explores from the trial.
        const double f_old = s.f;
        const double step_size = (x_trial - s.x).norm();

        s.x_old2 = s.x_old1;
        s.x_old1 = s.x;
        s.x = x_trial;
        s.f = f_trial;
        if(m > 0)
            s.c_ineq = c_ineq_trial;

        s.problem->gradient(s.x, s.g);
        if(m > 0)
        {
            Eigen::MatrixXd J_tmp(m, n);
            s.problem->constraint_jacobian(s.x, J_tmp);
            s.J_ineq = J_tmp;
        }

        ++s.iteration;

        // 5. KKT residual (N&W 2e Definition 12.1 + eq 12.34).
        //    Dual multipliers y_dual map directly to argmin mu_ineq
        //    (no sign flip; see detail/ccsa_dual_problem.h comment).
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
            for(int i = 0; i < m; ++i)
                s.c_dual_ref[i] = std::max(std::abs(s.c_ineq[i]), 1.0);
        }
        s.iteration = 0;
        s.x_old1 = x0;
        s.x_old2 = x0;

        const double ri = s.opts.rho_init.value_or(1.0);
        s.rho = ri;
        s.rhoc.setConstant(ri);
        s.feasible = false;
        s.infeasibility = 0.0;
        s.y_dual.setZero();
    }

    template <typename P>
    void reset_clear(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        reset(s, x0);
        const int n = static_cast<int>(x0.size());
        constexpr double inf_val = std::numeric_limits<double>::infinity();
        const double si = s.opts.sigma_init.value_or(0.5);
        for(int j = 0; j < n; ++j)
        {
            if(s.lower[j] > -inf_val && s.upper[j] < inf_val)
                s.sigma[j] = si * (s.upper[j] - s.lower[j]);
            else
                s.sigma[j] = 1.0;
        }
    }
};

}

#endif
