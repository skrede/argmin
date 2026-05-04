#ifndef HPP_GUARD_ARGMIN_SOLVER_KRAFT_SLSQP_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_KRAFT_SLSQP_POLICY_H

// Kraft 1988 SLSQP policy for basic_solver.
//
// Implements Sequential Least Squares Programming (SLSQP) faithful to
// Kraft (1988) DFVLR-FB 88-28: dense damped BFGS Hessian of the
// Lagrangian, Kraft's LSQ/LSEI cascade for the QP subproblem with
// native box-bound handling, L1 merit function for the line search,
// and second-order correction on infeasibility.
//
// Each step: (1) pull the BFGS Hessian B directly (no rebuild),
// (2) solve the QP via the kraft_lsq_qp_solver cascade, (3) update
// the L1 merit penalty sigma from the QP multipliers, (4) backtrack
// on the merit function, (5) damped BFGS update of B using the
// Lagrangian gradient difference.
//
// Differences from the earlier L-BFGS variant:
//   - detail::compact_lbfgs is replaced with detail::dense_ldl_bfgs
//     (packed L D L^T BFGS approximation with Powell-damped rank-1
//     updates per Fletcher & Powell 1974; Shanno 1978 initial-Hessian
//     rescaling theta = y^T y / s^T y is applied only on the first
//     push after construction or reset, after which the LDL absorbs
//     curvature evolution incrementally at O(n^2) per push). The
//     packed factor is consumed directly by the QP via factor_to_E_and_f
//     (skipping LLT(B) at the QP call site).
//   - detail::active_set_qp_solver is replaced with
//     detail::kraft_lsq_qp_solver, eliminating the Phase-1 feasibility
//     projection and the box-bound stalling that required feasibility
//     restoration earlier.
//
// Supports: unconstrained, equality, inequality, box, and mixed
// constraints. When the problem is only differentiable, constraints
// default to empty and the solver reduces to BFGS with a QP step
// computation.
//
// Reference: Kraft, D. (1988) "A Software Package for Sequential
//            Quadratic Programming", DFVLR-FB 88-28, Deutsche
//            Forschungs- und Versuchsanstalt fuer Luft- und Raumfahrt
//            (dense BFGS, LSQ/LSEI cascade, L1 merit, SOC).
//            N&W Chapter 18 (SQP methods), Procedure 18.2 (damped
//            BFGS), Section 18.3 (L1 merit function).
//            K&W Section 10.4 (penalty methods, augmented Lagrangian).

#include "argmin/detail/dense_ldl_bfgs.h"
#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/kraft_lsq_qp.h"
#include "argmin/detail/kraft_lsq_qp_recovery.h"
#include "argmin/detail/lagrangian.h"
#include "argmin/detail/merit_function.h"
#include "argmin/line_search/armijo.h"
#include "argmin/options/qp_options.h"
#include "argmin/line_search/options.h"
#include "argmin/result/step_result.h"
#include "argmin/solver/options.h"
#include "argmin/types.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <algorithm>

namespace argmin
{

template <int N = dynamic_dimension>
struct kraft_slsqp_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = kraft_slsqp_policy<M>;

    struct options_type
    {
        std::optional<double> initial_penalty{};       // penalty weight (default: 1.0, Kraft 1988)
        std::optional<double> penalty_growth{};        // penalty multiplier (default: 10.0, Kraft 1988)
        // Minimum constraint violation at x_k below which the second-order
        // correction (Maratos retry) is skipped on Armijo failure. Below this
        // threshold the linearization is consistent enough that the SoC step
        // adds work without materially changing the search direction.
        // Default 1e-3 (Kraft 1988 §2.2.4 / N&W §18.3 conventional choice);
        // the prior 1e-8 effectively always fired the retry.
        std::optional<double> soc_min_violation{};
        line_search_options line_search{};             // Embedded line search params
        qp_options qp{};                               // QP subproblem params
        // BFGS-reset-on-LS-failure cap. On Armijo line-search
        // exhaustion, reset the BFGS Hessian to identity (Shanno-
        // rescaled on next push) and retry the QP up to
        // bfgs_reset_max times. After exhaustion, return a null
        // step with diagnostics.bfgs_reset_count populated. Default
        // 5 = NLopt slsqp.c:1890-1895 ireset parity.
        //
        // Reference: NLopt slsqp.c:1890-1895 (ireset loop);
        //            PITFALLS.md Section L (line-search exhaustion).
        // argmin variant: replaces the K3 force-continue
        // (alpha = 1e-4) which corrupted BFGS curvature by
        // stepping along an unaccepted direction.
        // Adopted from: NLopt slsqp.c:1890-1895.
        std::size_t bfgs_reset_max{5};
        std::uint16_t stall_window{50};
    };

    options_type options{};

    template <typename P = void>
    struct state_type
    {
        static constexpr int M = [] {
            if constexpr(has_constraint_count<P>) return constraint_count_v<P>;
            else return dynamic_dimension;
        }();

        const P* problem{nullptr};
        Eigen::Vector<double, N> x;
        Eigen::Vector<double, N> g;
        Eigen::VectorXd c_eq;
        Eigen::VectorXd c_ineq;
        Eigen::Matrix<double, Eigen::Dynamic, N> J_eq;
        Eigen::Matrix<double, Eigen::Dynamic, N> J_ineq;
        Eigen::Vector<double, N> lower;
        Eigen::Vector<double, N> upper;
        Eigen::VectorXd lambda;
        double objective_value{};
        double sigma{1.0};
        detail::dense_ldl_bfgs<double, N> hessian;
        detail::kraft_lsq_qp_recovery_solver<double, N> qp_solver;

        // Pre-allocated constraint workspace (sized in init(), reused in step())
        Eigen::VectorXd c_all;
        Eigen::MatrixXd J_all;
        Eigen::VectorXd b_eq_workspace;
        Eigen::VectorXd b_ineq_workspace;

        // Per-step buffers reused across step() invocations to avoid heap
        // allocation in the BFGS curvature-pair update path. J_all_old
        // holds J(x_k) for the Lagrangian gradient at the previous
        // iterate; populated by copying s.J_all just before the
        // J(x_{k+1}) re-evaluation overwrites it, eliminating the
        // separate constraint_jacobian(x_old, ...) call that the SLSQP
        // BFGS pair previously required (one J-eval/iter instead of two).
        // lam_buf / lam_eq_buf / lam_ineq_buf hold per-step QP multiplier
        // copies so the multiplier scatter is allocation-free.
        Eigen::MatrixXd J_all_old;
        Eigen::VectorXd lam_buf;
        Eigen::VectorXd lam_eq_buf;
        Eigen::VectorXd lam_ineq_buf;

        // Per-step / per-line-search trial buffers. Sized in init() to
        // (n) or (n_eq + n_ineq); reused on every step() and on every
        // Armijo backtrack inside the line-search lambdas (phi_ls /
        // phi_soc / merit). Eliminates the heap allocation that the
        // earlier inline `Eigen::Vector<double, N> x_trial = ...`
        // expression triggered on every merit-function evaluation,
        // which dominated wall time on small constrained HS problems
        // where Armijo can call phi() ~5-20x per outer iteration.
        Eigen::Vector<double, N> x_trial_buf;
        Eigen::Vector<double, N> p_lo_buf;
        Eigen::Vector<double, N> p_hi_buf;
        Eigen::Vector<double, N> p_buf;
        Eigen::Vector<double, N> p_combined_buf;
        Eigen::Vector<double, N> x_old_buf;
        Eigen::Vector<double, N> g_old_buf;
        Eigen::Vector<double, N> grad_L_old_buf;
        Eigen::Vector<double, N> grad_L_new_buf;
        Eigen::Vector<double, N> sk_buf;
        Eigen::Vector<double, N> yk_buf;

        // Pre-factored Hessian buffers for the kraft_lsq_qp_recovery_solver
        // direct path: E = sqrt(D) * L^T (upper triangular) and f =
        // -D^{-1/2} * L^{-1} * g, both produced in O(n^2) by
        // detail::dense_ldl_bfgs::factor_to_E_and_f. Replaces the
        // O(n^3 / 3) Eigen::LLT(B) the QP solver runs internally on every
        // solve when given a dense B. Sized in init().
        Eigen::Matrix<double, N, N> E_buf;
        Eigen::Vector<double, N> f_buf;
        Eigen::VectorXd c_trial_buf;
        Eigen::VectorXd b_eq_soc_buf;
        Eigen::VectorXd b_ineq_soc_buf;

        // Shared buffer for the lambda_eq / mu_ineq scatter in the
        // null-step, deferred-guard reset, and end-of-step KKT residual
        // paths. At most one of those paths fires per step() call and
        // none survive across calls, so the same buffer pair is safe
        // to reuse instead of zero-allocating fresh VectorXds.
        Eigen::VectorXd kkt_lambda_eq_buf;
        Eigen::VectorXd kkt_mu_ineq_buf;

        // Cumulative count of phi(alpha) calls made by the Armijo
        // backtracker across every step() invocation since init/reset.
        // Counts both the main merit-function line search and the
        // second-order-correction line search if SOC fires. Incremented
        // inside phi_ls and phi_soc lambdas. Exposed so benchmarks can
        // divide by iteration count to get "average backtracks per
        // step" without needing policy-internal instrumentation.
        // Rationale: cartan's UR3e perf profile showed ~2.6x more
        // absolute FK work per pose in argmin_slsqp than in
        // nlopt_slsqp; the merit-function line search is the suspected
        // source and this counter lets us measure the hypothesis
        // directly.
        std::uint64_t line_search_calls{0};

        std::uint32_t iteration{0};
        int n_eq{0};
        int n_ineq{0};
        int n{0};
    };

    template <typename Problem, typename Convergence>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<double, N>& x0,
                             const solver_options<Convergence>& opts,
                             const options_type& policy_opts)
    {
        options = policy_opts;
        return init(problem, x0, opts);
    }

    template <typename Problem, typename Convergence = default_convergence>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<double, N>& x0,
                             const solver_options<Convergence>& /*opts*/)
    {
        const int n = problem.dimension();
        state_type<Problem> s;
        s.problem = &problem;

        s.n = n;
        s.x = x0;
        s.g.setZero(n);
        s.objective_value = problem.value(x0);
        problem.gradient(x0, s.g);

        // Per-step / per-line-search trial buffers.
        s.x_trial_buf.resize(n);
        s.p_lo_buf.resize(n);
        s.p_hi_buf.resize(n);
        s.p_buf.resize(n);
        s.p_combined_buf.resize(n);
        s.x_old_buf.resize(n);
        s.g_old_buf.resize(n);
        s.grad_L_old_buf.resize(n);
        s.grad_L_new_buf.resize(n);
        s.sk_buf.resize(n);
        s.yk_buf.resize(n);

        // Bounds
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

        // Constraints
        if constexpr(constrained<Problem>)
        {
            s.n_eq = problem.num_equality();
            s.n_ineq = problem.num_inequality();

            const int m = s.n_eq + s.n_ineq;
            if(m > 0)
            {
                s.c_all.resize(m);
                s.J_all.resize(m, n);
                s.b_eq_workspace.resize(s.n_eq);
                s.b_ineq_workspace.resize(s.n_ineq);
                s.J_all_old.resize(m, n);
                s.lam_buf.resize(m);
                s.lam_eq_buf.resize(s.n_eq);
                s.lam_ineq_buf.resize(s.n_ineq);
                s.c_trial_buf.resize(m);
                s.b_eq_soc_buf.resize(s.n_eq);
                s.b_ineq_soc_buf.resize(s.n_ineq);
                s.kkt_lambda_eq_buf.resize(s.n_eq);
                s.kkt_mu_ineq_buf.resize(s.n_ineq);

                problem.constraints(x0, s.c_all);
                s.c_eq = s.c_all.head(s.n_eq);
                s.c_ineq = s.c_all.tail(s.n_ineq);

                problem.constraint_jacobian(x0, s.J_all);
                s.J_eq = s.J_all.topRows(s.n_eq);
                s.J_ineq = s.J_all.bottomRows(s.n_ineq);

                s.lambda = Eigen::VectorXd::Zero(m);
            }
        }
        else
        {
            s.n_eq = 0;
            s.n_ineq = 0;
            s.c_all.resize(0);
            s.J_all.resize(0, n);
            s.c_eq.resize(0);
            s.c_ineq.resize(0);
            s.J_eq.resize(0, n);
            s.J_ineq.resize(0, n);
            s.lambda.resize(0);
        }

        // Packed LDL^T BFGS Hessian (Fletcher-Powell 1974). Initialized
        // at L = I, D = I (B_0 = I) and updated by Powell-damped rank-1
        // LDL updates per push (NLopt slsqp.c slsqpb_); replaces the
        // adaptive-theta compact L-BFGS path that rebuilt B from scratch
        // every push at O(M * n^2) cost.
        s.hessian = detail::dense_ldl_bfgs<double, N>(n);
        s.E_buf.resize(n, n);
        s.f_buf.resize(n);
        s.sigma = options.initial_penalty.value_or(1.0);
        s.iteration = 0;

        // Pre-allocate kraft_lsq_qp_solver workspace. Size finite-bound
        // counts to n so the allocator covers any runtime pattern of
        // finite/infinite box bounds without re-allocating.
        s.qp_solver.resize(n, s.n_eq, s.n_ineq, n, n);

        return s;
    }

    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        // Invariant at top of step: s.g, s.c_eq/c_ineq, s.J_eq/J_ineq are
        // already fresh at s.x. Maintained by init() / reset() / the prior
        // step()'s post-update block at the bottom of this function, which
        // writes gradient and constraint data for the accepted new iterate
        // before returning. basic_solver::step_n does not mutate state.x
        // between policy.step() calls, so no top-of-step re-evaluation is
        // needed.
        const int n = s.n;

        // Pre-factor the BFGS Hessian into E = sqrt(D) * L^T and f =
        // -D^{-1/2} * L^{-1} * g directly off the packed L, D factors
        // maintained by detail::dense_ldl_bfgs. The recovery solver's
        // direct path consumes (E, f) at no further factorization cost,
        // skipping the O(n^3 / 3) Eigen::LLT(B) the previous code ran
        // inside kraft_lsq_qp_solver::solve on every step.
        // Reference: NLopt slsqp.c lsq_ (lines 1234-1340) for the
        // E = sqrt(D) * L^T reconstruction; Kraft 1988 DFVLR-FB 88-28
        // Section 2.2.3 (BFGS update), Section 3.2 (LSQ cast).
        s.hessian.factor_to_E_and_f(s.E_buf, s.g, s.f_buf);

        s.b_eq_workspace = -s.c_eq;
        s.b_ineq_workspace = -s.c_ineq;

        // Box bounds on the step p: p_lo <= p <= p_hi.
        // The Kraft LSQ cascade handles infinite bounds natively by
        // skipping the augmented +I / -I rows, so we do not need to
        // clip to a large finite surrogate here.
        s.p_lo_buf.noalias() = s.lower - s.x;
        s.p_hi_buf.noalias() = s.upper - s.x;

        auto qp_res = s.qp_solver.solve_with_factored_hessian(
            s.E_buf, s.f_buf, s.g,
            s.J_eq, s.b_eq_workspace,
            s.J_ineq, s.b_ineq_workspace,
            s.p_lo_buf, s.p_hi_buf);

        s.p_buf = qp_res.x;
        Eigen::Vector<double, N>& p = s.p_buf;

        double p_norm = p.norm();
        if(p_norm < 1e-15)
        {
            // Zero QP direction: the LSQ/LSEI cascade returned p = 0,
            // which happens either at a true KKT point (correct solver
            // behavior at a stationary point) or under active-set
            // degeneracy. is_null_step = true exempts
            // step_tolerance_criterion from firing stalled on iter 2;
            // kkt_residual via the Full E-measure lets
            // objective_tolerance_criterion declare convergence when
            // the iterate is a true KKT point; constraint_violation
            // reports L-infinity primal feasibility for dimensional
            // consistency with kkt_residual. Mirrors nw_sqp_policy's
            // null-step return so the two SQP policies behave
            // consistently at near-optimum initial points.
            //
            // Reference: N&W 2e Section 18.3 (SQP null-step semantics);
            //            Definition 12.1 + eq. 12.34 (full KKT
            //            first-order optimality E-measure).
            s.kkt_lambda_eq_buf.setZero(s.n_eq);
            s.kkt_mu_ineq_buf.setZero(s.n_ineq);
            if constexpr(constrained<P>)
            {
                if(qp_res.lambda.size() >= s.n_eq + s.n_ineq)
                {
                    if(s.n_eq > 0)
                        s.kkt_lambda_eq_buf = qp_res.lambda.head(s.n_eq);
                    if(s.n_ineq > 0)
                        s.kkt_mu_ineq_buf = qp_res.lambda.segment(s.n_eq, s.n_ineq);
                }
            }
            // Seed s.lambda from the current QP multipliers so the
            // Lagrangian-gradient helper reads fresh multipliers
            // (matches the kkt_residual leg's multiplier source and
            // nw_sqp's null-step convention at :289-326).
            if constexpr(constrained<P>)
            {
                if(qp_res.lambda.size() >= s.n_eq + s.n_ineq)
                {
                    if(s.n_eq > 0)
                        s.lambda.head(s.n_eq) = qp_res.lambda.head(s.n_eq);
                    if(s.n_ineq > 0)
                        s.lambda.segment(s.n_eq, s.n_ineq) =
                            qp_res.lambda.segment(s.n_eq, s.n_ineq);
                }
            }
            double kkt_null = detail::kkt_residual<double,
                                                   Eigen::Dynamic,
                                                   Eigen::Dynamic,
                                                   Eigen::Dynamic>(
                s.g, s.J_eq, s.J_ineq,
                s.kkt_lambda_eq_buf, s.kkt_mu_ineq_buf,
                s.c_eq, s.c_ineq);

            return step_result<double>{
                .objective_value = s.objective_value,
                // argmin variant: report ||grad f - A^T lambda|| (Lagrangian
                // gradient) instead of raw ||grad f||; rationale: KKT
                // first-order optimality (N&W eq. 12.34).
                .gradient_norm = lagrangian_gradient_norm(s),
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
                .is_null_step = true,
                .constraint_violation = detail::primal_feasibility_inf(s.c_eq, s.c_ineq),
                .x_norm = s.x.norm(),
                .kkt_residual = kkt_null,
            };
        }

        // Update penalty parameter sigma from the QP multipliers.
        // Ensures sigma >= |lambda|_inf + delta so the QP step is a
        // descent direction for the L1 merit under the linearized
        // KKT conditions. Monotone (never decreases).
        //
        // Reference: N&W eq. 18.36 (sufficient penalty for descent).
        double constraint_viol_0 = 0.0;
        if(s.c_eq.size() > 0)
            constraint_viol_0 += s.c_eq.cwiseAbs().sum();
        if(s.c_ineq.size() > 0)
            constraint_viol_0 += (-s.c_ineq).cwiseMax(0.0).sum();

        if(qp_res.lambda.size() > 0)
        {
            const double lambda_max = qp_res.lambda.cwiseAbs().maxCoeff();
            if(lambda_max + 0.5 > s.sigma)
                s.sigma = lambda_max + 1.0;
        }

        // Iter-0 cold-start calibration of sigma. The monotone rule
        // above bumps sigma only when lambda_max + 0.5 > sigma; on
        // problems where the cold-start sigma_0 = 1.0 already exceeds
        // lambda_max it leaves sigma at 1.0, which on objective-
        // dominated initial points (HS071 from x_0 = (1, 5, 5, 1))
        // produces an iter-0 step that satisfies merit-decrease
        // against an under-weighted violation term and parks the
        // iterate strongly infeasible. Force a magnitude-aware floor
        // at iter-0 only.
        //
        // The cold-start is additionally gated on a strict-violation
        // threshold (||c_0||_1 > 1e-6): when iter-0 is already
        // (near-)feasible the K-factor problem-scale floor degenerates
        // (denom -> eps; magnitude_floor explodes) and over-penalises
        // the linearised constraint perturbation produced by the QP
        // step. HS026 from x_0 = (-2.6, 2, 2) hits c_0 = 0 exactly and
        // is the canonical canary; without this gate the cold-start
        // drives sigma to ~2e14 and the line search rejects every
        // descent direction. The gate keeps the cold-start active only
        // on the case it is designed for (objective-dominated AND
        // strictly infeasible iter-0).
        //
        // argmin variant: combines lambda-floor + K-factor floor at
        // iter-0; preserves the existing monotone rule for iter > 0.
        // The K-factor problem-scale floor is the magnitude-aware
        // companion that closes objective-dominated cold-starts where
        // ||lambda||_inf is small but |f_0| / ||c_0||_1 is large.
        //
        // Reference: N&W 2e Section 18.3 / eq. 18.36 (lambda-floor for
        //            descent on the L1 merit);
        //            Kraft 1988 DFVLR-FB 88-28 Section 2.2.6 (sigma
        //            update companion).
        constexpr double cold_start_violation_floor = 1e-6;
        if(s.iteration == 0 && qp_res.lambda.size() > 0
           && constraint_viol_0 > cold_start_violation_floor)
        {
            s.sigma = detail::calibrate_initial_penalty(
                s.sigma, qp_res.lambda, s.objective_value,
                constraint_viol_0);
        }

        // L1 merit function for the line search (Kraft 1988, N&W 18.3).
        auto merit = [&](const Eigen::Vector<double, N>& xk) -> double {
            double fval = s.problem->value(xk);
            double viol = 0.0;

            if constexpr(constrained<P>)
            {
                if(s.n_eq + s.n_ineq > 0)
                {
                    s.problem->constraints(xk, s.c_trial_buf);
                    auto ceq = s.c_trial_buf.head(s.n_eq);
                    auto cineq = s.c_trial_buf.tail(s.n_ineq);

                    if(ceq.size() > 0)
                        viol += ceq.cwiseAbs().sum();
                    if(cineq.size() > 0)
                        viol += (-cineq).cwiseMax(0.0).sum();
                }
            }

            return fval + s.sigma * viol;
        };

        double merit_0 = merit(s.x);

        // h4-weighted directional derivative of the L1 merit (Kraft
        // 1988 §3.4). When the QP wrapper produced the step via
        // augmentation (relaxation_factor > 0), the augmented step
        // targets a (1 - s_aug)-scaled relaxation of the original
        // linearized constraints; weighting the violation term by
        // h4 = 1 - s_aug gives the slope of the merit along p that
        // reflects the residual violation the step actually attacks.
        // For the direct path (relaxation_factor = 0), h4 = 1 and
        // dphi_merit reduces to the standard unweighted slope.
        //
        // The h4 slope feeds the Armijo line search below; rejection
        // of the augmented step is deferred to a post-line-search
        // guard so an alpha-shrunk version of p has a chance to
        // satisfy the merit decrease test even when the unit-step
        // slope is non-negative. Only when the line search and the
        // second-order correction both fail to find a decreasing
        // step do we reset BFGS and null-step on the augmented path.
        //
        // Reference: Kraft, D. (1988). DFVLR-FB 88-28, §3 (line
        //            search) and §3.4 (Inconsistent Linearization).
        //            Nocedal & Wright (2006). Numerical Optimization,
        //            2e, §18.3 (L1 merit function for SQP line search).
        const double h4 = 1.0 - qp_res.relaxation_factor;
        double dphi_merit = s.g.dot(p) - s.sigma * constraint_viol_0 * h4;

        // If the L1 merit directional derivative is non-negative, the
        // penalty parameter sigma is too small to make p a descent
        // direction. Solve directly for the minimum sigma that makes
        // dphi_merit strictly negative and bump sigma to that value:
        //
        //   dphi_merit < 0  <=>  sigma > g^T p / (viol * h4)
        //
        // Set sigma_target = g^T p / (viol * h4) + 1, which yields
        // dphi_merit_new = -viol * h4 < 0 when viol * h4 > 0.
        // Caps sigma at 1e10 to bound runaway growth on near-feasible
        // iterates. Replaces an earlier dphi_merit = -1e-8 clamp that
        // forced a fake-descent slope rather than fixing the penalty.
        //
        // Reference: Powell (1978) "A fast algorithm for nonlinearly
        //            constrained optimization calculations", §6;
        //            N&W 2e Section 18.5 eq. 18.36.
        if(dphi_merit >= 0.0)
        {
            const double viol_h4 = constraint_viol_0 * h4;
            if(viol_h4 > 1e-12)
            {
                const double sigma_target = s.g.dot(p) / viol_h4 + 1.0;
                if(sigma_target > s.sigma)
                {
                    s.sigma = std::min(sigma_target, 1e10);
                    dphi_merit = s.g.dot(p) - s.sigma * viol_h4;
                }
            }
        }

        // Backtracking Armijo line search on the L1 merit.
        auto phi_ls = [&](double alpha) {
            ++s.line_search_calls;
            s.x_trial_buf.noalias() = s.x + alpha * p;
            for(int i = 0; i < n; ++i)
            {
                if(s.x_trial_buf[i] < s.lower[i]) s.x_trial_buf[i] = s.lower[i];
                if(s.x_trial_buf[i] > s.upper[i]) s.x_trial_buf[i] = s.upper[i];
            }
            return merit(s.x_trial_buf);
        };

        auto ls = armijo(phi_ls, merit_0, dphi_merit, options.line_search);
        double alpha = ls.alpha;
        bool ls_success = ls.success;

        // Second-order correction (Kraft 1988 Section 2.2.4,
        // N&W Section 18.3 "Maratos effect").
        //
        // If the Armijo line search rejects the full step because
        // the linearization underestimates the constraint curvature
        // (Maratos effect), re-solve the QP with an updated RHS
        // that subtracts the nonlinear constraint residual at the
        // trial point x + p. Using the ORIGINAL Jacobian J_k (not
        // J_{k+1}) preserves the QP solver factorizations.
        //
        // RHS update:
        //   b_eq_soc  = -c_eq(x + p)  + J_eq(x)  * p
        //   b_ineq_soc = -c_ineq(x + p) + J_ineq(x) * p
        //
        // The correction step dp is added to p and the line search
        // is retried with the combined direction.
        const double soc_min_violation = options.soc_min_violation.value_or(1e-3);
        if(!ls.success && constraint_viol_0 > soc_min_violation)
        {
            if constexpr(constrained<P>)
            {
                if(s.n_eq + s.n_ineq > 0)
                {
                    s.x_trial_buf.noalias() = s.x + p;
                    for(int i = 0; i < n; ++i)
                    {
                        if(s.x_trial_buf[i] < s.lower[i]) s.x_trial_buf[i] = s.lower[i];
                        if(s.x_trial_buf[i] > s.upper[i]) s.x_trial_buf[i] = s.upper[i];
                    }

                    s.problem->constraints(s.x_trial_buf, s.c_trial_buf);
                    auto c_eq_trial = s.c_trial_buf.head(s.n_eq);
                    auto c_ineq_trial = s.c_trial_buf.tail(s.n_ineq);

                    if(s.n_eq > 0)
                        s.b_eq_soc_buf.noalias() = -c_eq_trial + s.J_eq * p;
                    if(s.n_ineq > 0)
                        s.b_ineq_soc_buf.noalias() = -c_ineq_trial + s.J_ineq * p;

                    // Reuse the (E, f) factored on this step's main QP
                    // solve: the Hessian and gradient have not changed
                    // between the main solve and this SOC retry, only
                    // the constraint RHS (b_eq_soc_buf, b_ineq_soc_buf).
                    auto soc_res = s.qp_solver.solve_with_factored_hessian(
                        s.E_buf, s.f_buf, s.g,
                        s.J_eq, s.b_eq_soc_buf, s.J_ineq, s.b_ineq_soc_buf,
                        s.p_lo_buf, s.p_hi_buf);

                    if(soc_res.status == detail::qp_status::optimal)
                    {
                        s.p_combined_buf.noalias() = p + soc_res.x;
                        auto phi_soc = [&](double a) {
                            ++s.line_search_calls;
                            s.x_trial_buf.noalias() = s.x + a * s.p_combined_buf;
                            for(int i = 0; i < n; ++i)
                            {
                                if(s.x_trial_buf[i] < s.lower[i]) s.x_trial_buf[i] = s.lower[i];
                                if(s.x_trial_buf[i] > s.upper[i]) s.x_trial_buf[i] = s.upper[i];
                            }
                            return merit(s.x_trial_buf);
                        };
                        auto ls_soc = armijo(phi_soc, merit_0, dphi_merit,
                                             options.line_search);
                        if(ls_soc.success && ls_soc.alpha > alpha)
                        {
                            p = s.p_combined_buf;
                            alpha = ls_soc.alpha;
                            ls_success = true;
                        }
                    }
                }
            }
        }

        // Deferred descent guard for the augmented-QP path (Kraft 1988
        // §3.4 inconsistent-linearization recovery). If the Armijo line
        // search and the second-order correction retry both failed to
        // find a step that decreases the L1 merit, the augmented
        // direction is not a usable descent direction even after alpha
        // shrinking. Reset BFGS to identity and return as a null step
        // so the next outer iter restarts with B = I.
        //
        // The direct path (relaxation_factor = 0) is intentionally not
        // null-stepped here: it keeps the alpha=1e-4 fallback below,
        // matching SLSQP's behavior of taking a small forced step on
        // rare unit-step descent failures rather than discarding the
        // BFGS history. Resetting BFGS only on the augmented path
        // localizes the guard to cells where inconsistent linearization
        // is the actual mechanism (Kraft 1988 §3 line-search recovery).
        if(!ls_success && qp_res.relaxation_factor > 0.0)
        {
            s.hessian.reset();

            s.kkt_lambda_eq_buf.setZero(s.n_eq);
            s.kkt_mu_ineq_buf.setZero(s.n_ineq);
            if constexpr(constrained<P>)
            {
                if(qp_res.lambda.size() >= s.n_eq + s.n_ineq)
                {
                    if(s.n_eq > 0)
                        s.kkt_lambda_eq_buf = qp_res.lambda.head(s.n_eq);
                    if(s.n_ineq > 0)
                        s.kkt_mu_ineq_buf = qp_res.lambda.segment(s.n_eq, s.n_ineq);
                }
            }
            // Seed s.lambda from current QP multipliers (consistency
            // with the kkt_residual leg; mirrors null-step convention).
            if constexpr(constrained<P>)
            {
                if(qp_res.lambda.size() >= s.n_eq + s.n_ineq)
                {
                    if(s.n_eq > 0)
                        s.lambda.head(s.n_eq) = qp_res.lambda.head(s.n_eq);
                    if(s.n_ineq > 0)
                        s.lambda.segment(s.n_eq, s.n_ineq) =
                            qp_res.lambda.segment(s.n_eq, s.n_ineq);
                }
            }
            double kkt_reset = detail::kkt_residual<double,
                                                    Eigen::Dynamic,
                                                    Eigen::Dynamic,
                                                    Eigen::Dynamic>(
                s.g, s.J_eq, s.J_ineq,
                s.kkt_lambda_eq_buf, s.kkt_mu_ineq_buf,
                s.c_eq, s.c_ineq);

            return step_result<double>{
                .objective_value = s.objective_value,
                // argmin variant: report ||grad f - A^T lambda|| (Lagrangian
                // gradient) instead of raw ||grad f||; rationale: KKT
                // first-order optimality (N&W eq. 12.34).
                .gradient_norm = lagrangian_gradient_norm(s),
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
                .is_null_step = true,
                .constraint_violation = detail::primal_feasibility_inf(s.c_eq, s.c_ineq),
                .x_norm = s.x.norm(),
                .kkt_residual = kkt_reset,
                // Augmented-path null step performed exactly one BFGS
                // reset (line 676) before returning. Surface the count
                // so callers can compose
                //   is_null_step && diagnostics.bfgs_reset_count > 0
                // to detect reset-driven nullification.
                .diagnostics = solver_diagnostics{.bfgs_reset_count = 1},
            };
        }

        // BFGS-reset-on-LS-failure retry (NLopt slsqp.c:1890-1895
        // ireset pattern). On unit-step Armijo failure (after the
        // augmented-path null-step branch above has already returned
        // for relaxation_factor > 0 cases), reset the BFGS Hessian
        // to identity (Shanno-rescaled on next push) and retry the
        // direct-path QP solve plus Armijo line search up to
        // options.bfgs_reset_max times. After exhaustion, return a
        // null step with diagnostics.bfgs_reset_count populated so
        // the caller can detect cap exhaustion via:
        //   is_null_step && diagnostics.bfgs_reset_count >= options.bfgs_reset_max
        //
        // Replaces the K3 force-continue (alpha = 1e-4) at this site
        // which corrupted BFGS curvature by stepping along an
        // unaccepted direction. Curvature corruption from the K3 hack
        // was the root cause of the C8 audit gap recorded in
        // PITFALLS.md Section L.
        //
        // Reference: NLopt slsqp.c:1890-1895 (ireset loop);
        //            PITFALLS.md Section L (line-search exhaustion).
        // Adopted from: NLopt slsqp.c:1890-1895.
        std::size_t reset_count = 0;
        while(!ls_success && reset_count < options.bfgs_reset_max)
        {
            s.hessian.reset();
            ++reset_count;

            // Re-factor the (now identity) BFGS Hessian into (E, f)
            // and re-solve the direct-path QP. The constraint
            // workspaces (b_eq_workspace, b_ineq_workspace) and the
            // box-bound deltas (p_lo_buf, p_hi_buf) are unchanged
            // across resets since s.x, s.c_eq, s.c_ineq, s.lower,
            // s.upper are not mutated until the iterate update at
            // the bottom of step(). The factor-reuse path
            // (solve_with_factored_hessian) is correct because (E, f)
            // is freshly recomputed from the reset Hessian on every
            // retry.
            s.hessian.factor_to_E_and_f(s.E_buf, s.g, s.f_buf);
            qp_res = s.qp_solver.solve_with_factored_hessian(
                s.E_buf, s.f_buf, s.g,
                s.J_eq, s.b_eq_workspace,
                s.J_ineq, s.b_ineq_workspace,
                s.p_lo_buf, s.p_hi_buf);

            // Inconsistent-linearization fall-through after reset:
            // the kraft_lsq_qp recovery solver promoted the QP into
            // the augmented (relaxation) path. This signals the
            // constraint linearisation is structurally inconsistent
            // and another Hessian reset cannot fix it. Exit the loop
            // and fall through to the cap-exhausted null-step return
            // below; the caller composes
            //   is_null_step && diagnostics.bfgs_reset_count > 0
            // to recognise the recovery path.
            if(qp_res.relaxation_factor > 0.0)
                break;

            p = qp_res.x;
            p_norm = p.norm();
            // Post-reset zero-direction: the QP at the reset Hessian
            // returned p = 0. This is either a true KKT point (the
            // reset confirms we were at one) or active-set degeneracy
            // that no further reset will resolve. Exit the loop and
            // return cap-style null step with reset_count populated.
            if(p_norm < 1e-15)
                break;

            // Re-evaluate the L1 merit slope at the reset direction.
            // h4 = 1 - relaxation_factor = 1 in the direct path.
            const double h4_retry = 1.0 - qp_res.relaxation_factor;
            dphi_merit = s.g.dot(p) - s.sigma * constraint_viol_0 * h4_retry;
            if(dphi_merit >= 0.0)
            {
                const double viol_h4 = constraint_viol_0 * h4_retry;
                if(viol_h4 > 1e-12)
                {
                    const double sigma_target = s.g.dot(p) / viol_h4 + 1.0;
                    if(sigma_target > s.sigma)
                    {
                        s.sigma = std::min(sigma_target, 1e10);
                        dphi_merit = s.g.dot(p) - s.sigma * viol_h4;
                    }
                }
            }

            auto ls_retry = armijo(phi_ls, merit_0, dphi_merit, options.line_search);
            alpha = ls_retry.alpha;
            ls_success = ls_retry.success;
        }

        if(!ls_success)
        {
            // Cap-exhausted (or fall-through) null step. Refresh the
            // KKT-residual buffers from the most recent qp_res
            // multipliers and surface the cumulative reset count.
            // Mirrors the augmented-path null-step return shape.
            s.kkt_lambda_eq_buf.setZero(s.n_eq);
            s.kkt_mu_ineq_buf.setZero(s.n_ineq);
            if constexpr(constrained<P>)
            {
                if(qp_res.lambda.size() >= s.n_eq + s.n_ineq)
                {
                    if(s.n_eq > 0)
                        s.kkt_lambda_eq_buf = qp_res.lambda.head(s.n_eq);
                    if(s.n_ineq > 0)
                        s.kkt_mu_ineq_buf = qp_res.lambda.segment(s.n_eq, s.n_ineq);
                }
            }
            if constexpr(constrained<P>)
            {
                if(qp_res.lambda.size() >= s.n_eq + s.n_ineq)
                {
                    if(s.n_eq > 0)
                        s.lambda.head(s.n_eq) = qp_res.lambda.head(s.n_eq);
                    if(s.n_ineq > 0)
                        s.lambda.segment(s.n_eq, s.n_ineq) =
                            qp_res.lambda.segment(s.n_eq, s.n_ineq);
                }
            }
            const double kkt_capped = detail::kkt_residual<double,
                                                           Eigen::Dynamic,
                                                           Eigen::Dynamic,
                                                           Eigen::Dynamic>(
                s.g, s.J_eq, s.J_ineq,
                s.kkt_lambda_eq_buf, s.kkt_mu_ineq_buf,
                s.c_eq, s.c_ineq);

            return step_result<double>{
                .objective_value = s.objective_value,
                .gradient_norm = lagrangian_gradient_norm(s),
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
                .is_null_step = true,
                .constraint_violation = detail::primal_feasibility_inf(s.c_eq, s.c_ineq),
                .x_norm = s.x.norm(),
                .kkt_residual = kkt_capped,
                .diagnostics = solver_diagnostics{.bfgs_reset_count = reset_count},
            };
        }

        // Update iterate
        s.x_old_buf = s.x;
        Eigen::Vector<double, N>& x_old = s.x_old_buf;
        double old_f = s.objective_value;
        s.x = s.x + alpha * p;

        for(int i = 0; i < n; ++i)
        {
            if(s.x[i] < s.lower[i]) s.x[i] = s.lower[i];
            if(s.x[i] > s.upper[i]) s.x[i] = s.upper[i];
        }

        s.objective_value = s.problem->value(s.x);

        s.g_old_buf = s.g;
        Eigen::Vector<double, N>& g_old = s.g_old_buf;
        s.problem->gradient(s.x, s.g);

        if constexpr(constrained<P>)
        {
            if(s.n_eq + s.n_ineq > 0)
            {
                // Cache J(x_k) into J_all_old before the next-iter
                // Jacobian overwrites s.J_all. The BFGS curvature-pair
                // update below needs J(x_old) = J(x_k) to compute
                // grad_L_old = g_k - J(x_k)^T lambda; with this cache
                // there is no second constraint_jacobian(x_old, ...) call.
                // First-step initialization is correct: init() at line
                // 264 populates s.J_all with J(x_0) and the first step's
                // x_old is x_0, so the cache holds J(x_0) on entry to
                // the BFGS block.
                s.J_all_old = s.J_all;

                s.problem->constraints(s.x, s.c_all);
                s.c_eq = s.c_all.head(s.n_eq);
                s.c_ineq = s.c_all.tail(s.n_ineq);

                s.problem->constraint_jacobian(s.x, s.J_all);
                s.J_eq = s.J_all.topRows(s.n_eq);
                s.J_ineq = s.J_all.bottomRows(s.n_ineq);
            }
        }

        // BFGS update driven by the Lagrangian gradient difference.
        //
        // y_k = grad_L(x_{k+1}, lam) - grad_L(x_k, lam) with
        // grad_L(x, lam) = g(x) - A(x)^T lam. Each gradient is
        // evaluated at its own iterate's Jacobian so that y_k picks
        // up the second-order constraint curvature term
        // (A_{k+1} - A_k)^T lam alongside the objective curvature
        // g_{k+1} - g_k. This is Kraft's SLSQP variant rather than
        // the "fixed Jacobian" modification in N&W Section 18.4.
        //
        // Reference: Kraft 1988 DFVLR-FB 88-28 Section 2.2.3 (SLSQP
        //            Hessian of the Lagrangian);
        //            N&W eq. 18.13 (Lagrangian Hessian).
        s.grad_L_old_buf = g_old;
        s.grad_L_new_buf = s.g;
        Eigen::Vector<double, N>& grad_L_old = s.grad_L_old_buf;
        Eigen::Vector<double, N>& grad_L_new = s.grad_L_new_buf;

        if constexpr(constrained<P>)
        {
            if(s.n_eq + s.n_ineq > 0 && qp_res.lambda.size() > 0)
            {
                const int m_total = s.n_eq + s.n_ineq;
                const int lam_take = std::min(m_total,
                                              static_cast<int>(qp_res.lambda.size()));
                if(s.lam_buf.size() < lam_take) s.lam_buf.resize(lam_take);
                s.lam_buf.head(lam_take) = qp_res.lambda.head(lam_take);

                if(lam_take == m_total)
                    s.lambda = s.lam_buf.head(m_total);

                // Single combined J^T lambda subtraction across all
                // (eq, ineq) constraints. The prior split into two
                // separate GEMVs (eq then ineq) ran two BLAS calls and
                // walked the buffers twice; J_all / J_all_old are
                // already laid out as the eq-then-ineq concatenation,
                // so a single J^T * lambda matches the storage and
                // halves the GEMV count. Mirrors NLopt slsqp.c
                // slsqpb_'s u[i] = g[i] - sum_j a[j,i]*r[j] loop.
                if(lam_take == m_total)
                {
                    grad_L_old.noalias() -= s.J_all_old.topRows(m_total).transpose()
                                           * s.lam_buf.head(m_total);
                    grad_L_new.noalias() -= s.J_all.topRows(m_total).transpose()
                                           * s.lam_buf.head(m_total);
                }
                else
                {
                    if(s.n_eq > 0 && lam_take >= s.n_eq)
                    {
                        s.lam_eq_buf.head(s.n_eq) = s.lam_buf.head(s.n_eq);
                        grad_L_old.noalias() -= s.J_all_old.topRows(s.n_eq).transpose()
                                               * s.lam_eq_buf.head(s.n_eq);
                        grad_L_new.noalias() -= s.J_eq.transpose()
                                               * s.lam_eq_buf.head(s.n_eq);
                    }
                }
            }
        }

        s.sk_buf.noalias() = s.x - x_old;
        s.yk_buf.noalias() = grad_L_new - grad_L_old;
        Eigen::Vector<double, N>& sk = s.sk_buf;
        Eigen::Vector<double, N>& yk = s.yk_buf;

        // Skip BFGS updates on near-zero or non-positive curvature
        // pairs. In SLSQP the Lagrangian gradient difference y_k can
        // easily have s^T y < 0 when the constraint Hessian contribution
        // (A_{k+1} - A_k)^T lam dominates the objective curvature; the
        // dense_ldl_bfgs::push call below applies Powell damping per
        // N&W eq. 18.22-18.24 internally, but the explicit guard here
        // keeps the policy-step semantics legible and avoids the
        // damping logic for the trivial null-step / non-curvature case.
        //
        // Reference: Kraft 1988 DFVLR-FB 88-28 Section 2.2.3;
        //            N&W Procedure 18.2 damping guard.
        const double sTy = sk.dot(yk);
        if(sk.norm() > 1e-15 && sTy > 0.0)
        {
            // dense_ldl_bfgs::push applies the Shanno (1978) initial
            // scaling theta = y^T y / s^T y on the first call after
            // construction or reset (N&W eq. 6.20), then evolves the
            // packed L D L^T factor by Powell-damped rank-1 LDL updates
            // for every subsequent pair (Fletcher-Powell 1974). The
            // damping safeguard ensures B remains SPD even when the
            // Lagrangian curvature shifts toward a degenerate
            // constrained optimum (HS026 family).
            s.hessian.push(sk, yk);
        }

        ++s.iteration;

        // KKT residual: full first-order optimality error E(x, lambda,
        // mu) as the L-infinity maximum over five legs (stationarity,
        // primal equality feasibility, primal inequality feasibility,
        // dual feasibility, complementarity). Multipliers come from the
        // QP solution; equality multipliers occupy the first n_eq
        // entries of qp_res.lambda and inequality multipliers follow.
        // When m == 0 the helper collapses to ||grad_f||_inf.
        //
        // Reference: N&W 2e Definition 12.1 (KKT conditions:
        //            stationarity, primal feasibility, dual feasibility,
        //            complementarity); eq. 12.34 (Lagrangian
        //            stationarity leg).
        s.kkt_lambda_eq_buf.setZero(s.n_eq);
        s.kkt_mu_ineq_buf.setZero(s.n_ineq);
        if constexpr(constrained<P>)
        {
            if(qp_res.lambda.size() >= s.n_eq + s.n_ineq)
            {
                if(s.n_eq > 0)
                    s.kkt_lambda_eq_buf = qp_res.lambda.head(s.n_eq);
                if(s.n_ineq > 0)
                    s.kkt_mu_ineq_buf = qp_res.lambda.segment(s.n_eq, s.n_ineq);
            }
        }
        double kkt = detail::kkt_residual<double,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic>(
            s.g, s.J_eq, s.J_ineq,
            s.kkt_lambda_eq_buf, s.kkt_mu_ineq_buf,
            s.c_eq, s.c_ineq);

        // Primal feasibility (L-infinity) reported into
        // step_result.constraint_violation for dimensional consistency with
        // step_result.kkt_residual. L1-merit internal paths keep L1 per
        // N&W eq. 15.24 (merit semantics distinct from reporting semantics).
        //
        // Reference: N&W 2e Definition 12.1 (KKT primal feasibility).
        return step_result<double>{
            .objective_value = s.objective_value,
            // argmin variant: report ||grad f - A^T lambda|| (Lagrangian
            // gradient) instead of raw ||grad f||; rationale: KKT
            // first-order optimality (N&W eq. 12.34).
            .gradient_norm = lagrangian_gradient_norm(s),
            .step_size = sk.norm(),
            .objective_change = s.objective_value - old_f,
            .improved = s.objective_value < old_f,
            .constraint_violation = detail::primal_feasibility_inf(s.c_eq, s.c_ineq),
            .x_norm = s.x.norm(),
            .kkt_residual = kkt,
            // Surface BFGS-reset count for telemetry. Zero on the
            // common success path (no resets fired); non-zero only
            // when the retry loop above performed at least one
            // reset before Armijo accepted. Caller telemetry can
            // distinguish "step accepted on first try" from
            // "step accepted only after BFGS curvature was discarded".
            .diagnostics = solver_diagnostics{.bfgs_reset_count = reset_count},
        };
    }

    // Hot start -- preserves BFGS curvature information.
    template <typename P>
    void reset(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        s.x = x0;
        s.objective_value = s.problem->value(x0);
        s.problem->gradient(x0, s.g);
        if constexpr(constrained<P>)
        {
            if(s.n_eq + s.n_ineq > 0)
            {
                s.problem->constraints(x0, s.c_all);
                s.c_eq = s.c_all.head(s.n_eq);
                s.c_ineq = s.c_all.tail(s.n_ineq);

                s.problem->constraint_jacobian(x0, s.J_all);
                s.J_eq = s.J_all.topRows(s.n_eq);
                s.J_ineq = s.J_all.bottomRows(s.n_ineq);
            }
        }
        s.iteration = 0;
    }

    // Cold restart -- clears BFGS curvature information (B := I).
    template <typename P>
    void reset_clear(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        reset(s, x0);
        s.hessian.reset();
        s.sigma = options.initial_penalty.value_or(1.0);
    }

private:
    // Lagrangian gradient norm for step_result.gradient_norm. Replaces
    // raw ||grad f|| reporting which silently misfires ftol-based
    // convergence on near-feasible points where lambda is non-trivial.
    // Mirrors the nw_sqp_policy helper at file-level parity for the
    // SQP family.
    //
    // Reference: N&W 2e Section 12.3 / eq. 12.34 (KKT first-order
    //            optimality measure).
    // Adopted from: argmin nw_sqp_policy.h:599-612 (in-tree pattern).
    template <typename P>
    static double lagrangian_gradient_norm(const state_type<P>& s)
    {
        const int n = static_cast<int>(s.x.size());
        const int m = s.n_eq + s.n_ineq;
        if(m == 0)
            return s.g.norm();

        Eigen::Matrix<double, Eigen::Dynamic, N> A(m, n);
        if(s.n_eq > 0) A.topRows(s.n_eq) = s.J_eq;
        if(s.n_ineq > 0) A.bottomRows(s.n_ineq) = s.J_ineq;
        return detail::lagrangian_gradient(s.g, A, s.lambda).norm();
    }
};

}

#endif
