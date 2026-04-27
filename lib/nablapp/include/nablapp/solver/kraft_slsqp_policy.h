#ifndef HPP_GUARD_NABLAPP_SOLVER_KRAFT_SLSQP_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_KRAFT_SLSQP_POLICY_H

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
//   - detail::compact_lbfgs is replaced with detail::adaptive_bfgs
//     (compact L-BFGS form with per-push Shanno (1978) initial-Hessian
//     rescaling theta = y^T y / s^T y on each accepted pair; N&W
//     eq. 6.20; N&W Section 7.2; Kraft 1988 DFVLR-FB 88-28
//     Section 2.2.3), eliminating the per-step dense_hessian() rebuild
//     which was the dominant per-step cost at IK scale (n = 4-200)
//     while retaining a dense B suitable for the LSQ/LSEI QP subproblem.
//   - detail::active_set_qp_solver is replaced with
//     detail::kraft_lsq_qp_solver, eliminating the Phase-1 feasibility
//     projection and the box-bound stalling that required feasibility
//     restoration in Phase 24.
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

#include "nablapp/detail/adaptive_bfgs.h"
#include "nablapp/detail/kkt_residual.h"
#include "nablapp/detail/kraft_lsq_qp.h"
#include "nablapp/detail/lagrangian.h"
#include "nablapp/line_search/armijo.h"
#include "nablapp/options/qp_options.h"
#include "nablapp/line_search/options.h"
#include "nablapp/result/step_result.h"
#include "nablapp/solver/options.h"
#include "nablapp/types.h"

#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <algorithm>

namespace nablapp
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
        line_search_options line_search{};             // Embedded line search params
        qp_options qp{};                               // QP subproblem params
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
        detail::adaptive_bfgs<double, N, 10> hessian;
        detail::kraft_lsq_qp_solver<double, N> qp_solver;

        // Pre-allocated constraint workspace (sized in init(), reused in step())
        Eigen::VectorXd c_all;
        Eigen::MatrixXd J_all;
        Eigen::VectorXd b_eq_workspace;
        Eigen::VectorXd b_ineq_workspace;

        // Per-step buffers reused across step() invocations to avoid heap
        // allocation in the BFGS curvature-pair update path. J_all_old is
        // populated by problem.constraint_jacobian(x_old, ...) inside step()
        // when computing the Lagrangian gradient at the previous iterate;
        // lam_buf / lam_eq_buf / lam_ineq_buf hold per-step QP multiplier
        // copies so the multiplier scatter is allocation-free.
        Eigen::MatrixXd J_all_old;
        Eigen::VectorXd lam_buf;
        Eigen::VectorXd lam_eq_buf;
        Eigen::VectorXd lam_ineq_buf;

        // Cumulative count of phi(alpha) calls made by the Armijo
        // backtracker across every step() invocation since init/reset.
        // Counts both the main merit-function line search and the
        // second-order-correction line search if SOC fires. Incremented
        // inside phi_ls and phi_soc lambdas. Exposed so benchmarks can
        // divide by iteration count to get "average backtracks per
        // step" without needing policy-internal instrumentation.
        // Rationale: cartan's UR3e perf profile showed ~2.6x more
        // absolute FK work per pose in nablapp_slsqp than in
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

        // Adaptive-theta dense BFGS: identity baseline, theta recomputed
        // on every push (L-BFGS-style) to prevent Hessian drift on SQP
        // problems with shifting Lagrangian curvature (HS026 family).
        s.hessian = detail::adaptive_bfgs<double, N, 10>(n);
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

        // Direct access to the in-place BFGS Hessian -- no dense_hessian()
        // rebuild at the QP call site. The dense B is refreshed by
        // detail::adaptive_bfgs::push() at the end of step() on an
        // accepted curvature pair (Shanno 1978 / N&W eq. 6.20 rescaling;
        // N&W Section 7.2; Kraft 1988 DFVLR-FB 88-28 Section 2.2.3).
        const auto& B = s.hessian.hessian();

        s.b_eq_workspace = -s.c_eq;
        s.b_ineq_workspace = -s.c_ineq;

        // Box bounds on the step p: p_lo <= p <= p_hi.
        // The Kraft LSQ cascade handles infinite bounds natively by
        // skipping the augmented +I / -I rows, so we do not need to
        // clip to a large finite surrogate here.
        Eigen::Vector<double, N> p_lo = (Eigen::Vector<double, N>(s.lower) -
                                          Eigen::Vector<double, N>(s.x)).eval();
        Eigen::Vector<double, N> p_hi = (Eigen::Vector<double, N>(s.upper) -
                                          Eigen::Vector<double, N>(s.x)).eval();

        auto qp_res = s.qp_solver.solve(B, Eigen::Vector<double, N>(s.g),
                                         s.J_eq, s.b_eq_workspace,
                                         s.J_ineq, s.b_ineq_workspace,
                                         p_lo, p_hi);

        Eigen::Vector<double, N> p = qp_res.x;

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
            Eigen::VectorXd lambda_eq_null = Eigen::VectorXd::Zero(s.n_eq);
            Eigen::VectorXd mu_ineq_null = Eigen::VectorXd::Zero(s.n_ineq);
            if constexpr(constrained<P>)
            {
                if(qp_res.lambda.size() >= s.n_eq + s.n_ineq)
                {
                    if(s.n_eq > 0)
                        lambda_eq_null = qp_res.lambda.head(s.n_eq);
                    if(s.n_ineq > 0)
                        mu_ineq_null = qp_res.lambda.segment(s.n_eq, s.n_ineq);
                }
            }
            double kkt_null = detail::kkt_residual<double,
                                                   Eigen::Dynamic,
                                                   Eigen::Dynamic,
                                                   Eigen::Dynamic>(
                s.g, s.J_eq, s.J_ineq,
                lambda_eq_null, mu_ineq_null,
                s.c_eq, s.c_ineq);

            return step_result<double>{
                .objective_value = s.objective_value,
                .gradient_norm = s.g.norm(),
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

        // L1 merit function for the line search (Kraft 1988, N&W 18.3).
        auto merit = [&](const Eigen::Vector<double, N>& xk) -> double {
            double fval = s.problem->value(xk);
            double viol = 0.0;

            if constexpr(constrained<P>)
            {
                if(s.n_eq + s.n_ineq > 0)
                {
                    Eigen::VectorXd c_all(s.n_eq + s.n_ineq);
                    s.problem->constraints(xk, c_all);
                    auto ceq = c_all.head(s.n_eq);
                    auto cineq = c_all.tail(s.n_ineq);

                    if(ceq.size() > 0)
                        viol += ceq.cwiseAbs().sum();
                    if(cineq.size() > 0)
                        viol += (-cineq).cwiseMax(0.0).sum();
                }
            }

            return fval + s.sigma * viol;
        };

        double merit_0 = merit(s.x);

        double dphi_merit = s.g.dot(p) - s.sigma * constraint_viol_0;

        if(dphi_merit >= 0.0)
            dphi_merit = -1e-8;

        // Backtracking Armijo line search on the L1 merit.
        auto phi_ls = [&](double alpha) {
            ++s.line_search_calls;
            Eigen::Vector<double, N> x_trial = s.x + alpha * p;
            for(int i = 0; i < n; ++i)
            {
                if(x_trial[i] < s.lower[i]) x_trial[i] = s.lower[i];
                if(x_trial[i] > s.upper[i]) x_trial[i] = s.upper[i];
            }
            return merit(x_trial);
        };

        auto ls = armijo(phi_ls, merit_0, dphi_merit, options.line_search);
        double alpha = ls.alpha;

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
        if(!ls.success && constraint_viol_0 > 1e-8)
        {
            if constexpr(constrained<P>)
            {
                if(s.n_eq + s.n_ineq > 0)
                {
                    Eigen::Vector<double, N> x_trial = s.x + p;
                    for(int i = 0; i < n; ++i)
                    {
                        if(x_trial[i] < s.lower[i]) x_trial[i] = s.lower[i];
                        if(x_trial[i] > s.upper[i]) x_trial[i] = s.upper[i];
                    }

                    Eigen::VectorXd c_trial(s.n_eq + s.n_ineq);
                    s.problem->constraints(x_trial, c_trial);
                    auto c_eq_trial = c_trial.head(s.n_eq);
                    auto c_ineq_trial = c_trial.tail(s.n_ineq);

                    Eigen::VectorXd b_eq_soc = s.n_eq > 0
                        ? (-c_eq_trial + s.J_eq * p).eval()
                        : Eigen::VectorXd{};
                    Eigen::VectorXd b_ineq_soc = s.n_ineq > 0
                        ? (-c_ineq_trial + s.J_ineq * p).eval()
                        : Eigen::VectorXd{};

                    auto soc_res = s.qp_solver.solve(
                        B, Eigen::Vector<double, N>(s.g),
                        s.J_eq, b_eq_soc, s.J_ineq, b_ineq_soc,
                        p_lo, p_hi);

                    if(soc_res.status == detail::qp_status::optimal)
                    {
                        Eigen::Vector<double, N> p_combined = p + soc_res.x;
                        auto phi_soc = [&](double a) {
                            ++s.line_search_calls;
                            Eigen::Vector<double, N> x_trial = s.x + a * p_combined;
                            for(int i = 0; i < n; ++i)
                            {
                                if(x_trial[i] < s.lower[i]) x_trial[i] = s.lower[i];
                                if(x_trial[i] > s.upper[i]) x_trial[i] = s.upper[i];
                            }
                            return merit(x_trial);
                        };
                        auto ls_soc = armijo(phi_soc, merit_0, dphi_merit,
                                             options.line_search);
                        if(ls_soc.success && ls_soc.alpha > alpha)
                        {
                            p = p_combined;
                            alpha = ls_soc.alpha;
                        }
                    }
                }
            }
        }

        if(alpha < 1e-15)
            alpha = 1e-4;

        // Update iterate
        Eigen::Vector<double, N> x_old = s.x;
        double old_f = s.objective_value;
        s.x = s.x + alpha * p;

        for(int i = 0; i < n; ++i)
        {
            if(s.x[i] < s.lower[i]) s.x[i] = s.lower[i];
            if(s.x[i] > s.upper[i]) s.x[i] = s.upper[i];
        }

        s.objective_value = s.problem->value(s.x);

        Eigen::Vector<double, N> g_old = s.g;
        s.problem->gradient(s.x, s.g);

        if constexpr(constrained<P>)
        {
            if(s.n_eq + s.n_ineq > 0)
            {
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
        Eigen::Vector<double, N> grad_L_old = g_old;
        Eigen::Vector<double, N> grad_L_new = s.g;

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

                s.problem->constraint_jacobian(x_old, s.J_all_old);

                if(s.n_eq > 0 && lam_take >= s.n_eq)
                {
                    s.lam_eq_buf.head(s.n_eq) = s.lam_buf.head(s.n_eq);
                    grad_L_old.noalias() -= s.J_all_old.topRows(s.n_eq).transpose()
                                           * s.lam_eq_buf.head(s.n_eq);
                    grad_L_new.noalias() -= s.J_eq.transpose()
                                           * s.lam_eq_buf.head(s.n_eq);
                }

                if(s.n_ineq > 0 && lam_take >= s.n_eq + s.n_ineq)
                {
                    s.lam_ineq_buf.head(s.n_ineq) = s.lam_buf.segment(s.n_eq, s.n_ineq);
                    grad_L_old.noalias() -= s.J_all_old.bottomRows(s.n_ineq).transpose()
                                           * s.lam_ineq_buf.head(s.n_ineq);
                    grad_L_new.noalias() -= s.J_ineq.transpose()
                                           * s.lam_ineq_buf.head(s.n_ineq);
                }
            }
        }

        Eigen::Vector<double, N> sk = s.x - x_old;
        Eigen::Vector<double, N> yk = grad_L_new - grad_L_old;

        // Skip BFGS updates on non-positive curvature pairs.
        // In SLSQP the Lagrangian gradient difference y_k can easily
        // have s^T y < 0 when the constraint Hessian contribution
        // (A_{k+1} - A_k)^T lam dominates the objective curvature --
        // feeding such a pair into the BFGS formula distorts B
        // instead of improving it. adaptive_bfgs::push does its own
        // s^T y > 0 guard, but checking here keeps the semantics
        // explicit in the policy step.
        //
        // Reference: Kraft 1988 DFVLR-FB 88-28 Section 2.2.3;
        //            N&W Procedure 18.2 damping guard.
        const double sTy = sk.dot(yk);
        if(sk.norm() > 1e-15 && sTy > 0.0)
        {
            // adaptive_bfgs::push recomputes theta = y^T y / s^T y from
            // the latest pair on every call and rebuilds B via the
            // compact L-BFGS representation over the stored history.
            // This is L-BFGS-with-adaptive-theta semantics, which
            // tracks local curvature even as the SQP iterate moves
            // towards a degenerate constrained optimum (e.g. HS026).
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
        Eigen::VectorXd lambda_eq_kkt = Eigen::VectorXd::Zero(s.n_eq);
        Eigen::VectorXd mu_ineq_kkt = Eigen::VectorXd::Zero(s.n_ineq);
        if constexpr(constrained<P>)
        {
            if(qp_res.lambda.size() >= s.n_eq + s.n_ineq)
            {
                if(s.n_eq > 0)
                    lambda_eq_kkt = qp_res.lambda.head(s.n_eq);
                if(s.n_ineq > 0)
                    mu_ineq_kkt = qp_res.lambda.segment(s.n_eq, s.n_ineq);
            }
        }
        double kkt = detail::kkt_residual<double,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic>(
            s.g, s.J_eq, s.J_ineq,
            lambda_eq_kkt, mu_ineq_kkt,
            s.c_eq, s.c_ineq);

        // Primal feasibility (L-infinity) reported into
        // step_result.constraint_violation for dimensional consistency with
        // step_result.kkt_residual. L1-merit internal paths keep L1 per
        // N&W eq. 15.24 (merit semantics distinct from reporting semantics).
        //
        // Reference: N&W 2e Definition 12.1 (KKT primal feasibility).
        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = s.g.norm(),
            .step_size = sk.norm(),
            .objective_change = s.objective_value - old_f,
            .improved = s.objective_value < old_f,
            .constraint_violation = detail::primal_feasibility_inf(s.c_eq, s.c_ineq),
            .x_norm = s.x.norm(),
            .kkt_residual = kkt,
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
};

}

#endif
