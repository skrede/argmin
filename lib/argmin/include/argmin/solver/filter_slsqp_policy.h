#ifndef HPP_GUARD_ARGMIN_SOLVER_FILTER_SLSQP_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_FILTER_SLSQP_POLICY_H

// Filter SQP policy for basic_solver.
//
// Mirrors kraft_slsqp_policy in QP solver (kraft_lsq_qp), Hessian
// (dense_ldl_bfgs, packed L D L^T), and constraint handling. Differs
// in globalization:
// Fletcher-Leyffer 2002 filter acceptance with Wachter-Biegler 2006
// switching condition replaces the L1 merit function, eliminating the
// penalty parameter sigma.
//
// Filter acceptance partitions iterations into f-type (near-feasible,
// Armijo on f) and h-type (infeasible, filter dominance test). A
// second-order correction fires on severe Maratos cases where the
// filter rejects the full step despite high constraint violation.
// Feasibility restoration via hybrid L1-then-QP recovers when the
// filter blocks all step sizes.
//
// Reference: Fletcher & Leyffer 2002, "Nonlinear programming without
//            a penalty function", Math. Program. 91:239-269.
//            Wachter & Biegler 2006, "On the implementation of an
//            interior-point filter line-search algorithm for large-scale
//            nonlinear programming", Math. Program. 106:25-57.
//            N&W Section 15.5 (filter methods).
//            Kraft 1988 DFVLR-FB 88-28 (QP solver, Hessian update).

#include "argmin/detail/dense_ldl_bfgs.h"
#include "argmin/detail/filter_acceptance.h"
#include "argmin/detail/filter_restoration.h"
#include "argmin/detail/kkt_residual.h"
#include "argmin/detail/kraft_lsq_qp.h"
#include "argmin/detail/kraft_lsq_qp_recovery.h"
#include "argmin/detail/lagrangian.h"
#include "argmin/detail/merit_function.h"
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
#include <algorithm>

namespace argmin
{

template <int N = dynamic_dimension>
struct filter_slsqp_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = filter_slsqp_policy<M>;

    struct options_type
    {
        line_search_options line_search{};
        qp_options qp{};
        detail::restoration_strategy restoration{detail::restoration_strategy::hybrid};
        std::uint16_t max_restoration_steps{10};
        double soc_violation_threshold{1e-8};
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
        detail::dense_ldl_bfgs<double, N> hessian;
        detail::kraft_lsq_qp_recovery_solver<double, N> qp_solver;
        detail::filter_set<double> filter;

        Eigen::VectorXd c_all;
        Eigen::MatrixXd J_all;
        Eigen::VectorXd b_eq_workspace;
        Eigen::VectorXd b_ineq_workspace;

        // Per-step buffers reused across step() invocations to avoid heap
        // allocation in the BFGS curvature-pair update path.
        Eigen::MatrixXd J_all_old;
        Eigen::VectorXd lam_buf;
        Eigen::VectorXd lam_eq_buf;
        Eigen::VectorXd lam_ineq_buf;

        // Pre-factored Hessian buffers for the dense_ldl_bfgs ->
        // kraft_lsq_qp_recovery_solver direct path. See kraft_slsqp_policy
        // for the rationale: factor_to_E_and_f writes E = sqrt(D) * L^T
        // and f = -D^{-1/2} L^{-1} g at O(n^2), replacing the O(n^3 / 3)
        // LLT(B) the QP solver runs internally on every solve.
        Eigen::Matrix<double, N, N> E_buf;
        Eigen::Vector<double, N> f_buf;

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
                             const solver_options<Convergence>&)
    {
        const int n = problem.dimension();
        state_type<Problem> s;
        s.problem = &problem;

        s.n = n;
        s.x = x0;
        s.g.setZero(n);
        s.objective_value = problem.value(x0);
        problem.gradient(x0, s.g);

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

        double h_0 = 0.0;
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
                h_0 = detail::constraint_violation(s.c_eq, s.c_ineq);
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

        s.hessian = detail::dense_ldl_bfgs<double, N>(n);
        s.E_buf.resize(n, n);
        s.f_buf.resize(n);
        s.iteration = 0;

        // Initialize filter with h_max per Wachter-Biegler 2006 eq. (8).
        s.filter.initialize(1e4 * std::max(1.0, h_0));

        s.qp_solver.resize(n, s.n_eq, s.n_ineq, n, n);

        return s;
    }

    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        const int n = s.n;

        // Factor the BFGS LDL Hessian into (E, f) directly off the
        // packed L, D factors -- skipping the O(n^3 / 3) LLT(B) the QP
        // solver would otherwise run on every solve. See
        // detail::dense_ldl_bfgs::factor_to_E_and_f.
        s.hessian.factor_to_E_and_f(s.E_buf, s.g, s.f_buf);

        // QP subproblem: identical to kraft_slsqp.
        s.b_eq_workspace = -s.c_eq;
        s.b_ineq_workspace = -s.c_ineq;

        Eigen::Vector<double, N> p_lo = (Eigen::Vector<double, N>(s.lower) -
                                          Eigen::Vector<double, N>(s.x)).eval();
        Eigen::Vector<double, N> p_hi = (Eigen::Vector<double, N>(s.upper) -
                                          Eigen::Vector<double, N>(s.x)).eval();

        auto qp_res = s.qp_solver.solve_with_factored_hessian(
            s.E_buf, s.f_buf, Eigen::Vector<double, N>(s.g),
            s.J_eq, s.b_eq_workspace,
            s.J_ineq, s.b_ineq_workspace,
            p_lo, p_hi);

        Eigen::Vector<double, N> p = qp_res.x;
        double p_norm = p.norm();
        if(p_norm < 1e-15)
        {
            // Zero step with high constraint violation: attempt restoration
            // before giving up, since the QP may be infeasible at box bounds.
            //
            // Dual convention: step_result.constraint_violation reports the
            // L-infinity primal feasibility (dimensionally consistent with
            // kkt_residual per N&W 2e Definition 12.1); filter_set entries
            // retain L1 h_k per Fletcher-Leyffer 2002 Section 2 dominance
            // ordering. The `h_zero > 1e-8` restoration trigger keeps the
            // L1 form for consistency with filter semantics at that gate.
            double h_zero_l1 = detail::constraint_violation(s.c_eq, s.c_ineq);
            if constexpr(constrained<P>)
            {
                if(h_zero_l1 > 1e-8 && s.n_eq + s.n_ineq > 0)
                {
                    auto rest = run_restoration_(s);
                    if(rest.success)
                    {
                        // Capture x before the restoration assignment so
                        // step_size below reports the actual primal step
                        // norm (||rest.x - x_old||) rather than the
                        // residual constraint violation.
                        const double rest_step_norm = (rest.x - s.x).norm();
                        s.x = rest.x;
                        s.objective_value = s.problem->value(s.x);
                        s.problem->gradient(s.x, s.g);

                        s.problem->constraints(s.x, s.c_all);
                        s.c_eq = s.c_all.head(s.n_eq);
                        s.c_ineq = s.c_all.tail(s.n_ineq);
                        s.problem->constraint_jacobian(s.x, s.J_all);
                        s.J_eq = s.J_all.topRows(s.n_eq);
                        s.J_ineq = s.J_all.bottomRows(s.n_ineq);

                        const double h_rest_l1 = detail::constraint_violation(s.c_eq, s.c_ineq);
                        s.filter.add(s.objective_value, h_rest_l1);

                        ++s.iteration;
                        return step_result<double>{
                            .objective_value = s.objective_value,
                            .gradient_norm = s.g.norm(),
                            .step_size = rest_step_norm,
                            .objective_change = s.objective_value - s.objective_value,
                            .improved = false,
                            .constraint_violation = detail::primal_feasibility_inf(s.c_eq, s.c_ineq),
                            .x_norm = s.x.norm(),
                        };
                    }
                }
            }

            // Null step: QP zero direction with high constraint
            // violation; filter blocks all trial steps. Kraft 1988
            // Section 2.2.3.
            //
            // is_null_step=true exempts this iterate from
            // step_tolerance stall detection so basic_solver does not
            // double-report the stall via the tolerance criterion on
            // top of the policy-reported solver_status::stalled below.
            //
            // Null-step branch: x_{k+1} = x_k so the active-set LS
            // re-estimate is measured at the current iterate with no
            // staleness. Populating kkt_residual here lets the outer
            // convergence check terminate on a true KKT point even when
            // the QP returns zero at the optimum (otherwise the composite
            // gate falls back to gradient_norm = ||grad_f||_inf which is
            // not zero at constrained optima and blocks termination).
            Eigen::VectorXd lambda_eq_null = Eigen::VectorXd::Zero(s.n_eq);
            Eigen::VectorXd mu_ineq_null = Eigen::VectorXd::Zero(s.n_ineq);
            if constexpr(constrained<P>)
            {
                if(s.n_eq + s.n_ineq > 0)
                {
                    Eigen::Vector<double, N> g_fixed = s.g;
                    Eigen::VectorXd lambda_reest =
                        detail::estimate_multipliers_active_set<double,
                                                                N,
                                                                Eigen::Dynamic,
                                                                Eigen::Dynamic>(
                            g_fixed, s.J_eq, s.J_ineq, s.c_ineq);
                    if(s.n_eq > 0)
                        lambda_eq_null = lambda_reest.head(s.n_eq);
                    if(s.n_ineq > 0)
                        mu_ineq_null = lambda_reest.segment(s.n_eq, s.n_ineq);
                }
            }
            double kkt_null = detail::kkt_residual<double,
                                                   Eigen::Dynamic,
                                                   Eigen::Dynamic,
                                                   Eigen::Dynamic>(
                s.g, s.J_eq, s.J_ineq,
                lambda_eq_null, mu_ineq_null,
                s.c_eq, s.c_ineq);
            ++s.iteration;
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
                .policy_status = h_zero_l1 > 1e-8
                    ? std::optional{solver_status::stalled}
                    : std::nullopt,
            };
        }

        // Current constraint violation and objective.
        double h_k = detail::constraint_violation(s.c_eq, s.c_ineq);
        double f_k = s.objective_value;
        double grad_f_dot_p = s.g.dot(p);

        // Switching condition: f-type when near-feasible with sufficient
        // predicted f-decrease.
        // Reference: Wachter & Biegler 2006 eq. (14).
        bool f_type = detail::is_f_type_iteration(h_k, grad_f_dot_p, s.filter.h_max());

        // Add current point to filter on h-type iterations.
        // Reference: Wachter & Biegler 2006 Algorithm A.
        if(!f_type)
            s.filter.add(f_k, h_k);

        // Backtracking with filter acceptance or Armijo f-descent.
        double alpha = 1.0;
        bool accepted = false;
        double f_trial = f_k;
        double h_trial = h_k;
        Eigen::VectorXd c_eq_trial, c_ineq_trial;

        const double c1 = options.line_search.c1;
        const double rho = options.line_search.rho;
        const auto max_ls = options.line_search.max_iterations;

        for(std::uint16_t ls = 0; ls < max_ls; ++ls)
        {
            Eigen::Vector<double, N> x_trial = s.x + alpha * p;
            for(int i = 0; i < n; ++i)
            {
                if(x_trial[i] < s.lower[i]) x_trial[i] = s.lower[i];
                if(x_trial[i] > s.upper[i]) x_trial[i] = s.upper[i];
            }

            f_trial = s.problem->value(x_trial);
            ++s.line_search_calls;

            if constexpr(constrained<P>)
            {
                if(s.n_eq + s.n_ineq > 0)
                {
                    Eigen::VectorXd c_trial(s.n_eq + s.n_ineq);
                    s.problem->constraints(x_trial, c_trial);
                    c_eq_trial = c_trial.head(s.n_eq);
                    c_ineq_trial = c_trial.tail(s.n_ineq);
                    h_trial = detail::constraint_violation(c_eq_trial, c_ineq_trial);
                }
                else
                {
                    h_trial = 0.0;
                }
            }
            else
            {
                h_trial = 0.0;
            }

            if(f_type)
            {
                // Armijo f-descent check.
                if(f_trial <= f_k + c1 * alpha * grad_f_dot_p)
                {
                    accepted = true;
                    break;
                }
            }
            else
            {
                // Filter acceptance check.
                if(s.filter.is_acceptable(f_trial, h_trial))
                {
                    accepted = true;
                    break;
                }
            }

            alpha *= rho;
        }

        // Second-order correction for severe Maratos effect.
        // Fires when the step was rejected AND the current point has
        // significant constraint violation, indicating the linearization
        // underestimated constraint curvature.
        //
        // Reference: Wachter & Biegler 2006 Section 2.4;
        //            N&W Section 18.3 (Maratos effect).
        if(!accepted && h_k > options.soc_violation_threshold)
        {
            if constexpr(constrained<P>)
            {
                if(s.n_eq + s.n_ineq > 0)
                {
                    Eigen::Vector<double, N> x_full = s.x + p;
                    for(int i = 0; i < n; ++i)
                    {
                        if(x_full[i] < s.lower[i]) x_full[i] = s.lower[i];
                        if(x_full[i] > s.upper[i]) x_full[i] = s.upper[i];
                    }

                    Eigen::VectorXd c_full(s.n_eq + s.n_ineq);
                    s.problem->constraints(x_full, c_full);
                    auto c_eq_full = c_full.head(s.n_eq);
                    auto c_ineq_full = c_full.tail(s.n_ineq);

                    Eigen::VectorXd b_eq_soc = s.n_eq > 0
                        ? (-c_eq_full + s.J_eq * p).eval()
                        : Eigen::VectorXd{};
                    Eigen::VectorXd b_ineq_soc = s.n_ineq > 0
                        ? (-c_ineq_full + s.J_ineq * p).eval()
                        : Eigen::VectorXd{};

                    // Reuse the (E, f) factored on this step's main QP
                    // solve: Hessian and gradient unchanged, only the
                    // constraint RHS differs (SOC retry).
                    auto soc_res = s.qp_solver.solve_with_factored_hessian(
                        s.E_buf, s.f_buf, Eigen::Vector<double, N>(s.g),
                        s.J_eq, b_eq_soc, s.J_ineq, b_ineq_soc,
                        p_lo, p_hi);

                    if(soc_res.status == detail::qp_status::optimal)
                    {
                        Eigen::Vector<double, N> p_soc = p + soc_res.x;
                        double alpha_soc = 1.0;

                        for(std::uint16_t ls = 0; ls < max_ls; ++ls)
                        {
                            Eigen::Vector<double, N> x_trial = s.x + alpha_soc * p_soc;
                            for(int i = 0; i < n; ++i)
                            {
                                if(x_trial[i] < s.lower[i]) x_trial[i] = s.lower[i];
                                if(x_trial[i] > s.upper[i]) x_trial[i] = s.upper[i];
                            }

                            double f_soc = s.problem->value(x_trial);
                            ++s.line_search_calls;

                            double h_soc = 0.0;
                            Eigen::VectorXd c_eq_soc_trial, c_ineq_soc_trial;
                            if(s.n_eq + s.n_ineq > 0)
                            {
                                Eigen::VectorXd c_soc(s.n_eq + s.n_ineq);
                                s.problem->constraints(x_trial, c_soc);
                                c_eq_soc_trial = c_soc.head(s.n_eq);
                                c_ineq_soc_trial = c_soc.tail(s.n_ineq);
                                h_soc = detail::constraint_violation(c_eq_soc_trial, c_ineq_soc_trial);
                            }

                            bool soc_accepted = false;
                            if(f_type)
                            {
                                if(f_soc <= f_k + c1 * alpha_soc * grad_f_dot_p)
                                    soc_accepted = true;
                            }
                            else
                            {
                                if(s.filter.is_acceptable(f_soc, h_soc))
                                    soc_accepted = true;
                            }

                            if(soc_accepted)
                            {
                                p = p_soc;
                                alpha = alpha_soc;
                                f_trial = f_soc;
                                h_trial = h_soc;
                                c_eq_trial = c_eq_soc_trial;
                                c_ineq_trial = c_ineq_soc_trial;
                                accepted = true;
                                break;
                            }

                            alpha_soc *= rho;
                        }
                    }
                }
            }
        }

        // Feasibility restoration: recover from filter-blocked iterates.
        //
        // Reference: Wachter & Biegler 2006 Section 3.
        if(!accepted)
        {
            if constexpr(constrained<P>)
            {
                if(s.n_eq + s.n_ineq > 0)
                {
                    auto restoration_result = run_restoration_(s);
                    if(restoration_result.success)
                    {
                        // Capture x before the restoration assignment so
                        // step_size below reports the actual primal step
                        // norm rather than zero.
                        const double rest_step_norm =
                            (restoration_result.x - s.x).norm();
                        s.x = restoration_result.x;
                        s.objective_value = s.problem->value(s.x);
                        s.problem->gradient(s.x, s.g);

                        s.problem->constraints(s.x, s.c_all);
                        s.c_eq = s.c_all.head(s.n_eq);
                        s.c_ineq = s.c_all.tail(s.n_ineq);
                        s.problem->constraint_jacobian(s.x, s.J_all);
                        s.J_eq = s.J_all.topRows(s.n_eq);
                        s.J_ineq = s.J_all.bottomRows(s.n_ineq);

                        const double h_rest_l1 = detail::constraint_violation(s.c_eq, s.c_ineq);
                        s.filter.add(s.objective_value, h_rest_l1);

                        ++s.iteration;
                        return step_result<double>{
                            .objective_value = s.objective_value,
                            .gradient_norm = s.g.norm(),
                            .step_size = rest_step_norm,
                            .objective_change = s.objective_value - f_k,
                            .improved = s.objective_value < f_k,
                            .constraint_violation = detail::primal_feasibility_inf(s.c_eq, s.c_ineq),
                            .x_norm = s.x.norm(),
                        };
                    }

                    ++s.iteration;
                    return step_result<double>{
                        .objective_value = s.objective_value,
                        .gradient_norm = s.g.norm(),
                        .step_size = 0.0,
                        .objective_change = 0.0,
                        .improved = false,
                        .constraint_violation = detail::primal_feasibility_inf(s.c_eq, s.c_ineq),
                        .x_norm = s.x.norm(),
                        .policy_status = solver_status::stalled,
                    };
                }
            }
        }

        // Accept step.
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

        // Multiplier estimation.
        // Reference: N&W eq. 18.15.
        if constexpr(constrained<P>)
        {
            if(s.n_eq + s.n_ineq > 0)
            {
                Eigen::MatrixXd A_full(s.n_eq + s.n_ineq, n);
                if(s.n_eq > 0)
                    A_full.topRows(s.n_eq) = s.J_eq;
                if(s.n_ineq > 0)
                    A_full.bottomRows(s.n_ineq) = s.J_ineq;
                Eigen::VectorXd g_dyn = s.g;
                s.lambda = detail::estimate_multipliers(g_dyn, A_full);
            }
        }

        // BFGS update using Lagrangian gradient difference.
        // Reference: Kraft 1988 Section 2.2.3; N&W eq. 18.13.
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

                s.problem->constraint_jacobian(x_old, s.J_all_old);

                // Single combined J^T lambda subtraction for the BFGS
                // curvature pair (eq + ineq fused). Matches NLopt
                // slsqp.c slsqpb_'s u-update loop. Falls back to the
                // eq-only path when the QP returned partial multipliers.
                if(lam_take == m_total && m_total > 0)
                {
                    grad_L_old.noalias() -= s.J_all_old.topRows(m_total).transpose()
                                           * s.lam_buf.head(m_total);
                    grad_L_new.noalias() -= s.J_all.topRows(m_total).transpose()
                                           * s.lam_buf.head(m_total);
                }
                else if(s.n_eq > 0 && lam_take >= s.n_eq)
                {
                    s.lam_eq_buf.head(s.n_eq) = s.lam_buf.head(s.n_eq);
                    grad_L_old.noalias() -= s.J_all_old.topRows(s.n_eq).transpose()
                                           * s.lam_eq_buf.head(s.n_eq);
                    grad_L_new.noalias() -= s.J_eq.transpose()
                                           * s.lam_eq_buf.head(s.n_eq);
                }
            }
        }

        Eigen::Vector<double, N> sk = s.x - x_old;
        Eigen::Vector<double, N> yk = grad_L_new - grad_L_old;

        const double sTy = sk.dot(yk);
        if(sk.norm() > 1e-15 && sTy > 0.0)
            s.hessian.push(sk, yk);

        ++s.iteration;

        // Primal feasibility (L-infinity) reported into
        // step_result.constraint_violation for dimensional consistency
        // with step_result.kkt_residual per N&W 2e Definition 12.1.
        // Filter set internals (filter.add / is_acceptable elsewhere in
        // this policy) retain L1 per Fletcher-Leyffer 2002 Section 2.
        double h_new = 0.0;
        if constexpr(constrained<P>)
            h_new = detail::primal_feasibility_inf(s.c_eq, s.c_ineq);

        // KKT residual: full first-order optimality error E(x, lambda,
        // mu) using least-squares multiplier estimates (N&W eq. 18.15).
        // filter_slsqp uses L-BFGS with no explicit multiplier update
        // from the QP, so s.lambda is the only multiplier source.
        // Equality multipliers occupy the first n_eq entries of
        // s.lambda; inequality multipliers follow.
        //
        // Reference: N&W 2e Definition 12.1 (KKT conditions:
        //            stationarity, primal feasibility, dual feasibility,
        //            complementarity); eq. 12.34 (Lagrangian
        //            stationarity leg); eq. 18.15 (multiplier LS).
        // Active-set multiplier re-estimation at x_{k+1} for the kkt leg.
        // The store at :559 (s.lambda = estimate_multipliers(...)) is the
        // raw LS fit used for BFGS curvature-pair construction above; the
        // kkt path here uses a separate active-set LS that detects binding
        // inequalities (|c_ineq[i]| < 1e-8), restricts the LS to the
        // equality + active rows, and clips mu_ineq to >= 0. Plain LS +
        // cwiseMax breaks on optima with parallel inequality gradients
        // (HS024: row 2 = -row 3 of J_ineq at x*) because the min-norm
        // split between parallel rows is not KKT-valid after sign
        // projection.
        //
        // Reference: N&W 2e Section 18.3 + Algorithm 18.3 (working-set
        //            identification);
        //            eq. 18.15 (least-squares lambda);
        //            Definition 12.1 (KKT dual feasibility).
        Eigen::VectorXd lambda_eq_kkt = Eigen::VectorXd::Zero(s.n_eq);
        Eigen::VectorXd mu_ineq_kkt = Eigen::VectorXd::Zero(s.n_ineq);
        if constexpr(constrained<P>)
        {
            if(s.n_eq + s.n_ineq > 0)
            {
                Eigen::Vector<double, N> g_fixed = s.g;
                Eigen::VectorXd lambda_reest =
                    detail::estimate_multipliers_active_set<double,
                                                            N,
                                                            Eigen::Dynamic,
                                                            Eigen::Dynamic>(
                        g_fixed, s.J_eq, s.J_ineq, s.c_ineq);
                if(s.n_eq > 0)
                    lambda_eq_kkt = lambda_reest.head(s.n_eq);
                if(s.n_ineq > 0)
                    mu_ineq_kkt = lambda_reest.segment(s.n_eq, s.n_ineq);
            }
        }
        double kkt = detail::kkt_residual<double,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic>(
            s.g, s.J_eq, s.J_ineq,
            lambda_eq_kkt, mu_ineq_kkt,
            s.c_eq, s.c_ineq);

        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = s.g.norm(),
            .step_size = sk.norm(),
            .objective_change = s.objective_value - old_f,
            .improved = s.objective_value < old_f,
            .constraint_violation = h_new,
            .x_norm = s.x.norm(),
            .kkt_residual = kkt,
        };
    }

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
        s.filter.clear();
        s.iteration = 0;
    }

    template <typename P>
    void reset_clear(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        reset(s, x0);
        s.hessian.reset();
        double h_0 = 0.0;
        if constexpr(constrained<P>)
            h_0 = detail::constraint_violation(s.c_eq, s.c_ineq);
        s.filter.initialize(1e4 * std::max(1.0, h_0));
    }

private:
    // Adapter wrapping a argmin problem type for the restoration
    // functions, which expect eval_constraints(x, c_eq, c_ineq) and
    // eval_constraint_jacobians(x, J_eq, J_ineq) methods.
    template <typename P>
    struct restoration_adapter
    {
        const P& problem;
        int n_eq;
        int n_ineq;

        void eval_constraints(const Eigen::Vector<double, N>& x,
                              Eigen::VectorXd& c_eq,
                              Eigen::VectorXd& c_ineq) const
        {
            Eigen::VectorXd c_all(n_eq + n_ineq);
            problem.constraints(x, c_all);
            c_eq = c_all.head(n_eq);
            c_ineq = c_all.tail(n_ineq);
        }

        void eval_constraint_jacobians(const Eigen::Vector<double, N>& x,
                                       Eigen::MatrixXd& J_eq,
                                       Eigen::MatrixXd& J_ineq) const
        {
            Eigen::MatrixXd J_all(n_eq + n_ineq, x.size());
            problem.constraint_jacobian(x, J_all);
            J_eq = J_all.topRows(n_eq);
            J_ineq = J_all.bottomRows(n_ineq);
        }
    };

    // Run the configured restoration strategy.
    //
    // Reference: Wachter & Biegler 2006 Section 3.
    template <typename P>
    detail::restoration_result<double, N> run_restoration_(state_type<P>& s)
    {
        restoration_adapter<P> adapter{*s.problem, s.n_eq, s.n_ineq};

        // sigma for L1 restoration: max(|lambda|) + 1e-4
        // Reference: N&W eq. 18.36 (same formula as kraft_slsqp update_penalty).
        double sigma_restore = 1e4;
        if(s.lambda.size() > 0)
            sigma_restore = s.lambda.cwiseAbs().maxCoeff() + 1e-4;

        auto strategy = options.restoration;
        auto max_steps = options.max_restoration_steps;

        if(strategy == detail::restoration_strategy::l1_penalty ||
           strategy == detail::restoration_strategy::hybrid)
        {
            auto result = detail::restore_l1<double, N>(
                adapter, s.x, s.g, s.lower, s.upper,
                sigma_restore, max_steps);
            if(result.success)
                return result;
        }

        if(strategy == detail::restoration_strategy::feasibility_qp ||
           strategy == detail::restoration_strategy::hybrid)
        {
            return detail::restore_feasibility_qp<double, N>(
                adapter, s.qp_solver, s.x, s.lower, s.upper, max_steps);
        }

        return {s.x, detail::constraint_violation(s.c_eq, s.c_ineq), false, 0};
    }
};

}

#endif
