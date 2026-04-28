#ifndef HPP_GUARD_NABLAPP_SOLVER_FILTER_NW_SQP_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_FILTER_NW_SQP_POLICY_H

// Dense BFGS SQP with Fletcher-Leyffer 2002 filter acceptance.
//
// Mirrors nw_sqp_policy in QP solver (active_set_qp_solver) and Hessian
// (adaptive_bfgs, Shanno-rescaled compact L-BFGS per N&W eq. 6.20 /
// Section 7.2; Kraft 1988 DFVLR-FB 88-28 Section 2.2.3) but replaces
// user-tuned L1 merit globalization with a filter-based acceptance
// test. The filter maintains a set of non-dominated (f, h) pairs.
// Trial points must be filter-acceptable AND satisfy an Armijo
// condition on an automatically-derived L1 merit. The penalty sigma
// is computed from QP multipliers (no user tuning).
//
// The switching condition distinguishes f-type iterations (near-feasible,
// Armijo on objective suffices) from h-type iterations (infeasible, aim
// to reduce constraint violation via filter acceptance).
//
// Reference: Fletcher & Leyffer 2002, "Nonlinear programming without a
//            penalty function", Math. Program. 91:239-269;
//            Wachter & Biegler 2006, "On the implementation of an
//            interior-point filter line-search algorithm", Section 2.3;
//            N&W Section 15.5 (filter methods);
//            N&W Chapter 18 (dense BFGS SQP).

#include "nablapp/detail/lagrangian.h"
#include "nablapp/detail/merit_function.h"
#include "nablapp/detail/adaptive_bfgs.h"
#include "nablapp/detail/active_set_qp.h"
#include "nablapp/detail/kkt_residual.h"
#include "nablapp/detail/filter_acceptance.h"
#include "nablapp/detail/filter_restoration.h"
#include "nablapp/detail/bound_projection.h"
#include "nablapp/options/qp_options.h"
#include "nablapp/line_search/options.h"
#include "nablapp/result/step_result.h"
#include "nablapp/solver/options.h"
#include "nablapp/types.h"

#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace nablapp
{

template <int N = dynamic_dimension>
struct filter_nw_sqp_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = filter_nw_sqp_policy<M>;

    struct options_type
    {
        line_search_options line_search{};
        detail::restoration_strategy restoration{detail::restoration_strategy::hybrid};
        std::uint16_t max_restoration_steps{10};
        double soc_violation_threshold{1e-8};
        std::uint16_t stall_window{50};
    };

    options_type options{};

    template <typename P = void>
    struct state_type
    {
        const P* problem{nullptr};
        Eigen::Vector<double, N> x;
        Eigen::Vector<double, N> g;
        Eigen::VectorXd c_eq;
        Eigen::VectorXd c_ineq;
        Eigen::MatrixXd J_eq;
        Eigen::MatrixXd J_ineq;
        Eigen::Vector<double, N> lower;
        Eigen::Vector<double, N> upper;
        Eigen::VectorXd lambda;
        double objective_value{};
        // Shanno-rescaled compact L-BFGS operator (N&W eq. 6.20 /
        // Section 7.2; Kraft 1988 DFVLR-FB 88-28 Section 2.2.3).
        // Adaptive theta rebuilds B from the stored history with a
        // fresh theta = y^T y / s^T y on every push, tracking local
        // Lagrangian curvature rather than freezing at iteration 0.
        detail::adaptive_bfgs<double, N, 10> hessian;
        detail::filter_set<double> filter;

        // Automatically-derived penalty for L1 merit acceptance.
        // Unlike nw_sqp_policy, sigma is NOT exposed to users. It is
        // computed from QP multipliers each step (N&W eq. 18.36).
        double sigma{1.0};

        Eigen::VectorXd c_all;
        Eigen::MatrixXd J_all;
        Eigen::MatrixXd J_all_old;  // Saved Jacobian at x_k for BFGS update (N&W eq. 18.13)

        std::uint32_t iteration{0};
        int n_eq{0};
        int n_ineq{0};
        int n{0};
    };

    template <typename Problem, typename Convergence = default_convergence>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<double, N>& x0,
                             const solver_options<Convergence>& /*opts*/)
    {
        static_assert(differentiable<Problem>,
                      "filter_nw_sqp_policy requires differentiable<Problem>");
        static_assert(constrained<Problem>,
                      "filter_nw_sqp_policy requires constrained<Problem>");

        state_type<Problem> s;
        s.problem = &problem;
        s.n = problem.dimension();
        s.n_eq = problem.num_equality();
        s.n_ineq = problem.num_inequality();
        const int m = s.n_eq + s.n_ineq;

        s.x = x0;
        s.g.setZero(s.n);
        s.objective_value = problem.value(x0);
        problem.gradient(x0, s.g);

        s.c_all.resize(m);
        s.J_all.resize(m, s.n);
        s.J_all_old.setZero(m, s.n);
        if(m > 0)
            problem.constraints(x0, s.c_all);
        s.c_eq = s.c_all.head(s.n_eq);
        s.c_ineq = s.c_all.tail(s.n_ineq);

        if(m > 0)
            problem.constraint_jacobian(x0, s.J_all);
        s.J_eq = s.J_all.topRows(s.n_eq);
        s.J_ineq = s.J_all.bottomRows(s.n_ineq);

        if constexpr(bound_constrained<Problem>)
        {
            s.lower = problem.lower_bounds();
            s.upper = problem.upper_bounds();
        }
        else
        {
            constexpr double inf = std::numeric_limits<double>::infinity();
            s.lower = Eigen::Vector<double, N>::Constant(s.n, -inf);
            s.upper = Eigen::Vector<double, N>::Constant(s.n, inf);
        }

        s.hessian = detail::adaptive_bfgs<double, N, 10>(s.n);
        s.sigma = 1.0;
        s.iteration = 0;

        // Initialize filter with h_max based on initial constraint violation.
        // Reference: Wachter & Biegler 2006, eq. (8).
        double h_0 = detail::constraint_violation(s.c_eq, s.c_ineq);
        s.filter.initialize(1e4 * std::max(1.0, h_0));

        // Initial multiplier estimate via least-squares (N&W eq. 18.15).
        if(m > 0)
        {
            Eigen::MatrixXd A_all(m, s.n);
            if(s.n_eq > 0) A_all.topRows(s.n_eq) = s.J_eq;
            if(s.n_ineq > 0) A_all.bottomRows(s.n_ineq) = s.J_ineq;
            s.lambda = detail::estimate_multipliers(
                Eigen::VectorXd(s.g), A_all);
        }
        else
        {
            s.lambda = Eigen::VectorXd::Zero(0);
        }

        return s;
    }

    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        const int n = s.n;
        const int m = s.n_eq + s.n_ineq;

        // Re-evaluate at current x (skip first iteration; init did it).
        if(s.iteration != 0)
        {
            s.objective_value = s.problem->value(s.x);
            s.problem->gradient(s.x, s.g);
            if(m > 0)
            {
                s.problem->constraints(s.x, s.c_all);
                s.c_eq = s.c_all.head(s.n_eq);
                s.c_ineq = s.c_all.tail(s.n_ineq);
                s.problem->constraint_jacobian(s.x, s.J_all);
                s.J_eq = s.J_all.topRows(s.n_eq);
                s.J_ineq = s.J_all.bottomRows(s.n_ineq);
            }
        }

        // --- 1. Build and solve QP subproblem (N&W eq. 18.12) ---
        Eigen::MatrixXd A_eq = s.J_eq;
        Eigen::VectorXd b_eq = -s.c_eq;
        Eigen::MatrixXd A_ineq = s.J_ineq;
        Eigen::VectorXd b_ineq = -s.c_ineq;

        Eigen::VectorXd p0 = Eigen::VectorXd::Zero(n);
        if(s.n_eq > 0 && s.c_eq.norm() > 1e-15)
        {
            Eigen::MatrixXd AAt = A_eq * A_eq.transpose();
            auto qr = AAt.colPivHouseholderQr();
            Eigen::VectorXd w = qr.solve(b_eq);
            p0 = A_eq.transpose() * w;
        }

        qp_options qp_opts;
        qp_opts.max_iterations = std::uint16_t{200};
        qp_opts.tolerance = 1e-12;

        detail::qp_result<double> qp;
        bool has_bounds = has_finite_box(s.lower, s.upper);

        if(has_bounds)
        {
            Eigen::VectorXd p_lower = (Eigen::VectorXd(s.lower)
                - Eigen::VectorXd(s.x)).eval();
            Eigen::VectorXd p_upper = (Eigen::VectorXd(s.upper)
                - Eigen::VectorXd(s.x)).eval();
            p0 = p0.cwiseMax(p_lower).cwiseMin(p_upper);
            qp = detail::solve_qp(Eigen::MatrixXd(s.hessian.hessian()), Eigen::VectorXd(s.g),
                                  A_eq, b_eq, A_ineq, b_ineq,
                                  p_lower, p_upper, p0, qp_opts);
        }
        else
        {
            qp = detail::solve_qp(Eigen::MatrixXd(s.hessian.hessian()), Eigen::VectorXd(s.g),
                                  A_eq, b_eq, A_ineq, b_ineq,
                                  p0, qp_opts);
        }

        Eigen::VectorXd p = Eigen::VectorXd(qp.x);

        if(p.norm() < 1e-15)
        {
            // Null step: QP returned a zero direction (degeneracy or
            // active-set cycling). N&W 2e S18.4.
            //
            // is_null_step=true exempts this iterate from
            // step_tolerance stall detection so basic_solver gives
            // the policy another iteration to break the degeneracy
            // rather than flagging solver_status::stalled on iter 2.
            //
            // Dual convention: step_result.constraint_violation reports
            // L-infinity primal feasibility (dimensionally consistent with
            // kkt_residual per N&W 2e Definition 12.1); filter_set entries
            // elsewhere in this policy retain L1 h_k per Fletcher-Leyffer
            // 2002 Section 2 filter dominance ordering.
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
            if(m > 0)
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
            double kkt_null = detail::kkt_residual<double,
                                                   Eigen::Dynamic,
                                                   Eigen::Dynamic,
                                                   Eigen::Dynamic>(
                s.g, s.J_eq, s.J_ineq,
                lambda_eq_null, mu_ineq_null,
                s.c_eq, s.c_ineq);
            return step_result<double>{
                .objective_value = s.objective_value,
                .gradient_norm = lagrangian_gradient_norm(s),
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
                .is_null_step = true,
                .constraint_violation = detail::primal_feasibility_inf(s.c_eq, s.c_ineq),
                .kkt_residual = kkt_null,
            };
        }

        // --- 2. Extract QP multipliers and update penalty ---
        Eigen::VectorXd lambda_new = Eigen::VectorXd::Zero(m);
        if(m > 0 && qp.lambda.size() >= m)
            lambda_new = qp.lambda.head(m);
        else if(m > 0 && qp.lambda.size() > 0)
            lambda_new.head(qp.lambda.size()) = qp.lambda;

        // Auto-derive penalty sigma from QP multipliers (N&W eq. 18.36).
        // This replaces the user-tunable penalty of nw_sqp_policy.
        // Additionally increase sigma if the directional derivative of the
        // L1 merit is non-negative, which indicates the penalty is too low
        // for the current step to be a descent direction on the merit.
        s.sigma = detail::update_penalty(s.sigma, lambda_new);
        double h_cur = detail::constraint_violation(s.c_eq, s.c_ineq);
        double dphi_check = Eigen::VectorXd(s.g).dot(p) - s.sigma * h_cur;
        if(dphi_check >= 0.0 && h_cur > 1e-12)
            s.sigma = std::max(s.sigma,
                std::abs(Eigen::VectorXd(s.g).dot(p)) / h_cur + 1e-4);

        // --- 3. Filter + merit acceptance line search ---
        //
        // The filter prevents cycling (Fletcher & Leyffer 2002).
        // The L1 merit Armijo ensures constraint satisfaction via the
        // automatically derived penalty sigma.
        //
        // Reference: Wachter & Biegler 2006, Section 2.3;
        //            N&W Section 18.5 (L1 merit Armijo).
        double f_k = s.objective_value;
        double h_k = detail::constraint_violation(s.c_eq, s.c_ineq);
        double grad_f_dot_p = Eigen::VectorXd(s.g).dot(p);

        // Check switching condition to determine iteration type.
        bool f_type = detail::is_f_type_iteration(
            h_k, grad_f_dot_p, s.filter.h_max());

        // Add current point to filter for h-type iterations.
        if(!f_type)
            s.filter.add(f_k, h_k);

        // L1 merit at current point.
        double phi0 = detail::l1_merit(f_k, s.c_eq, s.c_ineq, s.sigma);
        double dphi0 = detail::l1_merit_directional_derivative(
            grad_f_dot_p, s.c_eq, s.c_ineq, s.sigma);

        double alpha = 1.0;
        constexpr double c1 = 1e-4;
        constexpr double rho_shrink = 0.5;
        const int max_ls = options.line_search.max_iterations;

        Eigen::Vector<double, N> x_trial(n);
        Eigen::VectorXd c_eq_trial, c_ineq_trial;
        double f_trial = f_k;
        double h_trial = h_k;
        bool accepted = false;

        Eigen::VectorXd c_trial_all(m);

        for(int ls = 0; ls < max_ls; ++ls)
        {
            x_trial = s.x + alpha * Eigen::Vector<double, N>(p);
            if(has_bounds)
                x_trial = detail::project(x_trial, s.lower, s.upper);

            f_trial = s.problem->value(x_trial);
            if(m > 0)
            {
                s.problem->constraints(x_trial, c_trial_all);
                c_eq_trial = c_trial_all.head(s.n_eq);
                c_ineq_trial = c_trial_all.tail(s.n_ineq);
            }
            else
            {
                c_eq_trial = Eigen::VectorXd{};
                c_ineq_trial = Eigen::VectorXd{};
            }
            h_trial = detail::constraint_violation(c_eq_trial, c_ineq_trial);

            // Filter acceptance check.
            bool filter_ok = s.filter.is_acceptable(f_trial, h_trial);

            // L1 merit Armijo check.
            double phi_trial = detail::l1_merit(
                f_trial, c_eq_trial, c_ineq_trial, s.sigma);
            bool merit_ok = phi_trial <= phi0 + c1 * alpha * dphi0;

            if(f_type)
            {
                // F-type: require both filter acceptance and merit Armijo.
                if(filter_ok && merit_ok)
                {
                    accepted = true;
                    break;
                }
            }
            else
            {
                // H-type: filter acceptance OR merit Armijo suffices.
                if(filter_ok || merit_ok)
                {
                    accepted = true;
                    break;
                }
            }

            alpha *= rho_shrink;
        }

        // Second-order correction (SOC) if rejected and infeasible.
        if(!accepted && h_k > options.soc_violation_threshold)
        {
            Eigen::VectorXd b_eq_soc = -c_eq_trial;
            Eigen::VectorXd b_ineq_soc = -c_ineq_trial;
            Eigen::VectorXd p_soc_0 = Eigen::VectorXd::Zero(n);

            detail::qp_result<double> qp_soc;
            if(has_bounds)
            {
                Eigen::VectorXd p_lower_soc = (Eigen::VectorXd(s.lower)
                    - Eigen::VectorXd(x_trial)).eval();
                Eigen::VectorXd p_upper_soc = (Eigen::VectorXd(s.upper)
                    - Eigen::VectorXd(x_trial)).eval();
                qp_soc = detail::solve_qp(Eigen::MatrixXd(s.hessian.hessian()), Eigen::VectorXd(s.g),
                                           A_eq, b_eq_soc, A_ineq, b_ineq_soc,
                                           p_lower_soc, p_upper_soc, p_soc_0,
                                           qp_opts);
            }
            else
            {
                qp_soc = detail::solve_qp(Eigen::MatrixXd(s.hessian.hessian()), Eigen::VectorXd(s.g),
                                           A_eq, b_eq_soc, A_ineq, b_ineq_soc,
                                           p_soc_0, qp_opts);
            }

            Eigen::Vector<double, N> x_soc = x_trial
                + Eigen::Vector<double, N>(Eigen::VectorXd(qp_soc.x));
            if(has_bounds)
                x_soc = detail::project(x_soc, s.lower, s.upper);

            double f_soc = s.problem->value(x_soc);
            Eigen::VectorXd c_soc_all(m);
            Eigen::VectorXd c_eq_soc, c_ineq_soc;
            if(m > 0)
            {
                s.problem->constraints(x_soc, c_soc_all);
                c_eq_soc = c_soc_all.head(s.n_eq);
                c_ineq_soc = c_soc_all.tail(s.n_ineq);
            }
            double h_soc = detail::constraint_violation(c_eq_soc, c_ineq_soc);

            double phi_soc = detail::l1_merit(
                f_soc, c_eq_soc, c_ineq_soc, s.sigma);
            if(s.filter.is_acceptable(f_soc, h_soc)
               && phi_soc <= phi0 + c1 * dphi0)
            {
                x_trial = x_soc;
                f_trial = f_soc;
                h_trial = h_soc;
                c_eq_trial = c_eq_soc;
                c_ineq_trial = c_ineq_soc;
                accepted = true;
            }
        }

        // If still rejected, attempt feasibility restoration.
        // Reference: Wachter & Biegler 2006, Section 3.
        if(!accepted)
        {
            auto rest = run_restoration_(s);
            if(rest.success)
            {
                // Capture x before the restoration assignment so
                // step_size below reports the actual primal step
                // norm rather than zero.
                const double rest_step_norm = (rest.x - s.x).norm();
                s.x = rest.x;
                s.objective_value = s.problem->value(s.x);
                s.problem->gradient(s.x, s.g);
                if(m > 0)
                {
                    s.problem->constraints(s.x, s.c_all);
                    s.c_eq = s.c_all.head(s.n_eq);
                    s.c_ineq = s.c_all.tail(s.n_ineq);
                    s.problem->constraint_jacobian(s.x, s.J_all);
                    s.J_eq = s.J_all.topRows(s.n_eq);
                    s.J_ineq = s.J_all.bottomRows(s.n_ineq);
                }
                const double h_restored_l1 = detail::constraint_violation(s.c_eq, s.c_ineq);
                s.filter.add(s.objective_value, h_restored_l1);
                ++s.iteration;
                return step_result<double>{
                    .objective_value = s.objective_value,
                    .gradient_norm = lagrangian_gradient_norm(s),
                    .step_size = rest_step_norm,
                    .objective_change = s.objective_value - f_k,
                    .improved = h_restored_l1 < h_k,
                    .constraint_violation = detail::primal_feasibility_inf(s.c_eq, s.c_ineq),
                };
            }

            // Null step: feasibility restoration exhausted without
            // reducing constraint violation.
            //
            // is_null_step=true prevents step_tolerance_criterion
            // from labelling the zero step as a stall; the iterate
            // is carried forward without movement while the outer
            // loop retains flexibility (restoration may succeed on
            // a future iteration once the filter envelope relaxes).
            //
            // step_result.constraint_violation reports L-infinity primal
            // feasibility (kkt_residual-consistent). The local h_k (L1)
            // is consumed by filter dominance checks per Fletcher-Leyffer
            // 2002 Section 2.
            ++s.iteration;
            return step_result<double>{
                .objective_value = s.objective_value,
                .gradient_norm = lagrangian_gradient_norm(s),
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
                .is_null_step = true,
                .constraint_violation = detail::primal_feasibility_inf(s.c_eq, s.c_ineq),
            };
        }

        // --- 4. Accept step and compute BFGS update ---
        Eigen::Vector<double, N> x_old = s.x;
        Eigen::VectorXd g_old = Eigen::VectorXd(s.g);
        double f_old = s.objective_value;

        // Save Jacobian at x_k before re-evaluation for BFGS y-vector (N&W eq. 18.13).
        if(m > 0)
            s.J_all_old.noalias() = s.J_all;

        s.x = x_trial;
        s.objective_value = f_trial;
        s.c_eq = c_eq_trial;
        s.c_ineq = c_ineq_trial;
        s.problem->gradient(s.x, s.g);
        if(m > 0)
        {
            s.problem->constraint_jacobian(s.x, s.J_all);
            s.J_eq = s.J_all.topRows(s.n_eq);
            s.J_ineq = s.J_all.bottomRows(s.n_ineq);
        }

        Eigen::VectorXd sk = (Eigen::VectorXd(s.x)
            - Eigen::VectorXd(x_old)).eval();

        // Lagrangian gradient at new and old points for BFGS y-vector.
        // N&W eq. 18.13: grad_L_new uses Jacobian at x_{k+1}, grad_L_old uses Jacobian at x_k.
        Eigen::VectorXd grad_L_new, grad_L_old;
        if(m > 0)
        {
            grad_L_new = detail::lagrangian_gradient(
                Eigen::VectorXd(s.g), s.J_all, lambda_new);
            grad_L_old = detail::lagrangian_gradient(
                g_old, s.J_all_old, lambda_new);
        }
        else
        {
            grad_L_new = Eigen::VectorXd(s.g);
            grad_L_old = g_old;
        }

        Eigen::VectorXd yk = grad_L_new - grad_L_old;

        // --- 5. BFGS curvature push (adaptive_bfgs with Shanno rescaling) ---
        // Skip BFGS updates on non-positive curvature pairs.
        // In filter-SQP the Lagrangian gradient difference y_k can
        // easily have s^T y < 0 when the constraint Hessian contribution
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
            Eigen::Vector<double, N> sk_fixed = sk;
            Eigen::Vector<double, N> yk_fixed = yk;
            s.hessian.push(sk_fixed, yk_fixed);
        }

        // --- 6. Update multipliers and iteration ---
        s.lambda = lambda_new;
        ++s.iteration;

        double phi_new = detail::l1_merit(
            s.objective_value, s.c_eq, s.c_ineq, s.sigma);

        // Active-set multiplier re-estimation at x_{k+1}. The QP-derived
        // lambda_new satisfies the linearised KKT at x_k, not at x_{k+1}
        // where grad_f and the constraint Jacobian have moved; reusing
        // lambda_new at the new iterate produces a stationarity leg that
        // oscillates with the remaining problem curvature. Active-set LS
        // detects binding inequalities (|c_ineq[i]| < 1e-8) and solves
        // the reduced LS problem on just the equality + active rows,
        // with mu_ineq clamped to >= 0 for dual feasibility. Plain LS +
        // cwiseMax fails on optima with parallel inequality gradients
        // (HS024: row 2 = -row 3 of J_ineq at x*) because the min-norm
        // split between the parallel rows is not KKT-valid after sign
        // projection. lambda_reest is local to this kkt evaluation;
        // BFGS curvature-pair construction above at :521-524 uses
        // lambda_new (QP multipliers) so inter-iteration BFGS semantics
        // are unchanged.
        //
        // Reference: N&W 2e Section 18.3 + Algorithm 18.3 (working-set
        //            identification);
        //            eq. 18.15 (least-squares lambda);
        //            Definition 12.1 (KKT dual feasibility);
        //            eq. 12.34 (Lagrangian stationarity leg).
        Eigen::VectorXd lambda_eq_kkt = Eigen::VectorXd::Zero(s.n_eq);
        Eigen::VectorXd mu_ineq_kkt = Eigen::VectorXd::Zero(s.n_ineq);
        if(m > 0)
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
        double kkt = detail::kkt_residual<double,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic,
                                          Eigen::Dynamic>(
            s.g, s.J_eq, s.J_ineq,
            lambda_eq_kkt, mu_ineq_kkt,
            s.c_eq, s.c_ineq);

        // Primal feasibility (L-infinity) reported into step_result for
        // dimensional consistency with kkt_residual per N&W 2e Definition
        // 12.1. The loop-local h_trial (L1) above was consumed by
        // filter.is_acceptable at the line-search step per Fletcher-Leyffer
        // 2002 Section 2 dominance ordering.
        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = grad_L_new.norm(),
            .step_size = sk.norm(),
            .objective_change = s.objective_value - f_old,
            .improved = phi_new < phi0,
            .constraint_violation = detail::primal_feasibility_inf(s.c_eq, s.c_ineq),
            .kkt_residual = kkt,
        };
    }

    // Hot start: preserves BFGS Hessian, but clears the filter envelope.
    //
    // The filter is a per-run dominance set: entries added during one
    // solve represent (objective, violation) pairs that were rejected
    // along that trajectory. Carrying them across to a new run on the
    // same state object causes the canonical Wachter-Biegler oscillation
    // -- a converged near-optimum entry from the prior run dominates
    // virtually every trial point of the new run, the filter rejects
    // every line-search step, and the outer loop stalls at alpha -> 0.
    //
    // BFGS Hessian curvature, on the other hand, IS transferable across
    // runs that approach the same local model, so reset() preserves it.
    //
    // Reference: Wachter & Biegler 2006, Section 3.3 (filter
    //            re-initialization between independent runs);
    //            N&W 2e Section 15.4 (filter SQP semantics).
    template <typename Problem>
    void reset(state_type<Problem>& s,
               const Eigen::Vector<double, N>& x0)
    {
        s.x = x0;
        s.objective_value = s.problem->value(x0);
        s.problem->gradient(x0, s.g);
        const int m = s.n_eq + s.n_ineq;
        if(m > 0)
        {
            s.problem->constraints(x0, s.c_all);
            s.c_eq = s.c_all.head(s.n_eq);
            s.c_ineq = s.c_all.tail(s.n_ineq);
            s.problem->constraint_jacobian(x0, s.J_all);
            s.J_eq = s.J_all.topRows(s.n_eq);
            s.J_ineq = s.J_all.bottomRows(s.n_ineq);
        }
        s.filter.clear();
        s.iteration = 0;
    }

    // Cold restart: clears BFGS Hessian, filter envelope (with new
    // h_max derived from the new x0), and penalty.
    template <typename Problem>
    void reset_clear(state_type<Problem>& s,
                     const Eigen::Vector<double, N>& x0)
    {
        reset(s, x0);
        s.hessian.reset();
        const double h_0 = detail::constraint_violation(s.c_eq, s.c_ineq);
        s.filter.initialize(1e4 * std::max(1.0, h_0));
        s.sigma = 1.0;
        s.lambda.setZero();
    }

private:
    static bool has_finite_box(const Eigen::Vector<double, N>& lower,
                               const Eigen::Vector<double, N>& upper)
    {
        constexpr double inf = std::numeric_limits<double>::infinity();
        for(int i = 0; i < lower.size(); ++i)
        {
            if(lower[i] > -inf || upper[i] < inf)
                return true;
        }
        return false;
    }

    template <typename Problem>
    static double lagrangian_gradient_norm(const state_type<Problem>& s)
    {
        const int m = s.n_eq + s.n_ineq;
        if(m == 0)
            return Eigen::VectorXd(s.g).norm();

        Eigen::MatrixXd A(m, s.n);
        if(s.n_eq > 0) A.topRows(s.n_eq) = s.J_eq;
        if(s.n_ineq > 0) A.bottomRows(s.n_ineq) = s.J_ineq;
        return detail::lagrangian_gradient(
            Eigen::VectorXd(s.g), A, s.lambda).norm();
    }

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

    // Reference: Wachter & Biegler 2006 Section 3.
    template <typename P>
    detail::restoration_result<double, N> run_restoration_(state_type<P>& s)
    {
        restoration_adapter<P> adapter{*s.problem, s.n_eq, s.n_ineq};

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

        // feasibility QP via solve_qp free function
        if(strategy == detail::restoration_strategy::feasibility_qp ||
           strategy == detail::restoration_strategy::hybrid)
        {
            Eigen::Vector<double, N> x = s.x;
            for(std::uint16_t step = 0; step < max_steps; ++step)
            {
                Eigen::VectorXd c_eq, c_ineq;
                adapter.eval_constraints(x, c_eq, c_ineq);
                double h_k = detail::constraint_violation(c_eq, c_ineq);
                if(h_k < 1e-12)
                    return {x, h_k, true, step};

                Eigen::MatrixXd J_eq_r, J_ineq_r;
                adapter.eval_constraint_jacobians(x, J_eq_r, J_ineq_r);

                auto n = x.size();
                Eigen::MatrixXd H = Eigen::MatrixXd::Identity(n, n);
                Eigen::VectorXd g_zero = Eigen::VectorXd::Zero(n);
                Eigen::VectorXd x0_qp = Eigen::VectorXd::Zero(n);

                Eigen::VectorXd neg_c_eq = (-c_eq).eval();
                Eigen::VectorXd neg_c_ineq = (-c_ineq).eval();
                auto qp_res = detail::solve_qp(H, g_zero,
                    J_eq_r, neg_c_eq, J_ineq_r, neg_c_ineq, x0_qp);

                Eigen::Vector<double, N> p = qp_res.x;
                if(p.norm() < 1e-15)
                    return {x, h_k, false, step};

                double alpha = 1.0;
                while(alpha > 1e-10)
                {
                    Eigen::Vector<double, N> x_trial =
                        (x + alpha * p).cwiseMax(s.lower).cwiseMin(s.upper);
                    Eigen::VectorXd c_eq_t, c_ineq_t;
                    adapter.eval_constraints(x_trial, c_eq_t, c_ineq_t);
                    if(detail::constraint_violation(c_eq_t, c_ineq_t) < h_k)
                    {
                        x = x_trial;
                        break;
                    }
                    alpha *= 0.5;
                }
                if(alpha <= 1e-10)
                    return {x, h_k, false, step};
            }
            Eigen::VectorXd c_eq_f, c_ineq_f;
            adapter.eval_constraints(x, c_eq_f, c_ineq_f);
            double h_f = detail::constraint_violation(c_eq_f, c_ineq_f);
            return {x, h_f, h_f < 1e-12, max_steps};
        }

        return {s.x, detail::constraint_violation(s.c_eq, s.c_ineq), false, 0};
    }
};

}

#endif
