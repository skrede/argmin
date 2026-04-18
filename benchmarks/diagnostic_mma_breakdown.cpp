// Per-iteration trace harness for MMA and GCMMA on pre-fix HEAD.
//
// Runs:
//   1. mma_policy on HS043         -- confirms unconditional-accept
//                                     failure mode (null-step trap at
//                                     early iter; step_size drops to
//                                     zero under the grafted merit
//                                     backtrack while the KKT residual
//                                     is still large).
//   2. gcmma_policy on HS035       -- confirms conservativity-loop-driven
//                                     classification drift (reaches the
//                                     optimum but terminates as stalled
//                                     rather than ftol_reached, because
//                                     step_result::kkt_residual is not
//                                     populated and the raw gradient-
//                                     norm report remains large at a
//                                     constrained KKT point).
//   3. gcmma_policy on HS043       -- confirms non-convergence under
//                                     asymptote-tightening conservativity
//                                     (the tighten_factor mechanism is
//                                     not the Svanberg 2002 per-component
//                                     raa_i growth; convergence stalls
//                                     well short of the optimum).
//   4. mma_policy on HS076         -- one-time sign-convention probe:
//                                     logs the subproblem dual y
//                                     magnitudes versus the active-set
//                                     least-squares multiplier estimate,
//                                     so the correctness surgery can pin
//                                     down the y -> mu_ineq mapping sign.
//
// Output: CSV-formatted rows to stderr (one row per (problem, policy,
// iter)) plus a summary probe line for HS076.
//
// Throwaway: delete after the mma/gcmma correctness pass lands.
//
// References
//   Svanberg 1987, "The method of moving asymptotes -- a new method for
//     structural optimization", Int. J. Numer. Methods Eng. 24,
//     Section 5 (MMA dual KKT sign convention).
//   Svanberg 2002, "A class of globally convergent optimization methods
//     based on conservative convex separable approximations", SIAM J.
//     Optim. 12(2), Section 4.2 (GCMMA conservativity / raa-growth).
//   Nocedal and Wright, "Numerical Optimization" 2e, Definition 12.1
//     (KKT conditions; inequality-constraint multiplier sign);
//     Algorithm 18.3 (active-set least-squares multiplier estimate).
//
// Sign convention (matches lib/nablapp/include/nablapp/detail/
// lagrangian.h:42-60 and formulation/concepts.h): c_ineq(x) >= 0
// feasible; mu_ineq >= 0 by KKT dual feasibility. The MMA subproblem
// uses the mirror form g_i = -c_ineq_i <= 0 with dual y_i >= 0. The
// identity -J_ineq^T y = J_ineq^T mu_ineq then maps y directly into
// nablapp's mu_ineq without a sign flip (verified in the probe below).

#include "nablapp/detail/kkt_residual.h"
#include "nablapp/detail/mma_subproblem.h"
#include "nablapp/detail/lagrangian.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/solver/gcmma_policy.h"
#include "nablapp/solver/mma_policy.h"
#include "nablapp/solver/options.h"
#include "nablapp/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <string_view>

namespace
{

// Replicates detail::mma_dual_solve but captures the dual variable y at
// convergence so the sign-convention probe has access to the subproblem
// multiplier. The lib-side dual solve is a free function whose y is
// strictly local; Plan 01 may not edit library code, so the harness
// carries its own copy. The structure mirrors detail/mma_subproblem.h
// (the dual_solve free function) verbatim in numerics; only the return
// shape changes.
//
// Reference: Svanberg 1987, dual formulation; Svanberg 2002, Section 5.
template <typename Scalar, int N, int M>
struct probe_dual_result
{
    Eigen::Vector<Scalar, N> x_opt;
    Eigen::Vector<Scalar, M> y;
};

template <typename Scalar, int N, int M>
probe_dual_result<Scalar, N, M> probe_dual_solve_with_multipliers(
    const nablapp::detail::mma_coeffs<Scalar, N, M>& coeffs,
    const Eigen::Vector<Scalar, N>& L,
    const Eigen::Vector<Scalar, N>& U,
    const Eigen::Vector<Scalar, N>& x_min,
    const Eigen::Vector<Scalar, N>& x_max,
    int max_iter = 50,
    Scalar tol = Scalar(1e-9),
    Scalar backtrack = Scalar(0.95))
{
    constexpr Scalar eps = Scalar(1e-10);

    const int n = static_cast<int>(L.size());
    const int m = static_cast<int>(coeffs.ri.size());

    probe_dual_result<Scalar, N, M> out;
    out.x_opt.resize(n);
    out.y.resize(m);

    if(m == 0)
    {
        for(int j = 0; j < n; ++j)
        {
            const Scalar sp = std::sqrt(coeffs.p0[j]);
            const Scalar sq = std::sqrt(coeffs.q0[j]);
            out.x_opt[j] = (sp * L[j] + sq * U[j]) / (sp + sq);
            out.x_opt[j] = std::clamp(out.x_opt[j],
                std::max(L[j] + eps, x_min[j]),
                std::min(U[j] - eps, x_max[j]));
        }
        return out;
    }

    Eigen::Vector<Scalar, M> y = Eigen::Vector<Scalar, M>::Ones(m);
    Eigen::Vector<Scalar, M> grad(m);
    Eigen::Matrix<Scalar, M, M> negH(m, m);

    for(int iter = 0; iter < max_iter; ++iter)
    {
        for(int j = 0; j < n; ++j)
        {
            Scalar pj = coeffs.p0[j];
            Scalar qj = coeffs.q0[j];
            for(int i = 0; i < m; ++i)
            {
                pj += y[i] * coeffs.pi(i, j);
                qj += y[i] * coeffs.qi(i, j);
            }
            pj = std::max(pj, eps);
            qj = std::max(qj, eps);

            const Scalar sp = std::sqrt(pj);
            const Scalar sq = std::sqrt(qj);
            out.x_opt[j] = (sp * L[j] + sq * U[j]) / (sp + sq);
            out.x_opt[j] = std::clamp(out.x_opt[j],
                std::max(L[j] + eps, x_min[j]),
                std::min(U[j] - eps, x_max[j]));
        }

        for(int i = 0; i < m; ++i)
        {
            grad[i] = coeffs.ri[i];
            for(int j = 0; j < n; ++j)
            {
                const Scalar ux = U[j] - out.x_opt[j];
                const Scalar xl = out.x_opt[j] - L[j];
                grad[i] += coeffs.pi(i, j) / ux + coeffs.qi(i, j) / xl;
            }
        }

        Scalar max_proj = Scalar(0);
        for(int i = 0; i < m; ++i)
        {
            if(y[i] > eps)
                max_proj = std::max(max_proj, std::abs(grad[i]));
            else
                max_proj = std::max(max_proj, std::max(grad[i], Scalar(0)));
        }
        if(max_proj < tol)
            break;

        negH.setZero();
        for(int j = 0; j < n; ++j)
        {
            Scalar pj = coeffs.p0[j];
            Scalar qj = coeffs.q0[j];
            for(int i = 0; i < m; ++i)
            {
                pj += y[i] * coeffs.pi(i, j);
                qj += y[i] * coeffs.qi(i, j);
            }
            pj = std::max(pj, eps);
            qj = std::max(qj, eps);

            Scalar aj = U[j] - out.x_opt[j];
            Scalar bj = out.x_opt[j] - L[j];
            aj = std::max(aj, eps);
            bj = std::max(bj, eps);

            const Scalar wj = std::max(
                Scalar(2) * pj / (aj * aj * aj)
                + Scalar(2) * qj / (bj * bj * bj),
                eps);

            for(int i1 = 0; i1 < m; ++i1)
            {
                const Scalar v1 = coeffs.pi(i1, j) / (aj * aj)
                                - coeffs.qi(i1, j) / (bj * bj);
                for(int i2 = i1; i2 < m; ++i2)
                {
                    const Scalar v2 = coeffs.pi(i2, j) / (aj * aj)
                                    - coeffs.qi(i2, j) / (bj * bj);
                    const Scalar val = v1 * v2 / wj;
                    negH(i1, i2) += val;
                    if(i1 != i2)
                        negH(i2, i1) += val;
                }
            }
        }
        negH.diagonal().array() += eps;

        Eigen::LDLT<Eigen::Matrix<Scalar, M, M>> ldlt(std::move(negH));
        Eigen::Vector<Scalar, M> dy = ldlt.solve(grad);

        Scalar alpha = Scalar(1);
        for(int i = 0; i < m; ++i)
        {
            if(dy[i] < Scalar(0) && y[i] > Scalar(0))
            {
                const Scalar max_step = y[i] / (-dy[i]) * backtrack;
                alpha = std::min(alpha, max_step);
            }
        }
        alpha = std::max(alpha, eps);

        y += alpha * dy;
        y = y.cwiseMax(Scalar(0));
    }

    out.y = y;
    return out;
}

// Emits one CSV data row to stderr. Column schema matches the header
// printed in main(); keep in sync.
void print_row(std::string_view problem,
               std::string_view policy,
               std::uint32_t iter,
               double f,
               double grad_norm,
               double cv_linf,
               double kkt,
               double step_size,
               bool is_null_step,
               int policy_status_code)
{
    std::fprintf(stderr,
        "%.*s,%.*s,%u,%.6e,%.6e,%.6e,%.6e,%.6e,%s,%d\n",
        static_cast<int>(problem.size()), problem.data(),
        static_cast<int>(policy.size()), policy.data(),
        iter, f, grad_norm, cv_linf, kkt, step_size,
        is_null_step ? "null" : "step",
        policy_status_code);
}

// Returns the active-set LS multiplier restricted to the inequality
// block for an inequality-only problem (no equality constraints).
//
// Reference: Nocedal and Wright 2e, Algorithm 18.3 (active-set
//            multiplier estimate); lib/nablapp/include/nablapp/
//            detail/lagrangian.h:137-193.
template <int N, int MC>
Eigen::Vector<double, MC> active_set_mu_ineq(
    const Eigen::Vector<double, N>& grad_f,
    const Eigen::Matrix<double, MC, N>& J_ineq,
    const Eigen::Vector<double, MC>& c_ineq,
    double active_tol = 1e-6)
{
    const Eigen::Index n_ineq = J_ineq.rows();

    Eigen::Matrix<double, 0, N> J_eq_empty(0, grad_f.size());

    Eigen::Vector<double, Eigen::Dynamic> lambda_full =
        nablapp::detail::estimate_multipliers_active_set<double, N,
            Eigen::Dynamic, MC>(
                grad_f, J_eq_empty, J_ineq, c_ineq, active_tol);

    // Layout is [lambda_eq ; mu_ineq] with n_eq = 0 here.
    Eigen::Vector<double, MC> mu(n_ineq);
    for(Eigen::Index i = 0; i < n_ineq; ++i)
        mu[i] = lambda_full[i];
    return mu;
}

// Drives basic_solver::step() up to `budget` iterations with disabled
// convergence termination (thresholds set tight so the solver runs the
// full budget and the harness captures every iterate). Emits a CSV row
// per step.
//
// Kept generic over problem and policy so the three failure-mode runs
// share a single code path.
template <typename Problem, typename Policy>
void run_case(std::string_view problem_label,
              std::string_view policy_label,
              Problem problem,
              Policy policy,
              std::uint32_t budget)
{
    constexpr int N = Problem::problem_dimension;
    constexpr int MC = Problem::constraint_count;

    auto x0 = problem.initial_point();

    nablapp::solver_options opts;
    opts.max_iterations = budget;
    opts.set_gradient_threshold(1e-14);
    opts.set_objective_threshold(1e-18);
    opts.set_step_threshold(1e-18);
    opts.set_stationarity_threshold(1e-14);

    nablapp::basic_solver solver{policy, problem, x0, opts};

    // Scratch vectors for the inequality-only kkt_residual call.
    [[maybe_unused]] Eigen::Vector<double, 0> lambda_eq_empty{};
    [[maybe_unused]] Eigen::Vector<double, 0> c_eq_empty{};
    [[maybe_unused]] Eigen::Matrix<double, 0, N> J_eq_empty(0, problem.dimension());

    for(std::uint32_t i = 0; i < budget; ++i)
    {
        const auto sr = solver.step();
        const auto& s = solver.state();

        // GCMMA wraps an MMA state inside state_type::mma_state; that
        // inner state is where g and J_ineq live. Dispatch at compile
        // time so the mma_policy direct-state path keeps its zero
        // indirection too.
        const auto& inner = [&]() -> const auto&
        {
            if constexpr(requires { s.mma_state; })
                return s.mma_state;
            else
                return s;
        }();

        // cv_Linf reported out-of-band: policies at HEAD still use the
        // L1 constraint_violation helper in step_result. Recompute here.
        const double cv_linf =
            nablapp::detail::primal_feasibility_inf<double,
                Eigen::Dynamic, MC>(inner.c_eq, inner.c_ineq);

        // KKT-local multiplier estimate via Algorithm 18.3; this is the
        // surrogate for the subproblem dual y when y is unavailable
        // through the public policy API (y is a private member of
        // mma_subproblem_solver at HEAD).
        const Eigen::Vector<double, MC> mu_ineq =
            active_set_mu_ineq<N, MC>(inner.g, inner.J_ineq, inner.c_ineq);

        const double kkt =
            nablapp::detail::kkt_residual<double, N, 0, MC>(
                inner.g, J_eq_empty, inner.J_ineq,
                lambda_eq_empty, mu_ineq,
                c_eq_empty, inner.c_ineq);

        const int policy_status_code = sr.policy_status
            ? static_cast<int>(*sr.policy_status)
            : -1;

        print_row(problem_label, policy_label,
                  i + 1,
                  sr.objective_value,
                  sr.gradient_norm,
                  cv_linf,
                  kkt,
                  sr.step_size,
                  sr.is_null_step,
                  policy_status_code);

        if(sr.policy_status)
            break;
    }
}

// Sign-convention probe on HS076.
//
// Drives mma_policy for 30 iterations, then at the final iterate:
//   1. Recomputes the subproblem coefficients via a harness-local
//      mma_coefficients call using the policy's current asymptotes.
//   2. Calls the harness-local probe_dual_solve_with_multipliers so
//      the dual variable y is exposed.
//   3. Compares y component-wise to active_set_mu_ineq on the same
//      (grad_f, J_ineq, c_ineq) state.
//   4. Emits the comparison and a single A1-FINDING line.
//
// Reference: Svanberg 1987 Section 5 (subproblem dual = inequality
//            multiplier up to sign convention); Nocedal and Wright
//            2e Algorithm 18.3 (active-set LS multiplier estimate).
void run_sign_convention_probe()
{
    using Problem = nablapp::hs076<double>;
    constexpr int N = Problem::problem_dimension;
    constexpr int MC = Problem::constraint_count;

    Problem problem;
    auto x0 = problem.initial_point();

    nablapp::solver_options opts;
    opts.max_iterations = 30;
    opts.set_gradient_threshold(1e-14);
    opts.set_objective_threshold(1e-18);
    opts.set_step_threshold(1e-18);
    opts.set_stationarity_threshold(1e-14);

    nablapp::mma_policy<N> policy;
    nablapp::basic_solver solver{policy, problem, x0, opts};

    for(std::uint32_t i = 0; i < opts.max_iterations; ++i)
    {
        const auto sr = solver.step();

        // Also emit the per-iter trace for HS076 MMA so the probe's
        // residual context is visible alongside the other runs.
        const auto& s = solver.state();
        const double cv_linf =
            nablapp::detail::primal_feasibility_inf<double,
                Eigen::Dynamic, MC>(s.c_eq, s.c_ineq);
        const Eigen::Vector<double, MC> mu_ineq_est =
            active_set_mu_ineq<N, MC>(s.g, s.J_ineq, s.c_ineq);

        Eigen::Matrix<double, 0, N> J_eq_empty(0, problem.dimension());
        Eigen::Vector<double, 0> lambda_eq_empty{};
        Eigen::Vector<double, 0> c_eq_empty{};

        const double kkt =
            nablapp::detail::kkt_residual<double, N, 0, MC>(
                s.g, J_eq_empty, s.J_ineq,
                lambda_eq_empty, mu_ineq_est,
                c_eq_empty, s.c_ineq);

        const int policy_status_code = sr.policy_status
            ? static_cast<int>(*sr.policy_status)
            : -1;

        print_row("hs076", "mma",
                  i + 1,
                  sr.objective_value,
                  sr.gradient_norm,
                  cv_linf,
                  kkt,
                  sr.step_size,
                  sr.is_null_step,
                  policy_status_code);

        if(sr.policy_status)
            break;
    }

    // ------- Sign-convention probe block -------
    const auto& s = solver.state();
    const int n = static_cast<int>(s.x.size());
    const int m = static_cast<int>(s.c_ineq.size());

    // Replicate the MMA coefficient build from the current state, using
    // the same asymptotes and g_mma = -c_ineq transform as the policy.
    nablapp::detail::mma_coeffs<double, N, MC> coeffs;
    coeffs.p0.resize(n);
    coeffs.q0.resize(n);
    coeffs.pi.resize(m, n);
    coeffs.qi.resize(m, n);
    coeffs.r0.resize(1);
    coeffs.ri.resize(m);

    constexpr double epsilon = 1e-7;

    double r0_val = s.f;
    for(int j = 0; j < n; ++j)
    {
        const double ux = s.U[j] - s.x[j];
        const double xl = s.x[j] - s.L[j];

        coeffs.p0[j] = ux * ux * std::max(s.g[j], 0.0) + epsilon;
        coeffs.q0[j] = xl * xl * std::max(-s.g[j], 0.0) + epsilon;

        r0_val -= coeffs.p0[j] / ux + coeffs.q0[j] / xl;
    }
    coeffs.r0[0] = r0_val;

    Eigen::Vector<double, MC> g_mma = -s.c_ineq;
    Eigen::Matrix<double, MC, N> dg_mma = -s.J_ineq;

    for(int i = 0; i < m; ++i)
    {
        double ri_val = g_mma[i];
        for(int j = 0; j < n; ++j)
        {
            const double ux = s.U[j] - s.x[j];
            const double xl = s.x[j] - s.L[j];

            coeffs.pi(i, j) = ux * ux * std::max(dg_mma(i, j), 0.0) + epsilon;
            coeffs.qi(i, j) = xl * xl * std::max(-dg_mma(i, j), 0.0) + epsilon;

            ri_val -= coeffs.pi(i, j) / ux + coeffs.qi(i, j) / xl;
        }
        coeffs.ri[i] = ri_val;
    }

    // Pass alpha = L, beta = U so the harness probe mirrors the policy's
    // (L_inner, U_inner, L_inner, U_inner) GCMMA-style call shape; MMA
    // uses (L, U, alpha, beta) with alpha/beta as bound-safe inner box.
    // Here we want the unrestricted dual y, so pass the asymptote pair.
    Eigen::Vector<double, N> alpha = s.L;
    Eigen::Vector<double, N> beta = s.U;
    for(int j = 0; j < n; ++j)
    {
        alpha[j] = std::max(alpha[j], s.lower[j]);
        beta[j] = std::min(beta[j], s.upper[j]);
    }

    auto probe = probe_dual_solve_with_multipliers<double, N, MC>(
        coeffs, s.L, s.U, alpha, beta);

    // Tight active tolerance mirrors the SQP-policy default (1e-6 /
    // 1e-8). At a stalled but non-KKT iterate this often classifies
    // zero constraints as active, so a looser second pass is run
    // below to cross-check.
    Eigen::Vector<double, MC> mu_est_tight =
        active_set_mu_ineq<N, MC>(s.g, s.J_ineq, s.c_ineq, 1e-6);
    Eigen::Vector<double, MC> mu_est_loose =
        active_set_mu_ineq<N, MC>(s.g, s.J_ineq, s.c_ineq, 1e-2);

    // Log the per-constraint context: c_ineq value (negative = infeasible
    // in the nablapp c >= 0 convention), harness-probe y, and both
    // active-set multipliers.
    std::fprintf(stderr, "# ---- HS076 sign-convention probe ----\n");
    std::fprintf(stderr, "# iterate: f=%.6e, ||grad||=%.6e\n",
                 s.f, s.g.lpNorm<Eigen::Infinity>());
    std::fprintf(stderr, "# c_ineq values (>= 0 feasible under nablapp "
                         "sign convention)\n");
    for(int i = 0; i < m; ++i)
        std::fprintf(stderr, "#   c_ineq[%d] = %.6e\n", i, s.c_ineq[i]);
    std::fprintf(stderr, "# y_probe (harness-local subproblem dual from "
                         "replicated dual_solve)\n");
    for(int i = 0; i < m; ++i)
        std::fprintf(stderr, "#   y_probe[%d] = %.6e\n", i, probe.y[i]);
    std::fprintf(stderr, "# mu_est_tight (active_tol=1e-6)\n");
    for(int i = 0; i < m; ++i)
        std::fprintf(stderr, "#   mu_est_tight[%d] = %.6e\n",
                     i, mu_est_tight[i]);
    std::fprintf(stderr, "# mu_est_loose (active_tol=1e-2)\n");
    for(int i = 0; i < m; ++i)
        std::fprintf(stderr, "#   mu_est_loose[%d] = %.6e\n",
                     i, mu_est_loose[i]);

    // Evaluate sign alignment on the loose estimate (tight one is too
    // selective to pin an active component when the solver stalled
    // before satisfying |c_ineq| < 1e-6). A "positive-only y" result
    // under the nablapp c_ineq >= 0 convention already confirms no
    // sign flip is needed: the subproblem dual y is non-negative by
    // construction (dual_solve projects to the non-negative orthant),
    // and nablapp's mu_ineq is likewise >= 0 by KKT dual feasibility
    // (N&W 2e Def 12.1). Sign agreement is therefore definitional;
    // the magnitude comparison below confirms the mapping is y -> mu
    // and not y -> -mu.
    int same_sign = 0;
    int opposite_sign = 0;
    double max_rel_diff = 0.0;
    bool probe_has_nonzero = false;
    for(int i = 0; i < m; ++i)
    {
        const double a = probe.y[i];
        const double b = mu_est_loose[i];
        if(std::abs(a) > 1e-8)
            probe_has_nonzero = true;
        if(std::abs(a) + std::abs(b) < 1e-8)
            continue;
        if((a >= 0.0) == (b >= 0.0))
            ++same_sign;
        else
            ++opposite_sign;

        const double denom = std::max(std::abs(a) + std::abs(b), 1e-12);
        const double diff = std::abs(a - b) / denom;
        max_rel_diff = std::max(max_rel_diff, diff);
    }

    std::fprintf(stderr, "# HS076 A1-probe: "
                         "same_sign_components=%d, opposite_sign_components=%d, "
                         "max_rel_diff=%.6e\n",
                 same_sign, opposite_sign, max_rel_diff);

    // The subproblem solver projects y to the non-negative orthant
    // after each Newton step (see lib/nablapp/include/nablapp/detail/
    // mma_subproblem.h line 376: y_.cwiseMax(0)). If y_probe has any
    // strictly positive entry at a point where the solver has made
    // meaningful progress, the "no sign flip" mapping is confirmed:
    // positive y directly maps to non-negative mu_ineq (KKT dual
    // feasibility under c_ineq >= 0 convention).
    if(probe_has_nonzero && opposite_sign == 0)
    {
        std::fprintf(stderr,
            "# A1-FINDING: y direct map to mu_ineq confirmed; "
            "correctness surgery uses mu_ineq = y (no sign flip).\n");
    }
    else if(opposite_sign > 0 && same_sign == 0)
    {
        std::fprintf(stderr,
            "# A1-FINDING: y requires sign flip to map to mu_ineq; "
            "correctness surgery must use mu_ineq = -y.\n");
    }
    else if(opposite_sign > 0 && same_sign > 0)
    {
        std::fprintf(stderr,
            "# A1-FINDING: mixed sign agreement "
            "(same=%d opposite=%d); probe is inconclusive -- rerun at a "
            "different iterate or tighten active_tol.\n",
            same_sign, opposite_sign);
    }
    else
    {
        std::fprintf(stderr,
            "# A1-FINDING: all components numerically zero at this "
            "iterate; probe degraded -- rerun with looser convergence "
            "threshold so at least one inequality is active.\n");
    }
}

}

int main()
{
    // CSV header row. Keep column order in sync with print_row().
    std::fprintf(stderr,
        "problem,policy,iter,f,grad_norm,cv_Linf,kkt_residual,"
        "step_size,is_null_step,policy_status\n");

    // Run 1: MMA on HS043 -- inequality-only, n=4, m=3.
    using Hs043 = nablapp::hs043<double>;
    run_case<Hs043, nablapp::mma_policy<Hs043::problem_dimension>>(
        "hs043", "mma",
        Hs043{},
        nablapp::mma_policy<Hs043::problem_dimension>{},
        30);

    // Run 2: GCMMA on HS035 -- inequality-and-bound, n=3, m=1.
    using Hs035 = nablapp::hs035<double>;
    run_case<Hs035, nablapp::gcmma_policy<Hs035::problem_dimension>>(
        "hs035", "gcmma",
        Hs035{},
        nablapp::gcmma_policy<Hs035::problem_dimension>{},
        50);

    // Run 3: GCMMA on HS043 -- cap at 100 iter; full 10000-iter run is
    // unnecessary for the breakdown analysis, the drift is clear in the
    // first 50 outer iterations.
    run_case<Hs043, nablapp::gcmma_policy<Hs043::problem_dimension>>(
        "hs043", "gcmma",
        Hs043{},
        nablapp::gcmma_policy<Hs043::problem_dimension>{},
        100);

    // Run 4: MMA on HS076 plus sign-convention probe.
    run_sign_convention_probe();

    return 0;
}
