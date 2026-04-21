// Per-outer-iteration CSV trace harness for MMA and GCMMA
// asymptote-oscillation investigation on HS043, HS076, and HS035
// (control), plus an MMA HS043 move_limit A/B sweep at {0.5, 0.7, 0.9}.
//
// Runs:
//   1. mma_policy on HS043, default move_limit = 0.9, 10000 outer iters.
//   2. mma_policy on HS076, default move_limit = 0.9, 10000 outer iters.
//   3. gcmma_policy on HS076, default options, 10000 outer iters.
//   4. mma_policy on HS043, move_limit = 0.5, 200 outer iters (A/B a).
//   5. mma_policy on HS043, move_limit = 0.7, 200 outer iters (A/B b).
//   6. mma_policy on HS043, move_limit = 0.9, 200 outer iters (A/B c).
//   7. mma_policy on HS035, default, 100 outer iters (converging
//      control so non-pathological rows are available for comparison).
//
// Output: one CSV file per run, written via std::ofstream to the
// logs/ subdirectory of the diagnostic artifact tree. Progress lines
// go to stderr via std::fprintf. The harness creates the logs/
// subdirectory at startup via std::filesystem::create_directories.
//
// Throwaway: delete after the mma/gcmma globalization question is
// resolved.
//
// inner_iter column: real GCMMA inner count in [0, max_inner]; -1 =
//                    MMA (no inner loop); -2 = sentinel for "inner
//                    cap hit" (GCMMA reached max_inner_iterations
//                    without meeting the conservativity test, so
//                    step_result.is_null_step is true).
//
// References
//   Svanberg 1987, "The method of moving asymptotes -- a new method for
//     structural optimization", Int. J. Numer. Methods Eng. 24,
//     Section 3 (asymptote schedule on sign-flip detection),
//     Section 5 (move-limit subproblem bounds, eq. 5.2-5.5),
//     Theorem 5.1 (Cauchy-sequence convergence).
//   Svanberg 2002, "A class of globally convergent optimization methods
//     based on conservative convex separable approximations", SIAM J.
//     Optim. 12(2), Section 3 (structured regularization),
//     Section 4.2 (rho-growth schedule and inner conservativity loop).
//   Nocedal and Wright, "Numerical Optimization" 2e, Definition 12.1
//     (KKT conditions).
//
// Sign convention (matches lib/nablapp/include/nablapp/detail/
// lagrangian.h:42-60 and formulation/concepts.h): c_ineq(x) >= 0
// feasible; mu_ineq >= 0 by KKT dual feasibility. The MMA subproblem
// uses the mirror form g_i = -c_ineq_i <= 0 with dual y_i >= 0. The
// identity -J_ineq^T y = J_ineq^T mu_ineq then maps y directly into
// nablapp's mu_ineq without a sign flip.

#include "nablapp/detail/mma_subproblem.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/solver/gcmma_policy.h"
#include "nablapp/solver/mma_policy.h"
#include "nablapp/solver/options.h"
#include "nablapp/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <cmath>
#include <cstdio>
#include <string>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <filesystem>
#include <string_view>

namespace
{

constexpr std::string_view csv_header =
    "problem,policy,iter,inner_iter,f,f_trial,f_approx_at_trial,"
    "step_size,x_norm,sign_flip_count,asymptote_factor_mean,"
    "L_U_width_min,L_U_width_max,rho,rhoc_max,raa_0,raa_max,"
    "kkt_residual,constraint_violation_Linf,inner_max_hit,"
    "policy_status\n";

// Builds a solver_options with the suppressed-convergence thresholds
// from diagnostic_mma_breakdown.cpp:304-309. The tight thresholds let
// the harness observe the full pathology without the solver exiting
// on ftol/xtol/stationarity by side effect.
nablapp::solver_options<> make_suppressed_convergence_opts(std::uint32_t budget)
{
    nablapp::solver_options<> opts;
    opts.max_iterations = budget;
    opts.set_gradient_threshold(1e-14);
    opts.set_objective_threshold(1e-18);
    opts.set_step_threshold(1e-18);
    opts.set_stationarity_threshold(1e-14);
    return opts;
}

// Drives a basic_solver for `budget` outer iterations, emitting one
// CSV row per outer iter. Generic over Problem and Solver so MMA and
// GCMMA, as well as explicit-policy-opts A/B variants, share one
// emission loop. The compile-time dispatch on requires { s.mma_state; }
// distinguishes GCMMA (which nests an mma_policy state) from MMA.
template <typename Problem, typename Solver>
void emit_trace(std::string_view problem_label,
                std::string_view policy_label,
                Problem problem,
                Solver& solver,
                std::uint32_t budget,
                std::ofstream& out)
{
    constexpr int N = Problem::problem_dimension;
    constexpr int MC = Problem::constraint_count;

    // Shadow rho for the Svanberg 2002 "as-if" computation
    // (research-only instrumentation; not a proposed fix). Init
    // matches NLopt mma.c:209 rho_init=1.0 and floor at
    // MMA_RHOMIN=1e-5 per mma.c:36-41.
    double shadow_rho = 1.0;
    double shadow_rhoc_max = 1.0;
    constexpr double shadow_rho_min = 1e-5;

    out << std::setprecision(17);

    for(std::uint32_t i = 0; i < budget; ++i)
    {
        // Pre-step snapshot (captures x_k, x_old1_k, x_old2_k, L_k,
        // U_k before solver.step() mutates state).
        const auto pre = solver.state();

        const auto sr = solver.step();
        const auto& s = solver.state();

        // Compile-time dispatch: GCMMA state nests an MMA state under
        // s.mma_state; MMA policy has no mma_state field.
        const auto& inner_pre = [&]() -> const auto&
        {
            if constexpr(requires { pre.mma_state; })
                return pre.mma_state;
            else
                return pre;
        }();
        const auto& inner_post = [&]() -> const auto&
        {
            if constexpr(requires { s.mma_state; })
                return s.mma_state;
            else
                return s;
        }();

        const std::uint32_t iter = i;

        // inner_iter encoding:
        //   MMA (no inner loop)           -> -1.
        //   GCMMA, non-null-step           -> -1 (inner loop converged;
        //                                       per-outer count not
        //                                       exposed through the
        //                                       public step_result under
        //                                       the per-outer-only seam).
        //   GCMMA, null-step (cap hit)     -> -2 sentinel.
        int inner_iter = -1;
        if constexpr(requires { s.mma_state; })
        {
            if(sr.is_null_step)
                inner_iter = -2;
        }

        // f at outer-iter entry and objective at trial point
        // (== state.x after step()).
        const double f = problem.value(inner_pre.x);
        const double f_trial = problem.value(inner_post.x);

        // f_approx_at_trial via harness-local MMA approximant: build
        // coefficients at pre-step state (raa_0 = 0, raa = 0 baseline)
        // and evaluate at the accepted trial point using pre-step L/U.
        Eigen::Vector<double, N> grad_pre;
        grad_pre.resize(inner_pre.x.size());
        problem.gradient(inner_pre.x, grad_pre);

        Eigen::Vector<double, MC> g_mma_pre = -inner_pre.c_ineq;
        Eigen::Matrix<double, MC, N> dg_mma_pre = -inner_pre.J_ineq;

        const Eigen::Vector<double, MC> raa_zero =
            Eigen::Vector<double, MC>::Zero(inner_pre.c_ineq.size());

        auto coeffs = nablapp::detail::mma_coefficients<double, N, MC>(
            inner_pre.x, f, grad_pre, g_mma_pre, dg_mma_pre,
            inner_pre.L, inner_pre.U, 0.0, raa_zero);

        const double f_approx_at_trial =
            nablapp::detail::mma_subproblem_value<double, N, MC>(
                coeffs, inner_post.x, inner_pre.L, inner_pre.U);

        const double step_size = (inner_post.x - inner_pre.x).norm();
        const double x_norm = inner_post.x.norm();

        // Asymptote sign-flip + factor-mean summary over the pre-step
        // oscillation history (x_k - x_old1) * (x_old1 - x_old2).
        std::uint32_t sign_flip_count = 0;
        double asymptote_factor_sum = 0.0;
        for(int j = 0; j < N; ++j)
        {
            const double d1 = inner_pre.x[j] - inner_pre.x_old1[j];
            const double d2 = inner_pre.x_old1[j] - inner_pre.x_old2[j];
            const double product = d1 * d2;
            if(product < 0.0)
            {
                ++sign_flip_count;
                asymptote_factor_sum += 0.7;
            }
            else if(product > 0.0)
            {
                asymptote_factor_sum += 1.2;
            }
            else
            {
                asymptote_factor_sum += 1.0;
            }
        }
        const double asymptote_factor_mean =
            asymptote_factor_sum / static_cast<double>(N);

        // Post-step asymptote box widths.
        Eigen::Vector<double, N> widths = inner_post.U - inner_post.L;
        const double L_U_width_min = widths.minCoeff();
        const double L_U_width_max = widths.maxCoeff();

        // Shadow rho update (NLopt mma.c:363-370 formula). The denom
        // `dd.wval` is a subproblem-dual weight the harness does not
        // have access to through the public API; using 1.0 makes the
        // shadow rho a first-order indicator of "would have grown"
        // rather than an exact simulation.
        const double fcur = f_trial;
        const double fapprox = f_approx_at_trial;
        constexpr double wval = 1.0;
        if(fcur > fapprox)
            shadow_rho = std::min(10.0 * shadow_rho,
                                  1.1 * (shadow_rho + (fcur - fapprox) / wval));
        else
            shadow_rho = std::max(0.1 * shadow_rho, shadow_rho_min);
        shadow_rhoc_max = shadow_rho;

        // GCMMA raa readouts; NaN for MMA.
        double raa_0 = std::numeric_limits<double>::quiet_NaN();
        double raa_max = std::numeric_limits<double>::quiet_NaN();
        if constexpr(requires { s.raa_0; })
        {
            raa_0 = s.raa_0;
            if constexpr(requires { s.raa; })
                raa_max = (MC > 0) ? static_cast<double>(s.raa.maxCoeff()) : 0.0;
        }

        const double kkt = sr.kkt_residual.value_or(
            std::numeric_limits<double>::quiet_NaN());

        // Primal feasibility L-infty on inequality block
        // (HS043/HS076/HS035 are all inequality-only; c_ineq >= 0
        // feasible under nablapp convention, violation = max(0, -c)).
        double cv_linf = 0.0;
        for(int k = 0; k < MC; ++k)
        {
            const double c_k = static_cast<double>(inner_post.c_ineq[k]);
            cv_linf = std::max(cv_linf, std::max(0.0, -c_k));
        }

        const int inner_max_hit = (inner_iter == -2) ? 1 : 0;

        const int policy_status =
            sr.policy_status ? static_cast<int>(*sr.policy_status) : -1;

        out << problem_label << ',' << policy_label << ',' << iter << ','
            << inner_iter << ',' << f << ',' << f_trial << ','
            << f_approx_at_trial << ',' << step_size << ',' << x_norm << ','
            << sign_flip_count << ',' << asymptote_factor_mean << ','
            << L_U_width_min << ',' << L_U_width_max << ',' << shadow_rho
            << ',' << shadow_rhoc_max << ',' << raa_0 << ',' << raa_max
            << ',' << kkt << ',' << cv_linf << ',' << inner_max_hit << ','
            << policy_status << '\n';

        // Do NOT break on sr.policy_status: the traces target the full
        // budget pathology. Only exit on budget via the for-loop.
    }
}

// Default-options variant: build solver with policy defaults, emit trace.
template <typename Problem, typename Policy>
void run_case_default(std::string_view problem_label,
                      std::string_view policy_label,
                      Problem problem,
                      Policy policy,
                      std::uint32_t budget,
                      std::ofstream& out)
{
    auto x0 = problem.initial_point();
    auto opts = make_suppressed_convergence_opts(budget);
    nablapp::basic_solver solver{policy, problem, x0, opts};
    emit_trace(problem_label, policy_label, problem, solver, budget, out);
}

// Explicit-policy-opts variant: threads policy_opts through the
// basic_solver(policy, problem, x0, opts, policy_opts) ctor.
template <typename Problem, typename Policy, typename PolicyOpts>
void run_case_with_policy_opts(std::string_view problem_label,
                               std::string_view policy_label,
                               Problem problem,
                               Policy policy,
                               const PolicyOpts& policy_opts,
                               std::uint32_t budget,
                               std::ofstream& out)
{
    auto x0 = problem.initial_point();
    auto opts = make_suppressed_convergence_opts(budget);
    nablapp::basic_solver solver{policy, problem, x0, opts, policy_opts};
    emit_trace(problem_label, policy_label, problem, solver, budget, out);
}

std::ofstream open_csv(std::string_view log_dir, std::string_view leafname)
{
    std::string path{log_dir};
    path.append(leafname);
    std::ofstream f{path};
    f << csv_header;
    f << std::setprecision(17);
    return f;
}

}

int main(int argc, char** argv)
{
    // Default: ./logs/ relative to the current working directory.  No
    // planning-tool path embedded in source.  Runners override the
    // destination via --log-dir <path>.
    std::string log_dir = "./logs/";
    for(int i = 1; i + 1 < argc; ++i)
    {
        if(std::string_view{argv[i]} == "--log-dir")
        {
            log_dir = argv[i + 1];
            // Append a trailing '/' if missing so path concatenation
            // works identically to the prior file-scope-constant
            // behavior (open_csv just appends the leaf name).
            if(!log_dir.empty() && log_dir.back() != '/')
                log_dir.push_back('/');
            break;
        }
    }

    std::filesystem::create_directories(log_dir);

    std::fprintf(stderr, "Running diagnostic_mma_asymptote_trace: 7 runs\n");

    // Run 1: MMA HS043, default, 10000 iters.
    {
        std::fprintf(stderr,
            "  [1/7] mma_policy HS043 default move_limit 10000 iters\n");
        auto out = open_csv(log_dir, "mma_hs043_trace.csv");
        run_case_default("hs043", "mma",
                         nablapp::hs043<double>{},
                         nablapp::mma_policy<4>{},
                         10000u, out);
    }

    // Run 2: MMA HS076, default, 10000 iters.
    {
        std::fprintf(stderr,
            "  [2/7] mma_policy HS076 default move_limit 10000 iters\n");
        auto out = open_csv(log_dir, "mma_hs076_trace.csv");
        run_case_default("hs076", "mma",
                         nablapp::hs076<double>{},
                         nablapp::mma_policy<4>{},
                         10000u, out);
    }

    // Run 3: GCMMA HS076, default, 10000 iters.
    {
        std::fprintf(stderr,
            "  [3/7] gcmma_policy HS076 default 10000 iters\n");
        auto out = open_csv(log_dir, "gcmma_hs076_trace.csv");
        run_case_default("hs076", "gcmma",
                         nablapp::hs076<double>{},
                         nablapp::gcmma_policy<4>{},
                         10000u, out);
    }

    // Runs 4-6: MMA HS043 move_limit A/B sweep, 200 iters each.
    struct ab_entry
    {
        double ml;
        std::string_view leaf;
        int idx;
    };
    const ab_entry sweep[] = {
        {0.5, "mma_hs043_ab_ml050.csv", 4},
        {0.7, "mma_hs043_ab_ml070.csv", 5},
        {0.9, "mma_hs043_ab_ml090.csv", 6},
    };
    for(const auto& entry : sweep)
    {
        std::fprintf(stderr,
            "  [%d/7] mma_policy HS043 move_limit=%.2f 200 iters\n",
            entry.idx, entry.ml);
        auto out = open_csv(log_dir, entry.leaf);
        typename nablapp::mma_policy<4>::options_type policy_opts;
        policy_opts.move_limit = entry.ml;
        run_case_with_policy_opts("hs043", "mma",
                                  nablapp::hs043<double>{},
                                  nablapp::mma_policy<4>{},
                                  policy_opts,
                                  200u, out);
    }

    // Run 7: MMA HS035 control, default, 100 iters.
    {
        std::fprintf(stderr,
            "  [7/7] mma_policy HS035 default 100 iters (control)\n");
        auto out = open_csv(log_dir, "mma_hs035_control.csv");
        run_case_default("hs035", "mma",
                         nablapp::hs035<double>{},
                         nablapp::mma_policy<3>{},
                         100u, out);
    }

    // Runs 8-15: GCMMA HS076 K_saturation sweep, 1000 iters each.
    // Tests the Signal 2 consecutive-inner-cap-hit threshold at
    // K values bracketing the paper-form-calibrated default (5).
    // Below-5 values are diagnostic-only (confirm problem direction);
    // >=5 values are ship candidates, smallest-K-closing-HS076 wins
    // per the selection rule in the accompanying verification doc.
    const std::uint16_t K_values[] = {1, 2, 3, 5, 10, 15, 20, 25};
    int k_idx = 8;
    for(auto K : K_values)
    {
        std::fprintf(stderr,
            "  [%d/15] gcmma_policy HS076 K_saturation=%u 1000 iters\n",
            k_idx, static_cast<unsigned>(K));
        char leaf_buf[32];
        std::snprintf(leaf_buf, sizeof(leaf_buf),
                      "gcmma_hs076_K%02u.csv",
                      static_cast<unsigned>(K));
        auto out = open_csv(log_dir, leaf_buf);
        typename nablapp::gcmma_policy<4>::options_type policy_opts;
        policy_opts.raa_saturated_stall_consecutive_count = K;
        run_case_with_policy_opts("hs076", "gcmma",
                                  nablapp::hs076<double>{},
                                  nablapp::gcmma_policy<4>{},
                                  policy_opts,
                                  1000u, out);
        ++k_idx;
    }

    std::fprintf(stderr, "Done. Wrote 15 CSV files under %s\n",
                 log_dir.c_str());
    return 0;
}
