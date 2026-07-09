#ifndef HPP_GUARD_ARGMIN_BENCHMARKS_BENCH_ARGMIN_H
#define HPP_GUARD_ARGMIN_BENCHMARKS_BENCH_ARGMIN_H

// Nablapp solver benchmark harness.
//
// Provides run_argmin_solver() to run a single policy on a problem and
// produce a benchmark_result, and run_all_argmin_solvers() to run all
// applicable argmin policies on a given problem.

#include "bench_config.h"
#include "counting_problem.h"
#include "benchmark_result.h"
#include "trace_entry.h"
#include "problem_registry.h"

#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/lm_policy.h"
#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/isres_policy.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/mma_policy.h"
#include "argmin/solver/gcmma_policy.h"
#include "argmin/solver/ccsa_quadratic_policy.h"
#include "argmin/solver/alternative/gcmma/rho_wval_policy.h"
#include "argmin/solver/alternative/gcmma/raa_augmented_policy.h"
#include "argmin/solver/alternative/gcmma/move_limit_shrink_policy.h"
#include "argmin/solver/cmaes_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/multistart_policy.h"
#include "argmin/solver/restarting_policy.h"
#include "argmin/solver/projected_gn_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/byrd_lbfgsb_policy.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_trsqp_policy.h"
#include "argmin/solver/projected_gradient_gn_policy.h"
#include "argmin/solver/augmented_lagrangian_policy.h"

#include "argmin/formulation/concepts.h"
#include "argmin/result/status.h"
#include "argmin/result/solve_result.h"
#include "argmin/test_functions/problem_class.h"

#include <benchmark/benchmark.h>

#include <cstdio>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace argmin::bench
{

namespace detail
{

template <typename Policy, int N, typename = void>
struct rebind_policy { using type = Policy; };

template <typename Policy, int N>
struct rebind_policy<Policy, N, std::void_t<typename Policy::template rebind<N>>>
{ using type = typename Policy::template rebind<N>; };

template <typename Policy, int N>
using rebind_policy_t = typename rebind_policy<Policy, N>::type;

[[nodiscard]] inline auto status_string(solver_status s) -> std::string_view
{
    switch(s)
    {
    case solver_status::converged:        return "converged";
    case solver_status::max_iterations:   return "max_iterations";
    case solver_status::budget_exhausted: return "budget_exhausted";
    case solver_status::stalled:          return "stalled";
    case solver_status::diverged:         return "diverged";
    case solver_status::ftol_reached:     return "ftol_reached";
    case solver_status::xtol_reached:     return "xtol_reached";
    case solver_status::maxeval_reached:  return "maxeval_reached";
    case solver_status::roundoff_limited:   return "roundoff_limited";
    case solver_status::trust_region_step_rejected: return "trust_region_step_rejected";
    case solver_status::objective_stalled: return "objective_stalled";
    case solver_status::time_limit_reached: return "time_limit_reached";
    case solver_status::aborted:           return "aborted";
    case solver_status::running:           return "running";
    }
    return "unknown";
}

[[nodiscard]] inline auto cap_status_string(std::string_view status,
                                            const eval_counts& counts,
                                            const bench_config& config) -> std::string
{
    if(status == "time_limit_reached")
        return "wall";
    if(status == "budget_exhausted" || status == "maxeval_reached")
        return "f_eval";
    if(config.max_f_evals > 0 && counts.f >= config.max_f_evals)
        return "f_eval";
    return std::string{counts.cap_status()};
}

}

// Run a single argmin solver policy on a problem, collecting result and
// optional per-iteration trace.
//
// Policy must be a valid step_budget_solver policy.
// Problem must provide value, gradient, dimension, optimal_value, initial_point.
//
// bench uses `default_convergence` uniformly across solver families; each
// solver sets its inner `objective_tolerance_criterion::stationarity_threshold`
// directly, and the composite kkt_residual E-measure (N&W 2e Definition 12.1,
// eq. 12.34) carries primal feasibility into the single termination test.
template <typename Policy, typename Convergence = default_convergence,
          typename Problem, typename... PolicyOpts>
auto run_argmin_solver(std::string_view solver_name,
                        std::string_view problem_name,
                        const Problem& prob,
                        int max_iterations,
                        bool collect_trace,
                        std::vector<trace_entry>& trace,
                        const bench_config& config,
                        PolicyOpts&&... policy_opts) -> benchmark_result
{
    // bench_config consumption: mode::library_defaults preserves existing
    // byte-identical behavior on the pre-existing column set. The new
    // {f,g,c,J}_evals columns are populated from the counting_problem<P>
    // wrapper that intercepts every problem callback the policy makes;
    // solver_iters continues to come from the policy's native iter count
    // for diagnostic parity with per-library expectations.
    eval_counts counts;
    counts.set_max_f_evals(config.max_f_evals);
    counting_problem<Problem> wrapped{prob, counts};

    static constexpr int N = problem_dimension_v<Problem>;
    using rebound_policy = detail::rebind_policy_t<Policy, N>;

    // Tolerance / iteration budget sourced from bench_config so library_defaults
    // (ftol_rel = xtol_rel = 1e-12, max_iter = 10000) preserves the prior
    // argmin_bench behavior byte-for-byte while publication mode tightens
    // both gates to 1e-16 so the DM tau-grid down to 1e-12 is observable.
    //
    // Wall-time gap: argmin's solver_options does not currently expose a
    // wall-clock budget knob. config.max_wall_time_s is intentionally NOT
    // enforced here; the gap is documented in the publication-mode
    // methodology write-up. A future seed may add a wall-clock check at
    // the step_budget_solver step-loop level if non-trivial work justifies it.
    auto x0 = prob.initial_point();
    solver_options<Convergence> opts{};
    opts.max_iterations = max_iterations;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(config.ftol_rel);
    opts.set_step_threshold(config.xtol_rel);

    if constexpr(constrained<Problem>)
    {
        // Constrained solvers: the raw objective gradient is nonzero at a
        // constrained optimum (only the Lagrangian gradient vanishes), and
        // penalty methods (augmented Lagrangian) cycle the augmented gradient
        // during multiplier updates. The composite kkt_residual E-measure
        // (N&W 2e Definition 12.1 / eq. 12.34) is the proper stationarity
        // test for constrained problems; it folds primal feasibility and
        // dual feasibility into a single scalar that vanishes at a KKT
        // point. Relax objective/step gates and drive termination through
        // the stationarity test via objective_tolerance_criterion's
        // kkt_residual leg.
        //
        // gradient_threshold is NOT set here: the current SQP and
        // augmented-lagrangian policies write raw ||grad_f||_inf into
        // step_result.gradient_norm rather than the Lagrangian gradient
        // norm, so a loose gradient_tolerance fires prematurely on
        // non-optimum iterates where ||grad_f|| drops but primal
        // feasibility has not converged. Policy-level gradient_norm
        // semantics is the subject of seeds SEED-003 / SEED-004 queued
        // for Phase 36 (SQP) / Phase 37 (auglag). Until then the
        // composite kkt_residual alone gates the constrained regime.
        //
        // The relaxed objective/step gates (1e-6) are correctness-driven
        // per the rationale above and stay constant across modes; the
        // stationarity_threshold scales with config.ftol_rel so publication
        // mode (1e-16) drives the kkt_residual gate tighter than
        // library_defaults (1e-12 → 1e-4 effective).
        opts.set_objective_threshold(1e-6);
        opts.set_step_threshold(1e-6);
        constexpr double stationarity_scale = 1e8;  // 1e-12 ftol -> 1e-4 stationarity
        opts.set_stationarity_threshold(config.ftol_rel * stationarity_scale);
    }

    if(collect_trace)
    {
        // Step loop with trace collection (per D-09); rows populate the
        // D-C3 12-column publication schema. f_best is a running-min over
        // f_current; accuracy is |f_current - prob.optimal_value()|;
        // step_norm pulls from step_result.step_size (the policy-native
        // step length); kkt_residual pulls from step_result.kkt_residual
        // (argmin's 31.1 E-measure composite, NaN when the policy does
        // not populate it).
        trace.clear();
        trace.resize(static_cast<std::size_t>(max_iterations));

        const double f_star = prob.optimal_value();
        double f_best_running = std::numeric_limits<double>::infinity();

        using wrapped_t = counting_problem<Problem>;
        step_budget_solver<rebound_policy, N, wrapped_t> solver(wrapped, x0, opts,
                                    std::forward<PolicyOpts>(policy_opts)...);

        auto t0 = std::chrono::steady_clock::now();

        int iters = 0;
        solver_status final_status = solver_status::running;
        double final_obj{};

        for(int i = 0; i < max_iterations; ++i)
        {
            auto sr = solver.step();
            ++iters;

            const double f_current = static_cast<double>(sr.objective_value);
            f_best_running = std::min(f_best_running, f_current);

            const auto t_now = std::chrono::steady_clock::now();
            const auto wall_us_now = std::chrono::duration_cast<std::chrono::microseconds>(
                t_now - t0).count();

            trace[static_cast<std::size_t>(i)] = trace_entry{
                .iter         = i,
                .f_evals      = counts.f,
                .g_evals      = counts.g,
                .c_evals      = counts.c,
                .J_evals      = counts.J,
                .wall_us      = wall_us_now,
                .f_current    = f_current,
                .f_best       = f_best_running,
                .accuracy     = std::abs(f_current - f_star),
                .cv           = static_cast<double>(sr.constraint_violation),
                .step_norm    = static_cast<double>(sr.step_size),
                .kkt_residual = sr.kkt_residual.has_value()
                                    ? static_cast<double>(*sr.kkt_residual)
                                    : std::numeric_limits<double>::quiet_NaN(),
            };

            final_obj = f_current;

            if(sr.policy_status)
            {
                final_status = *sr.policy_status;
                break;
            }

            auto conv = opts.convergence.check(sr, static_cast<std::uint32_t>(i + 1));
            if(conv)
            {
                final_status = *conv;
                break;
            }
        }

        if(final_status == solver_status::running)
            final_status = solver_status::max_iterations;

        auto t1 = std::chrono::steady_clock::now();
        auto wall_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        trace.resize(static_cast<std::size_t>(iters));

        // Read final_obj from the trace tail rather than the local variable.
        // With release-mode LTO the compiler has been observed to emit a
        // denormal value (~1e-322) at the .final_objective field even when
        // the local final_obj held the correct value at the loop break -- a
        // concrete reproduction is gcmma on hs043 where the trace's last
        // f_current is -44.000 but the summary's final_objective lands at
        // 1.037537856266618e-322. The trace is data we wrote ourselves and
        // does not pass through whatever read pattern the optimizer is
        // miscompiling. benchmark::DoNotOptimize on final_obj is insufficient.
        const double final_obj_safe = trace.empty()
            ? final_obj
            : static_cast<double>(trace.back().f_current);
        benchmark::DoNotOptimize(final_obj_safe);

        // Final-iter cv pulled from the trace tail (the per-iter trace is
        // the in-process record of what the policy reported on each step);
        // the trace is empty only on degenerate problems that never enter
        // the step loop, in which case 0.0 stands in.
        const double final_cv = trace.empty()
            ? 0.0
            : trace.back().cv;

        const auto status = detail::status_string(final_status);
        return benchmark_result{
            .solver = solver_name,
            .library = "argmin",
            .problem = problem_name,
            .pclass = prob.pclass,
            .dimension = prob.dimension(),
            .seed = config.seed,
            .mode = (config.the_mode == bench_config::mode::publication)
                        ? std::string_view{"publication"}
                        : std::string_view{"library_defaults"},
            .solver_iters = iters,
            .f_evals = counts.f,
            .g_evals = counts.g,
            .c_evals = counts.c,
            .J_evals = counts.J,
            .wall_time_us = wall_us,
            .final_objective = final_obj_safe,
            .known_optimum = prob.optimal_value(),
            .accuracy = std::abs(final_obj_safe - prob.optimal_value()),
            .constraint_violation = final_cv,
            .status = status,
            .cap_status = detail::cap_status_string(status, counts, config),
            .solve_wall_time_us = wall_us,
            .end_to_end_wall_time_us = wall_us,
        };
    }
    else
    {
        // Use solve(opts) so the active opts.convergence (including any
        // per-solver stationarity_threshold overrides) gates termination.
        // The no-args solve() path falls through to stored_convergence_
        // which is always default-constructed default_convergence.
        using wrapped_t = counting_problem<Problem>;
        step_budget_solver<rebound_policy, N, wrapped_t> solver(wrapped, x0, opts,
                                    std::forward<PolicyOpts>(policy_opts)...);

        auto t0 = std::chrono::steady_clock::now();
        auto result = solver.solve(opts);
        auto t1 = std::chrono::steady_clock::now();

        auto wall_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        benchmark::DoNotOptimize(result);

        const auto status = detail::status_string(result.status);
        return benchmark_result{
            .solver = solver_name,
            .library = "argmin",
            .problem = problem_name,
            .pclass = prob.pclass,
            .dimension = prob.dimension(),
            .seed = config.seed,
            .mode = (config.the_mode == bench_config::mode::publication)
                        ? std::string_view{"publication"}
                        : std::string_view{"library_defaults"},
            .solver_iters = static_cast<int>(result.function_evaluations),
            .f_evals = counts.f,
            .g_evals = counts.g,
            .c_evals = counts.c,
            .J_evals = counts.J,
            .wall_time_us = wall_us,
            .final_objective = result.objective_value,
            .known_optimum = prob.optimal_value(),
            .accuracy = std::abs(result.objective_value - prob.optimal_value()),
            .constraint_violation = static_cast<double>(result.constraint_violation),
            .status = status,
            .cap_status = detail::cap_status_string(status, counts, config),
            .solve_wall_time_us = wall_us,
            .end_to_end_wall_time_us = wall_us,
        };
    }
}

// Run all applicable argmin solver policies on a problem.
//
// Uses concepts to determine which solvers are applicable:
//   - Unconstrained: lbfgsb_policy
//   - Bound-constrained: lbfgsb_policy, bobyqa_policy
//   - Inequality/equality/mixed: kraft_slsqp_policy, augmented_lagrangian_policy
//   - Inequality: ccsa_quadratic_policy (inequality only, no equality)
//   - Equality/mixed: nw_sqp_policy
//   - Global: cmaes_policy, bobyqa_policy (if has bounds)
template <typename Problem>
void run_all_argmin_solvers(
    std::string_view problem_name,
    const Problem& prob,
    int max_iterations,
    bool collect_trace,
    std::vector<benchmark_result>& results,
    std::vector<std::vector<trace_entry>>& traces,
    const bench_config& config,
    std::uint64_t seed = 42)
{
    (void)config;  // forwarded to run_argmin_solver below; this scope only
                   // dispatches policies. Future plans branch here on
                   // config.the_mode for solver-set selection per CONTEXT D-A3.

    auto run = [&]<typename Policy>(std::string_view name, Policy) {
        std::vector<trace_entry> trace;
        auto r = run_argmin_solver<Policy>(
            name, problem_name, prob, max_iterations, collect_trace, trace,
            config);
        results.push_back(r);
        traces.push_back(std::move(trace));
    };

    auto run_constrained = [&]<typename Policy>(std::string_view name, Policy) {
        std::vector<trace_entry> trace;
        auto r = run_argmin_solver<Policy, default_convergence>(
            name, problem_name, prob, max_iterations, collect_trace, trace,
            config);
        results.push_back(r);
        traces.push_back(std::move(trace));
    };

    constexpr bool is_bound = bound_constrained<Problem>;
    constexpr bool is_constrained = constrained<Problem>;
    constexpr bool is_global = has_class(Problem::pclass, problem_class::global);

    // L-BFGS-B: unconstrained or bound-constrained (not general constrained).
    if constexpr(differentiable<Problem> && !is_constrained)
        run("lbfgsb", lbfgsb_policy<>{});

    // Byrd 1995 L-BFGS-B variant: gate identical to lbfgsb_policy.
    if constexpr(differentiable<Problem> && !is_constrained)
        run("byrd_lbfgsb", byrd_lbfgsb_policy<>{});

    // Levenberg-Marquardt: unconstrained least-squares.
    if constexpr(least_squares<Problem> && !is_constrained)
        run("lm", lm_policy<>{});

    // BOBYQA: bound-constrained or global with bounds.
    if constexpr(is_bound && !is_constrained)
        run("bobyqa", bobyqa_policy<>{});

    // Multi-start over BOBYQA: gate mirrors bobyqa (bound-constrained, not general).
    if constexpr(is_bound && !is_constrained)
        run("multistart_bobyqa", multistart_policy<bobyqa_policy<>>{});

    // Projected GN: bound-constrained least-squares (no general constraints).
    if constexpr(least_squares<Problem> && is_bound && !is_constrained)
    {
        run("projected_gn", projected_gn_policy<>{});
        run("projected_gradient_gn", projected_gradient_gn_policy<>{});
    }

    // SLSQP: any constrained problem. The line-search SQP family is
    // single-mode after the empirical _fast collapse; one dispatch per
    // policy.
    if constexpr(is_constrained && differentiable<Problem> && is_bound)
    {
        run_constrained("kraft_slsqp_accurate", kraft_slsqp_policy_accurate<>{});
    }

    // MMA family: inequality-constrained (no equality).
    if constexpr(is_constrained && differentiable<Problem> && is_bound)
    {
        if constexpr(requires { prob.num_equality(); })
        {
            if(prob.num_equality() == 0 && prob.num_inequality() > 0)
            {
                {
                    std::vector<trace_entry> trace;
                    auto r = run_argmin_solver<mma_policy<>, default_convergence>(
                        "mma", problem_name, prob, max_iterations, collect_trace, trace,
                        config);
                    results.push_back(r);
                    traces.push_back(std::move(trace));
                }
                {
                    std::vector<trace_entry> trace;
                    auto r = run_argmin_solver<gcmma_policy<>, default_convergence>(
                        "gcmma", problem_name, prob, max_iterations, collect_trace, trace,
                        config);
                    results.push_back(r);
                    traces.push_back(std::move(trace));
                }
                {
                    std::vector<trace_entry> trace;
                    auto r = run_argmin_solver<ccsa_quadratic_policy<>, default_convergence>(
                        "ccsa_quadratic", problem_name, prob, max_iterations, collect_trace, trace,
                        config);
                    results.push_back(r);
                    traces.push_back(std::move(trace));
                }
                // Alternative GCMMA variants (research artifacts; preserved
                // for the published comparison even after the production
                // gcmma_policy is aliased to one of them). gcmma above
                // emits as the alias winner; alt rows below preserve the
                // losing variants for the comparison table.
                {
                    std::vector<trace_entry> trace;
                    auto r = run_argmin_solver<
                        alternative::gcmma::move_limit_shrink_policy<>,
                        default_convergence>(
                        "gcmma_move_limit_shrink", problem_name, prob,
                        max_iterations, collect_trace, trace, config);
                    results.push_back(r);
                    traces.push_back(std::move(trace));
                }
                {
                    std::vector<trace_entry> trace;
                    auto r = run_argmin_solver<
                        alternative::gcmma::raa_augmented_policy<>,
                        default_convergence>(
                        "gcmma_raa_augmented", problem_name, prob,
                        max_iterations, collect_trace, trace, config);
                    results.push_back(r);
                    traces.push_back(std::move(trace));
                }
            }
        }
    }

    // N&W SQP: equality or mixed constrained. Single-mode after the
    // _fast collapse on the line-search SQP family; the runtime
    // equality-presence guard remains.
    if constexpr(is_constrained && differentiable<Problem> && is_bound)
    {
        if constexpr(requires { prob.num_equality(); })
        {
            if(prob.num_equality() > 0)
            {
                run_constrained("nw_sqp_accurate", nw_sqp_policy_accurate<>{});
            }
        }
    }

    // Filter SLSQP: any constrained problem (filter-based L-BFGS SQP).
    // Single-mode after the _fast collapse on the line-search SQP family.
    if constexpr(is_constrained && differentiable<Problem> && is_bound)
    {
        run_constrained("filter_slsqp_accurate", filter_slsqp_policy_accurate<>{});
    }

    // Filter NW SQP: any constrained problem (filter-based dense BFGS SQP).
    // Single-mode after the _fast collapse on the line-search SQP family.
    if constexpr(is_constrained && differentiable<Problem> && is_bound)
    {
        run_constrained("filter_nw_sqp_accurate", filter_nw_sqp_policy_accurate<>{});
    }

    // TR-SQP (Byrd-Omojokun composite step): tr_sqp at parity with the
    // line-search SQP families on every applicable constrained cell.
    if constexpr(is_constrained && differentiable<Problem> && is_bound)
    {
        run_constrained("tr_sqp_fast",     tr_sqp_policy_fast<>{});
        run_constrained("tr_sqp_accurate", tr_sqp_policy_accurate<>{});
    }

    // Filter trust-region SQP (Byrd-Omojokun composite step paired with
    // Fletcher-Leyffer filter acceptance per Wachter-Biegler 2005/2006).
    // Both modes dispatched on every constrained differentiable bound-aware
    // cell, mirroring the tr_sqp registration pattern.
    if constexpr(is_constrained && differentiable<Problem> && is_bound)
    {
        run_constrained("filter_trsqp_fast",     filter_trsqp_policy_fast<>{});
        run_constrained("filter_trsqp_accurate", filter_trsqp_policy_accurate<>{});
    }

    // Augmented Lagrangian (with L-BFGS-B inner): any constrained problem.
    if constexpr(is_constrained && differentiable<Problem> && is_bound)
        run_constrained("augmented_lagrangian", augmented_lagrangian_policy<>{});

    // CMA-ES: global problems with bounds.
    // IPOP restarts are essential for multimodal landscapes — without them
    // CMA-ES collapses into a single basin and the covariance degenerates.
    if constexpr(is_global && is_bound)
    {
        typename cmaes_policy<>::options_type cmaes_opts{};
        cmaes_opts.seed = seed;
        cmaes_opts.restart = cmaes_policy<>::restart_strategy::ipop;
        std::vector<trace_entry> trace;
        auto r = run_argmin_solver<cmaes_policy<>>(
            "cmaes", problem_name, prob, max_iterations, collect_trace, trace,
            config, cmaes_opts);
        results.push_back(r);
        traces.push_back(std::move(trace));
    }

    // restarting_policy<cmaes_policy<...>>: gate mirrors CMA-ES (global + bounds).
    // Plumb the per-row seed through options.inner so the decorator's inner
    // cmaes_policy uses the caller-visible seed rather than the deterministic
    // default fallback that restarting_policy::init() emits when
    // options.inner.seed is empty. The decorator's own restart policy
    // replaces CMA-ES IPOP (avoids redundant double-IPOP wrapping); the inner
    // restart_strategy is therefore left at its default.
    //
    // MaxPopulation=4096 (vs the cmaes_policy<> default of
    // dynamic_dimension -> MaxPop=512) so the decorator's deep IPOP
    // escalation can grow lambda past 512 on multimodal 2D cells under
    // the publication-mode max_iterations budget. Without this widening,
    // the inner policy's per-step buffer cap fires first and the harness
    // either segfaults (Release) or trips Eigen's MaxColsAtCompileTime
    // assertion (Debug).
    //
    // Reference:
    //   Auger, A. & Hansen, N. (2005). A Restart CMA Evolution Strategy
    //   with Increasing Population Size. CEC 2005. IPOP-CMA-ES doubles
    //   lambda at each restart trigger; with initial lambda = 8 (n=2
    //   bounded default of 4*n) the per-decorator restart_count needs
    //   ~9-10 doublings to cross 4096. Under the publication-mode
    //   max_iterations budget the decorator can occasionally exceed
    //   that ceiling on the most stagnation-prone cells; the catch
    //   below converts the actionable std::runtime_error from
    //   cmaes_policy::reset into a stalled bench row instead of an
    //   aborted process. The ceiling is intentionally finite -- the
    //   per-step buffers are stack-allocated up to MaxPopulation, so a
    //   gratuitously wide cap inflates per-policy memory residency for
    //   cells that never need it.
    if constexpr(is_global && is_bound)
    {
        typename restarting_policy<cmaes_policy<dynamic_dimension, 4096>>::options_type rcmaes_opts{};
        rcmaes_opts.inner.seed = seed;
        std::vector<trace_entry> trace;
        try
        {
            auto r = run_argmin_solver<restarting_policy<cmaes_policy<dynamic_dimension, 4096>>>(
                "restarting_cmaes", problem_name, prob, max_iterations, collect_trace, trace,
                config, rcmaes_opts);
            results.push_back(r);
            traces.push_back(std::move(trace));
        }
        catch(const std::runtime_error&)
        {
            // The decorator-side IPOP escalation drove the inner
            // cmaes_policy past its compile-time MaxPop cap; the
            // policy's reset/init guard threw rather than allow the
            // silent buffer-overflow UB. Skip the row for this
            // (problem, seed) cell -- subsequent seeds on the same
            // problem may still complete and contribute to the
            // 5-seed median.
        }
    }

    // COBYLA: constrained derivative-free (requires bounds + constraint values).
    if constexpr(constrained_values<Problem> && is_bound)
        run_constrained("cobyla", cobyla_policy{});

    // ISRES: global constrained (requires bounds + constraint values).
    //
    // Bench-row decision: keep ONE row pointing at the production alias.
    // ISRES alternative-variant per-row split is intentionally NOT
    // emitted from this harness. The wider argmin_bench produces the
    // single 'isres' row resolved through the production alias in
    // solver/isres_policy.h (currently aliased to
    // alternative::isres::nlopt_faithful_policy). Empirical per-variant
    // comparison runs through benchmarks/micro_isres.cpp instead;
    // bench_argmin.h stays variant-agnostic to keep the publication CSV
    // schema stable for downstream consumers (Dolan-More profiles,
    // phaseXX-baseline.csv, etc.).
    if constexpr(is_global && constrained_values<Problem> && is_bound)
    {
        typename isres_policy<>::options_type isres_opts{};
        isres_opts.seed = seed;
        std::vector<trace_entry> trace;
        auto r = run_argmin_solver<isres_policy<>, default_convergence>(
            "isres", problem_name, prob, max_iterations, collect_trace, trace,
            config, isres_opts);
        results.push_back(r);
        traces.push_back(std::move(trace));
    }
}

}

#endif
