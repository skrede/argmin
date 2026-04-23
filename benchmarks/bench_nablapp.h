#ifndef HPP_GUARD_NABLAPP_BENCHMARKS_BENCH_NABLAPP_H
#define HPP_GUARD_NABLAPP_BENCHMARKS_BENCH_NABLAPP_H

// Nablapp solver benchmark harness.
//
// Provides run_nablapp_solver() to run a single policy on a problem and
// produce a benchmark_result, and run_all_nablapp_solvers() to run all
// applicable nablapp policies on a given problem.

#include "bench_config.h"
#include "counting_problem.h"
#include "benchmark_result.h"
#include "trace_entry.h"
#include "problem_registry.h"

#include "nablapp/solver/basic_solver.h"
#include "nablapp/solver/convergence.h"
#include "nablapp/solver/lm_policy.h"
#include "nablapp/solver/cobyla_policy.h"
#include "nablapp/solver/isres_policy.h"
#include "nablapp/solver/mma_policy.h"
#include "nablapp/solver/gcmma_policy.h"
#include "nablapp/solver/bobyqa_policy.h"
#include "nablapp/solver/cmaes_policy.h"
#include "nablapp/solver/lbfgsb_policy.h"
#include "nablapp/solver/nw_sqp_policy.h"
#include "nablapp/solver/multistart_policy.h"
#include "nablapp/solver/restarting_policy.h"
#include "nablapp/solver/projected_gn_policy.h"
#include "nablapp/solver/kraft_slsqp_policy.h"
#include "nablapp/solver/byrd_lbfgsb_policy.h"
#include "nablapp/solver/filter_nw_sqp_policy.h"
#include "nablapp/solver/filter_slsqp_policy.h"
#include "nablapp/solver/projected_gradient_gn_policy.h"
#include "nablapp/solver/augmented_lagrangian_policy.h"

#include "nablapp/formulation/concepts.h"
#include "nablapp/result/status.h"
#include "nablapp/result/solve_result.h"
#include "nablapp/test_functions/problem_class.h"

#include <benchmark/benchmark.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <vector>

namespace nablapp::bench
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
    case solver_status::objective_stalled: return "objective_stalled";
    case solver_status::time_limit_reached: return "time_limit_reached";
    case solver_status::aborted:           return "aborted";
    case solver_status::running:           return "running";
    }
    return "unknown";
}

}

// Run a single nablapp solver policy on a problem, collecting result and
// optional per-iteration trace.
//
// Policy must be a valid basic_solver policy.
// Problem must provide value, gradient, dimension, optimal_value, initial_point.
//
// bench uses `default_convergence` uniformly across solver families; each
// solver sets its inner `objective_tolerance_criterion::stationarity_threshold`
// directly, and the composite kkt_residual E-measure (N&W 2e Definition 12.1,
// eq. 12.34) carries primal feasibility into the single termination test.
template <typename Policy, typename Convergence = default_convergence,
          typename Problem, typename... PolicyOpts>
auto run_nablapp_solver(std::string_view solver_name,
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
    counting_problem<Problem> wrapped{prob, counts};

    static constexpr int N = problem_dimension_v<Problem>;
    using rebound_policy = detail::rebind_policy_t<Policy, N>;

    auto x0 = prob.initial_point();
    solver_options<Convergence> opts{};
    opts.max_iterations = max_iterations;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

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
        opts.set_objective_threshold(1e-6);
        opts.set_step_threshold(1e-6);
        opts.set_stationarity_threshold(1e-4);
    }

    if(collect_trace)
    {
        // Step loop with trace collection (per D-09); rows populate the
        // D-C3 12-column publication schema. f_best is a running-min over
        // f_current; accuracy is |f_current - prob.optimal_value()|;
        // step_norm pulls from step_result.step_size (the policy-native
        // step length); kkt_residual pulls from step_result.kkt_residual
        // (nablapp's 31.1 E-measure composite, NaN when the policy does
        // not populate it).
        trace.clear();
        trace.resize(static_cast<std::size_t>(max_iterations));

        const double f_star = prob.optimal_value();
        double f_best_running = std::numeric_limits<double>::infinity();

        using wrapped_t = counting_problem<Problem>;
        basic_solver<rebound_policy, N, wrapped_t> solver(wrapped, x0, opts,
                                    std::forward<PolicyOpts>(policy_opts)...);

        auto t0 = std::chrono::high_resolution_clock::now();

        int iters = 0;
        solver_status final_status = solver_status::running;
        double final_obj{};

        for(int i = 0; i < max_iterations; ++i)
        {
            auto sr = solver.step();
            ++iters;

            const double f_current = static_cast<double>(sr.objective_value);
            f_best_running = std::min(f_best_running, f_current);

            const auto t_now = std::chrono::high_resolution_clock::now();
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

        auto t1 = std::chrono::high_resolution_clock::now();
        auto wall_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        trace.resize(static_cast<std::size_t>(iters));

        benchmark::DoNotOptimize(final_obj);

        return benchmark_result{
            .solver = solver_name,
            .library = "nablapp",
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
            .final_objective = final_obj,
            .known_optimum = prob.optimal_value(),
            .accuracy = std::abs(final_obj - prob.optimal_value()),
            .status = detail::status_string(final_status),
        };
    }
    else
    {
        // Use solve(opts) so the active opts.convergence (including any
        // per-solver stationarity_threshold overrides) gates termination.
        // The no-args solve() path falls through to stored_convergence_
        // which is always default-constructed default_convergence.
        using wrapped_t = counting_problem<Problem>;
        basic_solver<rebound_policy, N, wrapped_t> solver(wrapped, x0, opts,
                                    std::forward<PolicyOpts>(policy_opts)...);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = solver.solve(opts);
        auto t1 = std::chrono::high_resolution_clock::now();

        auto wall_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        benchmark::DoNotOptimize(result);

        return benchmark_result{
            .solver = solver_name,
            .library = "nablapp",
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
            .status = detail::status_string(result.status),
        };
    }
}

// Run all applicable nablapp solver policies on a problem.
//
// Uses concepts to determine which solvers are applicable:
//   - Unconstrained: lbfgsb_policy
//   - Bound-constrained: lbfgsb_policy, bobyqa_policy
//   - Inequality/equality/mixed: kraft_slsqp_policy, augmented_lagrangian_policy
//   - Inequality: mma_policy (inequality only, no equality)
//   - Equality/mixed: nw_sqp_policy
//   - Global: cmaes_policy, bobyqa_policy (if has bounds)
template <typename Problem>
void run_all_nablapp_solvers(
    std::string_view problem_name,
    const Problem& prob,
    int max_iterations,
    bool collect_trace,
    std::vector<benchmark_result>& results,
    std::vector<std::vector<trace_entry>>& traces,
    const bench_config& config,
    std::uint64_t seed = 42)
{
    (void)config;  // forwarded to run_nablapp_solver below; this scope only
                   // dispatches policies. Future plans branch here on
                   // config.the_mode for solver-set selection per CONTEXT D-A3.

    auto run = [&]<typename Policy>(std::string_view name, Policy) {
        std::vector<trace_entry> trace;
        auto r = run_nablapp_solver<Policy>(
            name, problem_name, prob, max_iterations, collect_trace, trace,
            config);
        results.push_back(r);
        traces.push_back(std::move(trace));
    };

    auto run_constrained = [&]<typename Policy>(std::string_view name, Policy) {
        std::vector<trace_entry> trace;
        auto r = run_nablapp_solver<Policy, default_convergence>(
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
        run("projected_gn", projected_gn_policy{});
        run("projected_gradient_gn", projected_gradient_gn_policy{});
    }

    // SLSQP: any constrained problem.
    if constexpr(is_constrained && differentiable<Problem> && is_bound)
        run_constrained("kraft_slsqp", kraft_slsqp_policy<>{});

    // MMA: inequality-constrained (no equality).
    if constexpr(is_constrained && differentiable<Problem> && is_bound)
    {
        if constexpr(requires { prob.num_equality(); })
        {
            // Only use MMA when problem has no equality constraints.
            // Check at compile time if possible via static member, otherwise runtime.
            if(prob.num_equality() == 0 && prob.num_inequality() > 0)
            {
                std::vector<trace_entry> trace;
                auto r = run_nablapp_solver<mma_policy<>, default_convergence>(
                    "mma", problem_name, prob, max_iterations, collect_trace, trace,
                    config);
                results.push_back(r);
                traces.push_back(std::move(trace));
            }
        }
    }

    // GCMMA: globally-convergent MMA; gate mirrors MMA (inequality-only).
    if constexpr(is_constrained && differentiable<Problem> && is_bound)
    {
        if constexpr(requires { prob.num_equality(); })
        {
            if(prob.num_equality() == 0 && prob.num_inequality() > 0)
            {
                std::vector<trace_entry> trace;
                auto r = run_nablapp_solver<gcmma_policy<>, default_convergence>(
                    "gcmma", problem_name, prob, max_iterations, collect_trace, trace,
                    config);
                results.push_back(r);
                traces.push_back(std::move(trace));
            }
        }
    }

    // N&W SQP: equality or mixed constrained.
    if constexpr(is_constrained && differentiable<Problem> && is_bound)
    {
        if constexpr(requires { prob.num_equality(); })
        {
            if(prob.num_equality() > 0)
            {
                std::vector<trace_entry> trace;
                auto r = run_nablapp_solver<nw_sqp_policy<>, default_convergence>(
                    "nw_sqp", problem_name, prob, max_iterations, collect_trace, trace,
                    config);
                results.push_back(r);
                traces.push_back(std::move(trace));
            }
        }
    }

    // Filter SLSQP: any constrained problem (filter-based L-BFGS SQP).
    if constexpr(is_constrained && differentiable<Problem> && is_bound)
        run_constrained("filter_slsqp", filter_slsqp_policy<>{});

    // Filter NW SQP: any constrained problem (filter-based dense BFGS SQP).
    if constexpr(is_constrained && differentiable<Problem> && is_bound)
        run_constrained("filter_nw_sqp", filter_nw_sqp_policy<>{});

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
        auto r = run_nablapp_solver<cmaes_policy<>>(
            "cmaes", problem_name, prob, max_iterations, collect_trace, trace,
            config, cmaes_opts);
        results.push_back(r);
        traces.push_back(std::move(trace));
    }

    // restarting_policy<cmaes_policy<>>: gate mirrors CMA-ES (global + bounds).
    // Instantiate with default inner options so the decorator's own restart policy
    // replaces CMA-ES IPOP (avoids redundant double-IPOP wrapping).
    if constexpr(is_global && is_bound)
        run("restarting_cmaes", restarting_policy<cmaes_policy<>>{});

    // COBYLA: constrained derivative-free (requires bounds + constraint values).
    if constexpr(constrained_values<Problem> && is_bound)
        run_constrained("cobyla", cobyla_policy{});

    // ISRES: global constrained (requires bounds + constraint values).
    if constexpr(is_global && constrained_values<Problem> && is_bound)
    {
        typename isres_policy<>::options_type isres_opts{};
        isres_opts.seed = seed;
        std::vector<trace_entry> trace;
        auto r = run_nablapp_solver<isres_policy<>, default_convergence>(
            "isres", problem_name, prob, max_iterations, collect_trace, trace,
            config, isres_opts);
        results.push_back(r);
        traces.push_back(std::move(trace));
    }
}

}

#endif
