#ifndef HPP_GUARD_NABLAPP_BENCHMARKS_BENCH_NABLAPP_H
#define HPP_GUARD_NABLAPP_BENCHMARKS_BENCH_NABLAPP_H

// Nablapp solver benchmark harness.
//
// Provides run_nablapp_solver() to run a single policy on a problem and
// produce a benchmark_result, and run_all_nablapp_solvers() to run all
// applicable nablapp policies on a given problem.

#include "benchmark_result.h"
#include "trace_entry.h"
#include "problem_registry.h"

#include "nablapp/solver/basic_solver.h"
#include "nablapp/solver/convergence.h"
#include "nablapp/solver/cobyla_policy.h"
#include "nablapp/solver/isres_policy.h"
#include "nablapp/solver/bobyqa_policy.h"
#include "nablapp/solver/cmaes_policy.h"
#include "nablapp/solver/lbfgsb_policy.h"
#include "nablapp/solver/mma_policy.h"
#include "nablapp/solver/nw_sqp_policy.h"
#include "nablapp/solver/kraft_slsqp_policy.h"
#include "nablapp/solver/augmented_lagrangian_policy.h"

#include "nablapp/formulation/concepts.h"
#include "nablapp/result/status.h"
#include "nablapp/result/solve_result.h"
#include "nablapp/test_functions/problem_class.h"

#include <benchmark/benchmark.h>

#include <chrono>
#include <cmath>
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
// Convergence selects the convergence policy: default_convergence for
// unconstrained solvers, constrained_convergence for constrained solvers.
template <typename Policy, typename Convergence = default_convergence,
          typename Problem, typename... PolicyOpts>
auto run_nablapp_solver(std::string_view solver_name,
                        std::string_view problem_name,
                        const Problem& prob,
                        int max_iterations,
                        bool collect_trace,
                        std::vector<trace_entry>& trace,
                        PolicyOpts&&... policy_opts) -> benchmark_result
{
    static constexpr int N = problem_dimension_v<Problem>;
    using rebound_policy = detail::rebind_policy_t<Policy, N>;

    auto x0 = prob.initial_point();
    solver_options<Convergence> opts{};
    opts.max_iterations = max_iterations;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    if constexpr(requires(solver_options<Convergence>& o) { o.set_feasibility_threshold(1e-4); })
    {
        opts.set_feasibility_threshold(1e-4);
        // Relax gradient threshold for constrained solvers: the raw objective
        // gradient is nonzero at a constrained optimum (only the Lagrangian
        // gradient vanishes), and penalty methods (augmented Lagrangian)
        // cycle the augmented gradient during multiplier updates.
        opts.set_gradient_threshold(1e-2);
        // Relax objective tolerance and disable stationarity gate. Penalty
        // methods produce objective jumps that prevent tight ftol from
        // firing; the feasibility gate above is the primary convergence
        // signal for constrained solvers.
        opts.set_objective_threshold(1e-6);
        opts.set_step_threshold(1e-6);
        std::get<objective_tolerance_criterion>(
            opts.convergence.inner.criteria).stationarity_threshold = 1e2;
    }

    if(collect_trace)
    {
        // Step loop with trace collection (per D-09).
        trace.clear();
        trace.resize(static_cast<std::size_t>(max_iterations));

        basic_solver<rebound_policy, N, Problem> solver(prob, x0, opts,
                                    std::forward<PolicyOpts>(policy_opts)...);

        auto t0 = std::chrono::high_resolution_clock::now();

        int iters = 0;
        solver_status final_status = solver_status::running;
        double final_obj{};

        for(int i = 0; i < max_iterations; ++i)
        {
            auto sr = solver.step();
            ++iters;

            trace[static_cast<std::size_t>(i)] = trace_entry{
                .iteration = i,
                .objective_value = sr.objective_value,
                .gradient_norm = sr.gradient_norm,
                .constraint_violation = sr.constraint_violation,
            };

            final_obj = sr.objective_value;

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
            .f_evals = iters,
            .g_evals = iters,
            .wall_time_us = wall_us,
            .final_objective = final_obj,
            .known_optimum = prob.optimal_value(),
            .accuracy = std::abs(final_obj - prob.optimal_value()),
            .status = detail::status_string(final_status),
        };
    }
    else
    {
        // Use solve() for simplicity when no trace needed.
        basic_solver<rebound_policy, N, Problem> solver(prob, x0, opts,
                                    std::forward<PolicyOpts>(policy_opts)...);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = solver.solve();
        auto t1 = std::chrono::high_resolution_clock::now();

        auto wall_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        benchmark::DoNotOptimize(result);

        return benchmark_result{
            .solver = solver_name,
            .library = "nablapp",
            .problem = problem_name,
            .pclass = prob.pclass,
            .dimension = prob.dimension(),
            .f_evals = static_cast<int>(result.function_evaluations),
            .g_evals = static_cast<int>(result.function_evaluations),
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
    std::uint64_t seed = 42)
{
    auto run = [&]<typename Policy>(std::string_view name, Policy) {
        std::vector<trace_entry> trace;
        auto r = run_nablapp_solver<Policy>(
            name, problem_name, prob, max_iterations, collect_trace, trace);
        results.push_back(r);
        traces.push_back(std::move(trace));
    };

    auto run_constrained = [&]<typename Policy>(std::string_view name, Policy) {
        std::vector<trace_entry> trace;
        auto r = run_nablapp_solver<Policy, constrained_convergence>(
            name, problem_name, prob, max_iterations, collect_trace, trace);
        results.push_back(r);
        traces.push_back(std::move(trace));
    };

    constexpr bool is_bound = bound_constrained<Problem>;
    constexpr bool is_constrained = constrained<Problem>;
    constexpr bool is_global = has_class(Problem::pclass, problem_class::global);

    // L-BFGS-B: unconstrained or bound-constrained (not general constrained).
    if constexpr(differentiable<Problem> && !is_constrained)
        run("lbfgsb", lbfgsb_policy<>{});

    // BOBYQA: bound-constrained or global with bounds.
    if constexpr(is_bound && !is_constrained)
        run("bobyqa", bobyqa_policy<>{});

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
                auto r = run_nablapp_solver<mma_policy<>, constrained_convergence>(
                    "mma", problem_name, prob, max_iterations, collect_trace, trace);
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
                auto r = run_nablapp_solver<nw_sqp_policy<>, constrained_convergence>(
                    "nw_sqp", problem_name, prob, max_iterations, collect_trace, trace);
                results.push_back(r);
                traces.push_back(std::move(trace));
            }
        }
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
        auto r = run_nablapp_solver<cmaes_policy<>>(
            "cmaes", problem_name, prob, max_iterations, collect_trace, trace,
            cmaes_opts);
        results.push_back(r);
        traces.push_back(std::move(trace));
    }

    // COBYLA: constrained derivative-free (requires bounds + constraint values).
    if constexpr(constrained_values<Problem> && is_bound)
        run_constrained("cobyla", cobyla_policy{});

    // ISRES: global constrained (requires bounds + constraint values).
    if constexpr(is_global && constrained_values<Problem> && is_bound)
    {
        typename isres_policy<>::options_type isres_opts{};
        isres_opts.seed = seed;
        std::vector<trace_entry> trace;
        auto r = run_nablapp_solver<isres_policy<>, constrained_convergence>(
            "isres", problem_name, prob, max_iterations, collect_trace, trace,
            isres_opts);
        results.push_back(r);
        traces.push_back(std::move(trace));
    }
}

}

#endif
