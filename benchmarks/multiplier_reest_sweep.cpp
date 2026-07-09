// Strided active-set Lagrange-multiplier re-estimation sweep harness.
//
// Parametric Google Benchmark sweep across the cross product of:
//   - policy:        kraft_slsqp, nw_sqp, filter_slsqp, filter_nw_sqp
//                    (single-mode after the line-search _fast collapse)
//   - problem:       HS071 (n=4, m_eq=1, m_ineq=1, mixed)
//                    HS026 (n=3, m_eq=1, m_ineq=0, equality + curvature)
//                    HS028 (n=3, m_eq=1, m_ineq=0, equality, f* = 0)
//                    HS043 (n=4, m_eq=0, m_ineq=3, inequality)
//   - stride k:      {1, 2, 3, 5, 8, 10} via ArgsProduct
//
// Each cell solves the problem under the (policy, k) configuration and
// reports per-step Google Benchmark wall median over 5 reps. The stride
// k is read from state.range(0); the policy's options_type
// .multiplier_reest_every_k is assigned before each solve. Counters
// surface iter_count and the policy-status code for SR analysis; the
// median wall is used to verify the k=1 single-mode default against
// strided alternatives.
//
// The harness is read-only on the argmin library and is bench-only —
// it is not linked into argmin::argmin and consumers do not pick it
// up via FetchContent.
//
// Reference: Hock, W. and Schittkowski, K. (1981) "Test Examples for
//            Nonlinear Programming Codes", Lecture Notes in Economics
//            and Mathematical Systems vol. 187, Springer.
//            Bertsekas 1996 §4.2 (stale-multiplier reuse rationale).
//            N&W 2e §18.3 + Algorithm 18.3 (working-set identification).

#include "argmin/solver/sqp_mode.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace
{

bool acceptable_status(argmin::solver_status s)
{
    using argmin::solver_status;
    return s == solver_status::converged
        || s == solver_status::stalled
        || s == solver_status::xtol_reached
        || s == solver_status::ftol_reached;
}

// One sweep cell: instantiate a step_budget_solver with Policy{}, set the
// problem's multiplier_reest_every_k stride from state.range(0), and
// run the solve in the timed region. Iter count and final-status
// surface as state.counters; wall comes from Google Benchmark.
//
// Problem is a concrete instantiation (e.g., hs026<double>); the type
// is named directly rather than via a template-template-parameter
// because BENCHMARK_TEMPLATE forwards type arguments, not template
// arguments.
template <typename Policy, typename Problem>
void sweep_cell(benchmark::State& state)
{
    Problem problem;
    const auto x0 = problem.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-10);

    const std::size_t k = static_cast<std::size_t>(state.range(0));

    // Warmup once outside the timed region to absorb any first-touch
    // allocator pages and lazy Eigen specializations on the deepest
    // template path.
    {
        Policy policy;
        policy.options.multiplier_reest_every_k = k;
        argmin::step_budget_solver solver{policy, problem, x0, opts};
        auto warm = solver.solve();
        benchmark::DoNotOptimize(warm);
    }

    std::uint32_t iters_last = 0;
    int status_last = 0;
    double objective_last = 0.0;
    double accuracy_last = 0.0;
    double cv_last = 0.0;
    double correctness_ok_last = 0.0;
    for(auto _ : state)
    {
        Policy policy;
        policy.options.multiplier_reest_every_k = k;
        argmin::step_budget_solver solver{policy, problem, x0, opts};
        auto result = solver.solve();
        benchmark::DoNotOptimize(result);
        iters_last = static_cast<std::uint32_t>(result.iterations);
        status_last = static_cast<int>(result.status);
        objective_last = result.objective_value;
        accuracy_last = std::abs(result.objective_value - problem.optimal_value());
        cv_last = solver.constraint_violation();
        const bool objective_ok = std::isfinite(accuracy_last) && accuracy_last <= 1e-6;
        const bool feasibility_ok = std::isfinite(cv_last) && cv_last <= 1e-6;
        correctness_ok_last =
            (objective_ok && feasibility_ok && acceptable_status(result.status)) ? 1.0 : 0.0;
    }

    state.counters["iter_count"] = benchmark::Counter(
        static_cast<double>(iters_last));
    state.counters["status_code"] = benchmark::Counter(
        static_cast<double>(status_last));
    state.counters["objective"] = benchmark::Counter(objective_last);
    state.counters["accuracy"] = benchmark::Counter(accuracy_last);
    state.counters["constraint_violation"] = benchmark::Counter(cv_last);
    state.counters["correctness_gate_present"] = benchmark::Counter(1.0);
    state.counters["correctness_ok"] = benchmark::Counter(correctness_ok_last);
}

// Stride axis: k in {1, 2, 3, 5, 8, 10}. Repetitions=5 with
// ReportAggregatesOnly=true so each (policy, mode, problem, k) cell
// emits median + mean + stddev rows instead of 5 individual rows.
inline void configure_stride_axis(::benchmark::Benchmark* b)
{
    b->ArgsProduct({{1, 2, 3, 5, 8, 10}})
     ->Repetitions(5)
     ->ReportAggregatesOnly(true)
     ->Unit(benchmark::kMicrosecond);
}

}

// Aliases pulling in the single-mode policy specializations for each
// problem's compile-time dimension. Each problem fixes its dimension
// at the type level (HS026/HS028 = 3, HS043/HS071 = 4); the alias
// list is generated by hand because BENCHMARK_TEMPLATE takes type
// arguments individually rather than a parameter pack.

using kraft_3   = argmin::kraft_slsqp_policy_accurate<3>;
using kraft_4   = argmin::kraft_slsqp_policy_accurate<4>;

using nw_3      = argmin::nw_sqp_policy_accurate<3>;
using nw_4      = argmin::nw_sqp_policy_accurate<4>;

using fslsqp_3  = argmin::filter_slsqp_policy_accurate<3>;
using fslsqp_4  = argmin::filter_slsqp_policy_accurate<4>;

using fnw_3     = argmin::filter_nw_sqp_policy_accurate<3>;
using fnw_4     = argmin::filter_nw_sqp_policy_accurate<4>;

// Per-problem concrete instantiations (named here so BENCHMARK_TEMPLATE
// can take them by type).
using hs026_d = argmin::hs026<double>;
using hs028_d = argmin::hs028<double>;
using hs043_d = argmin::hs043<double>;
using hs071_d = argmin::hs071<double>;

// HS026 (n=3, equality only, curvature-driven convergence)
BENCHMARK_TEMPLATE(sweep_cell, kraft_3,  hs026_d)
    ->Name("kraft_slsqp_accurate/HS026")  ->Apply(configure_stride_axis);
BENCHMARK_TEMPLATE(sweep_cell, nw_3,     hs026_d)
    ->Name("nw_sqp_accurate/HS026")       ->Apply(configure_stride_axis);
BENCHMARK_TEMPLATE(sweep_cell, fslsqp_3, hs026_d)
    ->Name("filter_slsqp_accurate/HS026") ->Apply(configure_stride_axis);
BENCHMARK_TEMPLATE(sweep_cell, fnw_3,    hs026_d)
    ->Name("filter_nw_sqp_accurate/HS026")->Apply(configure_stride_axis);

// HS028 (n=3, equality only, f* = 0)
BENCHMARK_TEMPLATE(sweep_cell, kraft_3,  hs028_d)
    ->Name("kraft_slsqp_accurate/HS028")  ->Apply(configure_stride_axis);
BENCHMARK_TEMPLATE(sweep_cell, nw_3,     hs028_d)
    ->Name("nw_sqp_accurate/HS028")       ->Apply(configure_stride_axis);
BENCHMARK_TEMPLATE(sweep_cell, fslsqp_3, hs028_d)
    ->Name("filter_slsqp_accurate/HS028") ->Apply(configure_stride_axis);
BENCHMARK_TEMPLATE(sweep_cell, fnw_3,    hs028_d)
    ->Name("filter_nw_sqp_accurate/HS028")->Apply(configure_stride_axis);

// HS043 (n=4, inequality only)
BENCHMARK_TEMPLATE(sweep_cell, kraft_4,  hs043_d)
    ->Name("kraft_slsqp_accurate/HS043")  ->Apply(configure_stride_axis);
BENCHMARK_TEMPLATE(sweep_cell, nw_4,     hs043_d)
    ->Name("nw_sqp_accurate/HS043")       ->Apply(configure_stride_axis);
BENCHMARK_TEMPLATE(sweep_cell, fslsqp_4, hs043_d)
    ->Name("filter_slsqp_accurate/HS043") ->Apply(configure_stride_axis);
BENCHMARK_TEMPLATE(sweep_cell, fnw_4,    hs043_d)
    ->Name("filter_nw_sqp_accurate/HS043")->Apply(configure_stride_axis);

// HS071 (n=4, mixed equality + inequality)
BENCHMARK_TEMPLATE(sweep_cell, kraft_4,  hs071_d)
    ->Name("kraft_slsqp_accurate/HS071")  ->Apply(configure_stride_axis);
BENCHMARK_TEMPLATE(sweep_cell, nw_4,     hs071_d)
    ->Name("nw_sqp_accurate/HS071")       ->Apply(configure_stride_axis);
BENCHMARK_TEMPLATE(sweep_cell, fslsqp_4, hs071_d)
    ->Name("filter_slsqp_accurate/HS071") ->Apply(configure_stride_axis);
BENCHMARK_TEMPLATE(sweep_cell, fnw_4,    hs071_d)
    ->Name("filter_nw_sqp_accurate/HS071")->Apply(configure_stride_axis);

int main(int argc, char** argv)
{
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::AddCustomContext("provenance_id", "multiplier-reest-sweep-local");
    ::benchmark::AddCustomContext("grid_id", "multiplier_reest_every_k");
    ::benchmark::AddCustomContext("grid_axis_multiplier_reest_every_k", "1,2,3,5,8,10");
    ::benchmark::AddCustomContext("selected_multiplier_reest_every_k", "1");
    ::benchmark::AddCustomContext(
        "correctness_witness",
        "each row emits correctness_ok from objective, feasibility, and status gates");
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
