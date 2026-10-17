#include "alloc_gate_support.h"

#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_trsqp_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"

#include "argmin/test_functions/small_dense.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>

namespace
{

namespace sensor = argmin::detail::bench;

using argmin::alloc_gate::check_steady_zero;
using argmin::alloc_gate::hs026_fixed_gate;
using argmin::alloc_gate::hs071_fixed;
using argmin::alloc_gate::hs071_fixed_gate;
using argmin::alloc_gate::run_armed;
using argmin::alloc_gate::undercap_rosenbrock;

argmin::solver_options<> sqp_gate_opts()
{
    argmin::solver_options<> opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-10);
    return opts;
}

// Assert the steps taken after a converged solve are allocation-free, in the
// scalar's own default options. The solve must have reached the solution, or
// the window would measure descent steps a second time.
template <template <typename, int> typename Policy, argmin::scalar Scalar>
void check_at_solution_zero()
{
    argmin::solver_options<argmin::default_convergence<Scalar>> opts;
    opts.max_iterations = 200;
    const auto r = sensor::measure_at_solution(
        Policy<Scalar, argmin::hs071<Scalar>::problem_dimension>{}, argmin::hs071<Scalar>{}, opts, 20);
    REQUIRE((r.solve_status == argmin::solver_status::converged
             || r.solve_status == argmin::solver_status::ftol_reached
             || r.solve_status == argmin::solver_status::stalled));
    CHECK(r.eigen_malloc == 0);
    CHECK(r.c_alloc == 0);
}

}

// nw_sqp across the fixed-N fixture family, spanning unconstrained (sd006)
// through mixed-constrained near-ceiling (sd024).
TEST_CASE("sqp_alloc_gate_nw", "[alloc-gate]")
{
    const auto opts = sqp_gate_opts();

    check_steady_zero(sensor::measure_steady(
        argmin::nw_sqp_policy<double, argmin::hs071<>::problem_dimension>{},
        argmin::hs071<>{}, opts, 10));
    check_steady_zero(sensor::measure_steady(
        argmin::nw_sqp_policy<double, argmin::sd006<>::problem_dimension>{},
        argmin::sd006<>{}, opts, 10));
    check_steady_zero(sensor::measure_steady(
        argmin::nw_sqp_policy<double, argmin::sd012<>::problem_dimension>{},
        argmin::sd012<>{}, opts, 10));
    check_steady_zero(sensor::measure_steady(
        argmin::nw_sqp_policy<double, argmin::sd024<>::problem_dimension>{},
        argmin::sd024<>{}, opts, 10));

    check_at_solution_zero<argmin::nw_sqp_policy, double>();
    check_at_solution_zero<argmin::nw_sqp_policy, float>();
}

// Runtime-m-below-cap nw_sqp gate: the declared constraint cap (12) exceeds the
// runtime constraint count (6), exercising the inline max-bounded multiplier
// storage carrying fewer active rows than capacity. The un-armed witness solve
// proves the under-cap fixture reaches its known optimum, so the zero-alloc
// pass is numerically non-vacuous.
TEST_CASE("sqp_alloc_gate_nw_bounded_m", "[alloc-gate]")
{
    const auto opts = sqp_gate_opts();

    check_steady_zero(sensor::measure_steady(
        argmin::nw_sqp_policy<double, undercap_rosenbrock::problem_dimension>{},
        undercap_rosenbrock{}, opts, 10));

    undercap_rosenbrock problem;
    auto x0 = problem.initial_point();
    argmin::step_budget_solver solver{
        argmin::nw_sqp_policy<double, undercap_rosenbrock::problem_dimension>{},
        problem, x0, opts};
    const auto result = solver.solve();
    const double f_gap = std::abs(result.objective_value - problem.optimal_value());
    CHECK(f_gap <= 1e-6);
    CHECK(result.constraint_violation <= 1e-6);
}

// filter_nw_sqp across the fixed-N fixture family.
TEST_CASE("sqp_alloc_gate_filter_nw", "[alloc-gate]")
{
    const auto opts = sqp_gate_opts();

    check_steady_zero(sensor::measure_steady(
        argmin::filter_nw_sqp_policy<double, argmin::hs071<>::problem_dimension>{},
        argmin::hs071<>{}, opts, 10));
    check_steady_zero(sensor::measure_steady(
        argmin::filter_nw_sqp_policy<double, argmin::sd012<>::problem_dimension>{},
        argmin::sd012<>{}, opts, 10));
    check_steady_zero(sensor::measure_steady(
        argmin::filter_nw_sqp_policy<double, argmin::sd024<>::problem_dimension>{},
        argmin::sd024<>{}, opts, 10));

    check_at_solution_zero<argmin::filter_nw_sqp_policy, double>();
    check_at_solution_zero<argmin::filter_nw_sqp_policy, float>();
}

// kraft_slsqp at fixed N: a steady-state step window plus a warm reset() held
// to the same zero-allocation bar as a steady step.
TEST_CASE("sqp_alloc_gate_kraft", "[alloc-gate]")
{
    hs071_fixed problem;
    Eigen::Vector<double, 4> x0{1.0, 5.0, 5.0, 1.0};
    const auto opts = sqp_gate_opts();

    argmin::step_budget_solver solver{argmin::kraft_slsqp_policy<double, 4>{},
                                      problem, x0, opts};

    // Warmup absorbs lazy first-push BFGS allocations.
    solver.step();
    solver.step();

    constexpr std::size_t hot_steps = 10;
    const auto counts = run_armed([&] {
        for(std::size_t i = 0; i < hot_steps; ++i)
            solver.step();
        solver.reset(x0);
        for(std::size_t i = 0; i < hot_steps; ++i)
            solver.step();
    });

    CHECK(counts.eigen == 0);
    CHECK(counts.c == 0);

    check_at_solution_zero<argmin::kraft_slsqp_policy, double>();
    check_at_solution_zero<argmin::kraft_slsqp_policy, float>();
}

// filter_slsqp at fixed N over the warm-started real-time regime: several ticks
// of a bounded progress-step run, each preceded by a warm reset kept strictly
// ahead of the zero-step restoration onset.
TEST_CASE("sqp_alloc_gate_filter_slsqp", "[alloc-gate]")
{
    hs071_fixed_gate problem;
    Eigen::Vector<double, 4> x0{1.0, 5.0, 5.0, 1.0};
    const auto opts = sqp_gate_opts();

    argmin::step_budget_solver solver{argmin::filter_slsqp_policy<double, 4>{},
                                      problem, x0, opts};

    solver.step();
    solver.step();

    constexpr std::size_t progress_steps = 6;
    constexpr std::size_t ticks = 3;
    const auto counts = run_armed([&] {
        for(std::size_t t = 0; t < ticks; ++t)
        {
            solver.reset(x0);
            for(std::size_t i = 0; i < progress_steps; ++i)
                solver.step();
        }
    });

    CHECK(counts.eigen == 0);
    CHECK(counts.c == 0);
}

// filter_trsqp at fixed N over the warm-started composite-step regime on the
// equality-only HS026 fixture (no box faces, so the free-set restart path is
// never engaged). The armed window spans >= 40 steps for a bounded-steady-state
// confirmation.
TEST_CASE("sqp_alloc_gate_filter_trsqp", "[alloc-gate]")
{
    hs026_fixed_gate problem;
    Eigen::Vector<double, 3> x0{-2.6, 2.0, 2.0};
    const auto opts = sqp_gate_opts();

    argmin::step_budget_solver solver{argmin::filter_trsqp_policy_accurate<double, 3>{},
                                      problem, x0, opts};

    solver.step();
    solver.step();

    constexpr std::size_t progress_steps = 7;
    constexpr std::size_t ticks = 6;
    const auto counts = run_armed([&] {
        for(std::size_t t = 0; t < ticks; ++t)
        {
            solver.reset(x0);
            for(std::size_t i = 0; i < progress_steps; ++i)
                solver.step();
        }
    });

    CHECK(counts.eigen == 0);
    CHECK(counts.c == 0);
}

// tr_sqp: the bare trust-region policy over the same HS026 composite-step
// window as filter_trsqp.
TEST_CASE("sqp_alloc_gate_tr_sqp", "[alloc-gate]")
{
    hs026_fixed_gate problem;
    Eigen::Vector<double, 3> x0{-2.6, 2.0, 2.0};
    const auto opts = sqp_gate_opts();

    argmin::step_budget_solver solver{
        argmin::tr_sqp_policy<double, 3, argmin::sqp_mode::accurate>{}, problem, x0, opts};

    solver.step();
    solver.step();

    constexpr std::size_t progress_steps = 7;
    constexpr std::size_t ticks = 6;
    const auto counts = run_armed([&] {
        for(std::size_t t = 0; t < ticks; ++t)
        {
            solver.reset(x0);
            for(std::size_t i = 0; i < progress_steps; ++i)
                solver.step();
        }
    });

    CHECK(counts.eigen == 0);
    CHECK(counts.c == 0);
}
