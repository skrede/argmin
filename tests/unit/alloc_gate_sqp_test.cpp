#include "alloc_gate_support.h"

#include "argmin/detail/lagrangian.h"
#include "argmin/detail/filter_restoration.h"

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
using argmin::alloc_gate::declared_degenerate_vertex;
using argmin::alloc_gate::degenerate_vertex;
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

// The first step after each unarmed reset lands on the vertex, where the multiplier estimate sees three active rows at
// N = 2 and takes the reduction to an independent row subset.
template <typename Policy, typename Problem>
void check_degenerate_vertex_zero(Policy policy, const Problem& problem)
{
    const Eigen::Vector2d x0 = problem.initial_point();
    argmin::step_budget_solver solver{policy, problem, x0, sqp_gate_opts()};
    REQUIRE(solver.solve().status == argmin::solver_status::converged);
    REQUIRE(solver.state().x.norm() < 1e-8);
    for(std::size_t tick = 0; tick < 3; ++tick)
    {
        solver.reset(x0);
        const auto counts = run_armed([&] {
            for(std::size_t i = 0; i < 5; ++i)
                solver.step();
        });
        CHECK(counts.eigen == 0);
        CHECK(counts.c == 0);
    }
}

// solver_core also builds the placeholder state from a destructor while unwinding, where a throwing allocation would
// terminate the program.
template <template <typename, int> typename Policy>
void check_rejected_start_zero()
{
    const argmin::hs071<> problem;
    const Eigen::VectorXd rejected = Eigen::VectorXd::Constant(3, 1.0);
    const auto opts = sqp_gate_opts();
    argmin::solver_status status = argmin::solver_status::running;
    const auto counts = run_armed([&] {
        argmin::step_budget_solver solver{Policy<double, argmin::hs071<>::problem_dimension>{}, problem, rejected, opts};
        status = solver.status();
    });
    CHECK(status == argmin::solver_status::invalid_problem);
    CHECK(counts.eigen == 0);
    CHECK(counts.c == 0);
}

// A counting window over the L1 restoration leaf itself, on the workspace the policy's state was sized with, reaches
// restoration without waiting for the filter to block a step. From the lower-bound corner of HS071 both constraint
// rows are violated, so every pass runs the descent, the backtracking and the step-budget exit.
template <typename Policy>
void check_restoration_zero(Policy policy)
{
    const hs071_fixed problem;
    const Eigen::Vector<double, 4> x0 = Eigen::Vector<double, 4>::Ones();
    auto s = policy.init(problem, x0, sqp_gate_opts());
    const argmin::detail::restoration_adapter<double, 4, hs071_fixed> adapter{s.n_eq, s.n_ineq, problem, s.restoration_ws.c_all, s.restoration_ws.J_all};
    argmin::detail::restoration_result<double, 4> r{};
    const auto counts = run_armed([&] {
        for(std::size_t i = 0; i < 3; ++i)
            r = argmin::detail::restore_l1<double, 4>(adapter, s.restoration_ws, x0, s.g, s.lower, s.upper);
    });
    CHECK(counts.eigen == 0);
    CHECK(counts.c == 0);
    CHECK(r.iterations_used == 10);
    CHECK(r.constraint_violation < 1e-4);
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

// The estimate reduces the vertex's three active rows inside a fixed-N workspace. tr_sqp, filter_trsqp and filter_slsqp
// step there with the constraint count undeclared; nw_sqp and filter_nw_sqp size their QP result storage from the
// declared count, so they step there with it declared. At -O3 with -march=native, filter_slsqp's filter rejects the
// converged vertex's rounding-sized step in one tick and enters restoration.
TEST_CASE("sqp_alloc_gate_degenerate_vertex", "[alloc-gate]")
{
    const Eigen::Vector2d g(2.0, 2.0);
    const Eigen::Matrix<double, Eigen::Dynamic, 2> J_eq(0, 2);
    const Eigen::Matrix<double, Eigen::Dynamic, 2> J_ineq = degenerate_vertex::normals();
    const Eigen::VectorXd c_ineq = Eigen::VectorXd::Zero(3);
    argmin::detail::kkt_multiplier_workspace<double, 2> ws;
    Eigen::VectorXd lam_eq(0);
    Eigen::VectorXd mu_ineq(3);
    const auto counts = run_armed([&] {
        for(std::size_t i = 0; i < 4; ++i)
            argmin::detail::compute_kkt_multipliers_active_set<double, 2>(g, J_eq, J_ineq, c_ineq, ws, lam_eq, mu_ineq);
    });
    CHECK(counts.eigen == 0);
    CHECK(counts.c == 0);
    CHECK((mu_ineq.array() != 0.0).count() == 2);

    check_degenerate_vertex_zero(argmin::tr_sqp_policy<double, 2>{}, degenerate_vertex{});
    check_degenerate_vertex_zero(argmin::filter_trsqp_policy<double, 2>{}, degenerate_vertex{});
    check_degenerate_vertex_zero(argmin::nw_sqp_policy<double, 2>{}, declared_degenerate_vertex{});
    check_degenerate_vertex_zero(argmin::filter_nw_sqp_policy<double, 2>{}, declared_degenerate_vertex{});
    check_degenerate_vertex_zero(argmin::filter_slsqp_policy<double, 2>{}, degenerate_vertex{});
}

// filter_slsqp and filter_nw_sqp share the L1 leaf of the hybrid restoration strategy.
TEST_CASE("sqp_alloc_gate_restoration", "[alloc-gate]")
{
    check_restoration_zero(argmin::filter_slsqp_policy<double, 4>{});
    check_restoration_zero(argmin::filter_nw_sqp_policy<double, 4>{});
}

// A start of the wrong length leaves each fixed-N solver holding only its placeholder state.
TEST_CASE("sqp_alloc_gate_rejected_start", "[alloc-gate]")
{
    check_rejected_start_zero<argmin::kraft_slsqp_policy>();
    check_rejected_start_zero<argmin::nw_sqp_policy>();
    check_rejected_start_zero<argmin::filter_slsqp_policy>();
    check_rejected_start_zero<argmin::filter_nw_sqp_policy>();
}
