// Zero-allocation correctness gates for the real-time-claimed policies.
//
// Every case arms the dual allocation sensor (Sensor A: the Eigen-native
// counting eigen_assert; Sensor B: the glibc C-allocator overrides on Linux)
// across a steady-state hot-loop window and asserts the armed window is
// allocation-free. These are argmin-only gates -- no NLopt, no external
// comparison library -- so they run under the plain `dev` preset that ships no
// comparison libraries.
//
// The gate target defines BOTH ARGMIN_BENCH_TRACE_ALLOC=1 (arming the LIVE
// sensor branch of the diagnostics header) and EIGEN_RUNTIME_NO_MALLOC (routing
// Eigen's runtime malloc gate into the counting hook). Without
// ARGMIN_BENCH_TRACE_ALLOC the header compiles to its no-op twin and every read
// is a vacuous 0; the alloc_sensor_liveness_canary case is the runtime proof
// that the LIVE branch is selected -- it counts an armed Eigen allocation and
// fails loud if the sensor is blind.
//
// This diagnostics header must precede any other Eigen-including header so its
// counting eigen_assert is defined before Eigen/Core is first parsed.
#include "argmin/detail/diagnostics/alloc_counter.h"
#include "argmin/detail/diagnostics/steady_state_driver.h"

#include "alloc_gate_fixtures.h"

#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/solver/filter_trsqp_policy.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/byrd_lbfgsb_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/augmented_lagrangian_policy.h"
#include "argmin/solver/projected_gn_policy.h"
#include "argmin/solver/projected_gradient_gn_policy.h"
#include "argmin/solver/lm_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/detail/lbfgsb_direction.h"

#include "argmin/test_functions/hock_schittkowski.h"
#include "argmin/test_functions/small_dense.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>

namespace
{

namespace sensor = argmin::detail::bench;

using argmin::alloc_gate::bounded_linear_ls_fixed2;
using argmin::alloc_gate::bounded_quadratic_fixed;
using argmin::alloc_gate::hs026_fixed_gate;
using argmin::alloc_gate::hs071_fixed;
using argmin::alloc_gate::hs071_fixed_gate;
using argmin::alloc_gate::rosenbrock_fixed;
using argmin::alloc_gate::rosenbrock_ls_fixed;
using argmin::alloc_gate::undercap_rosenbrock;

struct window_counts
{
    std::size_t eigen;
    std::size_t c;
};

// Arm the sensor, run the measurement body, disarm, and return the observed
// counts. body() runs entirely inside the armed window; anything it observes
// (path-entry flags, mu changes) is captured through its own captures.
template <typename Body>
window_counts run_armed(Body&& body)
{
    sensor::reset_alloc_count();
    sensor::arm_alloc_trace();
    body();
    sensor::disarm_alloc_trace();
    return {sensor::read_eigen_malloc_count(), sensor::read_c_alloc_count()};
}

argmin::solver_options<> sqp_gate_opts()
{
    argmin::solver_options<> opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-10);
    return opts;
}

argmin::solver_options<> lm_gate_opts()
{
    argmin::solver_options<> opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-12);
    opts.set_objective_threshold(1e-14);
    opts.set_step_threshold(1e-14);
    return opts;
}

// Assert an armed steady-state window drove a pre-convergence, allocation-free
// hot loop.
void check_steady_zero(const sensor::steady_state_result& r)
{
    CHECK(r.eigen_malloc == 0);
    CHECK(r.c_alloc == 0);
    CHECK_FALSE(r.terminated_early);
}

// MSVC-safe defeat of dead-store elimination on the canary allocation: writing
// the heap pointer to a volatile sink forces the allocation to be materialized
// without the GCC/Clang-only inline-asm escape.
inline void sink_pointer(const void* p) noexcept
{
    static volatile const void* sink;
    sink = p;
    (void)sink;
}

}

// Sensor liveness canary. An Eigen allocation inside an armed region MUST be
// counted by at least one sensor; this is the exact false-negative the retired
// global-operator-new gate let pass in Release. A pass here proves the LIVE
// sensor branch is compiled in (the gate target defines ARGMIN_BENCH_TRACE_ALLOC)
// -- against the no-op twin the read would be 0 and this fails.
TEST_CASE("alloc_sensor_liveness_canary", "[canary]")
{
    const auto counts = run_armed([] {
        Eigen::VectorXd v(100);
        v.setConstant(1.0);
        sink_pointer(v.data());
    });

    CHECK((counts.eigen + counts.c) >= 1);
}

// nw_sqp across the fixed-N fixture family, spanning unconstrained (sd006)
// through mixed-constrained near-ceiling (sd024).
TEST_CASE("sqp_alloc_gate_nw", "[alloc-gate]")
{
    const auto opts = sqp_gate_opts();

    check_steady_zero(sensor::measure_steady(
        argmin::nw_sqp_policy<argmin::hs071<>::problem_dimension>{},
        argmin::hs071<>{}, opts, 10));
    check_steady_zero(sensor::measure_steady(
        argmin::nw_sqp_policy<argmin::sd006<>::problem_dimension>{},
        argmin::sd006<>{}, opts, 10));
    check_steady_zero(sensor::measure_steady(
        argmin::nw_sqp_policy<argmin::sd012<>::problem_dimension>{},
        argmin::sd012<>{}, opts, 10));
    check_steady_zero(sensor::measure_steady(
        argmin::nw_sqp_policy<argmin::sd024<>::problem_dimension>{},
        argmin::sd024<>{}, opts, 10));
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
        argmin::nw_sqp_policy<undercap_rosenbrock::problem_dimension>{},
        undercap_rosenbrock{}, opts, 10));

    undercap_rosenbrock problem;
    auto x0 = problem.initial_point();
    argmin::step_budget_solver solver{
        argmin::nw_sqp_policy<undercap_rosenbrock::problem_dimension>{},
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
        argmin::filter_nw_sqp_policy<argmin::hs071<>::problem_dimension>{},
        argmin::hs071<>{}, opts, 10));
    check_steady_zero(sensor::measure_steady(
        argmin::filter_nw_sqp_policy<argmin::sd012<>::problem_dimension>{},
        argmin::sd012<>{}, opts, 10));
    check_steady_zero(sensor::measure_steady(
        argmin::filter_nw_sqp_policy<argmin::sd024<>::problem_dimension>{},
        argmin::sd024<>{}, opts, 10));
}

// Levenberg-Marquardt on the 2D Rosenbrock least-squares fixture and the 12D
// extended-Rosenbrock least-squares fixture.
TEST_CASE("alloc_gate_lm", "[alloc-gate]")
{
    const auto opts = lm_gate_opts();

    check_steady_zero(sensor::measure_steady(
        argmin::lm_policy<2>{}, rosenbrock_ls_fixed{}, opts, 10));
    check_steady_zero(sensor::measure_steady(
        argmin::lm_policy<argmin::sd_ls012<>::problem_dimension>{},
        argmin::sd_ls012<>{}, opts, 10));
}

// kraft_slsqp at fixed N: a steady-state step window plus a warm reset() held
// to the same zero-allocation bar as a steady step.
TEST_CASE("sqp_alloc_gate_kraft", "[alloc-gate]")
{
    hs071_fixed problem;
    Eigen::Vector<double, 4> x0{1.0, 5.0, 5.0, 1.0};
    const auto opts = sqp_gate_opts();

    argmin::step_budget_solver solver{argmin::kraft_slsqp_policy<4>{},
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
}

// filter_slsqp at fixed N over the warm-started real-time regime: several ticks
// of a bounded progress-step run, each preceded by a warm reset kept strictly
// ahead of the zero-step restoration onset.
TEST_CASE("sqp_alloc_gate_filter_slsqp", "[alloc-gate]")
{
    hs071_fixed_gate problem;
    Eigen::Vector<double, 4> x0{1.0, 5.0, 5.0, 1.0};
    const auto opts = sqp_gate_opts();

    argmin::step_budget_solver solver{argmin::filter_slsqp_policy<4>{},
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

    argmin::step_budget_solver solver{argmin::filter_trsqp_policy_accurate<3>{},
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
        argmin::tr_sqp_policy<3, argmin::sqp_mode::accurate>{}, problem, x0, opts};

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

namespace
{

// Shared body of the two L-BFGS-B-family gates: a wide-bounds all-free scenario
// and a bound-active scenario carrying a path-entry assertion so a zero-mode
// gate can never certify the bound-active claim vacuously.
template <typename Policy>
void run_lbfgsb_family_gate()
{
    const auto opts = [] {
        argmin::solver_options<> o;
        o.max_iterations = 200;
        o.set_gradient_threshold(1e-8);
        o.set_objective_threshold(1e-10);
        o.set_step_threshold(1e-10);
        return o;
    }();

    // Scenario 1: wide-bounds all-free fast path.
    {
        rosenbrock_fixed problem;
        Eigen::Vector<double, 2> x0{-1.2, 1.0};
        const Eigen::VectorXd x0_reset = x0;

        argmin::step_budget_solver solver{Policy{}, problem, x0, opts};
        solver.step();
        solver.step();

        constexpr std::size_t hot_steps = 10;
        const auto counts = run_armed([&] {
            for(std::size_t i = 0; i < hot_steps; ++i)
                solver.step();
            solver.reset(x0_reset);
            for(std::size_t i = 0; i < hot_steps; ++i)
                solver.step();
        });

        CHECK(counts.eigen == 0);
        CHECK(counts.c == 0);
    }

    // Scenario 2: bound-active path (GCP -> free-variable subspace ->
    // reduced_hessian / multiply).
    {
        bounded_quadratic_fixed problem;
        Eigen::Vector<double, 2> x0{0.1, 0.1};
        const Eigen::VectorXd x0_reset = x0;

        argmin::step_budget_solver solver{Policy{}, problem, x0, opts};
        solver.step();
        solver.step();

        std::size_t bound_active_steps = 0;
        auto observe_path = [&] {
            const auto& st = solver.state();
            if(!argmin::detail::all_variables_free<double, 2>(
                   st.x, st.g, st.lower, st.upper))
                ++bound_active_steps;
        };

        constexpr std::size_t hot_steps = 10;
        const auto counts = run_armed([&] {
            for(std::size_t i = 0; i < hot_steps; ++i)
            {
                solver.step();
                observe_path();
            }
            solver.reset(x0_reset);
            for(std::size_t i = 0; i < hot_steps; ++i)
            {
                solver.step();
                observe_path();
            }
        });

        // Path-entry assertion: without it a zero-mode gate would certify the
        // bound-active claim vacuously.
        CHECK(bound_active_steps > 0);
        CHECK(counts.eigen == 0);
        CHECK(counts.c == 0);
    }
}

}

TEST_CASE("alloc_gate_byrd_lbfgsb", "[alloc-gate]")
{
    run_lbfgsb_family_gate<argmin::byrd_lbfgsb_policy<2>>();
}

TEST_CASE("alloc_gate_lbfgsb", "[alloc-gate]")
{
    run_lbfgsb_family_gate<argmin::lbfgsb_policy<2>>();
}

// augmented_lagrangian outer method-of-multipliers path, armed across a window
// that includes at least one penalty (mu) reduction -- the former inner-solver
// reconstruction path (now a reset_clear on the persisted solver). A warm
// reset() sits inside the armed window too. The mu-change assertion keeps the
// zero-mode pass non-vacuous.
TEST_CASE("alloc_gate_augmented_lagrangian", "[alloc-gate]")
{
    argmin::hs071<> problem;
    Eigen::Vector<double, 4> x0 = problem.initial_point();

    argmin::solver_options<> opts;
    opts.max_iterations = 400;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    using policy_t = argmin::augmented_lagrangian_policy<
        argmin::lbfgsb_policy<4>, 4>;
    policy_t::options_type popts;

    argmin::step_budget_solver solver{policy_t{}, problem, x0, opts, popts};

    // Warm-up: construct the inner solver and clear first-penalty lazy work.
    for(int i = 0; i < 6; ++i)
        solver.step();

    constexpr std::size_t hot_steps = 40;
    bool mu_changed_in_window = false;
    auto step_track_mu = [&] {
        const double mu_before = solver.state().mu;
        solver.step();
        if(solver.state().mu != mu_before)
            mu_changed_in_window = true;
    };

    const auto counts = run_armed([&] {
        for(std::size_t i = 0; i < hot_steps; ++i)
            step_track_mu();
        solver.reset(x0);
        for(std::size_t i = 0; i < hot_steps; ++i)
            step_track_mu();
    });

    CHECK(mu_changed_in_window);
    CHECK(counts.eigen == 0);
    CHECK(counts.c == 0);
}

// Projected Gauss-Newton active-set path on a fixed-2 bounded linear
// least-squares fixture whose first coordinate pins at its upper bound, so the
// free-set is stable {1} across the armed window.
TEST_CASE("alloc_gate_projected_gn", "[alloc-gate]")
{
    bounded_linear_ls_fixed2 problem;
    Eigen::Vector<double, 2> x0{0.5, -1.0};
    argmin::solver_options<> opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-12);

    argmin::step_budget_solver solver{argmin::projected_gn_policy<2>{},
                                      problem, x0, opts};

    solver.step();
    solver.step();

    // Path-entry assertion: the first coordinate must be pinned at its upper
    // bound so the active-set identification drops it from the free set.
    CHECK(solver.state().x(0) >= 0.5 - 1e-9);

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
}

// Projected-gradient GN backtracking path against an active bound.
TEST_CASE("alloc_gate_projected_gradient_gn", "[alloc-gate]")
{
    bounded_linear_ls_fixed2 problem;
    Eigen::Vector<double, 2> x0{-1.0, -1.0};
    argmin::solver_options<> opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-12);

    argmin::step_budget_solver solver{argmin::projected_gradient_gn_policy<2>{},
                                      problem, x0, opts};

    solver.step();
    solver.step();

    // Path-entry assertion: the first coordinate is pinned at its upper bound,
    // so the projected-gradient line search runs against an active bound.
    CHECK(solver.state().x(0) >= 0.5 - 1e-9);

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
}
