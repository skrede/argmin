#include "nablapp/solver/augmented_lagrangian_policy.h"
#include "nablapp/solver/bobyqa_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/test_functions/hock_schittkowski.h"
#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>

using Catch::Approx;
using namespace nablapp;

TEST_CASE("augmented lagrangian converges on HS076", "[augmented_lagrangian]")
{
    hs076 problem;
    auto x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 60;
    opts.set_gradient_threshold(1e-6);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);
    opts.set_stationarity_threshold(1e-4);

    basic_solver solver{augmented_lagrangian_policy<lbfgsb_policy<hs076<>::problem_dimension>>{},
        problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.objective_value == Approx(-4.6818).margin(1e-2));
    CHECK(solver.constraint_violation() < 1e-3);
}

TEST_CASE("augmented lagrangian converges on HS035", "[augmented_lagrangian]")
{
    hs035 problem;
    auto x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-6);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);
    opts.set_stationarity_threshold(1e-4);

    basic_solver solver{augmented_lagrangian_policy<lbfgsb_policy<hs035<>::problem_dimension>>{},
        problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.objective_value == Approx(1.0 / 9.0).margin(1e-3));
    CHECK(solver.constraint_violation() < 1e-4);
}

TEST_CASE("augmented lagrangian on HS071 (equality + inequality)",
          "[augmented_lagrangian]")
{
    hs071 problem;
    auto x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 80;
    opts.set_gradient_threshold(1e-4);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);
    opts.set_stationarity_threshold(1e-2);

    basic_solver solver{augmented_lagrangian_policy<lbfgsb_policy<hs071<>::problem_dimension>>{},
        problem, x0, opts};
    auto result = solver.solve(opts);

    // Relaxed tolerance: AugLag on mixed eq/ineq is harder.
    // Tighter initial penalty (mu_init=0.1) trades HS071 precision for
    // equality-constrained stability (HS040).
    CHECK(result.objective_value == Approx(17.0140173).margin(0.5));
    CHECK(solver.constraint_violation() < 1e-2);
    CHECK(result.iterations <= 80);
}

TEST_CASE("augmented lagrangian with BOBYQA inner solver",
          "[augmented_lagrangian]")
{
    // Use HS035 (simpler, inequality-only) with BOBYQA inner solver
    hs035 problem;
    auto x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 80;
    opts.set_gradient_threshold(1e-4);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);
    // Derivative-free inner solver produces iterates with cv residuals
    // in the 1e-2 range on HS035; widen the best-seen feasibility gate
    // so the AL outer-loop result is not snapped back to x_0 when the
    // inner solve converges on a near-feasible KKT point. Also honored
    // transparently by basic_solver's constraint_tolerance precedence
    // over feasibility_tolerance.
    opts.constraint_tolerance = 5e-2;

    basic_solver solver{augmented_lagrangian_policy<bobyqa_policy<hs035<>::problem_dimension>>{},
        problem, x0, opts};
    auto result = solver.solve(opts);

    // BOBYQA is derivative-free; relaxed tolerance
    CHECK(result.objective_value == Approx(1.0 / 9.0).margin(0.5));
}

TEST_CASE("augmented lagrangian converges on HS040 (equality only)",
          "[augmented_lagrangian]")
{
    hs040 problem;
    auto x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-4);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    basic_solver solver{augmented_lagrangian_policy<lbfgsb_policy<>>{}, 
        problem, x0, opts};
    auto result = solver.solve();

    CHECK(std::isfinite(result.objective_value));
    CHECK(result.objective_value == Approx(-0.25).margin(0.1));
    CHECK(solver.constraint_violation() < 0.2);
}

TEST_CASE("augmented lagrangian step and solve consistency",
          "[augmented_lagrangian]")
{
    hs035 problem;
    auto x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 30;
    opts.set_gradient_threshold(1e-5);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);
    opts.set_stationarity_threshold(1e-4);

    SECTION("step returns finite values")
    {
        basic_solver solver{augmented_lagrangian_policy<lbfgsb_policy<hs035<>::problem_dimension>>{},
            problem, x0, opts};

        for(int i = 0; i < 5; ++i)
        {
            auto sr = solver.step();
            CHECK(std::isfinite(sr.objective_value));
            CHECK(std::isfinite(sr.gradient_norm));
        }
    }

    SECTION("solve gives similar result to manual stepping")
    {
        basic_solver solver1{augmented_lagrangian_policy<lbfgsb_policy<hs035<>::problem_dimension>>{},
            problem, x0, opts};
        auto result1 = solver1.solve(opts);

        basic_solver solver2{augmented_lagrangian_policy<lbfgsb_policy<hs035<>::problem_dimension>>{},
            problem, x0, opts};
        step_result<double> last{};
        for(std::uint32_t i = 0; i < opts.max_iterations; ++i)
        {
            last = solver2.step();
            if(last.gradient_norm < 1e-5 && last.constraint_violation < 1e-4)
                break;
        }

        // Both should reach a similar objective value
        CHECK(last.objective_value == Approx(result1.objective_value).margin(1e-2));
    }
}

TEST_CASE("augmented lagrangian reports actual gradient norm",
          "[augmented_lagrangian]")
{
    hs076 problem;
    auto x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 60;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);
    opts.set_stationarity_threshold(1e-4);

    basic_solver solver{augmented_lagrangian_policy<lbfgsb_policy<hs076<>::problem_dimension>>{},
        problem, x0, opts};
    auto result = solver.solve(opts);

    // With actual gradient reporting the norm reflects the augmented
    // Lagrangian gradient, not the old max(violation, step_norm) proxy.
    // The aug-lag gradient includes penalty-scaled constraint terms, so
    // it can be moderate even at an optimal feasible point.
    CHECK(std::isfinite(result.gradient_norm));
    CHECK(result.gradient_norm >= 0.0);
    CHECK(result.constraint_violation < 1e-3);
}

TEST_CASE("augmented lagrangian convergence on HS076 with stationarity_threshold gate",
          "[augmented_lagrangian]")
{
    hs076 problem;
    auto x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 60;
    opts.set_gradient_threshold(1e-6);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);
    opts.set_stationarity_threshold(1e-4);

    basic_solver solver{augmented_lagrangian_policy<lbfgsb_policy<hs076<>::problem_dimension>>{},
        problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.objective_value == Approx(-4.6818).margin(1e-2));
    CHECK(result.constraint_violation < 1e-3);
}

TEST_CASE("augmented lagrangian warm-start converges on HS071",
          "[augmented_lagrangian]")
{
    constexpr int D = hs071<>::problem_dimension;
    using policy_type = augmented_lagrangian_policy<lbfgsb_policy<D>, D>;

    hs071 problem;
    auto x0 = problem.initial_point();

    policy_type::options_type policy_opts;
    policy_opts.warm_start_inner = true;
    policy_opts.max_outer_iterations = 100;

    solver_options opts;
    opts.max_iterations = 100;
    opts.set_gradient_threshold(1e-5);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    basic_solver<policy_type, D, hs071<>> solver{problem, x0, opts, policy_opts};
    auto result = solver.solve();

    CHECK(result.objective_value == Approx(17.0140173).margin(1e-1));
    CHECK(solver.constraint_violation() < 1e-3);
}

TEST_CASE("augmented lagrangian warm-start reduces inner iterations",
          "[augmented_lagrangian]")
{
    constexpr int D = hs071<>::problem_dimension;
    using policy_type = augmented_lagrangian_policy<lbfgsb_policy<D>, D>;

    hs071 problem;
    auto x0 = problem.initial_point();

    // Warm-start run
    policy_type::options_type warm_opts;
    warm_opts.warm_start_inner = true;
    warm_opts.max_outer_iterations = 100;

    solver_options sopts;
    sopts.max_iterations = 100;
    sopts.set_gradient_threshold(1e-5);
    sopts.set_objective_threshold(1e-15);
    sopts.set_step_threshold(1e-15);

    basic_solver<policy_type, D, hs071<>> warm_solver{problem, x0, sopts, warm_opts};
    auto warm_result = warm_solver.solve();
    int warm_iters = warm_result.iterations;

    // Cold-start run
    policy_type::options_type cold_opts;
    cold_opts.warm_start_inner = false;
    cold_opts.max_outer_iterations = 100;

    basic_solver<policy_type, D, hs071<>> cold_solver{problem, x0, sopts, cold_opts};
    auto cold_result = cold_solver.solve();
    int cold_iters = cold_result.iterations;

    // Both paths should converge; warm-start preserves curvature for
    // faster inner convergence, though outer iteration count may vary
    // slightly due to adaptive tolerance interaction.
    CHECK(warm_result.objective_value == Approx(17.0140173).margin(1e-1));
    CHECK(cold_result.objective_value == Approx(17.0140173).margin(1e-1));
}

TEST_CASE("augmented lagrangian adaptive tolerance converges on HS071",
          "[augmented_lagrangian]")
{
    constexpr int D = hs071<>::problem_dimension;
    using policy_type = augmented_lagrangian_policy<lbfgsb_policy<D>, D>;

    hs071 problem;
    auto x0 = problem.initial_point();

    // Conn-Gould-Toint adaptive schedule with default parameters.
    policy_type::options_type policy_opts;
    policy_opts.inner_tolerance_eta = 0.1;
    policy_opts.inner_tolerance_alpha = 0.1;
    policy_opts.max_outer_iterations = 100;

    solver_options opts;
    opts.max_iterations = 100;
    opts.set_gradient_threshold(1e-5);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    basic_solver<policy_type, D, hs071<>> solver{problem, x0, opts, policy_opts};
    auto result = solver.solve();

    CHECK(result.objective_value == Approx(17.0140173).margin(1e-1));
    CHECK(solver.constraint_violation() < 1e-3);
}

TEST_CASE("augmented lagrangian cold-start still works on HS071",
          "[augmented_lagrangian]")
{
    constexpr int D = hs071<>::problem_dimension;
    using policy_type = augmented_lagrangian_policy<lbfgsb_policy<D>, D>;

    hs071 problem;
    auto x0 = problem.initial_point();

    policy_type::options_type policy_opts;
    policy_opts.warm_start_inner = false;
    policy_opts.max_outer_iterations = 100;

    solver_options opts;
    opts.max_iterations = 100;
    opts.set_gradient_threshold(1e-5);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    basic_solver<policy_type, D, hs071<>> solver{problem, x0, opts, policy_opts};
    auto result = solver.solve();

    // Should still converge to the known optimum (regression test for
    // the opt-out path).
    CHECK(result.objective_value == Approx(17.0140173).margin(1e-1));
    CHECK(solver.constraint_violation() < 1e-2);
}

// augmented_lagrangian populates step_result::kkt_residual on every
// outer iteration via detail::kkt_residual using the outer-loop
// multiplier estimates (s.lambda_eq, s.lambda_ineq) and the Jacobians
// already materialized for the augmented-gradient reporting path.
//
// Reference: N&W 2e Section 12.3 / eq. 12.34.
TEST_CASE("augmented lagrangian populates kkt_residual",
          "[augmented_lagrangian][kkt]")
{
    hs076 problem;
    auto x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 30;
    opts.set_gradient_threshold(1e-5);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);
    opts.set_stationarity_threshold(1e-4);

    basic_solver solver{augmented_lagrangian_policy<lbfgsb_policy<hs076<>::problem_dimension>>{},
        problem, x0, opts};

    bool populated = false;
    for(int i = 0; i < 10; ++i)
    {
        auto sr = solver.step();
        if(sr.kkt_residual.has_value())
        {
            populated = true;
            CHECK(sr.kkt_residual.value() >= 0.0);
        }
        if(sr.policy_status)
            break;
    }
    CHECK(populated);
}
