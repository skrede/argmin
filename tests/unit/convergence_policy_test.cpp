// Tests for convergence policy types and criterion composition.
//
// Validates each convergence criterion in isolation, then tests
// convergence_policy fold-expression composition.
//
// Reference: N&W 2e Section 2.2, K&W 2e Section 2.3.

#include "nablapp/solver/convergence.h"
#include "nablapp/result/step_result.h"
#include "nablapp/result/status.h"

#include <catch2/catch_test_macros.hpp>

using namespace nablapp;

// -- Test 1: gradient_tolerance_criterion fires on small gradient --

TEST_CASE("gradient_tolerance_criterion fires on small gradient",
          "[convergence_policy]")
{
    gradient_tolerance_criterion c{.threshold = 1e-6};
    step_result<double> r{.gradient_norm = 1e-7};
    auto status = c.check(r, 5);
    REQUIRE(status.has_value());
    CHECK(*status == solver_status::converged);
}

// -- Test 2: gradient_tolerance_criterion disabled with nullopt --

TEST_CASE("gradient_tolerance_criterion disabled returns nullopt",
          "[convergence_policy]")
{
    gradient_tolerance_criterion c{};
    step_result<double> r{.gradient_norm = 1e-20};
    auto status = c.check(r, 5);
    CHECK(!status.has_value());
}

// -- Test 3: objective_tolerance_criterion skips iteration 1 --

TEST_CASE("objective_tolerance_criterion skips iteration 1",
          "[convergence_policy]")
{
    objective_tolerance_criterion c{.threshold = 1e-10};
    step_result<double> r{.objective_change = 1e-15};

    auto status_iter1 = c.check(r, 1);
    CHECK(!status_iter1.has_value());

    auto status_iter2 = c.check(r, 2);
    REQUIRE(status_iter2.has_value());
    CHECK(*status_iter2 == solver_status::ftol_reached);
}

// -- Test 4: step_tolerance_criterion fires on small step --

TEST_CASE("step_tolerance_criterion fires on small step",
          "[convergence_policy]")
{
    step_tolerance_criterion c{.threshold = 1e-8};
    step_result<double> r{.step_size = 1e-9};

    auto status_iter1 = c.check(r, 1);
    CHECK(!status_iter1.has_value());

    auto status_iter2 = c.check(r, 2);
    REQUIRE(status_iter2.has_value());
    CHECK(*status_iter2 == solver_status::stalled);
}

// -- Test 5: objective_tolerance_rel_criterion fires on small relative change --

TEST_CASE("objective_tolerance_rel fires on small relative change",
          "[convergence_policy]")
{
    // Relative change: |0.005| / max(|100.0|, 1.0) = 5e-5 < 1e-4
    objective_tolerance_rel_criterion c{.threshold = 1e-4};
    step_result<double> r{
        .objective_value = 100.0,
        .objective_change = 0.005,
    };

    auto status = c.check(r, 5);
    REQUIRE(status.has_value());
    CHECK(*status == solver_status::ftol_reached);
}

TEST_CASE("objective_tolerance_rel does not fire on large relative change",
          "[convergence_policy]")
{
    // Relative change: |5.0| / max(|100.0|, 1.0) = 0.05 > 1e-4
    objective_tolerance_rel_criterion c{.threshold = 1e-4};
    step_result<double> r{
        .objective_value = 100.0,
        .objective_change = 5.0,
    };

    auto status = c.check(r, 5);
    CHECK(!status.has_value());
}

// -- Test 6: step_tolerance_rel_criterion fires on small relative step --

TEST_CASE("step_tolerance_rel fires on small relative step",
          "[convergence_policy]")
{
    // Relative step: 1e-9 / max(10.0, 1.0) = 1e-10 < 1e-8
    step_tolerance_rel_criterion c{.threshold = 1e-8};
    step_result<double> r{
        .step_size = 1e-9,
        .x_norm = 10.0,
    };

    auto status = c.check(r, 5);
    REQUIRE(status.has_value());
    CHECK(*status == solver_status::xtol_reached);
}

TEST_CASE("step_tolerance_rel does not fire on large relative step",
          "[convergence_policy]")
{
    // Relative step: 1.0 / max(10.0, 1.0) = 0.1 > 1e-8
    step_tolerance_rel_criterion c{.threshold = 1e-8};
    step_result<double> r{
        .step_size = 1.0,
        .x_norm = 10.0,
    };

    auto status = c.check(r, 5);
    CHECK(!status.has_value());
}

// -- Test 7: convergence_policy composes criteria, first match wins --

TEST_CASE("convergence_policy first match wins", "[convergence_policy]")
{
    convergence_policy<gradient_tolerance_criterion,
                       objective_tolerance_criterion> cp{
        .criteria = {
            gradient_tolerance_criterion{.threshold = 1e-6},
            objective_tolerance_criterion{.threshold = 1e-10},
        }
    };
    // Both would fire, but gradient is checked first
    step_result<double> r{
        .gradient_norm = 1e-7,
        .objective_change = 1e-11,
    };
    auto status = cp.check(r, 5);
    REQUIRE(status.has_value());
    CHECK(*status == solver_status::converged);
}

// -- Test 8: convergence_policy with all nullopt returns nullopt --

TEST_CASE("convergence_policy all disabled returns nullopt",
          "[convergence_policy]")
{
    convergence_policy<gradient_tolerance_criterion,
                       objective_tolerance_criterion,
                       step_tolerance_criterion> cp{};
    step_result<double> r{
        .gradient_norm = 1e-20,
        .step_size = 1e-20,
        .objective_change = 1e-20,
    };
    auto status = cp.check(r, 5);
    CHECK(!status.has_value());
}

// -- Test 9: convergence_policy with only gradient criterion --

TEST_CASE("convergence_policy gradient-only for derivative-free scenario",
          "[convergence_policy]")
{
    convergence_policy<gradient_tolerance_criterion> cp{
        .criteria = {gradient_tolerance_criterion{.threshold = 1e-3}}
    };
    step_result<double> r_above{.gradient_norm = 1e-2};
    CHECK(!cp.check(r_above, 5).has_value());

    step_result<double> r_below{.gradient_norm = 1e-4};
    auto status = cp.check(r_below, 5);
    REQUIRE(status.has_value());
    CHECK(*status == solver_status::converged);
}
