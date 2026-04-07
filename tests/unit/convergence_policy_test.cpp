// Tests for convergence policy types and criterion composition.
//
// Validates each convergence criterion in isolation, then tests
// convergence_policy fold-expression composition and
// constrained_convergence_policy feasibility gating.
//
// Reference: N&W 2e Section 2.2, K&W 2e Section 2.3,
//            N&W 2e Section 12.1 (KKT feasibility).

#include "nablapp/solver/options.h"
#include "nablapp/solver/convergence.h"

#include "nablapp/result/status.h"
#include "nablapp/result/step_result.h"

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

// -- Test 10: stationarity gate blocks ftol when gradient is large --

TEST_CASE("objective_tolerance_criterion stationarity gate blocks when gradient large",
          "[convergence_policy]")
{
    objective_tolerance_criterion c{
        .threshold = 1e-10,
        .stationarity_threshold = 1e-6,
    };
    step_result<double> r{
        .gradient_norm = 0.5,
        .objective_change = 1e-15,
    };
    auto status = c.check(r, 2);
    CHECK(!status.has_value());
}

// -- Test 11: stationarity gate passes when gradient is small --

TEST_CASE("objective_tolerance_criterion stationarity gate passes when gradient small",
          "[convergence_policy]")
{
    objective_tolerance_criterion c{
        .threshold = 1e-10,
        .stationarity_threshold = 1e-6,
    };
    step_result<double> r{
        .gradient_norm = 1e-8,
        .objective_change = 1e-15,
    };
    auto status = c.check(r, 2);
    REQUIRE(status.has_value());
    CHECK(*status == solver_status::ftol_reached);
}

// -- Test 12: stationarity gate uses sensible default when nullopt --

TEST_CASE("objective_tolerance_criterion stationarity gate uses default when nullopt",
          "[convergence_policy]")
{
    objective_tolerance_criterion c{.threshold = 1e-10};

    // gradient_norm below default gate (1e-8) -- should pass
    step_result<double> r_small{
        .gradient_norm = 1e-9,
        .objective_change = 1e-15,
    };
    auto status_pass = c.check(r_small, 2);
    REQUIRE(status_pass.has_value());
    CHECK(*status_pass == solver_status::ftol_reached);

    // gradient_norm above default gate (1e-8) -- should block
    step_result<double> r_large{
        .gradient_norm = 0.1,
        .objective_change = 1e-15,
    };
    auto status_block = c.check(r_large, 2);
    CHECK(!status_block.has_value());
}

// -- Test 13: objective_tolerance_rel_criterion stationarity gate --

TEST_CASE("objective_tolerance_rel_criterion stationarity gate blocks when gradient large",
          "[convergence_policy]")
{
    objective_tolerance_rel_criterion c{
        .threshold = 1e-10,
        .stationarity_threshold = 1e-6,
    };
    step_result<double> r{
        .objective_value = 1.0,
        .gradient_norm = 0.5,
        .objective_change = 1e-15,
    };
    auto status = c.check(r, 2);
    CHECK(!status.has_value());
}

// -- Test 14: set_objective_threshold propagates stationarity_threshold --

TEST_CASE("set_objective_threshold propagates stationarity_threshold",
          "[convergence_policy]")
{
    solver_options<> opts;
    opts.set_objective_threshold(1e-10);

    auto& crit = std::get<objective_tolerance_criterion>(opts.convergence.criteria);
    REQUIRE(crit.stationarity_threshold.has_value());
    CHECK(*crit.stationarity_threshold == 1e-10);
}

// -- Test 15: full policy with stationarity gate on stagnant non-stationary point --

TEST_CASE("convergence_policy with stationarity gate does not fire on stagnant non-stationary point",
          "[convergence_policy]")
{
    default_convergence conv{
        .criteria = {
            gradient_tolerance_criterion{.threshold = 1e-6},
            objective_tolerance_criterion{
                .threshold = 1e-10,
                .stationarity_threshold = 1e-6,
            },
            step_tolerance_criterion{.threshold = 1e-12},
            stall_tolerance_criterion{},
        }
    };

    step_result<double> r{
        .gradient_norm = 0.1,
        .step_size = 0.01,
        .objective_change = 1e-15,
    };

    auto status = conv.check(r, 5);
    CHECK(!status.has_value());
}

// -- constrained_convergence_policy tests --

TEST_CASE("constrained_convergence_policy blocks when infeasible",
          "[convergence_policy]")
{
    constrained_convergence_policy<default_convergence> cp{
        .inner = {.criteria = {
            gradient_tolerance_criterion{.threshold = 1e-6},
            objective_tolerance_criterion{},
            step_tolerance_criterion{},
            stall_tolerance_criterion{},
        }},
        .feasibility_threshold = 1e-4,
    };

    step_result<double> r{
        .gradient_norm = 1e-8,
        .constraint_violation = 0.5,
    };

    auto status = cp.check(r, 5);
    CHECK(!status.has_value());
}

TEST_CASE("constrained_convergence_policy passes when feasible",
          "[convergence_policy]")
{
    constrained_convergence_policy<default_convergence> cp{
        .inner = {.criteria = {
            gradient_tolerance_criterion{.threshold = 1e-6},
            objective_tolerance_criterion{},
            step_tolerance_criterion{},
            stall_tolerance_criterion{},
        }},
        .feasibility_threshold = 1e-4,
    };

    step_result<double> r{
        .gradient_norm = 1e-8,
        .constraint_violation = 1e-6,
    };

    auto status = cp.check(r, 5);
    REQUIRE(status.has_value());
    CHECK(*status == solver_status::converged);
}

TEST_CASE("constrained_convergence_policy with nullopt threshold delegates directly",
          "[convergence_policy]")
{
    constrained_convergence_policy<default_convergence> cp{
        .inner = {.criteria = {
            gradient_tolerance_criterion{.threshold = 1e-6},
            objective_tolerance_criterion{},
            step_tolerance_criterion{},
            stall_tolerance_criterion{},
        }},
        .feasibility_threshold = std::nullopt,
    };

    step_result<double> r{
        .gradient_norm = 1e-8,
        .constraint_violation = 100.0,
    };

    auto status = cp.check(r, 5);
    REQUIRE(status.has_value());
    CHECK(*status == solver_status::converged);
}

TEST_CASE("solver_options<constrained_convergence> convenience setters work",
          "[convergence_policy]")
{
    solver_options<constrained_convergence> opts;
    opts.set_gradient_threshold(1e-6);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-8);
    opts.set_feasibility_threshold(1e-4);

    auto& inner = opts.convergence.inner;
    CHECK(std::get<gradient_tolerance_criterion>(inner.criteria).threshold == 1e-6);
    CHECK(std::get<objective_tolerance_criterion>(inner.criteria).threshold == 1e-10);
    CHECK(std::get<step_tolerance_criterion>(inner.criteria).threshold == 1e-8);
    CHECK(opts.convergence.feasibility_threshold == 1e-4);
}
