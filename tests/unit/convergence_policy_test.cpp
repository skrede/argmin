// Tests for convergence policy types and criterion composition.
//
// Validates each convergence criterion in isolation and the
// convergence_policy fold-expression composition. Also covers the
// detail::primal_feasibility_inf helper used by constrained-policy
// step_result writes.
//
// Reference: N&W 2e Section 2.2, K&W 2e Section 2.3,
//            N&W 2e Definition 12.1 (KKT primal feasibility).

#include "argmin/detail/lagrangian.h"
#include "argmin/solver/options.h"
#include "argmin/solver/convergence.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace argmin;

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

// -- Test 2: gradient_tolerance_criterion carries a direct-value literature
// default (1e-5) and fires out of the box -- it is no longer inert by
// default (see convergence.h for the citation).

TEST_CASE("gradient_tolerance_criterion fires on default-constructed threshold",
          "[convergence_policy]")
{
    gradient_tolerance_criterion c{};
    step_result<double> r{.gradient_norm = 1e-20};
    auto status = c.check(r, 5);
    REQUIRE(status.has_value());
    CHECK(*status == solver_status::converged);
}

TEST_CASE("gradient_tolerance_criterion does not fire above the default threshold",
          "[convergence_policy]")
{
    gradient_tolerance_criterion c{};
    step_result<double> r{.gradient_norm = 1.0};
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

// -- Test 8: convergence_policy with default-constructed criteria does not
// fire when residuals are all well above the direct-value literature
// defaults (gradient_tolerance_criterion, objective_tolerance_criterion,
// and step_tolerance_criterion are no longer inert-by-default; see
// convergence.h for the citations).

TEST_CASE("convergence_policy with default thresholds does not fire on large residuals",
          "[convergence_policy]")
{
    convergence_policy<gradient_tolerance_criterion,
                       objective_tolerance_criterion,
                       step_tolerance_criterion> cp{};
    step_result<double> r{
        .gradient_norm = 1.0,
        .step_size = 1.0,
        .objective_change = 1.0,
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
    CHECK(crit.stationarity_threshold == 1e-10);
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

// Primal feasibility is carried inside kkt_residual per the full E-measure
// composite (N&W 2e eq. 12.34) and consumed by objective_tolerance_criterion
// via stationarity_threshold. The two TEST_CASEs below lock that behaviour.

TEST_CASE("objective_tolerance_criterion blocks ftol when kkt_residual is large",
          "[convergence_policy]")
{
    // When primal feasibility (carried inside kkt_residual per the full
    // E-measure) exceeds stationarity_threshold, ftol must not fire even
    // if objective_change is below threshold.
    //
    // Reference: N&W 2e Definition 12.1 (KKT optimality requires feasibility).
    objective_tolerance_criterion c{
        .threshold = 1e-10,
        .stationarity_threshold = 1e-4,
    };

    step_result<double> r{
        .objective_change = 1e-15,
        .kkt_residual = 0.5,
    };

    auto status = c.check(r, 5);
    CHECK(!status.has_value());
}

TEST_CASE("objective_tolerance_criterion fires when kkt_residual is small",
          "[convergence_policy]")
{
    // Counterpart to the block-when-large test: small kkt_residual below the
    // stationarity gate allows ftol_reached to fire.
    objective_tolerance_criterion c{
        .threshold = 1e-10,
        .stationarity_threshold = 1e-4,
    };

    step_result<double> r{
        .objective_change = 1e-15,
        .kkt_residual = 1e-6,
    };

    auto status = c.check(r, 5);
    REQUIRE(status.has_value());
    CHECK(*status == solver_status::ftol_reached);
}

// -- last_check_results accessor tests --
//
// The accessor exposes per-criterion outcomes populated by check() on
// every iteration, independent of the short-circuit return value.
// Reference: K&W 2e Section 4.4.

TEST_CASE("convergence_policy last_check_results captures every criterion",
          "[convergence_policy]")
{
    convergence_policy<gradient_tolerance_criterion,
                       step_tolerance_criterion> policy{};
    std::get<0>(policy.criteria).threshold = 1e-6;
    std::get<1>(policy.criteria).threshold = 1e-6;

    step_result<double> r{};
    r.gradient_norm = 1e-7;  // gradient criterion fires
    r.step_size = 1e-7;      // step criterion fires too
    r.objective_value = 0.0;
    r.constraint_violation = 0.0;
    r.objective_change = 0.0;

    auto status = policy.check(r, /*iteration=*/5);
    REQUIRE(status.has_value());
    CHECK(*status == solver_status::converged);  // gradient fires first

    const auto& results = policy.last_check_results();
    REQUIRE(results.size() == 2);
    REQUIRE(results[0].has_value());
    CHECK(*results[0] == solver_status::converged);  // gradient -> converged
    REQUIRE(results[1].has_value());
    CHECK(*results[1] == solver_status::stalled);    // step_tolerance -> stalled
}

TEST_CASE("convergence_policy reports first-firing criterion as status",
          "[convergence_policy]")
{
    convergence_policy<gradient_tolerance_criterion,
                       step_tolerance_criterion> policy{};
    std::get<0>(policy.criteria).threshold = 1e-6;
    std::get<1>(policy.criteria).threshold = 1e-6;

    step_result<double> r{};
    r.gradient_norm = 1e-7;  // gradient fires
    r.step_size = 1.0;       // step does not fire
    r.objective_change = 0.0;
    r.objective_value = 0.0;
    r.constraint_violation = 0.0;

    auto status = policy.check(r, /*iteration=*/5);
    REQUIRE(status.has_value());
    CHECK(*status == solver_status::converged);

    const auto& results = policy.last_check_results();
    REQUIRE(results[0].has_value());
    CHECK(*results[0] == solver_status::converged);
    CHECK(!results[1].has_value());
}

TEST_CASE("convergence_policy last_check_results reports mid-tuple fire when earlier criterion disabled",
          "[convergence_policy]")
{
    convergence_policy<gradient_tolerance_criterion,
                       step_tolerance_criterion> policy{};
    // Gradient criterion left at its direct-value default (1e-5); the
    // step_result below (gradient_norm = 1.0) is well above that default,
    // so it still does not fire.
    std::get<1>(policy.criteria).threshold = 1e-6;

    step_result<double> r{};
    r.gradient_norm = 1.0;   // would trigger if gradient were active
    r.step_size = 1e-7;      // step fires
    r.objective_change = 0.0;
    r.objective_value = 0.0;
    r.constraint_violation = 0.0;

    auto status = policy.check(r, /*iteration=*/5);
    REQUIRE(status.has_value());
    CHECK(*status == solver_status::stalled);

    const auto& results = policy.last_check_results();
    CHECK(!results[0].has_value());   // gradient disabled
    REQUIRE(results[1].has_value());
    CHECK(*results[1] == solver_status::stalled);
}

// NLopt-compatible alias: three-criterion policy exposing only the
// ftol_rel / xtol_rel / stall stops that NLopt SLSQP reports through
// slsqp.c:1140-1220. Reference: K&W 2e Section 4.4.
TEST_CASE("slsqp_compatible_convergence fires on relative ftol at iteration >= 2",
          "[convergence_policy]")
{
    slsqp_compatible_convergence policy{};
    std::get<0>(policy.criteria).threshold = 1e-10;
    std::get<0>(policy.criteria).stationarity_threshold = 1.0;

    step_result<double> r{};
    r.objective_value = 2.0;
    r.objective_change = 1e-12;
    r.gradient_norm = 1e-3;  // below stationarity gate (1.0), above default 1e-8
    r.step_size = 1.0;
    r.x_norm = 1.0;
    r.constraint_violation = 0.0;

    auto status = policy.check(r, /*iteration=*/5);
    REQUIRE(status.has_value());
    CHECK(*status == solver_status::ftol_reached);

    const auto& results = policy.last_check_results();
    REQUIRE(results.size() == 3);
    REQUIRE(results[0].has_value());
    CHECK(*results[0] == solver_status::ftol_reached);
    CHECK(!results[1].has_value());
    CHECK(!results[2].has_value());
}

TEST_CASE("slsqp_compatible_convergence fires on relative xtol at iteration >= 2",
          "[convergence_policy]")
{
    slsqp_compatible_convergence policy{};
    std::get<1>(policy.criteria).threshold = 1e-10;

    step_result<double> r{};
    r.objective_value = 2.0;
    r.objective_change = 1.0;            // not near ftol
    r.gradient_norm = 1.0;
    r.step_size = 1e-11;
    r.x_norm = 1.0;                       // step/x ratio = 1e-11 < 1e-10
    r.constraint_violation = 0.0;

    auto status = policy.check(r, /*iteration=*/5);
    REQUIRE(status.has_value());
    CHECK(*status == solver_status::xtol_reached);

    const auto& results = policy.last_check_results();
    REQUIRE(results.size() == 3);
    CHECK(!results[0].has_value());
    REQUIRE(results[1].has_value());
    CHECK(*results[1] == solver_status::xtol_reached);
    CHECK(!results[2].has_value());
}

TEST_CASE("slsqp_compatible_convergence does not carry a gradient_tolerance slot",
          "[convergence_policy]")
{
    slsqp_compatible_convergence policy{};

    // Compile-time guarantee that no gradient_tolerance_criterion is in
    // the tuple: std::get<gradient_tolerance_criterion>(policy.criteria)
    // would fail to compile. Runtime check: the array length matches the
    // three-criterion composition.
    const auto& results = policy.last_check_results();
    CHECK(results.size() == 3);
}

// -- detail::primal_feasibility_inf tests --
//
// Locks the L-infinity composition used at step_result.constraint_violation
// write sites across the 5 constrained policies. Dimensionally consistent
// with detail::kkt_residual legs 2 and 3 (both L-infinity).
//
// Reference: N&W 2e Definition 12.1 (KKT primal feasibility).
TEST_CASE("detail::primal_feasibility_inf returns L-infinity composite",
          "[lagrangian][primal_feasibility]")
{
    using Catch::Approx;

    SECTION("mixed eq + ineq with both legs non-zero")
    {
        Eigen::VectorXd c_eq(3); c_eq << 1.0, -2.0, 0.5;
        Eigen::VectorXd c_ineq(3); c_ineq << 2.0, -0.3, 1.0;
        CHECK(detail::primal_feasibility_inf(c_eq, c_ineq) == Approx(2.0));
    }
    SECTION("empty c_eq, non-empty c_ineq")
    {
        Eigen::VectorXd c_eq(0);
        Eigen::VectorXd c_ineq(2); c_ineq << -0.3, 1.0;
        CHECK(detail::primal_feasibility_inf(c_eq, c_ineq) == Approx(0.3));
    }
    SECTION("both empty returns zero")
    {
        Eigen::VectorXd c_eq(0), c_ineq(0);
        CHECK(detail::primal_feasibility_inf(c_eq, c_ineq) == Approx(0.0));
    }
}
