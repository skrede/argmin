#include "argmin/options/qp_options.h"
#include "argmin/options/cmaes_options.h"
#include "argmin/options/asymptote_options.h"
#include "argmin/options/trust_region_options.h"
#include "argmin/options/mma_subproblem_options.h"
#include "argmin/solver/cmaes_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/restarting_policy.h"
#include "argmin/solver/multistart_policy.h"
#include "argmin/solver/augmented_lagrangian_policy.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <optional>
#include <type_traits>

// These tests pin the direct-value option convention: literature-default
// fields expose their default directly (no std::optional / value_or
// indirection), so the default has a single source of truth read straight at
// the consumer. Type-identity static_asserts prove the field is a raw value
// type; value assertions pin the literature default.

using namespace argmin;

TEST_CASE("asymptote_options exposes literature defaults directly", "[options]")
{
    asymptote_options o{};
    static_assert(std::is_same_v<double, decltype(o.minimum_distance_fraction)>);
    static_assert(std::is_same_v<double, decltype(o.maximum_distance_fraction)>);
    CHECK(o.minimum_distance_fraction == 0.01);
    CHECK(o.maximum_distance_fraction == 10.0);
}

TEST_CASE("trust_region_options exposes literature defaults directly", "[options]")
{
    trust_region_options o{};
    static_assert(std::is_same_v<double, decltype(o.eta_good)>);
    static_assert(std::is_same_v<double, decltype(o.geometry_factor)>);
    CHECK(o.eta_good == 0.7);
    CHECK(o.eta_poor == 0.1);
    CHECK(o.expand_factor == 2.0);
    CHECK(o.shrink_factor == 0.5);
    CHECK(o.step_threshold == 0.5);
    CHECK(o.geometry_factor == 2.0);
}

TEST_CASE("mma_subproblem_options exposes literature defaults directly", "[options]")
{
    mma_subproblem_options o{};
    static_assert(std::is_same_v<std::uint16_t, decltype(o.dual_max_iterations)>);
    static_assert(std::is_same_v<double, decltype(o.dual_tolerance)>);
    static_assert(std::is_same_v<double, decltype(o.backtrack_factor)>);
    CHECK(o.dual_max_iterations == 50);
    CHECK(o.dual_tolerance == 1e-9);
    CHECK(o.backtrack_factor == 0.95);
}

TEST_CASE("cmaes_options exposes literature defaults directly", "[options]")
{
    cmaes_options o{};
    static_assert(std::is_same_v<double, decltype(o.sigma_collapse_threshold)>);
    static_assert(std::is_same_v<double, decltype(o.condition_number_limit)>);
    static_assert(std::is_same_v<double, decltype(o.objective_value_tolerance)>);
    static_assert(std::is_same_v<double, decltype(o.step_size_tolerance)>);
    CHECK(o.sigma_collapse_threshold == 1e-12);
    CHECK(o.condition_number_limit == 1e14);
    CHECK(o.objective_value_tolerance == 1e-12);
    CHECK(o.step_size_tolerance == 1e-12);
}

TEST_CASE("qp_options is direct-value (reference shape)", "[options]")
{
    qp_options o{};
    static_assert(std::is_same_v<std::uint16_t, decltype(o.max_iterations)>);
    static_assert(std::is_same_v<double, decltype(o.tolerance)>);
    CHECK(o.max_iterations == 200);
    CHECK(o.tolerance == 1e-12);
}

TEST_CASE("augmented_lagrangian options are direct-value and single-sourced",
          "[options]")
{
    using opts_t = augmented_lagrangian_policy<>::options_type;
    opts_t o{};
    static_assert(std::is_same_v<double, decltype(o.mu_init)>);
    static_assert(std::is_same_v<double, decltype(o.mu_decrease)>);
    static_assert(std::is_same_v<double, decltype(o.mu_min)>);
    static_assert(std::is_same_v<std::uint32_t, decltype(o.inner_max_iterations)>);
    static_assert(std::is_same_v<double, decltype(o.constraint_tolerance)>);
    static_assert(std::is_same_v<double, decltype(o.inner_gradient_tolerance)>);
    static_assert(std::is_same_v<double, decltype(o.feasibility_progress)>);
    static_assert(std::is_same_v<bool, decltype(o.warm_start_inner)>);
    static_assert(std::is_same_v<double, decltype(o.inner_tolerance_eta)>);
    static_assert(std::is_same_v<double, decltype(o.inner_tolerance_alpha)>);
    CHECK(o.mu_init == 0.1);
    CHECK(o.mu_decrease == 0.25);
    CHECK(o.mu_min == 1e-6);
    CHECK(o.inner_max_iterations == 200u);
    CHECK(o.constraint_tolerance == 1e-6);
    CHECK(o.inner_gradient_tolerance == 1e-6);
    CHECK(o.feasibility_progress == 0.25);
    CHECK(o.warm_start_inner == true);
    CHECK(o.inner_tolerance_eta == 0.1);
    // inner_tolerance_alpha's single source of truth is the value the
    // consumer previously reached via value_or(0.9); the field's default now
    // matches that runtime default exactly.
    CHECK(o.inner_tolerance_alpha == 0.9);
}

TEST_CASE("restarting_policy population_multiplier is direct-value", "[options]")
{
    using opts_t = restarting_policy<cmaes_policy<>>::options_type;
    opts_t o{};
    static_assert(std::is_same_v<double, decltype(o.population_multiplier)>);
    CHECK(o.population_multiplier == 2.0);
    // stagnation_limit is a genuinely-optional auto-formula override and stays
    // std::optional (unset means the 10 + ceil(30 n / lambda) auto formula).
    static_assert(std::is_same_v<std::optional<std::uint32_t>,
                                 decltype(o.stagnation_limit)>);
    CHECK_FALSE(o.stagnation_limit.has_value());
}

TEST_CASE("multistart_policy max_restarts is direct-value single-sourced",
          "[options]")
{
    using policy_t = multistart_policy<cmaes_policy<>>;
    using opts_t = policy_t::options_type;
    opts_t o{};
    static_assert(std::is_same_v<std::uint16_t, decltype(o.max_restarts)>);
    CHECK(o.max_restarts == policy_t::default_max_restarts);
    CHECK(o.max_restarts == 10);
    // stall_budget_per_restart is a genuinely-optional auto-formula override.
    static_assert(std::is_same_v<std::optional<std::uint32_t>,
                                 decltype(o.stall_budget_per_restart)>);
    CHECK_FALSE(o.stall_budget_per_restart.has_value());
}
