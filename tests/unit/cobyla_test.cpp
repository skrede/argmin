// COBYLA solver policy unit tests.
//
// Validates concept satisfaction, basic convergence on a simple constrained
// problem, HS024 convergence, and basic_solver_group compatibility.

#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/schedule/basic_solver_group.h"
#include "argmin/schedule/round_robin_schedule.h"
#include "argmin/test_functions/hock_schittkowski.h"
#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace argmin;

namespace
{

// Simple constrained problem: minimise x0^2 + x1^2
// subject to x0 + x1 >= 1 (argmin: c >= 0)
// with box bounds [-10, 10]^2.
//
// Satisfies constrained_values && bound_constrained but NOT constrained
// (no constraint_jacobian), proving COBYLA works without gradients.
//
// Optimal: x* = (0.5, 0.5), f* = 0.5.
struct simple_constrained
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        return x[0] * x[0] + x[1] * x[1];
    }

    void constraints(const Eigen::Vector<double, 2>& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = x[0] + x[1] - 1.0;
    }

    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    Eigen::Vector<double, 2> lower_bounds() const
    {
        Eigen::Vector<double, 2> lb;
        lb << -10.0, -10.0;
        return lb;
    }

    Eigen::Vector<double, 2> upper_bounds() const
    {
        Eigen::Vector<double, 2> ub;
        ub << 10.0, 10.0;
        return ub;
    }
};

}

// Concept satisfaction checks.
static_assert(constrained_values<simple_constrained>);
static_assert(!constrained<simple_constrained>);
static_assert(bound_constrained<simple_constrained>);

static_assert(constrained_values<hs071<>>);
static_assert(constrained<hs071<>>);

static_assert(constrained_values<hs024<>>);
static_assert(constrained<hs024<>>);

TEST_CASE("cobyla_policy: simple constrained 2D", "[cobyla]")
{
    simple_constrained problem;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    basic_solver solver{cobyla_policy{}, problem, x0, opts};
    auto result = solver.solve();

    CHECK(result.objective_value < 0.6);
    CHECK(solver.constraint_violation() < 1e-3);
}

TEST_CASE("cobyla_policy: HS024", "[cobyla][hs]")
{
    // HS024 in 2D, f* = -1.0 at x* = (3, sqrt(3)). Pre-Phase-33-hotfix
    // argmin returned f = -0.30 with cv ~ 0 (silent wrong-optimum, the
    // dominant motivator of the static-audit C1 finding). The previous
    // test bar `Approx(-1.0).margin(1.0)` and `cv < 0.5` was wide
    // enough to pass on that wrong answer.
    //
    // Phase 33 hotfix (Powell adaptive parmu + parmu-gated termination
    // + denormalized-simplex fix + tightened geometry threshold)
    // moves argmin to f = -0.652 with cv = 0 -- still not f*, but a
    // 50% closure of the gap. The remaining gap to f* = -1.0 requires
    // the v0.3.x faithfulness rewrite (static-audit C2 trstlp,
    // C3 select-replacement-vertex, C10 geometry step). The bar below
    // locks the hotfix gain (must beat the prior wrong-optimum f =
    // -0.30 by a wide margin) and the feasibility (must be exactly
    // feasible to match the corrected merit), while staying above f*
    // (cannot claim to reach the true optimum yet).
    //
    // When the v0.3.x rewrite lands, tighten this to
    // `Approx(-1.0).margin(0.01)` and `cv < 1e-6`.
    hs024 problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 2000;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    basic_solver solver{cobyla_policy{}, problem, x0, opts};
    auto result = solver.solve();

    INFO("HS024 objective: " << result.objective_value);
    INFO("HS024 cv:        " << solver.constraint_violation());
    // Beats the pre-hotfix wrong optimum f = -0.30 by a wide margin:
    CHECK(result.objective_value < -0.5);
    // Doesn't claim to hit f* = -1.0 (deferred to v0.3.x rewrite):
    CHECK(result.objective_value > -1.05);
    // Feasibility is now exact under adaptive parmu:
    CHECK(solver.constraint_violation() < 1e-4);
}

TEST_CASE("cobyla_policy: basic_solver_group compatibility", "[cobyla][solver_group]")
{
    simple_constrained problem;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    basic_solver_group<round_robin_schedule, dynamic_dimension, simple_constrained, cobyla_policy> group{
        problem, x0, opts};
    auto result = group.solve();

    CHECK(std::isfinite(result.objective_value));
}

// Verify that COBYLA returns normal (non-denormalized) objective values on all
// HS constrained problems. A denormalized value (nonzero but smaller than
// std::numeric_limits<double>::min() ~ 2.2e-308) indicates uninitialized memory
// or memory corruption rather than a legitimate computation result.
//
// Reference: IEEE 754-2019, Section 3.4 (subnormal numbers).
namespace
{

bool is_normal_or_zero(double v)
{
    return v == 0.0 || std::abs(v) >= std::numeric_limits<double>::min();
}

template <typename Problem>
void run_cobyla_denorm_check(const char* name)
{
    Problem problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 2000;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    basic_solver solver{cobyla_policy{}, problem, x0, opts};
    auto result = solver.solve();

    INFO(name << ": objective_value = " << result.objective_value);
    CHECK(std::isfinite(result.objective_value));
    CHECK(is_normal_or_zero(result.objective_value));
}

}

TEST_CASE("cobyla_policy: objective values are not denormalized on HS problems", "[cobyla][hs]")
{
    SECTION("HS006") { run_cobyla_denorm_check<hs006<>>("HS006"); }
    SECTION("HS007") { run_cobyla_denorm_check<hs007<>>("HS007"); }
    SECTION("HS024") { run_cobyla_denorm_check<hs024<>>("HS024"); }
    SECTION("HS026") { run_cobyla_denorm_check<hs026<>>("HS026"); }
    SECTION("HS028") { run_cobyla_denorm_check<hs028<>>("HS028"); }
    SECTION("HS035") { run_cobyla_denorm_check<hs035<>>("HS035"); }
    SECTION("HS039") { run_cobyla_denorm_check<hs039<>>("HS039"); }
    SECTION("HS040") { run_cobyla_denorm_check<hs040<>>("HS040"); }
    SECTION("HS043") { run_cobyla_denorm_check<hs043<>>("HS043"); }
    SECTION("HS048") { run_cobyla_denorm_check<hs048<>>("HS048"); }
    SECTION("HS050") { run_cobyla_denorm_check<hs050<>>("HS050"); }
    SECTION("HS051") { run_cobyla_denorm_check<hs051<>>("HS051"); }
    SECTION("HS071") { run_cobyla_denorm_check<hs071<>>("HS071"); }
    SECTION("HS076") { run_cobyla_denorm_check<hs076<>>("HS076"); }
}
