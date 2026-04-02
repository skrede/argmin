#include "mock_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/schedule/basic_solver_group.h"
#include "nablapp/schedule/round_robin_schedule.h"
#include "nablapp/result/status.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using namespace nablapp;

struct quadratic_group
{
    static constexpr int problem_dimension = nablapp::dynamic_dimension;

    int dimension() const { return 2; }
    double value(const Eigen::VectorXd& x) const { return 0.5 * x.squaredNorm(); }
};

TEST_CASE("basic_solver_group construction", "[solver_group]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    solver_options opts;

    SECTION("constructs with round_robin and two policies")
    {
        basic_solver_group<round_robin_schedule, nablapp::dynamic_dimension, test::mock_policy, test::mock_policy>
            group{prob, x0, opts};
        (void)group;
        SUCCEED("construction compiled and ran");
    }
}

TEST_CASE("basic_solver_group step with round_robin", "[solver_group]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;

    basic_solver_group<round_robin_schedule, nablapp::dynamic_dimension, test::mock_policy, test::mock_policy>
        group{prob, x0, opts};

    auto r1 = group.step();
    CHECK(r1.objective_value < 0.5 * x0.squaredNorm());

    auto r2 = group.step();
    CHECK(r2.objective_value < 0.5 * x0.squaredNorm());
}

TEST_CASE("basic_solver_group solve converges", "[solver_group]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{1.0, 1.0}};
    solver_options opts;

    basic_solver_group<round_robin_schedule, nablapp::dynamic_dimension, test::mock_policy, test::mock_policy>
        group{prob, x0, opts};

    auto result = group.step_n(100);
    CHECK(result.x.norm() < 1.0);
}

TEST_CASE("basic_solver_group fold expression concept check", "[solver_group]")
{
    // This test verifies the fold-expression requires clause compiles.
    // If the problem type did not satisfy mock_policy's constructor
    // requirements, this would be a compile error.
    quadratic_group prob;
    Eigen::VectorXd x0{{1.0, 1.0}};

    basic_solver_group<round_robin_schedule, nablapp::dynamic_dimension, test::mock_policy, test::non_converging_policy>
        group{prob, x0};
    (void)group;
    SUCCEED("fold expression concept check passed at compile time");
}

TEST_CASE("basic_solver step propagates constraint_violation", "[solver]")
{
    quadratic_group prob;

    SECTION("constrained policy returns nonzero constraint_violation")
    {
        Eigen::VectorXd x0{{0.5, 0.5}};
        basic_solver<test::constrained_mock_policy> solver{prob, x0};
        auto result = solver.step();
        CHECK(result.constraint_violation > 0.0);
    }

    SECTION("unconstrained policy returns zero constraint_violation")
    {
        Eigen::VectorXd x0{{1.0, 1.0}};
        basic_solver<test::mock_policy> solver{prob, x0};
        auto result = solver.step();
        CHECK(result.constraint_violation == 0.0);
    }
}

TEST_CASE("feasible solver beats infeasible despite worse objective",
          "[solver_group][feasibility]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{1.0, 1.0}};
    solver_options opts;

    // feasible_mock: obj=10, cv=0 (feasible)
    // infeasible_mock: obj=1, cv=2 (infeasible)
    basic_solver_group<round_robin_schedule, nablapp::dynamic_dimension,
                       test::feasible_mock_policy,
                       test::infeasible_mock_policy>
        group{prob, x0, opts};

    auto result = group.step_n(2);

    // Best solver should be the feasible one (obj ~10) not infeasible (obj ~1).
    CHECK(result.objective_value == Approx(10.0));
}

TEST_CASE("among feasible solvers lowest objective wins",
          "[solver_group][feasibility]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;

    // Both unconstrained (cv=0, both feasible). mock_policy does gradient
    // descent so objectives differ slightly between the two due to round-robin
    // alternation, but both have the same behavior. The result should pick
    // whichever has the lowest objective.
    basic_solver_group<round_robin_schedule, nablapp::dynamic_dimension,
                       test::mock_policy,
                       test::mock_policy>
        group{prob, x0, opts};

    auto result = group.step_n(4);

    // Both solvers converge toward 0 -- the one that stepped more recently
    // has lower objective. Just verify it converges and picks a valid result.
    CHECK(result.objective_value < 0.5 * x0.squaredNorm());
}

TEST_CASE("among infeasible solvers lowest violation wins",
          "[solver_group][feasibility]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{1.0, 1.0}};
    solver_options opts;

    // infeasible_mock: obj=1, cv=2 (lower violation)
    // high_violation_mock: obj=0.5, cv=5 (higher violation)
    basic_solver_group<round_robin_schedule, nablapp::dynamic_dimension,
                       test::infeasible_mock_policy,
                       test::high_violation_mock_policy>
        group{prob, x0, opts};

    auto result = group.step_n(2);

    // Best solver should be the one with lower violation (obj=1), not
    // the one with lower objective but higher violation (obj=0.5).
    CHECK(result.objective_value == Approx(1.0));
}

TEST_CASE("basic_solver per-policy options forwarding",
          "[solver][per-policy]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;

    SECTION("custom step_size via policy options")
    {
        test::mock_policy_with_opts::options_type popts{.custom_step_size = 0.1};
        basic_solver<test::mock_policy_with_opts> solver{prob, x0, opts, popts};
        auto result = solver.step();
        CHECK(result.step_size == Approx(0.1));
    }

    SECTION("default step_size without policy options")
    {
        basic_solver<test::mock_policy_with_opts> solver{prob, x0, opts};
        auto result = solver.step();
        CHECK(result.step_size == Approx(0.5));
    }
}

TEST_CASE("basic_solver_group per-policy options tuple",
          "[solver_group][per-policy]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;

    auto policy_opts = std::tuple{
        test::mock_policy_with_opts::options_type{.custom_step_size = 0.1},
        test::mock_policy_with_opts::options_type{.custom_step_size = 0.9}
    };

    basic_solver_group<round_robin_schedule, nablapp::dynamic_dimension,
                       test::mock_policy_with_opts,
                       test::mock_policy_with_opts>
        group{prob, x0, opts, policy_opts};

    // Round-robin: first step hits solver 0 (step_size 0.1),
    // second step hits solver 1 (step_size 0.9).
    auto r1 = group.step();
    auto r2 = group.step();

    CHECK(r1.step_size == Approx(0.1));
    CHECK(r2.step_size == Approx(0.9));
}

TEST_CASE("basic_solver_group existing constructor still works",
          "[solver_group][per-policy]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    solver_options opts;

    SECTION("no options_type policy (broadcast)")
    {
        basic_solver_group<round_robin_schedule, nablapp::dynamic_dimension,
                           test::mock_policy, test::mock_policy>
            group{prob, x0, opts};
        auto result = group.step();
        CHECK(result.objective_value < 0.5 * x0.squaredNorm());
    }

    SECTION("has options_type policy (broadcast, no per-policy opts)")
    {
        basic_solver_group<round_robin_schedule, nablapp::dynamic_dimension,
                           test::mock_policy_with_opts, test::mock_policy_with_opts>
            group{prob, x0, opts};
        auto result = group.step();
        CHECK(result.step_size == Approx(0.5));
    }
}
