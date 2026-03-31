#include "mock_policy.h"
#include "nablapp/schedule/basic_solver_group.h"
#include "nablapp/schedule/round_robin_schedule.h"
#include "nablapp/result/status.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using namespace nablapp;

struct quadratic_group
{
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
        basic_solver_group<round_robin_schedule, test::mock_policy, test::mock_policy>
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

    basic_solver_group<round_robin_schedule, test::mock_policy, test::mock_policy>
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

    basic_solver_group<round_robin_schedule, test::mock_policy, test::mock_policy>
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

    basic_solver_group<round_robin_schedule, test::mock_policy, test::non_converging_policy>
        group{prob, x0};
    (void)group;
    SUCCEED("fold expression concept check passed at compile time");
}
