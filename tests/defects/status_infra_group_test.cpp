#include "argmin/schedule/basic_solver_group.h"
#include "argmin/schedule/round_robin_schedule.h"

#include "argmin/solver/options.h"
#include "argmin/solver/lbfgsb_policy.h"

#include "argmin/result/status.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <cstdint>

using namespace argmin;

namespace
{

// |x - (1, 1)|^2 on [-5, 5]^2 with a NaN objective everywhere but the start (3, 2), so every L-BFGS-B line
// search fails.
struct poisoned_box_bowl
{
    static constexpr std::int32_t problem_dimension = 2;

    std::int32_t dimension() const { return 2; }

    double value(const Eigen::Vector2d& x) const
    {
        if(x != Eigen::Vector2d(3.0, 2.0))
            return std::numeric_limits<double>::quiet_NaN();
        return (x - Eigen::Vector2d::Ones()).squaredNorm();
    }

    void gradient(const Eigen::Vector2d& x, Eigen::Vector2d& g) const { g = 2.0 * (x - Eigen::Vector2d::Ones()); }

    Eigen::Vector2d lower_bounds() const { return Eigen::Vector2d::Constant(-5.0); }

    Eigen::Vector2d upper_bounds() const { return Eigen::Vector2d::Constant(5.0); }
};

using lbfgsb_pair = basic_solver_group<round_robin_schedule, 2, poisoned_box_bowl, lbfgsb_policy<double, 2>, lbfgsb_policy<double, 2>>;

solver_options<> fifty_iterations()
{
    solver_options<> opts;
    opts.max_iterations = 50;
    return opts;
}

}

TEST_CASE("a solver group stops once every member reports line_search_failed", "[known-defect][!shouldfail]")
{
    const poisoned_box_bowl problem{};
    lbfgsb_pair group{problem, Eigen::VectorXd(Eigen::Vector2d(3.0, 2.0)), fifty_iterations()};
    CHECK(group.solve().iterations == 2);
}

TEST_CASE("a solver group reports line_search_failed for a member whose line search failed", "[known-defect][!shouldfail]")
{
    const poisoned_box_bowl problem{};
    lbfgsb_pair group{problem, Eigen::VectorXd(Eigen::Vector2d(3.0, 2.0)), fifty_iterations()};
    static_cast<void>(group.solve());
    CHECK(group.results()[0].status == solver_status::line_search_failed);
}
