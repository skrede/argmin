#include "eigen_trap.h"

#include "argmin/solver/options.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/result/status.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <cstdint>

using namespace argmin;

namespace
{

struct evaluation_failure
{
};

// f = 0.5 |x - 1|^2 over a reported dimension, so a start of another length is rejected at construction.
template <typename Vector>
struct failing_bowl
{
    static constexpr std::int32_t problem_dimension = Vector::RowsAtCompileTime;

    std::int32_t n;
    bool fail;

    std::int32_t dimension() const { return n; }

    double value(const Vector& x) const
    {
        if(fail)
            throw evaluation_failure{};
        return 0.5 * (x.array() - 1.0).matrix().squaredNorm();
    }

    void gradient(const Vector& x, Vector& g) const { g = (x.array() - 1.0).matrix(); }
};

using fixed_bowl = failing_bowl<Eigen::Vector2d>;
using dynamic_bowl = failing_bowl<Eigen::VectorXd>;

}

TEST_CASE("a reset that throws while building a rejected-start state leaves the solver unbuilt and buildable", "[release-safety][start-length]")
{
    dynamic_bowl problem{.n = 2, .fail = false};
    step_budget_solver solver{lbfgsb_policy<double>{}, problem, Eigen::VectorXd::Constant(3, 0.5).eval(), solver_options<>{}};
    REQUIRE(solver.status() == solver_status::invalid_problem);

    const Eigen::VectorXd start = Eigen::VectorXd::Constant(2, 3.0);
    problem.fail = true;
    CHECK_THROWS_AS(solver.reset(start), evaluation_failure);
    CHECK(solver.status() == solver_status::invalid_problem);
    CHECK(solver.step().policy_status == solver_status::invalid_problem);

    problem.fail = false;
    CHECK(solver.reset(start) == solver_status::running);
    CHECK(solver.state().x == start);
    CHECK_FALSE(solver.step().policy_status == solver_status::invalid_problem);
}

TEST_CASE("resetting a rejected-start fixed-dimension solver from its own placeholder start equals resetting from a copy", "[release-safety][start-length]")
{
    const fixed_bowl problem{.n = 2, .fail = false};
    const Eigen::VectorXd rejected = Eigen::VectorXd::Constant(3, 0.5);
    step_budget_solver aliased{lbfgsb_policy<double, 2>{}, problem, rejected, solver_options<>{}};
    step_budget_solver copied{lbfgsb_policy<double, 2>{}, problem, rejected, solver_options<>{}};
    REQUIRE(aliased.status() == solver_status::invalid_problem);

    const Eigen::Vector2d placeholder = Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN());
    CHECK(aliased.reset(aliased.state().x) == copied.reset(placeholder));
    CHECK(aliased.status() == copied.status());
    CHECK(aliased.state().x.array().isNaN().all());
    CHECK(copied.state().x.array().isNaN().all());
}
