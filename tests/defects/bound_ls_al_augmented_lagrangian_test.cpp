#include "argmin/solver/options.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/augmented_lagrangian_policy.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <cstdint>

using namespace argmin;

namespace
{

// |x - (1, 1)|^2 under one linear row a^T x - b, an inequality (>= 0) or an equality. Off the point nan_except the
// objective is NaN unless nan_except is NaN itself, which disables the poison.
struct one_row_bowl
{
    static constexpr std::int32_t problem_dimension = 2;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_inequality() const { return equality ? 0 : 1; }

    std::int32_t num_equality() const { return equality ? 1 : 0; }

    double value(const Eigen::Vector2d& x) const
    {
        if(!nan_except.hasNaN() && x != nan_except)
            return std::numeric_limits<double>::quiet_NaN();
        return (x - Eigen::Vector2d::Ones()).squaredNorm();
    }

    void gradient(const Eigen::Vector2d& x, Eigen::Vector2d& g) const { g = 2.0 * (x - Eigen::Vector2d::Ones()); }

    void constraints(const Eigen::Vector2d& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = a.dot(x) - b;
    }

    void constraint_jacobian(const Eigen::Vector2d&, Eigen::MatrixXd& J) const { J = a.transpose(); }

    Eigen::Vector2d a;
    double b;
    bool equality;
    Eigen::Vector2d nan_except;
};

using al_policy = augmented_lagrangian_policy<lbfgsb_policy<double, 2>, 2>;
using al_solver = step_budget_solver<al_policy, 2, one_row_bowl>;

al_policy::options_type one_inner_solve_per_step()
{
    al_policy::options_type opts;
    opts.inner_chunk = 100000;
    return opts;
}

}

// 5 - x0 >= 0 holds at every iterate from the origin, so each outer step meets the feasibility-progress gate.
TEST_CASE("the augmented Lagrangian keeps its penalty parameter at feasible iterates", "[known-defect][!shouldfail]")
{
    const double no_poison = std::numeric_limits<double>::quiet_NaN();
    const one_row_bowl problem{{-1.0, 0.0}, -5.0, false, Eigen::Vector2d::Constant(no_poison)};
    al_solver solver{al_policy{}, problem, Eigen::Vector2d::Zero().eval(), solver_options<>{}, one_inner_solve_per_step()};
    const double initial_mu = solver.state().mu;
    for(std::int32_t k = 0; k < 5; ++k)
        static_cast<void>(solver.step());
    CHECK(solver.state().mu == initial_mu);
}

// Every point but the start (3, 2) has a NaN objective, so the inner L-BFGS-B solve ends line_search_failed.
TEST_CASE("the augmented Lagrangian reports a failure status when its inner solve fails", "[known-defect][!shouldfail]")
{
    const one_row_bowl problem{{1.0, 1.0}, 1.0, true, {3.0, 2.0}};
    al_solver solver{al_policy{}, problem, Eigen::Vector2d(3.0, 2.0), solver_options<>{}, one_inner_solve_per_step()};
    CHECK(solver.step().policy_status.has_value());
}
