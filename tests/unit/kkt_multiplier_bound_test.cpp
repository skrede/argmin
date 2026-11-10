#include "argmin/detail/lagrangian.h"
#include "argmin/detail/kkt_residual.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <limits>

namespace
{

constexpr double inf = std::numeric_limits<double>::infinity();

struct bound_estimate
{
    double mu;
    double kkt;
};

// The row x0 + x1 >= 1 is active at x = (0, 1), the point the box is placed against.
bound_estimate estimate_on_row(const Eigen::Vector2d& g, const Eigen::Vector2d& lower, const Eigen::Vector2d& upper)
{
    const Eigen::Matrix<double, Eigen::Dynamic, 2> J_eq(0, 2);
    const Eigen::Matrix<double, Eigen::Dynamic, 2> J_ineq{{1.0, 1.0}};
    const Eigen::VectorXd c_eq(0);
    const Eigen::VectorXd c_ineq = Eigen::VectorXd::Zero(1);
    const Eigen::Vector2d x(0.0, 1.0);
    argmin::detail::kkt_multiplier_workspace<double, 2> ws;
    Eigen::VectorXd lam_eq(0);
    Eigen::VectorXd mu_ineq(1);
    argmin::detail::compute_kkt_multipliers_active_set<double, 2>(g, J_eq, J_ineq, c_ineq, x, lower, upper, ws, lam_eq, mu_ineq);
    return {mu_ineq[0], argmin::detail::kkt_residual<double, 2, Eigen::Dynamic, Eigen::Dynamic>(g, J_eq, J_ineq, lam_eq, mu_ineq, c_eq, c_ineq, x, lower, upper)};
}

}

TEST_CASE("an active bound keeps its share of the gradient off the general multipliers", "[lagrangian][kkt-bound]")
{
    const bound_estimate at_lower = estimate_on_row(Eigen::Vector2d(3.0, 1.0), Eigen::Vector2d(0.0, -inf), Eigen::Vector2d(inf, inf));
    CHECK(at_lower.mu == 1.0);
    CHECK(at_lower.kkt == 0.0);
    const bound_estimate at_upper = estimate_on_row(Eigen::Vector2d(1.0, -2.0), Eigen::Vector2d(-inf, -inf), Eigen::Vector2d(inf, 1.0));
    CHECK(at_upper.mu == 1.0);
    CHECK(at_upper.kkt == 0.0);
}

TEST_CASE("a wrong-signed bound multiplier is not certified", "[lagrangian][kkt-bound]")
{
    const bound_estimate at_lower = estimate_on_row(Eigen::Vector2d(-1.0, 1.0), Eigen::Vector2d(0.0, -inf), Eigen::Vector2d(inf, inf));
    CHECK(at_lower.mu == 1.0);
    CHECK(at_lower.kkt == 2.0);
    const bound_estimate at_upper = estimate_on_row(Eigen::Vector2d(1.0, 2.0), Eigen::Vector2d(-inf, -inf), Eigen::Vector2d(inf, 1.0));
    CHECK(at_upper.mu == 1.0);
    CHECK(at_upper.kkt == 1.0);
}

TEST_CASE("a bound whose multiplier-slack product exceeds the threshold is not certified", "[kkt_residual][kkt-bound]")
{
    const Eigen::Matrix<double, Eigen::Dynamic, 2> J(0, 2);
    const Eigen::VectorXd none(0);
    const Eigen::Vector2d g(1000.0, 0.0);
    const Eigen::Vector2d lower(0.0, -inf);
    const Eigen::Vector2d upper(inf, inf);
    const auto kkt_at = [&](const Eigen::Vector2d& x) {
        return argmin::detail::kkt_residual<double, 2, Eigen::Dynamic, Eigen::Dynamic>(g, J, J, none, none, none, none, x, lower, upper);
    };
    CHECK(kkt_at(Eigen::Vector2d(1e-10, 0.0)) == (1000.0 - 1e-10) * 1e-10);
    CHECK(kkt_at(Eigen::Vector2d(0.0, 0.0)) == 0.0);
}
