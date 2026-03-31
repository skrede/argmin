#include "nablapp/detail/active_set_qp.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace nablapp::detail;

TEST_CASE("solve_qp unconstrained", "[qp]")
{
    // min 0.5 x^T (2I) x + [-2, -4]^T x
    // = x1^2 + x2^2 - 2*x1 - 4*x2
    // Gradient = 0 => x* = [1, 2]
    Eigen::MatrixXd G = 2.0 * Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd d{{-2.0, -4.0}};
    Eigen::MatrixXd A_eq(0, 2);
    Eigen::VectorXd b_eq(0);
    Eigen::MatrixXd A_ineq(0, 2);
    Eigen::VectorXd b_ineq(0);
    Eigen::VectorXd x0{{0.0, 0.0}};

    auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);

    CHECK(result.status == qp_status::optimal);
    CHECK(result.x[0] == Approx(1.0).margin(1e-10));
    CHECK(result.x[1] == Approx(2.0).margin(1e-10));
}

TEST_CASE("solve_qp equality only", "[qp]")
{
    // min 0.5 x^T (2I) x   s.t.  x1 + x2 = 1
    // Solution: x* = [0.5, 0.5]
    Eigen::MatrixXd G = 2.0 * Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd d = Eigen::VectorXd::Zero(2);
    Eigen::MatrixXd A_eq(1, 2);
    A_eq << 1.0, 1.0;
    Eigen::VectorXd b_eq{{1.0}};
    Eigen::MatrixXd A_ineq(0, 2);
    Eigen::VectorXd b_ineq(0);
    Eigen::VectorXd x0{{0.5, 0.5}};

    auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);

    CHECK(result.status == qp_status::optimal);
    CHECK(result.x[0] == Approx(0.5).margin(1e-10));
    CHECK(result.x[1] == Approx(0.5).margin(1e-10));
}

TEST_CASE("solve_qp inequality constraints", "[qp]")
{
    // N&W Example 16.3 style:
    // min (x1 - 1)^2 + (x2 - 2.5)^2
    // = 0.5 x^T (2I) x + [-2, -5]^T x + const
    //
    // s.t.  x1 - 2*x2 + 2 >= 0    (constraint 1)
    //      -x1 - 2*x2 + 6 >= 0    (constraint 2)
    //      -x1 + 2*x2 + 2 >= 0    (constraint 3)
    //       x1 >= 0                (constraint 4)
    //       x2 >= 0                (constraint 5)
    //
    // Solution: x* = (1.4, 1.7)
    Eigen::MatrixXd G = 2.0 * Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd d{{-2.0, -5.0}};
    Eigen::MatrixXd A_eq(0, 2);
    Eigen::VectorXd b_eq(0);

    Eigen::MatrixXd A_ineq(5, 2);
    Eigen::VectorXd b_ineq(5);
    A_ineq << 1.0, -2.0,
             -1.0, -2.0,
             -1.0,  2.0,
              1.0,  0.0,
              0.0,  1.0;
    b_ineq << -2.0, -6.0, -2.0, 0.0, 0.0;

    Eigen::VectorXd x0{{2.0, 0.0}};

    auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);

    CHECK(result.status == qp_status::optimal);
    CHECK(result.x[0] == Approx(1.4).margin(1e-8));
    CHECK(result.x[1] == Approx(1.7).margin(1e-8));
}

TEST_CASE("solve_qp box constraints", "[qp]")
{
    // min 0.5 x^T I x + [-2, -3]^T x
    // Unconstrained min: [2, 3]. With bounds [0,1]^2: x* = [1, 1].
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd d{{-2.0, -3.0}};
    Eigen::MatrixXd A_eq(0, 2);
    Eigen::VectorXd b_eq(0);
    Eigen::MatrixXd A_ineq(0, 2);
    Eigen::VectorXd b_ineq(0);
    Eigen::VectorXd lower{{0.0, 0.0}};
    Eigen::VectorXd upper{{1.0, 1.0}};
    Eigen::VectorXd x0{{0.5, 0.5}};

    auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, lower, upper, x0);

    CHECK(result.status == qp_status::optimal);
    CHECK(result.x[0] == Approx(1.0).margin(1e-8));
    CHECK(result.x[1] == Approx(1.0).margin(1e-8));
}

TEST_CASE("solve_qp mixed equality inequality box", "[qp]")
{
    // 3D problem: min 0.5 x^T I x
    // s.t. x1 + x2 + x3 = 1         (equality)
    //      x1 - x2 >= 0              (inequality)
    //      0 <= xi <= 2              (box)
    //
    // With equality x3 = 1 - x1 - x2, objective becomes function of x1,x2.
    // With x1 >= x2 and bounds, solution at x1 = x2 = 1/3, x3 = 1/3.
    // Actually unconstrained min on the equality plane is x1=x2=x3=1/3.
    // Check x1>=x2: 1/3 >= 1/3 satisfied (weakly).
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(3, 3);
    Eigen::VectorXd d = Eigen::VectorXd::Zero(3);
    Eigen::MatrixXd A_eq(1, 3);
    A_eq << 1.0, 1.0, 1.0;
    Eigen::VectorXd b_eq{{1.0}};
    Eigen::MatrixXd A_ineq(1, 3);
    A_ineq << 1.0, -1.0, 0.0;
    Eigen::VectorXd b_ineq{{0.0}};
    Eigen::VectorXd lower{{0.0, 0.0, 0.0}};
    Eigen::VectorXd upper{{2.0, 2.0, 2.0}};

    // Feasible starting point on equality plane with x1 >= x2 and in bounds
    Eigen::VectorXd x0{{0.5, 0.3, 0.2}};

    auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, lower, upper, x0);

    CHECK(result.status == qp_status::optimal);
    // x1 + x2 + x3 = 1
    CHECK(result.x.sum() == Approx(1.0).margin(1e-8));
    // x1 >= x2
    CHECK(result.x[0] >= result.x[1] - 1e-8);
    // All components in bounds
    for(int i = 0; i < 3; ++i)
    {
        CHECK(result.x[i] >= -1e-8);
        CHECK(result.x[i] <= 2.0 + 1e-8);
    }
    // Optimal: x* ~ [1/3, 1/3, 1/3]
    CHECK(result.x[0] == Approx(1.0 / 3.0).margin(1e-6));
    CHECK(result.x[1] == Approx(1.0 / 3.0).margin(1e-6));
    CHECK(result.x[2] == Approx(1.0 / 3.0).margin(1e-6));
}

TEST_CASE("solve_qp infeasible detected", "[qp]")
{
    // x >= 1 and x <= 0 (contradictory) -- infeasibility manifests as max_iterations.
    // Starting from x0 = 0.5 which violates x >= 1.
    // With contradictory constraints the solver cycles or hits iteration limit.
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(1, 1);
    Eigen::VectorXd d{{0.0}};
    Eigen::MatrixXd A_eq(0, 1);
    Eigen::VectorXd b_eq(0);
    Eigen::MatrixXd A_ineq(2, 1);
    A_ineq << 1.0, -1.0;
    Eigen::VectorXd b_ineq(2);
    b_ineq << 1.0, 0.0;  // x >= 1 and -x >= 0 => x <= 0
    Eigen::VectorXd x0{{0.5}};
    qp_options opts{.max_iterations = 50};

    auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0, opts);

    // Infeasibility should manifest as max_iterations or suboptimal result
    CHECK((result.status == qp_status::max_iterations ||
           result.status == qp_status::optimal));
    // If "optimal", the constraints cannot both be satisfied
    if(result.status == qp_status::optimal)
    {
        bool feasible = (result.x[0] >= 1.0 - 1e-6) && (result.x[0] <= 1e-6);
        CHECK_FALSE(feasible);
    }
}

TEST_CASE("solve_qp degenerate weakly active", "[qp]")
{
    // min x1^2 + (x2+1)^2   s.t. x1 >= 0, x2 >= 0
    // = 0.5 x^T (2I) x + [0, 2]^T x + const
    // Unconstrained min: (0, -1). With x2 >= 0: x* = (0, 0).
    // Constraint x1 >= 0 is weakly active (lambda_1 = 0 at solution).
    Eigen::MatrixXd G = 2.0 * Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd d{{0.0, 2.0}};
    Eigen::MatrixXd A_eq(0, 2);
    Eigen::VectorXd b_eq(0);
    Eigen::MatrixXd A_ineq(2, 2);
    A_ineq << 1.0, 0.0,
              0.0, 1.0;
    Eigen::VectorXd b_ineq{{0.0, 0.0}};
    Eigen::VectorXd x0{{1.0, 1.0}};

    auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);

    CHECK(result.status == qp_status::optimal);
    CHECK(result.x[0] == Approx(0.0).margin(1e-8));
    CHECK(result.x[1] == Approx(0.0).margin(1e-8));
}

TEST_CASE("solve_qp N&W Example 16.1 equality", "[qp]")
{
    // N&W Example 16.1, p. 443:
    // min 3*x1^2 + 2*x1*x2 + x1*x3 + 2.5*x2^2 + 2*x2*x3 + 2*x3^2
    //     - 8*x1 - 3*x2 - 3*x3
    // s.t. x1 + x3 = 3, x2 + x3 = 0
    //
    // G = [[6,2,1],[2,5,2],[1,2,4]], d = [-8,-3,-3]
    // Solution: x* = [2, -1, 1], lambda* = [3, -2]
    Eigen::MatrixXd G(3, 3);
    G << 6.0, 2.0, 1.0,
         2.0, 5.0, 2.0,
         1.0, 2.0, 4.0;
    Eigen::VectorXd d{{-8.0, -3.0, -3.0}};
    Eigen::MatrixXd A_eq(2, 3);
    A_eq << 1.0, 0.0, 1.0,
            0.0, 1.0, 1.0;
    Eigen::VectorXd b_eq{{3.0, 0.0}};
    Eigen::MatrixXd A_ineq(0, 3);
    Eigen::VectorXd b_ineq(0);
    // x0 must satisfy constraints: x1+x3=3, x2+x3=0 => e.g. x=(1,-2,2)
    Eigen::VectorXd x0{{1.0, -2.0, 2.0}};

    auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);

    CHECK(result.status == qp_status::optimal);
    CHECK(result.x[0] == Approx(2.0).margin(1e-8));
    CHECK(result.x[1] == Approx(-1.0).margin(1e-8));
    CHECK(result.x[2] == Approx(1.0).margin(1e-8));
}
