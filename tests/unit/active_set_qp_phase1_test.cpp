#include "argmin/detail/active_set_qp.h"

#include "argmin/options/qp_options.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>

using Catch::Approx;
using argmin::detail::qp_status;
using solver_type = argmin::detail::active_set_qp_solver<double>;

namespace
{

constexpr double inf = std::numeric_limits<double>::infinity();

// min 0.5 ||x||^2 + d^T x over A_eq x = b_eq and A_in x >= b_in.
struct qp
{
    Eigen::MatrixXd A_eq;
    Eigen::VectorXd b_eq;
    Eigen::MatrixXd A_in;
    Eigen::VectorXd b_in;
    Eigen::VectorXd d;
};

solver_type::result_type solve_from(const qp& p, const Eigen::VectorXd& x0)
{
    const Eigen::Index n = x0.size();
    solver_type solver(static_cast<int>(n), static_cast<int>(p.A_eq.rows() + p.A_in.rows()));
    return solver.solve(Eigen::MatrixXd::Identity(n, n).eval(), p.d, p.A_eq, p.b_eq, p.A_in, p.b_in, x0);
}

}

TEST_CASE("active_set_qp_solver solves from a start that violates an inequality row", "[qp]")
{
    // Solved by (1, 1) with multiplier 1; the start is the unconstrained minimizer, so the loop alone would stop there.
    const qp p{Eigen::MatrixXd(0, 2), Eigen::VectorXd(0), Eigen::MatrixXd{{1.0, 1.0}}, Eigen::VectorXd{{2.0}}, Eigen::VectorXd::Zero(2)};
    const auto r = solve_from(p, Eigen::VectorXd::Zero(2));
    REQUIRE(r.status == qp_status::optimal);
    CHECK((r.x - Eigen::Vector2d(1.0, 1.0)).cwiseAbs().maxCoeff() <= 1e-12);
    CHECK(r.lambda[0] == Approx(1.0).margin(1e-12));
}

TEST_CASE("active_set_qp_solver solves from a start outside its box", "[qp]")
{
    solver_type solver(2, 4);
    const auto r = solver.solve(Eigen::MatrixXd::Identity(2, 2).eval(), Eigen::VectorXd{{-3.0, -3.0}}, Eigen::MatrixXd(0, 2),
                                Eigen::VectorXd(0), Eigen::MatrixXd(0, 2), Eigen::VectorXd(0), Eigen::VectorXd::Zero(2).eval(),
                                Eigen::VectorXd{{1.0, 2.0}}, Eigen::VectorXd{{3.0, 3.0}});
    REQUIRE(r.status == qp_status::optimal);
    CHECK((r.x - Eigen::Vector2d(1.0, 2.0)).cwiseAbs().maxCoeff() <= 1e-12);
}

TEST_CASE("active_set_qp_solver keeps a dependent equality pair while it restores an inequality row", "[qp]")
{
    // x0 + x1 + x2 = 3 stated twice and x0 >= 2: solved by (2, 0.5, 0.5) with multipliers 0.5, 0 and 1.5.
    const qp p{Eigen::MatrixXd{{1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}}, Eigen::VectorXd{{3.0, 6.0}}, Eigen::MatrixXd{{1.0, 0.0, 0.0}},
               Eigen::VectorXd{{2.0}}, Eigen::VectorXd::Zero(3)};
    const auto r = solve_from(p, Eigen::VectorXd::Zero(3));
    REQUIRE(r.status == qp_status::optimal);
    CHECK((r.x - Eigen::Vector3d(2.0, 0.5, 0.5)).cwiseAbs().maxCoeff() <= 1e-12);
    CHECK(r.lambda[2] == Approx(1.5).margin(1e-12));
}

TEST_CASE("active_set_qp_solver reports infeasible when an inequality row contradicts the equality rows", "[qp]")
{
    const qp p{Eigen::MatrixXd{{1.0, 0.0}}, Eigen::VectorXd{{0.0}}, Eigen::MatrixXd{{2.0, 0.0}}, Eigen::VectorXd{{1.0}}, Eigen::VectorXd::Zero(2)};
    const auto r = solve_from(p, Eigen::VectorXd::Zero(2));
    CHECK(r.status == qp_status::infeasible);
    CHECK(r.iterations == 0);
}

TEST_CASE("active_set_qp_solver reports infeasible for inequality rows with no common point", "[qp]")
{
    const qp p{Eigen::MatrixXd(0, 2), Eigen::VectorXd(0), Eigen::MatrixXd{{1.0, 1.0}, {-1.0, -1.0}}, Eigen::VectorXd{{1.0, 0.0}},
               Eigen::VectorXd::Zero(2)};
    CHECK(solve_from(p, Eigen::VectorXd::Zero(2)).status == qp_status::infeasible);
}

TEST_CASE("active_set_qp_solver restores inequality rows whose right-hand sides are far from unit scale", "[qp]")
{
    // x0 + x1 >= 1e9 and x0 - x1 >= 1e9 meet at (1e9, 0), and the absent row and the bound at 1e20 never bind.
    const qp p{Eigen::MatrixXd(0, 2), Eigen::VectorXd(0), Eigen::MatrixXd{{1.0, 1.0}, {1.0, -1.0}, {0.0, 1.0}, {0.0, -1.0}},
               Eigen::VectorXd{{1e9, 1e9, -inf, -1e20}}, Eigen::VectorXd::Zero(2)};
    const auto r = solve_from(p, Eigen::VectorXd::Zero(2));
    REQUIRE(r.status == qp_status::optimal);
    CHECK(r.x[0] == Approx(1e9).epsilon(1e-12));
    CHECK(std::abs(r.x[1]) <= 1e-12 * 1e9);
}

TEST_CASE("active_set_qp_solver solves from a start that violates an inequality row in single precision", "[qp]")
{
    argmin::detail::active_set_qp_solver<float> solver(2, 1);
    const auto r = solver.solve(Eigen::MatrixXf::Identity(2, 2).eval(), Eigen::VectorXf::Zero(2).eval(), Eigen::MatrixXf(0, 2),
                                Eigen::VectorXf(0), Eigen::MatrixXf{{1.0f, 1.0f}}, Eigen::VectorXf{{2.0f}}, Eigen::VectorXf::Zero(2).eval());
    REQUIRE(r.status == qp_status::optimal);
    CHECK((r.x - Eigen::Vector2f(1.0f, 1.0f)).cwiseAbs().maxCoeff() <= 1e-5f);
}

TEST_CASE("active_set_qp_solver solves rows its dual solve fails on instead of reporting them infeasible", "[qp]")
{
    // A thin wedge of rows of norm about 1e-3 around the feasible optimum; the dual NNLS returns an unchecked u there.
    const qp p{Eigen::MatrixXd(0, 2), Eigen::VectorXd(0),
               Eigen::MatrixXd{{0.00088851983262812296, -2.9855235365240096e-05}, {-0.00088957322490335403, 2.9796172325970206e-05},
                               {0.00084302716795813047, 0.00075115226958537577}, {0.0018029771534392072, 0.0011043928579334077},
                               {-0.00072103306346573825, 0.00037490658196518025}},
               Eigen::VectorXd{{0.0034314305442807135, -0.003435363730948634, 0.0021419104735822779, 0.0047481155847467222,
                                -0.0035835173508430608}},
               Eigen::VectorXd{{3.3873458718784111, 4.5247996190001665}}};
    const auto r = solve_from(p, Eigen::VectorXd{{4.6463735623912887, 2.4160338929914991}});
    REQUIRE(r.status == qp_status::optimal);
    CHECK((p.A_in * r.x - p.b_in).minCoeff() >= -1e-12);
}

TEST_CASE("active_set_qp_solver does not report infeasible rows that meet in one point a rounding error away", "[qp]")
{
    const Eigen::MatrixXd A{{1.0, 0.0}, {-0.5048461045998576, 0.8632093666488737}, {-0.4902608213406994, -0.8715757724135882}};
    const Eigen::VectorXd xf{{0.1, 0.7}};
    const qp p{Eigen::MatrixXd(0, 2), Eigen::VectorXd(0), A, A * xf, Eigen::VectorXd{{1.0, -1.0}}};
    const auto r = solve_from(p, xf + Eigen::VectorXd{{1e-9, -2e-9}});
    REQUIRE(r.status == qp_status::optimal);
    CHECK((r.x - xf).cwiseAbs().maxCoeff() <= 1e-12);
}

TEST_CASE("active_set_qp_solver restores a row left out of its projection for its slack when the move violates it", "[qp]")
{
    // The wedge pair meets at x2 = 200, so the move violates x3 - x2 >= -150, a row first left out for its slack.
    const qp p{Eigen::MatrixXd(0, 3), Eigen::VectorXd(0), Eigen::MatrixXd{{1.0, 0.005, 0.0}, {-1.0, 0.005, 0.0}, {0.0, -1.0, 1.0}},
               Eigen::VectorXd{{1.0, 1.0, -150.0}}, Eigen::VectorXd::Zero(3)};
    const auto r = solve_from(p, Eigen::VectorXd::Zero(3));
    REQUIRE(r.status == qp_status::optimal);
    CHECK((r.x - Eigen::Vector3d(0.0, 200.0, 50.0)).cwiseAbs().maxCoeff() <= 1e-9);
    CHECK(r.primal_feasible);
}

TEST_CASE("active_set_qp_solver does not report infeasible a sharp wedge whose rows meet far from the start", "[qp]")
{
    const qp p{Eigen::MatrixXd(0, 2), Eigen::VectorXd(0), Eigen::MatrixXd{{1.0, 1e-8}, {-1.0, 1e-8}}, Eigen::VectorXd{{1.0, 1.0}},
               Eigen::VectorXd::Zero(2)};
    CHECK(solve_from(p, Eigen::VectorXd::Zero(2)).status != qp_status::infeasible);
}

TEST_CASE("active_set_qp_solver does not report infeasible a row the equality rows fix within its allowance", "[qp]")
{
    const qp p{Eigen::MatrixXd{{1.0, 0.0}}, Eigen::VectorXd{{0.0}}, Eigen::MatrixXd{{1.0, 0.0}}, Eigen::VectorXd{{1e-10}}, Eigen::VectorXd::Zero(2)};
    const auto r = solve_from(p, Eigen::VectorXd::Zero(2));
    CHECK(r.status == qp_status::optimal);
    CHECK(r.primal_feasible);
}

TEST_CASE("active_set_qp_solver returns the equality-projected start for inequality rows with no common point", "[qp]")
{
    const qp p{Eigen::MatrixXd{{0.0, 1.0}}, Eigen::VectorXd{{3.0}}, Eigen::MatrixXd{{1.0, 0.0}, {-1.0, 0.0}}, Eigen::VectorXd{{1.0, 0.0}},
               Eigen::VectorXd::Zero(2)};
    const auto r = solve_from(p, Eigen::VectorXd::Zero(2));
    REQUIRE(r.status == qp_status::infeasible);
    CHECK((r.x - Eigen::Vector2d(0.0, 3.0)).cwiseAbs().maxCoeff() <= 1e-15);
    CHECK_FALSE(r.primal_feasible);
}
