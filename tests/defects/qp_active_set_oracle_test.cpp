#include "oracle_tolerances.h"

#include "oracle/qp_oracle.h"

#include "argmin/qp/qp_types.h"
#include "argmin/qp/dense_active_set_qp.h"

#include "argmin/options/dense_active_set_qp_options.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <limits>

using namespace argmin;

namespace
{

// Gram matrix of a 4 x 4 factor with an eigenvalue of 8.6e-15, written exactly.
Eigen::Matrix4d round_off_singular_hessian()
{
    Eigen::Matrix4d P;
    P << 0x1.914p-8, -0x1.ep-8, -0x1.6p-14, -0x1.3d8p-6, -0x1.ep-8, 0x1.9019p+6, 0x1.9014p-1, 0x1.7cp-1, -0x1.6p-14, 0x1.9014p-1, 0x1.905p-8, 0x1.76p-8, -0x1.3d8p-6, 0x1.7cp-1, 0x1.76p-8, 0x1.32p-4;
    return P;
}

}

TEST_CASE("dense active-set QP refuses a row bounded below by +inf or reports it infeasible", "[known-defect][!shouldfail]")
{
    const double inf = std::numeric_limits<double>::infinity();
    dense_active_set_qp_solver<double> solver(2, 1);
    const auto outcome = solver.solve(Eigen::MatrixXd::Identity(2, 2), Eigen::VectorXd::Zero(2), Eigen::MatrixXd(Eigen::RowVector2d(1.0, 0.0)), Eigen::VectorXd::Constant(1, inf), Eigen::VectorXd::Constant(1, inf));
    CHECK((!outcome || outcome->status == qp_solve_status::primal_infeasible));
}

TEST_CASE("dense active-set QP refuses a Hessian singular to round-off or returns a finite x", "[known-defect][!shouldfail]")
{
    const Eigen::VectorXd none(0);
    dense_active_set_qp_solver<double> solver(4, 0);
    const auto outcome = solver.solve(round_off_singular_hessian(), Eigen::Vector4d(0.0, -0.25, -0.0009765625, 0.0), Eigen::MatrixXd(0, 4), none, none);
    CHECK((!outcome || outcome->x.allFinite()));
}

// P = b b^T for b = (2, 0, 84, 268) is singular on the subspace the equality rows leave free, and q has a component
// along it, so the problem is unbounded below.
TEST_CASE("dense active-set QP claims solved on a singular reduced Hessian only at a point that meets its rows", "[known-defect][!shouldfail]")
{
    const Eigen::Vector4d b(2.0, 0.0, 84.0, 268.0);
    Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
    A(0, 0) = 0.001953125;
    A(0, 2) = 0.0419921875;
    A(2, 1) = 0.03125;
    A(3, 1) = 0.0625;
    const Eigen::Vector4d q(0.0, 0.078125, 352.0, 0.0);
    const Eigen::Vector4d bounds = Eigen::Vector4d::Zero();
    dense_active_set_qp_solver<double> solver(4, 4);
    const auto outcome = solver.solve(b * b.transpose(), q, A, bounds, bounds);
    REQUIRE(outcome.has_value());
    REQUIRE(outcome->status == qp_solve_status::solved);
    const oracle::qp_verdict v = oracle::judge_qp(Eigen::MatrixXd(b * b.transpose()), q, A, bounds, bounds, *outcome, oracle::qp_tolerances_of(dense_active_set_qp_options<double>{}));
    CAPTURE(v.failing);
    CHECK(v.ok);
}
