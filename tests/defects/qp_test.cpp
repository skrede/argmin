#include "argmin/qp/qp_types.h"
#include "argmin/qp/dense_admm_qp.h"
#include "argmin/qp/dense_active_set_qp.h"

#include "argmin/options/dense_qp_options.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace argmin;

namespace
{

// P is positive semidefinite of rank 2 with integer entries, so its LDLT pivots are exactly (1, 0, 1).
Eigen::Matrix3d rank_deficient_hessian()
{
    Eigen::Matrix3d P;
    P << 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    return P;
}

expected<qp_result<double>, qp_error> solve_rank_deficient(const dense_qp_options<double>& opts)
{
    const Eigen::VectorXd none(0);
    dense_admm_qp_solver<double> solver(3, 0);
    return solver.solve(rank_deficient_hessian(), Eigen::Vector3d::Constant(-1.0), Eigen::MatrixXd(0, 3), none, none, opts);
}

bool solved(const expected<qp_result<double>, qp_error>& outcome)
{
    return outcome.has_value() && outcome->status == qp_solve_status::solved;
}

// min 0.5 |x|^2 on the equality row x0 + x1 = 1.
struct equality_row_qp
{
    equality_row_qp() : P(Eigen::Matrix2d::Identity()), q(Eigen::Vector2d::Zero()), A(Eigen::MatrixXd::Ones(1, 2)) {}

    Eigen::Matrix2d P;
    Eigen::Vector2d q;
    Eigen::MatrixXd A;
};

expected<qp_result<double>, qp_error> warm_solve_on_row(dense_active_set_qp_solver<double>& solver)
{
    const equality_row_qp d;
    const Eigen::VectorXd one = Eigen::VectorXd::Ones(1);
    REQUIRE_FALSE(solver.warm_start(Eigen::Vector2d(0.5, 0.5), Eigen::VectorXd::Zero(1)).has_value());
    return solver.solve(d.P, d.q, d.A, one, one);
}

}

TEST_CASE("dense ADMM solves the rank-deficient semidefinite QP with the convexity check off", "[qp]")
{
    dense_qp_options<double> opts;
    opts.validate_convexity = false;
    const auto outcome = solve_rank_deficient(opts);
    REQUIRE(solved(outcome));
    CHECK(std::abs(outcome->x[0] + outcome->x[1] - 1.0) <= 1e-3);
    CHECK(std::abs(outcome->x[2] - 1.0) <= 1e-3);
}

TEST_CASE("dense ADMM accepts a rank-deficient positive semidefinite Hessian under the default convexity check", "[known-defect][!shouldfail]")
{
    CHECK(solved(solve_rank_deficient(dense_qp_options<double>{})));
}

TEST_CASE("dense active-set QP solves from a warm start on its equality row", "[qp]")
{
    dense_active_set_qp_solver<double> solver(2, 1);
    const auto outcome = warm_solve_on_row(solver);
    REQUIRE(solved(outcome));
    CHECK((outcome->x - Eigen::Vector2d(0.5, 0.5)).cwiseAbs().maxCoeff() <= 1e-12);
}

TEST_CASE("dense active-set QP resolves after its equality row moves away from the retained point", "[known-defect][!shouldfail]")
{
    dense_active_set_qp_solver<double> solver(2, 1);
    REQUIRE(solved(warm_solve_on_row(solver)));
    const Eigen::VectorXd moved = Eigen::VectorXd::Constant(1, 1.5);
    const auto outcome = solver.resolve(Eigen::Vector2d::Zero(), moved, moved);
    CHECK((solved(outcome) && (outcome->x - Eigen::Vector2d(0.75, 0.75)).cwiseAbs().maxCoeff() <= 1e-12));
}

TEST_CASE("dense active-set QP solves an equality-constrained QP from its default start", "[known-defect][!shouldfail]")
{
    const equality_row_qp d;
    const Eigen::VectorXd one = Eigen::VectorXd::Ones(1);
    dense_active_set_qp_solver<double> solver(2, 1);
    CHECK(solved(solver.solve(d.P, d.q, d.A, one, one)));
}
