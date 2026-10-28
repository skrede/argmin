#include "oracle_tolerances.h"

#include "oracle/qp_oracle.h"

#include "argmin/qp/qp_types.h"
#include "argmin/qp/dense_admm_qp.h"
#include "argmin/qp/sparse_admm_qp.h"

#include "argmin/options/dense_qp_options.h"
#include "argmin/options/sparse_qp_options.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>

using namespace argmin;

namespace
{

constexpr double inf = std::numeric_limits<double>::infinity();

struct qp_data
{
    Eigen::MatrixXd P;
    Eigen::VectorXd q;
    Eigen::MatrixXd A;
    Eigen::VectorXd l;
    Eigen::VectorXd u;
};

qp_data unsatisfiable_row_qp()
{
    return {Eigen::MatrixXd::Identity(2, 2), Eigen::VectorXd::Zero(2), Eigen::RowVector2d(1.0, 0.0), Eigen::VectorXd::Constant(1, inf), Eigen::VectorXd::Constant(1, inf)};
}

// x >= 2 and x = 2 below the unconstrained minimizer 10: only the equality row may carry the multiplier.
qp_data duplicated_row_qp()
{
    return {Eigen::MatrixXd::Identity(1, 1), Eigen::VectorXd::Constant(1, -10.0), Eigen::Vector2d(1.0, 1.0), Eigen::Vector2d(2.0, 2.0), Eigen::Vector2d(inf, 2.0)};
}

qp_data contradictory_box_qp()
{
    return {Eigen::MatrixXd::Identity(1, 1), Eigen::VectorXd::Zero(1), Eigen::Vector2d(1.0, 1.0), Eigen::Vector2d(1.0, -inf), Eigen::Vector2d(inf, 0.0)};
}

Eigen::SparseMatrix<double> sparse(const Eigen::MatrixXd& dense)
{
    return dense.sparseView();
}

bool refused_or_certified(const expected<qp_result<double>, qp_error>& outcome)
{
    return !outcome || (outcome->status == qp_solve_status::primal_infeasible && outcome->x.allFinite());
}

oracle::qp_verdict judge_dense(const qp_data& d, const qp_result<double>& result)
{
    return oracle::judge_qp(d.P, d.q, d.A, d.l, d.u, result, oracle::qp_tolerances_of(dense_qp_options<double>{}));
}

oracle::qp_verdict judge_sparse(const qp_data& d, const qp_result<double>& result)
{
    return oracle::judge_qp(sparse(d.P), d.q, sparse(d.A), d.l, d.u, result, oracle::qp_tolerances_of(sparse_qp_options<double>{}));
}

// P = 0 with no rows: unbounded below along +x, then along -x once a resolve reverses the sign of q.
qp_result<double> reversed_resolve(dense_admm_qp_solver<double>& solver)
{
    const Eigen::VectorXd none(0);
    REQUIRE(solver.solve(Eigen::MatrixXd::Zero(1, 1), Eigen::VectorXd::Constant(1, -0.125), Eigen::MatrixXd(0, 1), none, none).has_value());
    const auto outcome = solver.resolve(Eigen::VectorXd::Constant(1, 0.068359375), none, none);
    REQUIRE(outcome.has_value());
    return *outcome;
}

}

TEST_CASE("dense ADMM refuses a row bounded below by +inf or certifies it infeasible with a finite x", "[known-defect][!shouldfail]")
{
    const qp_data d = unsatisfiable_row_qp();
    dense_admm_qp_solver<double> solver(2, 1);
    CHECK(refused_or_certified(solver.solve(d.P, d.q, d.A, d.l, d.u)));
}

TEST_CASE("sparse ADMM refuses a row bounded below by +inf or certifies it infeasible with a finite x", "[known-defect][!shouldfail]")
{
    const qp_data d = unsatisfiable_row_qp();
    sparse_admm_qp_solver<double> solver;
    CHECK(refused_or_certified(solver.solve(sparse(d.P), d.q, sparse(d.A), d.l, d.u)));
}

TEST_CASE("dense ADMM solves the duplicated-row QP at the shared bound", "[qp]")
{
    const qp_data d = duplicated_row_qp();
    dense_admm_qp_solver<double> solver(1, 2);
    const auto outcome = solver.solve(d.P, d.q, d.A, d.l, d.u);
    REQUIRE(outcome.has_value());
    CHECK(outcome->status == qp_solve_status::solved);
    CHECK(std::abs(outcome->x[0] - 2.0) < 1e-6);
}

TEST_CASE("dense ADMM polish keeps a multiplier off the one-sided row of a duplicated bound", "[known-defect][!shouldfail]")
{
    const qp_data d = duplicated_row_qp();
    dense_admm_qp_solver<double> solver(1, 2);
    const oracle::qp_verdict v = judge_dense(d, *solver.solve(d.P, d.q, d.A, d.l, d.u));
    CAPTURE(v.failing);
    CHECK(v.ok);
}

TEST_CASE("sparse ADMM polish keeps a multiplier off the one-sided row of a duplicated bound", "[known-defect][!shouldfail]")
{
    const qp_data d = duplicated_row_qp();
    sparse_admm_qp_solver<double> solver;
    const oracle::qp_verdict v = judge_sparse(d, *solver.solve(sparse(d.P), d.q, sparse(d.A), d.l, d.u));
    CAPTURE(v.failing);
    CHECK(v.ok);
}

TEST_CASE("dense ADMM reports dual_infeasible after a resolve that reverses the linear term", "[qp]")
{
    dense_admm_qp_solver<double> solver(1, 0);
    CHECK(reversed_resolve(solver).status == qp_solve_status::dual_infeasible);
}

TEST_CASE("dense ADMM returns an unbounded direction in x after a resolve that reverses the linear term", "[known-defect][!shouldfail]")
{
    dense_admm_qp_solver<double> solver(1, 0);
    const qp_data d{Eigen::MatrixXd::Zero(1, 1), Eigen::VectorXd::Constant(1, 0.068359375), Eigen::MatrixXd(0, 1), Eigen::VectorXd(0), Eigen::VectorXd(0)};
    const oracle::qp_verdict v = judge_dense(d, reversed_resolve(solver));
    CAPTURE(v.failing);
    CHECK(v.ok);
}

TEST_CASE("sparse ADMM returns a primal infeasibility certificate in y", "[known-defect][!shouldfail]")
{
    const qp_data d = contradictory_box_qp();
    sparse_admm_qp_solver<double> solver;
    const auto outcome = solver.solve(sparse(d.P), d.q, sparse(d.A), d.l, d.u);
    REQUIRE(outcome.has_value());
    REQUIRE(outcome->status == qp_solve_status::primal_infeasible);
    const oracle::qp_verdict v = judge_sparse(d, *outcome);
    CAPTURE(v.failing);
    CHECK(v.ok);
}

TEST_CASE("sparse ADMM keeps a non-finite warm start out of the iterate", "[known-defect][!shouldfail]")
{
    const Eigen::VectorXd none(0);
    const Eigen::SparseMatrix<double> P = sparse(Eigen::MatrixXd::Identity(1, 1));
    sparse_admm_qp_solver<double> solver;
    REQUIRE(solver.solve(P, Eigen::VectorXd::Ones(1), sparse(Eigen::MatrixXd(0, 1)), none, none).has_value());
    solver.warm_start(Eigen::VectorXd::Constant(1, std::numeric_limits<double>::quiet_NaN()), none);
    const auto outcome = solver.resolve(Eigen::VectorXd::Ones(1), none, none);
    CHECK((!outcome || outcome->x.allFinite()));
}
