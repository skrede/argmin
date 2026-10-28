#include "oracle_tolerances.h"

#include "oracle/qp_oracle.h"

#include "argmin/qp/dense_admm_qp.h"
#include "argmin/qp/sparse_admm_qp.h"
#include "argmin/qp/dense_active_set_qp.h"

#include "argmin/options/dense_qp_options.h"
#include "argmin/options/sparse_qp_options.h"
#include "argmin/options/dense_active_set_qp_options.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <catch2/catch_test_macros.hpp>

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

// The equality row x0 + x1 = 1 and the upper bound on x2 are active at the solution, so both
// multiplier signs are exercised.
qp_data convex_qp()
{
    qp_data d{Eigen::MatrixXd(3, 3), Eigen::VectorXd(3), Eigen::MatrixXd(4, 3), Eigen::VectorXd(4), Eigen::VectorXd(4)};
    d.P << 4.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 2.0;
    d.q << 1.0, 1.0, -1.0;
    d.A << 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    d.l << 1.0, 0.0, 0.0, 0.0;
    d.u << 1.0, 0.7, 0.7, 0.25;
    return d;
}

qp_data contradictory_bounds_qp()
{
    qp_data d{Eigen::MatrixXd::Identity(3, 3), Eigen::VectorXd::Zero(3), Eigen::MatrixXd::Zero(2, 3), Eigen::VectorXd(2), Eigen::VectorXd(2)};
    d.A(0, 0) = 1.0;
    d.A(1, 0) = 1.0;
    d.l << 1.0, -inf;
    d.u << inf, 0.0;
    return d;
}

qp_data unbounded_qp()
{
    qp_data d{Eigen::MatrixXd::Zero(2, 2), Eigen::VectorXd(2), Eigen::MatrixXd(1, 2), Eigen::VectorXd(1), Eigen::VectorXd(1)};
    d.P(0, 0) = 1.0;
    d.q << 0.0, -1.0;
    d.A << 0.0, 1.0;
    d.l << 0.0;
    d.u << inf;
    return d;
}

qp_result<double> dense_admm_solve(const qp_data& d)
{
    dense_admm_qp_solver<double> solver(static_cast<int>(d.P.rows()), static_cast<int>(d.A.rows()));
    return *solver.solve(d.P, d.q, d.A, d.l, d.u, dense_qp_options<double>{});
}

oracle::qp_verdict judge(const qp_data& d, const qp_result<double>& result)
{
    return oracle::judge_qp(d.P, d.q, d.A, d.l, d.u, result, oracle::qp_tolerances_of(dense_qp_options<double>{}));
}

qp_result<double> planted(qp_solve_status status, const Eigen::VectorXd& x, const Eigen::VectorXd& y)
{
    qp_result<double> result;
    result.status = status;
    result.x = x;
    result.y = y;
    return result;
}

}

TEST_CASE("QP oracle accepts a solved dense ADMM result", "[oracle]")
{
    const qp_result<double> result = dense_admm_solve(convex_qp());
    REQUIRE(result.status == qp_solve_status::solved);
    const oracle::qp_verdict v = judge(convex_qp(), result);
    CAPTURE(v.failing);
    CHECK(v.ok);
}

TEST_CASE("QP oracle rejects the solved dense ADMM result with x shifted", "[oracle]")
{
    qp_result<double> result = dense_admm_solve(convex_qp());
    result.x[0] += 1e-2;
    CHECK(judge(convex_qp(), result).failing == "stationarity");
}

TEST_CASE("QP oracle accepts a solved sparse ADMM result", "[oracle]")
{
    const qp_data d = convex_qp();
    const Eigen::SparseMatrix<double> P = d.P.sparseView();
    const Eigen::SparseMatrix<double> A = d.A.sparseView();
    sparse_admm_qp_solver<double> solver;
    const qp_result<double> result = *solver.solve(P, d.q, A, d.l, d.u);
    REQUIRE(result.status == qp_solve_status::solved);
    CHECK(oracle::judge_qp(P, d.q, A, d.l, d.u, result, oracle::qp_tolerances_of(sparse_qp_options<double>{})).ok);
}

TEST_CASE("QP oracle accepts a solved dense active-set result", "[oracle]")
{
    const qp_data d = convex_qp();
    dense_active_set_qp_solver<double> solver(3, 4);
    REQUIRE_FALSE(solver.warm_start(Eigen::Vector3d(0.5, 0.5, 0.0), Eigen::VectorXd()));
    const qp_result<double> result = *solver.solve(d.P, d.q, d.A, d.l, d.u);
    REQUIRE(result.status == qp_solve_status::solved);
    CHECK(oracle::judge_qp(d.P, d.q, d.A, d.l, d.u, result, oracle::qp_tolerances_of(dense_active_set_qp_options<double>{})).ok);
}

TEST_CASE("QP oracle rejects a planted solved status on contradictory bounds", "[oracle]")
{
    const qp_result<double> result = planted(qp_solve_status::solved, Eigen::VectorXd::Zero(3), Eigen::VectorXd::Zero(2));
    CHECK(judge(contradictory_bounds_qp(), result).failing == "primal_violation");
}

TEST_CASE("QP oracle accepts a planted primal infeasibility certificate", "[oracle]")
{
    const qp_result<double> result = planted(qp_solve_status::primal_infeasible, Eigen::VectorXd::Zero(3), Eigen::Vector2d(-1.0, 1.0));
    CHECK(judge(contradictory_bounds_qp(), result).ok);
}

TEST_CASE("QP oracle rejects a primal_infeasible claim whose y is no certificate", "[oracle]")
{
    const qp_result<double> result = planted(qp_solve_status::primal_infeasible, Eigen::VectorXd::Zero(3), Eigen::Vector2d(1.0, 1.0));
    CHECK(judge(contradictory_bounds_qp(), result).failing == "certificate_range");
}

TEST_CASE("dense ADMM reports primal_infeasible on contradictory bounds", "[oracle]")
{
    CHECK(dense_admm_solve(contradictory_bounds_qp()).status == qp_solve_status::primal_infeasible);
}

TEST_CASE("dense ADMM returns a primal infeasibility certificate in y", "[oracle][known-defect][!shouldfail]")
{
    const oracle::qp_verdict v = judge(contradictory_bounds_qp(), dense_admm_solve(contradictory_bounds_qp()));
    CAPTURE(v.failing);
    CHECK(v.ok);
}

TEST_CASE("QP oracle accepts the dense ADMM dual_infeasible iterate as an unbounded direction", "[oracle]")
{
    const qp_result<double> result = dense_admm_solve(unbounded_qp());
    REQUIRE(result.status == qp_solve_status::dual_infeasible);
    const oracle::qp_verdict v = judge(unbounded_qp(), result);
    CAPTURE(v.failing);
    CHECK(v.ok);
}
