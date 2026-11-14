#include "argmin/detail/sqp_common.h"

#include <Eigen/QR>
#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("the rank-deficient null-space projection matches the materialized Q1 projector", "[sqp_common]")
{
    Eigen::MatrixXd A(3, 4);
    A << 1.0, 2.0, 0.0, -1.0,
         0.0, 1.0, 3.0, 2.0,
         1.0, 3.0, 3.0, 1.0;
    Eigen::VectorXd x(4);
    x << 0.3, -1.2, 2.5, 0.7;
    const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A.transpose());
    const Eigen::MatrixXd Q1 = qr.householderQ() * Eigen::MatrixXd::Identity(4, qr.rank());
    const Eigen::VectorXd expected = x - Q1 * (Q1.transpose() * x);

    argmin::detail::null_space_qr_workspace<double> ws;
    ws.resize(4, 3);
    argmin::detail::project_off_range_qr<double, Eigen::Dynamic>(A, x, ws);

    CHECK(qr.rank() == 2);
    CHECK((x - expected).lpNorm<Eigen::Infinity>() < 1e-14);
    CHECK((A * x).lpNorm<Eigen::Infinity>() < 1e-14);
}
