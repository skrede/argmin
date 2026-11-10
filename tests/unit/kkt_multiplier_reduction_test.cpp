#include "argmin/detail/lagrangian.h"

#include <Eigen/Core>
#include <Eigen/QR>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <cstdint>

namespace
{

struct vertex_estimate
{
    Eigen::VectorXd lam_eq;
    Eigen::VectorXd mu_ineq;
};

template <int N>
Eigen::Vector<double, N> unbounded()
{
    return Eigen::Vector<double, N>::Constant(std::numeric_limits<double>::infinity());
}

template <int N>
vertex_estimate estimate_on_fixed_workspace(const Eigen::Vector<double, N>& g, const Eigen::Matrix<double, Eigen::Dynamic, N>& J_eq,
                                            const Eigen::Matrix<double, Eigen::Dynamic, N>& J_ineq, const Eigen::VectorXd& c_ineq)
{
    argmin::detail::kkt_multiplier_workspace<double, N> ws;
    vertex_estimate e{Eigen::VectorXd(J_eq.rows()), Eigen::VectorXd(J_ineq.rows())};
    argmin::detail::compute_kkt_multipliers_active_set<double, N>(g, J_eq, J_ineq, c_ineq, Eigen::Vector<double, N>::Zero(), -unbounded<N>(), unbounded<N>(), ws, e.lam_eq, e.mu_ineq);
    return e;
}

Eigen::MatrixXd stacked_active_rows(const Eigen::MatrixXd& J_eq, const Eigen::MatrixXd& J_ineq, const Eigen::VectorXd& c_ineq)
{
    Eigen::MatrixXd A(J_eq.rows() + (c_ineq.array() == 0.0).count(), J_eq.cols());
    A.topRows(J_eq.rows()) = J_eq;
    Eigen::Index row = J_eq.rows();
    for(Eigen::Index i = 0; i < J_ineq.rows(); ++i)
        if(c_ineq[i] == 0.0)
            A.row(row++) = J_ineq.row(i);
    return A;
}

double full_least_squares_residual(const Eigen::VectorXd& g, const Eigen::MatrixXd& A)
{
    const Eigen::VectorXd lambda = A.transpose().colPivHouseholderQr().solve(g);
    return (g - A.transpose() * lambda).norm();
}

template <int N>
void check_reduced_estimate(const Eigen::Vector<double, N>& g, const Eigen::Matrix<double, Eigen::Dynamic, N>& J_eq,
                            const Eigen::Matrix<double, Eigen::Dynamic, N>& J_ineq, const Eigen::VectorXd& c_ineq, std::int32_t rank)
{
    const Eigen::MatrixXd A = stacked_active_rows(J_eq, J_ineq, c_ineq);
    REQUIRE(A.rows() > N);
    const vertex_estimate e = estimate_on_fixed_workspace<N>(g, J_eq, J_ineq, c_ineq);
    const Eigen::VectorXd residual = g - J_eq.transpose() * e.lam_eq - J_ineq.transpose() * e.mu_ineq;
    CHECK(std::abs(residual.norm() - full_least_squares_residual(g, A)) < 1e-12);
    CHECK((e.lam_eq.array() != 0.0).count() + (e.mu_ineq.array() != 0.0).count() == rank);
    for(Eigen::Index i = 0; i < J_ineq.rows(); ++i)
        if(c_ineq[i] != 0.0)
            CHECK(e.mu_ineq[i] == 0.0);

    Eigen::VectorXd stacked(J_eq.rows() + J_ineq.rows());
    stacked << e.lam_eq, e.mu_ineq;
    CHECK(argmin::detail::estimate_multipliers_active_set<double, N>(g, J_eq, J_ineq, c_ineq, Eigen::Vector<double, N>::Zero(), -unbounded<N>(), unbounded<N>()) == stacked);
}

}

TEST_CASE("a degenerate vertex keeps the stationarity residual of the full least-squares multiplier estimate", "[lagrangian][kkt-vertex]")
{
    Eigen::Matrix<double, Eigen::Dynamic, 2> J_eq(0, 2);
    Eigen::Matrix<double, Eigen::Dynamic, 2> J_ineq(3, 2);
    J_ineq << 1.0, 0.0, 0.0, 1.0, 1.0, 1.0;
    check_reduced_estimate<2>(Eigen::Vector2d(2.0, 2.0), J_eq, J_ineq, Eigen::VectorXd::Zero(3), 2);
}

TEST_CASE("a rank-deficient degenerate active set refills dropped rows and keeps the least-squares residual", "[lagrangian][kkt-vertex]")
{
    Eigen::Matrix<double, Eigen::Dynamic, 3> J_eq(2, 3);
    J_eq << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0;
    Eigen::Matrix<double, Eigen::Dynamic, 3> J_ineq(3, 3);
    J_ineq << 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    Eigen::VectorXd c_ineq(3);
    c_ineq << 0.0, 0.0, 1.0;
    check_reduced_estimate<3>(Eigen::Vector3d(3.0, 1.0, 3.0), J_eq, J_ineq, c_ineq, 2);
}
