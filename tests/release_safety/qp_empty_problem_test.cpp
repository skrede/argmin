#include "eigen_trap.h"

#include "argmin/qp/qp_types.h"
#include "argmin/qp/dense_admm_qp.h"
#include "argmin/qp/dense_active_set_qp.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <cstdint>

using namespace argmin;

namespace
{

template <std::int32_t N, typename Solver>
qp_error unconstrained_rejection(Solver& solver, std::int32_t n)
{
    const Eigen::Matrix<double, N, N> P = Eigen::Matrix<double, N, N>::Identity(n, n);
    const Eigen::Vector<double, N> q = Eigen::Vector<double, N>::Zero(n);
    const Eigen::Matrix<double, Eigen::Dynamic, N> A(0, n);
    const Eigen::VectorXd bounds(0);
    const auto result = solver.solve(P, q, A, bounds, bounds);
    REQUIRE_FALSE(result.has_value());
    return result.error();
}

}

TEMPLATE_TEST_CASE("dense QP solvers reject an empty problem with a dimension mismatch", "[release-safety][qp]", dense_admm_qp_solver<double>, dense_active_set_qp_solver<double>)
{
    TestType solver(0, 0);
    CHECK(unconstrained_rejection<dynamic_dimension>(solver, 0) == qp_error::dimension_mismatch);
}

TEMPLATE_TEST_CASE("default-constructed dense QP solvers reject an empty problem and exceed capacity on any other", "[release-safety][qp]", dense_admm_qp_solver<double>, dense_active_set_qp_solver<double>)
{
    TestType solver;
    CHECK(unconstrained_rejection<dynamic_dimension>(solver, 0) == qp_error::dimension_mismatch);
    CHECK(unconstrained_rejection<dynamic_dimension>(solver, 2) == qp_error::capacity_exceeded);
}

TEMPLATE_TEST_CASE("default-constructed fixed-size dense QP solvers exceed capacity on their own dimension", "[release-safety][qp]", (dense_admm_qp_solver<double, 2>), (dense_active_set_qp_solver<double, 2>))
{
    TestType solver;
    CHECK(unconstrained_rejection<2>(solver, 2) == qp_error::capacity_exceeded);
}
