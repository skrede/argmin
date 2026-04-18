// EIGEN_RUNTIME_NO_MALLOC must be defined BEFORE any Eigen include
// to enable the runtime malloc-checking machinery.
#define EIGEN_RUNTIME_NO_MALLOC

#include "nablapp/test_functions/beale.h"
#include "nablapp/test_functions/booth.h"
#include "nablapp/test_functions/hock_schittkowski.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/solver/lbfgsb_policy.h"
#include "nablapp/solver/kraft_slsqp_policy.h"
#include "nablapp/solver/mma_policy.h"
#include "nablapp/types.h"
#include "nablapp/detail/compact_lbfgs.h"
#include "nablapp/detail/cauchy_point.h"
#include "nablapp/detail/bound_projection.h"
#include "nablapp/detail/subspace_minimization.h"
#include "nablapp/detail/active_set_qp.h"
#include "nablapp/detail/mma_subproblem.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <type_traits>

using namespace nablapp;

TEST_CASE("fixed-dim value+gradient: zero dynamic allocation", "[zero-copy]")
{
    beale<double> problem;
    Eigen::Vector<double, 2> x;
    x << 0.5, 0.5;
    Eigen::Vector<double, 2> g;

    Eigen::internal::set_is_malloc_allowed(false);

    double f = problem.value(x);
    problem.gradient(x, g);

    Eigen::Vector<double, 2> lower;
    lower << -10.0, -10.0;
    Eigen::Vector<double, 2> upper;
    upper << 10.0, 10.0;
    auto projected = detail::project(x, lower, upper);

    Eigen::internal::set_is_malloc_allowed(true);

    CHECK(std::isfinite(f));
    CHECK(std::isfinite(g[0]));
    CHECK(std::isfinite(g[1]));
    CHECK(projected[0] == x[0]);
}

TEST_CASE("fixed-dim booth value+gradient: zero dynamic allocation", "[zero-copy]")
{
    booth<double> problem;
    Eigen::Vector<double, 2> x;
    x << 1.0, 3.0;
    Eigen::Vector<double, 2> g;

    Eigen::internal::set_is_malloc_allowed(false);

    double f = problem.value(x);
    problem.gradient(x, g);

    Eigen::internal::set_is_malloc_allowed(true);

    CHECK(f < 1e-10);
    CHECK(std::abs(g[0]) < 1e-8);
    CHECK(std::abs(g[1]) < 1e-8);
}

TEST_CASE("fixed-dim solver state types are stack-allocated", "[zero-copy]")
{
    using solver_type = basic_solver<lbfgsb_policy<2>, 2, beale<double>>;
    using state_type = typename solver_type::state_type;

    static_assert(std::is_same_v<decltype(state_type::x),
                                 Eigen::Vector<double, 2>>,
                  "solver state x must be Vector<double, 2>");
    static_assert(std::is_same_v<decltype(state_type::g),
                                 Eigen::Vector<double, 2>>,
                  "solver state g must be Vector<double, 2>");

    beale<double> problem;
    Eigen::Vector<double, 2> x0;
    x0 << 0.5, 0.5;
    solver_options opts;
    opts.max_iterations = 3;

    solver_type solver{problem, x0, opts};
    auto result = solver.step();

    CHECK(std::isfinite(result.objective_value));
}

// Verify that the pre-allocated compact_lbfgs B*v product and two-loop
// recursion are allocation-free for fixed-dimension vectors after
// curvature pairs have been pushed.
TEST_CASE("fixed-dim compact_lbfgs operations: zero dynamic allocation", "[zero-copy]")
{
    constexpr int N = 2;
    detail::compact_lbfgs<double, N, 5> B;

    // Push a curvature pair (allocates internally on first push)
    Eigen::Vector<double, N> s;
    s << 0.1, 0.2;
    Eigen::Vector<double, N> y;
    y << 1.0, 2.0;
    B.push(s, y);

    Eigen::Vector<double, N> v;
    v << 1.0, -1.0;

    Eigen::internal::set_is_malloc_allowed(false);

    auto Bv = B.multiply(v);
    auto Hg = B.two_loop_recursion(v);

    Eigen::internal::set_is_malloc_allowed(true);

    CHECK(std::isfinite(Bv[0]));
    CHECK(std::isfinite(Hg[0]));
}

// Verify that the pre-allocated cauchy_point_solver is allocation-free
// for fixed-dimension problems after construction.
TEST_CASE("fixed-dim cauchy_point_solver: zero dynamic allocation", "[zero-copy]")
{
    constexpr int N = 2;
    detail::compact_lbfgs<double, N, 5> B;
    Eigen::Vector<double, N> s;
    s << 0.1, 0.2;
    Eigen::Vector<double, N> y;
    y << 1.0, 2.0;
    B.push(s, y);

    detail::cauchy_point_solver<double, N> solver{N};

    Eigen::Vector<double, N> x;
    x << 0.5, 0.5;
    Eigen::Vector<double, N> g;
    g << 1.0, -1.0;
    Eigen::Vector<double, N> lower;
    lower << -10.0, -10.0;
    Eigen::Vector<double, N> upper;
    upper << 10.0, 10.0;

    // Warm-up: first call may trigger lazy allocation in breakpoint vector
    solver.solve(x, g, lower, upper, B);

    Eigen::internal::set_is_malloc_allowed(false);

    const auto& result = solver.solve(x, g, lower, upper, B);

    Eigen::internal::set_is_malloc_allowed(true);

    CHECK(std::isfinite(result.x_cauchy[0]));
}

// Verify that the pre-allocated active_set_qp_solver is allocation-free
// for fixed-dimension QP solves (SLSQP hot path).
TEST_CASE("fixed-dim active_set_qp_solver: zero dynamic allocation", "[zero-copy]")
{
    constexpr int N = 3;
    detail::active_set_qp_solver<double, N> qp{N, 2};

    Eigen::Matrix<double, N, N> G = Eigen::Matrix<double, N, N>::Identity();
    Eigen::Vector<double, N> d;
    d << 1.0, 2.0, 3.0;
    Eigen::Matrix<double, Eigen::Dynamic, N> A_eq(0, N);
    Eigen::VectorXd b_eq(0);
    Eigen::Matrix<double, Eigen::Dynamic, N> A_ineq(1, N);
    A_ineq << 1.0, 1.0, 1.0;
    Eigen::VectorXd b_ineq(1);
    b_ineq << -5.0;
    Eigen::Vector<double, N> x0 = Eigen::Vector<double, N>::Zero();

    // Warm-up
    qp.solve(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);

    Eigen::internal::set_is_malloc_allowed(false);

    auto result = qp.solve(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);

    Eigen::internal::set_is_malloc_allowed(true);

    CHECK(result.status == detail::qp_status::optimal);
}

// Verify that the pre-allocated mma_subproblem_solver is allocation-free
// for fixed-dimension dual solves (MMA hot path).
//
// NOTE: This test covers N-dimension allocation paths (LDLT factorizations,
// workspace vectors, compact L-BFGS middle matrix). M-dimension paths
// (constraint vectors, Jacobian-sized matrices in policy state) remain
// dynamic until Plan 11 wires constraint_count_v<P> through the policies.
TEST_CASE("fixed-dim mma_subproblem_solver: zero dynamic allocation", "[zero-copy]")
{
    constexpr int N = 3;
    const int m = 1;
    detail::mma_subproblem_solver<double, N> sub{N, m};

    Eigen::Vector<double, N> x;
    x << 1.0, 2.0, 3.0;
    Eigen::Vector<double, N> grad_f;
    grad_f << 0.5, -0.3, 0.1;
    Eigen::VectorXd g(m);
    g << -0.5;
    Eigen::Matrix<double, Eigen::Dynamic, N> dg(m, N);
    dg << 1.0, 0.0, -1.0;
    Eigen::Vector<double, N> L;
    L << -1.0, 0.0, 1.0;
    Eigen::Vector<double, N> U;
    U << 3.0, 4.0, 5.0;

    Eigen::Vector<double, N> x_min;
    x_min << 0.0, 1.0, 2.0;
    Eigen::Vector<double, N> x_max;
    x_max << 2.0, 3.0, 4.0;

    // Svanberg 2002 Section 3 regularization arguments. Zero raa recovers
    // the Svanberg 1987 baseline (0.001 * |grad| stabilizer only).
    const double raa_0 = 0.0;
    Eigen::VectorXd raa = Eigen::VectorXd::Zero(m);

    // Warm-up: compute coefficients and solve once
    sub.compute_coefficients(x, 1.0, grad_f, g, dg, L, U, raa_0, raa);
    sub.dual_solve(L, U, x_min, x_max);

    // Second run should be allocation-free
    Eigen::internal::set_is_malloc_allowed(false);

    sub.compute_coefficients(x, 1.0, grad_f, g, dg, L, U, raa_0, raa);
    auto x_opt = sub.dual_solve(L, U, x_min, x_max);

    Eigen::internal::set_is_malloc_allowed(true);

    CHECK(std::isfinite(x_opt[0]));
}

// Verify that the pre-allocated subspace_minimizer is allocation-free
// for fixed-dimension subspace solves (L-BFGS-B hot path).
TEST_CASE("fixed-dim subspace_minimizer: zero dynamic allocation", "[zero-copy]")
{
    constexpr int N = 2;
    detail::compact_lbfgs<double, N, 5> B;
    Eigen::Vector<double, N> s;
    s << 0.1, 0.2;
    Eigen::Vector<double, N> y;
    y << 1.0, 2.0;
    B.push(s, y);

    detail::subspace_minimizer<double, N> sm{N};

    Eigen::Vector<double, N> x;
    x << 0.5, 0.5;
    Eigen::Vector<double, N> x_cauchy;
    x_cauchy << 0.4, 0.6;
    Eigen::Vector<double, N> g;
    g << 1.0, -1.0;
    Eigen::Vector<double, N> lower;
    lower << -10.0, -10.0;
    Eigen::Vector<double, N> upper;
    upper << 10.0, 10.0;
    std::vector<int> free_indices = {0, 1};

    // Warm-up
    sm.solve(x, x_cauchy, g, lower, upper, free_indices, B);

    Eigen::internal::set_is_malloc_allowed(false);

    auto result = sm.solve(x, x_cauchy, g, lower, upper, free_indices, B);

    Eigen::internal::set_is_malloc_allowed(true);

    CHECK(std::isfinite(result[0]));
}
