#include "alloc_gate_support.h"

#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"

#include "argmin/detail/active_set_qp.h"

#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <cstdint>

namespace
{

using argmin::alloc_gate::run_armed;
using argmin::alloc_gate::sqp_gate_opts;

constexpr double inf = std::numeric_limits<double>::infinity();

// Fixed-2 bowl minimized at (1, 1) on the row x0 + x1 <= 2. Its box is finite only below x0 and above x1, so two of
// the four box rows are absent from the row blocks the Kraft QP and the warm-start projection stack.
struct half_box_bowl
{
    static constexpr std::int32_t problem_dimension = 2;

    std::int32_t dimension() const { return 2; }
    std::int32_t num_equality() const { return 0; }
    std::int32_t num_inequality() const { return 1; }

    double value(const Eigen::Vector2d& x) const { return (x - Eigen::Vector2d(2.0, 2.0)).squaredNorm(); }

    void gradient(const Eigen::Vector2d& x, Eigen::Vector2d& g) const { g = 2.0 * (x - Eigen::Vector2d(2.0, 2.0)); }

    template <typename Values>
    void constraints(const Eigen::Vector2d& x, Values& c) const
    {
        c[0] = 2.0 - x[0] - x[1];
    }

    template <typename Jacobian>
    void constraint_jacobian(const Eigen::Vector2d&, Jacobian& J) const
    {
        J(0, 0) = -1.0;
        J(0, 1) = -1.0;
    }

    Eigen::Vector2d lower_bounds() const { return Eigen::Vector2d(0.0, -inf); }

    Eigen::Vector2d upper_bounds() const { return Eigen::Vector2d(inf, 1.5); }
};

struct declared_half_box_bowl : half_box_bowl
{
    static constexpr std::int32_t constraint_count = 1;
};

// The window opens on a freshly built solver, so it holds the first QP solve and every step after it.
template <typename Policy, typename Problem>
argmin::solver_status check_first_solve_zero(Policy policy, const Problem& problem, const Eigen::Vector<double, Problem::problem_dimension>& x0)
{
    argmin::step_budget_solver solver{policy, problem, x0, sqp_gate_opts()};
    argmin::solver_status status = argmin::solver_status::running;
    const auto counts = run_armed([&] { status = solver.solve().status; });
    CHECK(counts.eigen == 0);
    CHECK(counts.c == 0);
    return status;
}

// HS071's start violates its equality row, so the first QP starts off the linearized equality manifold and takes the
// phase-1 projection.
template <typename Policy>
void check_phase1_first_solve_zero(Policy policy)
{
    const argmin::hs071<> problem;
    const Eigen::Vector4d x0 = problem.initial_point();
    Eigen::VectorXd c(2);
    problem.constraints(x0, c);
    REQUIRE(c[0] != 0.0);
    check_first_solve_zero(policy, problem, x0);
}

}

// The start violates the row, so the first step also projects its warm start.
TEST_CASE("sqp_alloc_gate_first_solve_half_box", "[alloc-gate]")
{
    const half_box_bowl problem;
    const Eigen::Vector2d x0(2.0, 1.0);
    CHECK(check_first_solve_zero(argmin::kraft_slsqp_policy<double, 2>{}, problem, x0) == argmin::solver_status::converged);
    CHECK(check_first_solve_zero(argmin::filter_slsqp_policy<double, 2>{}, problem, x0) == argmin::solver_status::converged);
    const declared_half_box_bowl declared;
    CHECK(check_first_solve_zero(argmin::nw_sqp_policy<double, 2>{}, declared, x0) == argmin::solver_status::converged);
    CHECK(check_first_solve_zero(argmin::filter_nw_sqp_policy<double, 2>{}, declared, x0) == argmin::solver_status::converged);
}

TEST_CASE("sqp_alloc_gate_first_solve_phase1", "[alloc-gate]")
{
    check_phase1_first_solve_zero(argmin::nw_sqp_policy<double, 4>{});
    check_phase1_first_solve_zero(argmin::filter_nw_sqp_policy<double, 4>{});
}

// The dense active-set adapter delegates to this solver configuration, sized for its 2m mapped rows. The zero start is
// off the equality row, so the first solve takes the phase-1 projection.
TEST_CASE("qp_alloc_gate_active_set_phase1_first_solve", "[alloc-gate]")
{
    using solver_type = argmin::detail::active_set_qp_solver<double, 3, argmin::dynamic_dimension>;
    const Eigen::Matrix3d G = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d d(-1.0, -2.0, -3.0);
    const Eigen::Matrix<double, Eigen::Dynamic, 3> A_eq{{1.0, 1.0, 1.0}};
    const Eigen::VectorXd b_eq{{1.0}};
    const Eigen::Matrix<double, Eigen::Dynamic, 3> A_ineq{{1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}};
    const Eigen::VectorXd b_ineq{{-1.0, -1.0}};
    const Eigen::Vector3d x0 = Eigen::Vector3d::Zero();
    solver_type solver(3, 4);
    solver_type::result_type out;
    out.lambda.setZero(3);
    const auto counts = run_armed([&] { solver.solve_into(G, d, A_eq, b_eq, A_ineq, b_ineq, x0, argmin::qp_options<>{}, out); });
    CHECK(out.status == argmin::detail::qp_status::optimal);
    CHECK(counts.eigen == 0);
    CHECK(counts.c == 0);
}
