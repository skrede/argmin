#include "sqp_algorithm_problems.h"

#include "argmin/detail/kraft_lsq_qp_recovery.h"

#include "argmin/solver/options.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_trsqp_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <cstddef>
#include <cstdint>

using namespace argmin;
using defects::nowhere;

namespace
{

// -|x|^2 on [-1, 2]^2 with the inactive row 10 - x0 - x1 >= 0: every step has s^T y < 0.
defects::boxed_row_quadratic concave_box()
{
    return {{-2.0 * Eigen::MatrixXd::Identity(2, 2), Eigen::Vector2d::Zero(), Eigen::MatrixXd(0, 2), Eigen::VectorXd(0), Eigen::RowVector2d(-1.0, -1.0), Eigen::VectorXd::Constant(1, -10.0), nowhere},
            Eigen::Vector2d::Constant(-1.0),
            Eigen::Vector2d::Constant(2.0)};
}

template <typename Policy>
bool hessian_changes_on_negative_curvature()
{
    const defects::boxed_row_quadratic problem = concave_box();
    step_budget_solver solver{Policy{}, problem, Eigen::Vector2d(0.5, 0.25).eval(), solver_options<>{}};
    const Eigen::MatrixXd before = solver.state().hessian.hessian();
    solver.step();
    return (solver.state().hessian.hessian() - before).norm() > 0.0;
}

// 100 x0 + (x1 - 1)^2 on [0, 10]^2 with the inactive row 10 - x1 >= 0: only the x0 lower bound has a multiplier.
defects::boxed_row_quadratic bound_multiplier_problem()
{
    return {{Eigen::Vector2d(0.0, 2.0).asDiagonal(), Eigen::Vector2d(100.0, -2.0), Eigen::MatrixXd(0, 2), Eigen::VectorXd(0), Eigen::RowVector2d(0.0, -1.0), Eigen::VectorXd::Constant(1, -10.0), nowhere},
            Eigen::Vector2d::Zero(),
            Eigen::Vector2d::Constant(10.0)};
}

template <typename Policy>
double penalty_after_first_step()
{
    const defects::boxed_row_quadratic problem = bound_multiplier_problem();
    step_budget_solver solver{Policy{}, problem, Eigen::Vector2d::Zero().eval(), solver_options<>{}};
    solver.step();
    return solver.state().sigma;
}

template <typename Policy>
std::size_t second_order_corrections_at_feasible_iterates()
{
    const defects::rosenbrock_inside_row problem;
    step_budget_solver solver{Policy{}, problem, Eigen::Vector2d(-1.2, 1.0).eval(), solver_options<>{}};
    std::size_t count = 0;
    for(std::int32_t k = 0; k < 30; ++k)
    {
        const bool feasible = solver.state().x.sum() <= 10.0;
        const std::size_t retries = solver.step().diagnostics.soc_retry_count;
        if(feasible)
            count += retries;
    }
    return count;
}

// Two copies of the consistent row x0 + x1 = 1 in three variables: rank-deficient with fewer rows than unknowns.
double duplicated_row_relaxation_factor()
{
    detail::kraft_lsq_qp_recovery_solver<double, Eigen::Dynamic> qp;
    qp.resize(3, 2, 0, 0, 0);
    Eigen::MatrixXd A_eq(2, 3);
    A_eq << 1.0, 1.0, 0.0, 1.0, 1.0, 0.0;
    const double inf = std::numeric_limits<double>::infinity();
    const Eigen::VectorXd lower = Eigen::VectorXd::Constant(3, -inf);
    const Eigen::VectorXd upper = Eigen::VectorXd::Constant(3, inf);
    return qp.solve(Eigen::MatrixXd::Identity(3, 3), Eigen::VectorXd::Ones(3), A_eq, Eigen::VectorXd::Ones(2), Eigen::MatrixXd(0, 3), Eigen::VectorXd(0), lower, upper).relaxation_factor;
}

}

TEST_CASE("kraft_slsqp updates its Hessian after a negative-curvature step", "[known-defect][!shouldfail]")
{
    CHECK(hessian_changes_on_negative_curvature<kraft_slsqp_policy<>>());
}

TEST_CASE("filter_slsqp updates its Hessian after a negative-curvature step", "[known-defect][!shouldfail]")
{
    CHECK(hessian_changes_on_negative_curvature<filter_slsqp_policy<>>());
}

TEST_CASE("nw_sqp updates its Hessian after a negative-curvature step", "[known-defect][!shouldfail]")
{
    CHECK(hessian_changes_on_negative_curvature<nw_sqp_policy<>>());
}

TEST_CASE("filter_nw_sqp updates its Hessian after a negative-curvature step", "[known-defect][!shouldfail]")
{
    CHECK(hessian_changes_on_negative_curvature<filter_nw_sqp_policy<>>());
}

TEST_CASE("tr_sqp updates its Hessian after a negative-curvature step", "[known-defect][!shouldfail]")
{
    CHECK(hessian_changes_on_negative_curvature<tr_sqp_policy<>>());
}

TEST_CASE("filter_trsqp updates its Hessian after a negative-curvature step", "[known-defect][!shouldfail]")
{
    CHECK(hessian_changes_on_negative_curvature<filter_trsqp_policy<>>());
}

TEST_CASE("nw_sqp calibrates its first penalty on the constraint multipliers only", "[known-defect][!shouldfail]")
{
    CHECK(penalty_after_first_step<nw_sqp_policy<>>() < 10.0);
}

TEST_CASE("filter_nw_sqp calibrates its first penalty on the constraint multipliers only", "[known-defect][!shouldfail]")
{
    CHECK(penalty_after_first_step<filter_nw_sqp_policy<>>() < 10.0);
}

TEST_CASE("the SLSQP recovery QP skips the augmented retry for dependent equality rows fewer than the unknowns", "[known-defect][!shouldfail]")
{
    CHECK(duplicated_row_relaxation_factor() == 0.0);
}

TEST_CASE("kraft_slsqp fires no second-order correction at a feasible iterate under a linear row", "[known-defect][!shouldfail]")
{
    CHECK(second_order_corrections_at_feasible_iterates<kraft_slsqp_policy<>>() == 0);
}

TEST_CASE("filter_slsqp fires no second-order correction at a feasible iterate under a linear row", "[known-defect][!shouldfail]")
{
    CHECK(second_order_corrections_at_feasible_iterates<filter_slsqp_policy<>>() == 0);
}

TEST_CASE("nw_sqp fires no second-order correction at a feasible iterate under a linear row", "[known-defect][!shouldfail]")
{
    CHECK(second_order_corrections_at_feasible_iterates<nw_sqp_policy<>>() == 0);
}
