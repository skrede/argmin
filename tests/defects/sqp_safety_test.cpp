#include "sqp_safety_problems.h"

#include "argmin/detail/filter_acceptance.h"

#include "argmin/solver/options.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_trsqp_policy.h"

#include "argmin/result/status.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <limits>

using namespace argmin;
using defects::nowhere;

namespace
{

// Two equality rows on one variable, x - 1 = 0 and 2x - 2 = 0: more equalities than unknowns.
defects::row_quadratic overdetermined_equalities()
{
    return {Eigen::MatrixXd::Identity(1, 1), Eigen::VectorXd::Zero(1), Eigen::Vector2d(1.0, 2.0), Eigen::Vector2d(1.0, 2.0), Eigen::MatrixXd(0, 1), Eigen::VectorXd(0), nowhere};
}

template <typename Policy>
solver_status status_after_construction()
{
    const defects::row_quadratic problem = overdetermined_equalities();
    step_budget_solver solver{Policy{}, problem, Eigen::VectorXd::Constant(1, 3.0), solver_options<>{}};
    return solver.status();
}

// The inactive row x1 + 100 >= 0 keeps the problem constrained; a cutoff below 3 puts the minimizer in the NaN region.
defects::row_quadratic bowl_with_inactive_row(double nan_beyond)
{
    return defects::shifted_bowl(Eigen::MatrixXd(0, 2), Eigen::VectorXd(0), Eigen::RowVector2d(0.0, 1.0), Eigen::VectorXd::Constant(1, -100.0), nan_beyond);
}

template <typename Policy>
solve_result<double> solve_from_origin(const defects::row_quadratic& problem)
{
    const solver_options<> opts;
    step_budget_solver solver{Policy{}, problem, Eigen::Vector2d::Zero().eval(), opts};
    return solver.solve(opts);
}

// x0 - x1 - 1 = 0 from the origin: the full step lands at x0 = 3.5, beyond the NaN cutoff 2.5.
std::size_t first_step_soc_retries(double nan_beyond)
{
    const defects::row_quadratic problem = defects::shifted_bowl(Eigen::RowVector2d(1.0, -1.0), Eigen::VectorXd::Ones(1), Eigen::MatrixXd(0, 2), Eigen::VectorXd(0), nan_beyond);
    step_budget_solver solver{filter_slsqp_policy<>{}, problem, Eigen::Vector2d::Zero().eval(), solver_options<>{}};
    return solver.step().diagnostics.soc_retry_count;
}

}

TEST_CASE("kraft_slsqp rejects more equality rows than variables before its first step", "[known-defect][!shouldfail]")
{
    CHECK(status_after_construction<kraft_slsqp_policy<>>() == solver_status::invalid_problem);
}

TEST_CASE("filter_slsqp rejects more equality rows than variables before its first step", "[known-defect][!shouldfail]")
{
    CHECK(status_after_construction<filter_slsqp_policy<>>() == solver_status::invalid_problem);
}

TEST_CASE("tr_sqp and filter_trsqp converge to the bowl minimizer without a NaN region", "[sqp]")
{
    for(const solve_result<double>& r : {solve_from_origin<tr_sqp_policy<>>(bowl_with_inactive_row(nowhere)), solve_from_origin<filter_trsqp_policy<>>(bowl_with_inactive_row(nowhere))})
    {
        CHECK(r.status == solver_status::converged);
        CHECK((r.x - Eigen::Vector2d(3.0, 0.0)).norm() <= 1e-6);
    }
}

TEST_CASE("tr_sqp does not end converged after trial steps into a NaN objective", "[known-defect][!shouldfail]")
{
    CHECK(solve_from_origin<tr_sqp_policy<>>(bowl_with_inactive_row(0.5)).status != solver_status::converged);
}

TEST_CASE("filter_trsqp does not end converged after trial steps into a NaN objective", "[known-defect][!shouldfail]")
{
    CHECK(solve_from_origin<filter_trsqp_policy<>>(bowl_with_inactive_row(0.5)).status != solver_status::converged);
}

TEST_CASE("the filter rejects a trial with a NaN objective or a NaN violation", "[known-defect][!shouldfail]")
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    detail::filter_set<double> filter;
    filter.initialize(1e4);
    filter.add(1.0, 0.5);
    CHECK_FALSE((filter.is_acceptable(nan, 0.1) || filter.is_acceptable(0.0, nan)));
}

TEST_CASE("filter_slsqp fires no second-order correction on a finite full step that clears its row", "[sqp]")
{
    CHECK(first_step_soc_retries(nowhere) == 0);
}

TEST_CASE("filter_slsqp fires no second-order correction after a NaN full step", "[known-defect][!shouldfail]")
{
    CHECK(first_step_soc_retries(2.5) == 0);
}
