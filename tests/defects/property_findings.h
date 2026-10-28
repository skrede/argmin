#ifndef HPP_GUARD_ARGMIN_TESTS_DEFECTS_PROPERTY_FINDINGS_H
#define HPP_GUARD_ARGMIN_TESTS_DEFECTS_PROPERTY_FINDINGS_H

#include "oracle_tolerances.h"

#include "cases/linear_quadratic_problems.h"

#include "oracle/result_oracle.h"

#include "argmin/solver/options.h"
#include "argmin/solver/step_budget_solver.h"

#include <Eigen/Core>

namespace argmin::defects
{

// Shrunk by the property suite: a step from the start lands on the box, where x1 sits on its lower bound
// with a gradient pointing into the box.
inline cases::linear_least_squares box_binding_least_squares()
{
    cases::linear_least_squares p{Eigen::MatrixXd(5, 4), Eigen::VectorXd(5), Eigen::VectorXd(4), Eigen::VectorXd(4), Eigen::VectorXd(4)};
    p.M << -1.0, -0.669, -0.157, 0.424, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5;
    p.b << -0.454, 0.54, -1.136, -2.0, 0.188;
    p.start << -0.478, -0.239, -0.581, -0.999;
    p.lower << -0.578, -0.339, -0.681, -1.099;
    p.upper << -0.378, -0.139, -0.481, -0.899;
    return p;
}

// Shrunk by the property suite: the start violates both inequality rows, while points near the lower
// corner of the box satisfy them.
inline cases::linearly_constrained_quadratic infeasible_start_quadratic()
{
    cases::linearly_constrained_quadratic p;
    p.Q = Eigen::MatrixXd::Constant(3, 3, 3.0) + 0.1 * Eigen::MatrixXd::Identity(3, 3);
    p.c = Eigen::VectorXd::Constant(3, -1.0);
    p.start = Eigen::Vector3d(-0.976504, -0.937154, -1.0208985);
    p.lower = Eigen::VectorXd::Constant(3, -1.1);
    p.upper = Eigen::Vector3d(-0.756, -0.664, -0.819);
    p.E.resize(0, 3);
    p.e.resize(0);
    p.G.resize(2, 3);
    p.G << -0.697, -0.86, -0.819, -1.0, -1.0, -1.0;
    p.h = Eigen::Vector2d(2.376, 3.0);
    return p;
}

// Shrunk by the property suite from a feasible start: the run ends converged with the first row active
// and the second at slack 1e-3, where stationarity on the active row alone is 1.6e-5.
inline cases::linearly_constrained_quadratic two_row_quadratic()
{
    cases::linearly_constrained_quadratic p;
    p.Q = Eigen::Matrix2d{{0.89405, 0.00321}, {0.00321, 0.142354}};
    p.c = Eigen::Vector2d(-0.545, -0.748);
    p.start = Eigen::Vector2d(-0.572, -0.196);
    p.lower = Eigen::Vector2d(-0.845, -0.36);
    p.upper = Eigen::Vector2d(-0.272, 0.388);
    p.E.resize(0, 2);
    p.e.resize(0);
    p.G = Eigen::Matrix2d{{-0.72, -0.543}, {-0.306, -1.0}};
    p.h = Eigen::Vector2d(0.456268, 0.298032);
    return p;
}

template <typename Policy, typename Problem>
oracle::verdict<double> judge_finding(const Policy& policy, const Problem& problem, const solver_options<>& options)
{
    step_budget_solver solver{policy, problem, problem.initial_point(), options};
    const auto result = solver.solve(options);
    return oracle::judge(problem, result, oracle::tolerances_of(options));
}

}

#endif
