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

// Shrunk by the property suite, printed to six digits: one equality row, where the run ends converged at
// stationarity 1.03e-5.
inline cases::linearly_constrained_quadratic one_equality_quadratic()
{
    cases::linearly_constrained_quadratic p;
    p.Q = Eigen::MatrixXd{{0.752011, 0.179287, -0.121763, -0.235831, -0.354821},
                          {0.179287, 1.888812, 0.24738, -0.285365, 0.575907},
                          {-0.121763, 0.24738, 2.074395, -0.142072, 0.363666},
                          {-0.235831, -0.285365, -0.142072, 3.461062, 0.699774},
                          {-0.354821, 0.575907, 0.363666, 0.699774, 1.908781}};
    p.c = Eigen::VectorXd{{-0.565, 0.28, 0.196, -0.145, 0.788}};
    p.start = Eigen::VectorXd{{-0.003248, -0.505608, -0.46868, -0.083394, -0.25186}};
    p.lower = Eigen::VectorXd{{-0.217, -0.828, -1.266, -0.165, -1.315}};
    p.upper = Eigen::VectorXd{{1.015, 0.384, 0.02, 0.183, 0.048}};
    p.E = Eigen::MatrixXd{{0.805, -0.579, -0.104, 0.672, 0.362}};
    p.e = Eigen::VectorXd{{0.292329}};
    p.G.resize(0, 5);
    p.h.resize(0);
    return p;
}

// Shrunk by the property suite, printed to six digits: two equality and four inequality rows, where the
// run ends converged at stationarity 1.13e-5.
inline cases::linearly_constrained_quadratic mixed_row_quadratic()
{
    cases::linearly_constrained_quadratic p;
    p.Q = Eigen::Matrix4d{{1.066942, 0.386187, 0.231738, -0.293517},
                          {0.386187, 0.873598, 0.402186, -0.297213},
                          {0.231738, 0.402186, 0.577507, -0.279405},
                          {-0.293517, -0.297213, -0.279405, 0.433054}};
    p.c = Eigen::Vector4d(0.291, 0.366, 0.203, -0.244);
    p.start = Eigen::Vector4d(-0.951802, 0.526756, -0.278584, 0.8264225);
    p.lower = Eigen::Vector4d(-1.0, -0.296, -0.286, 0.365);
    p.upper = Eigen::Vector4d(0.108, 0.788, 0.126, 0.926);
    p.E = Eigen::MatrixXd{{0.249, -1.0, -0.229, -0.44}, {0.335, -0.291, 0.297, -0.035}};
    p.e = Eigen::Vector2d(-0.289581, -0.207695);
    p.G = Eigen::MatrixXd{{-1.0, -1.0, -1.0, -1.0}, {-0.203, 0.449, 0.631, -0.81}, {0.473, -0.769, -0.235, 0.044}, {-1.0, -1.0, -1.0, -1.0}};
    p.h = Eigen::Vector4d(-0.123, -0.438809, -0.279733, -0.123);
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
