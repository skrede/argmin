#include "sqp_safety_problems.h"

#include "argmin/detail/active_set_qp.h"

#include "argmin/solver/options.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/filter_nw_sqp_policy.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace argmin;
using defects::nowhere;

namespace
{

// x0 + x1 + 3 = 0 with x1 >= 2 and x1 <= 0: every linearization is inconsistent.
defects::row_quadratic inconsistent_rows()
{
    Eigen::MatrixXd G(2, 2);
    G << 0.0, 1.0, 0.0, -1.0;
    return {Eigen::MatrixXd::Identity(2, 2), Eigen::Vector2d(-10.0, -10.0), Eigen::RowVector2d(1.0, 1.0), Eigen::VectorXd::Constant(1, -3.0), G, Eigen::Vector2d(2.0, 0.0), nowhere};
}

step_result<double> nw_sqp_first_step_on_inconsistent_rows()
{
    const defects::row_quadratic problem = inconsistent_rows();
    step_budget_solver solver{nw_sqp_policy<>{}, problem, Eigen::Vector2d(0.0, -1.0), solver_options<>{}};
    return solver.step();
}

// min |x|^2 on x0 + x1 = 0 and x0 >= 1, solved by (1, -1).
defects::row_quadratic joint_rows()
{
    return {2.0 * Eigen::MatrixXd::Identity(2, 2), Eigen::Vector2d::Zero(), Eigen::RowVector2d(1.0, 1.0), Eigen::VectorXd::Zero(1), Eigen::RowVector2d(1.0, 0.0), Eigen::VectorXd::Ones(1), nowhere};
}

template <typename Policy>
solve_result<double> solve_joint_rows(std::uint32_t max_iterations)
{
    const defects::row_quadratic problem = joint_rows();
    solver_options<> opts;
    opts.max_iterations = max_iterations;
    step_budget_solver solver{Policy{}, problem, Eigen::Vector2d::Zero().eval(), opts};
    return solver.solve(opts);
}

bool at_joint_solution(const solve_result<double>& r)
{
    return r.status == solver_status::converged && (r.x - Eigen::Vector2d(1.0, -1.0)).cwiseAbs().maxCoeff() <= 1e-8;
}

}

TEST_CASE("nw_sqp makes no BFGS-reset retry when its QP is infeasible", "[known-defect][!shouldfail]")
{
    CHECK(nw_sqp_first_step_on_inconsistent_rows().diagnostics.bfgs_reset_count == 0);
}

TEST_CASE("nw_sqp reports a failure status on a step whose QP is infeasible", "[known-defect][!shouldfail]")
{
    const step_result<double> r = nw_sqp_first_step_on_inconsistent_rows();
    CHECK((r.policy_status.has_value() && *r.policy_status != solver_status::running && *r.policy_status != solver_status::converged));
}

TEST_CASE("filter_nw_sqp reports an infeasible linearization instead of running out of iterations", "[known-defect][!shouldfail]")
{
    const defects::row_quadratic problem = inconsistent_rows();
    solver_options<> opts;
    opts.max_iterations = 50;
    step_budget_solver solver{filter_nw_sqp_policy<>{}, problem, Eigen::Vector2d(0.0, -1.0), opts};
    CHECK(solver.solve(opts).status != solver_status::max_iterations);
}

TEST_CASE("kraft_slsqp solves the joint equality and inequality problem", "[sqp]")
{
    CHECK(at_joint_solution(solve_joint_rows<kraft_slsqp_policy<>>(20)));
}

TEST_CASE("nw_sqp solves the joint equality and inequality problem within 20 iterations", "[known-defect][!shouldfail]")
{
    CHECK(at_joint_solution(solve_joint_rows<nw_sqp_policy<>>(20)));
}

TEST_CASE("the active-set QP reaches the joint optimum from a seed off the equality row", "[known-defect][!shouldfail]")
{
    detail::active_set_qp_solver<double> qp(2, 2);
    const auto r = qp.solve(Eigen::MatrixXd::Identity(2, 2).eval(),
                            Eigen::VectorXd::Zero(2).eval(),
                            Eigen::MatrixXd::Ones(1, 2).eval(),
                            Eigen::VectorXd::Zero(1).eval(),
                            Eigen::MatrixXd(Eigen::RowVector2d(1.0, 0.0)),
                            Eigen::VectorXd::Ones(1).eval(),
                            Eigen::VectorXd(Eigen::Vector2d(1.0, 0.0)));
    CHECK((r.status == detail::qp_status::optimal && (r.x - Eigen::Vector2d(1.0, -1.0)).cwiseAbs().maxCoeff() <= 1e-12));
}
