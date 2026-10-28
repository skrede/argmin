#include "sqp_safety_problems.h"

#include "argmin/detail/active_set_qp.h"

#include "argmin/solver/options.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/filter_nw_sqp_policy.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <optional>

using namespace argmin;

namespace
{

// |x - (2, 3, 5)|^2 under x_i <= 1: from the origin the active-set QP adds one blocking row per iteration, so
// its main QP needs three iterations.
defects::row_quadratic three_blocking_rows()
{
    return {2.0 * Eigen::MatrixXd::Identity(3, 3), Eigen::Vector3d(-4.0, -6.0, -10.0), Eigen::MatrixXd(0, 3), Eigen::VectorXd(0), -Eigen::MatrixXd::Identity(3, 3), -Eigen::VectorXd::Ones(3), defects::nowhere};
}

qp_options<double> one_qp_iteration()
{
    qp_options<double> opts;
    opts.max_iterations = 1;
    return opts;
}

// x0^2 + 1 = 0 has no solution, so every feasibility restoration fails.
struct infeasible_equality
{
    static constexpr std::int32_t problem_dimension = 2;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_equality() const { return 1; }

    std::int32_t num_inequality() const { return 0; }

    double value(const Eigen::Vector2d& x) const { return (x - Eigen::Vector2d(3.0, 1.0)).squaredNorm(); }

    void gradient(const Eigen::Vector2d& x, Eigen::Vector2d& g) const { g = 2.0 * (x - Eigen::Vector2d(3.0, 1.0)); }

    void constraints(const Eigen::Vector2d& x, Eigen::VectorXd& c) const { c = Eigen::VectorXd::Constant(1, x[0] * x[0] + 1.0); }

    void constraint_jacobian(const Eigen::Vector2d& x, Eigen::MatrixXd& J) const { J = Eigen::RowVector2d(2.0 * x[0], 0.0); }
};

std::optional<step_result<double>> first_infeasible_null_step()
{
    using policy = filter_nw_sqp_policy<double, 2>;
    const infeasible_equality problem{};
    policy::options_type opts;
    opts.max_restoration_steps = 1;
    opts.bfgs_reset_max = 0;
    step_budget_solver<policy, 2, infeasible_equality> solver{policy{}, problem, Eigen::Vector2d(0.5, 0.1), solver_options<>{}, opts};
    for(std::int32_t k = 0; k < 200; ++k)
    {
        const step_result<double> r = solver.step();
        if(r.is_null_step && r.constraint_violation > 1e-6)
            return r;
    }
    return std::nullopt;
}

}

TEST_CASE("the three-row QP needs more than one active-set iteration", "[sqp]")
{
    const defects::row_quadratic p = three_blocking_rows();
    detail::active_set_qp_solver<double> qp(3, 3);
    const auto r = qp.solve(Eigen::MatrixXd::Identity(3, 3).eval(), p.c, p.E, p.e, p.G, p.h, Eigen::VectorXd::Zero(3).eval(), one_qp_iteration());
    CHECK(r.status == detail::qp_status::max_iterations);
}

TEST_CASE("nw_sqp reports a status on a step whose QP stopped at its iteration limit", "[known-defect][!shouldfail]")
{
    nw_sqp_policy<>::options_type opts;
    opts.qp = one_qp_iteration();
    const defects::row_quadratic problem = three_blocking_rows();
    step_budget_solver<nw_sqp_policy<>, dynamic_dimension, defects::row_quadratic> solver{nw_sqp_policy<>{}, problem, Eigen::VectorXd::Zero(3).eval(), solver_options<>{}, opts};
    CHECK(solver.step().policy_status.has_value());
}

TEST_CASE("filter_nw_sqp reaches an infeasible null step on a problem with no feasible point", "[sqp]")
{
    CHECK(first_infeasible_null_step().has_value());
}

TEST_CASE("filter_nw_sqp reports a restoration failure distinct from stalled", "[known-defect][!shouldfail]")
{
    const std::optional<step_result<double>> r = first_infeasible_null_step();
    REQUIRE(r.has_value());
    CHECK((r->policy_status.has_value() && *r->policy_status != solver_status::stalled));
}
