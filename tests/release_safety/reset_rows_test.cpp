#include "eigen_trap.h"

#include "argmin/solver/options.h"
#include "argmin/solver/lm_policy.h"
#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/projected_gn_policy.h"
#include "argmin/solver/filter_trsqp_policy.h"
#include "argmin/solver/augmented_lagrangian_policy.h"
#include "argmin/solver/projected_gradient_gn_policy.h"

#include "argmin/result/status.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <tuple>
#include <cstdint>
#include <initializer_list>

using namespace argmin;

namespace
{

using vector2 = Eigen::Vector2d;

const vector2 start(0.0, 0.0);

// The rows (1, 0), (0, 1), (1, 1), filled one coefficient at a time as a caller writing the handed buffer by index
// would, so a buffer sized for fewer rows than the current count is overrun.
Eigen::Matrix<double, 3, 2> normals()
{
    return (Eigen::Matrix<double, 3, 2>() << 1.0, 0.0, 0.0, 1.0, 1.0, 1.0).finished();
}

// f = 0.5 |x - (1, 2)|^2 in the [-5, 5]^2 box with the optional equality x0 + x1 = 3 and the first `ineq_rows` of
// 10 + n_i . x >= 0, all satisfied at the minimizer and the inequalities inactive there.
struct counted_rows
{
    static constexpr std::int32_t problem_dimension = 2;

    std::int32_t eq_rows;
    std::int32_t ineq_rows;
    mutable std::int32_t evaluations;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_equality() const { return eq_rows; }

    std::int32_t num_inequality() const { return ineq_rows; }

    double value(const vector2& x) const
    {
        ++evaluations;
        return 0.5 * (x - vector2(1.0, 2.0)).squaredNorm();
    }

    void gradient(const vector2& x, vector2& g) const
    {
        ++evaluations;
        g = x - vector2(1.0, 2.0);
    }

    template <typename Values>
    void constraints(const vector2& x, Values& c) const
    {
        ++evaluations;
        for(std::int32_t i = 0; i < eq_rows; ++i)
            c(i) = x.sum() - 3.0;
        for(std::int32_t i = 0; i < ineq_rows; ++i)
            c(eq_rows + i) = 10.0 + normals().row(i).dot(x);
    }

    template <typename Jacobian>
    void constraint_jacobian(const vector2&, Jacobian& J) const
    {
        ++evaluations;
        for(std::int32_t i = 0; i < eq_rows; ++i)
            J.row(i) = normals().row(2);
        for(std::int32_t i = 0; i < ineq_rows; ++i)
            J.row(eq_rows + i) = normals().row(i);
    }

    vector2 lower_bounds() const { return vector2::Constant(-5.0); }

    vector2 upper_bounds() const { return vector2::Constant(5.0); }
};

// The first `rows` of n_i . x - t_i with t = (1, 2, 3), all zero at (1, 2) in the [-5, 5]^2 box.
struct counted_residuals
{
    static constexpr std::int32_t problem_dimension = 2;

    std::int32_t rows;
    mutable std::int32_t evaluations;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_residuals() const { return rows; }

    double value(const vector2& x) const
    {
        ++evaluations;
        return 0.5 * (normals().topRows(rows) * x - Eigen::Vector3d(1.0, 2.0, 3.0).head(rows)).squaredNorm();
    }

    void residuals(const vector2& x, Eigen::VectorXd& r) const
    {
        ++evaluations;
        for(std::int32_t i = 0; i < rows; ++i)
            r(i) = normals().row(i).dot(x) - static_cast<double>(i + 1);
    }

    void jacobian(const vector2&, Eigen::MatrixXd& J) const
    {
        ++evaluations;
        for(std::int32_t i = 0; i < rows; ++i)
            J.row(i) = normals().row(i);
    }

    vector2 lower_bounds() const { return vector2::Constant(-5.0); }

    vector2 upper_bounds() const { return vector2::Constant(5.0); }
};

using constrained_policies = std::tuple<tr_sqp_policy<double, 2>, filter_trsqp_policy<double, 2>, augmented_lagrangian_policy<lbfgsb_policy<double, 2>>, cobyla_policy<double>>;
using least_squares_policies = std::tuple<lm_policy<double, 2>, projected_gn_policy<double, 2>, projected_gradient_gn_policy<double, 2>>;

template <typename Policy, typename Problem>
void check_solver_rejects(Problem problem, std::int32_t Problem::* count, std::int32_t changed)
{
    step_budget_solver solver{Policy{}, problem, start, solver_options<>{}};
    REQUIRE(solver.step().policy_status != solver_status::invalid_problem);
    const std::int32_t built = problem.*count;
    problem.*count = changed;
    const std::int32_t evaluations = problem.evaluations;
    CHECK(solver.reset(start) == solver_status::invalid_problem);
    CHECK(solver.reset_clear(start) == solver_status::invalid_problem);
    CHECK(solver.status() == solver_status::invalid_problem);
    CHECK(solver.step().policy_status == solver_status::invalid_problem);
    CHECK(problem.evaluations == evaluations);
    problem.*count = built;
    CHECK(solver.reset(start) == solver_status::running);
    CHECK(solver.step().policy_status != solver_status::invalid_problem);
}

template <typename Policy, typename Problem>
void check_policy_rejects(Problem problem, std::int32_t Problem::* count, std::int32_t changed)
{
    Policy policy;
    auto s = policy.init(problem, start, solver_options<>{});
    const std::int32_t built = problem.*count;
    problem.*count = changed;
    const std::int32_t evaluations = problem.evaluations;
    CHECK(policy.reset(s, start) == solver_status::invalid_problem);
    CHECK(policy.reset_clear(s, start) == solver_status::invalid_problem);
    CHECK(policy.step(s).policy_status == solver_status::invalid_problem);
    CHECK(problem.evaluations == evaluations);
    problem.*count = built;
    CHECK(policy.reset(s, start) == solver_status::running);
    CHECK(policy.step(s).policy_status != solver_status::invalid_problem);
}

const counted_rows two_inequalities{.eq_rows = 0, .ineq_rows = 2, .evaluations = 0};
const counted_residuals two_residuals{.rows = 2, .evaluations = 0};

}

TEMPLATE_LIST_TEST_CASE("a constrained reset rejects a changed row count until it is restored", "[release-safety][reset-rows]", constrained_policies)
{
    for(const std::int32_t rows : {3, 1})
        check_solver_rejects<TestType>(two_inequalities, &counted_rows::ineq_rows, rows);
    check_solver_rejects<TestType>(two_inequalities, &counted_rows::eq_rows, 1);
}

TEMPLATE_LIST_TEST_CASE("a constrained policy rejects a changed row count on a direct reset", "[release-safety][reset-rows]", constrained_policies)
{
    check_policy_rejects<TestType>(two_inequalities, &counted_rows::ineq_rows, 3);
}

TEMPLATE_LIST_TEST_CASE("a least-squares reset rejects a changed residual count until it is restored", "[release-safety][reset-rows]", least_squares_policies)
{
    for(const std::int32_t rows : {3, 1})
        check_solver_rejects<TestType>(two_residuals, &counted_residuals::rows, rows);
}

TEMPLATE_LIST_TEST_CASE("a least-squares policy rejects a changed residual count on a direct reset", "[release-safety][reset-rows]", least_squares_policies)
{
    check_policy_rejects<TestType>(two_residuals, &counted_residuals::rows, 3);
}
