#include "eigen_trap.h"

#include "argmin/solver/options.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"

#include "argmin/result/status.h"

#include "argmin/types.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <cstdint>

using namespace argmin;

namespace
{

using vector2 = Eigen::Vector2d;

// f = 0.5 |x - (1, 2)|^2 on the first `rows` of x0 + 10 >= 0, x1 + 10 >= 0 and 10 - x0 - x1 >= 0. Every row and the
// optional [-5, 5]^2 box are inactive at the minimizer, so no converging solve sits on a degenerate vertex.
template <std::int32_t Cap, bool Boxed>
struct capped_rows
{
    static constexpr std::int32_t problem_dimension = 2;
    static constexpr std::int32_t constraint_count = Cap;

    std::int32_t rows;
    mutable std::int32_t evaluations;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_equality() const { return 0; }

    std::int32_t num_inequality() const { return rows; }

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
        c = normals().topRows(rows) * x + Eigen::Vector3d::Constant(10.0).head(rows);
    }

    template <typename Jacobian>
    void constraint_jacobian(const vector2&, Jacobian& J) const
    {
        ++evaluations;
        J = normals().topRows(rows);
    }

    vector2 lower_bounds() const
        requires Boxed
    {
        return vector2::Constant(-5.0);
    }

    vector2 upper_bounds() const
        requires Boxed
    {
        return vector2::Constant(5.0);
    }

    static Eigen::Matrix<double, 3, 2> normals() { return (Eigen::Matrix<double, 3, 2>() << 1.0, 0.0, 0.0, 1.0, -1.0, -1.0).finished(); }
};

// f = 0.5 x^2 on one variable with the equality rows k (x - 1) = 0 for k = 1 .. rows, all satisfied at x = 1.
struct scaled_lines
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t rows;
    mutable std::int32_t evaluations;

    std::int32_t dimension() const { return 1; }

    std::int32_t num_equality() const { return rows; }

    std::int32_t num_inequality() const { return 0; }

    double value(const Eigen::VectorXd& x) const
    {
        ++evaluations;
        return 0.5 * x.squaredNorm();
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        ++evaluations;
        g = x;
    }

    template <typename Values>
    void constraints(const Eigen::VectorXd& x, Values& c) const
    {
        ++evaluations;
        c = slopes() * (x[0] - 1.0);
    }

    template <typename Jacobian>
    void constraint_jacobian(const Eigen::VectorXd&, Jacobian& J) const
    {
        ++evaluations;
        J = slopes();
    }

    Eigen::VectorXd slopes() const { return Eigen::VectorXd::LinSpaced(rows, 1.0, static_cast<double>(rows)); }
};

const vector2 inside(0.5, 0.5);

Eigen::VectorXd three()
{
    return Eigen::VectorXd::Constant(1, 3.0);
}

template <typename Policy, typename Problem, typename Start>
void check_rejected(const Problem& problem, const Start& x0)
{
    step_budget_solver solver{Policy{}, problem, x0, solver_options<>{}};
    CHECK(solver.status() == solver_status::invalid_problem);
    const std::int32_t evaluations = problem.evaluations;
    CHECK(solver.step().policy_status == solver_status::invalid_problem);
    CHECK(problem.evaluations == evaluations);
    step_budget_solver fresh{Policy{}, problem, x0, solver_options<>{}};
    CHECK(fresh.solve(solver_options<>{}).status == solver_status::invalid_problem);
}

template <typename Policy, typename Problem, typename Start, typename Minimizer>
void check_converges(const Problem& problem, const Start& x0, const Minimizer& minimizer)
{
    step_budget_solver solver{Policy{}, problem, x0, solver_options<>{}};
    CHECK(solver.status() == solver_status::running);
    const auto result = solver.solve(solver_options<>{});
    CHECK(result.status == solver_status::converged);
    CHECK((result.x - minimizer).norm() < 1e-6);
}

}

TEMPLATE_TEST_CASE("a line-search SQP policy rejects more constraint rows than its declared constraint count", "[release-safety][sqp-shape]", (nw_sqp_policy<double, 2>), (filter_nw_sqp_policy<double, 2>))
{
    SECTION("without a finite box")
    {
        check_rejected<TestType>(capped_rows<1, false>{.rows = 3, .evaluations = 0}, inside);
    }
    SECTION("with a finite box")
    {
        check_rejected<TestType>(capped_rows<1, true>{.rows = 3, .evaluations = 0}, inside);
    }
}

TEMPLATE_TEST_CASE("a line-search SQP policy solves a problem at or below its declared constraint count", "[release-safety][sqp-shape]", (nw_sqp_policy<double, 2>), (filter_nw_sqp_policy<double, 2>))
{
    const vector2 minimizer(1.0, 2.0);
    SECTION("three rows under a cap of three")
    {
        check_converges<TestType>(capped_rows<3, false>{.rows = 3, .evaluations = 0}, inside, minimizer);
    }
    SECTION("three rows and a finite box under a cap of three")
    {
        check_converges<TestType>(capped_rows<3, true>{.rows = 3, .evaluations = 0}, inside, minimizer);
    }
    SECTION("two rows under a cap of three")
    {
        check_converges<TestType>(capped_rows<3, false>{.rows = 2, .evaluations = 0}, inside, minimizer);
    }
}

TEMPLATE_TEST_CASE("a line-search SQP policy reset rejects rows past the cap or a changed row count without evaluating", "[release-safety][sqp-shape]", (nw_sqp_policy<double, 2>), (filter_nw_sqp_policy<double, 2>))
{
    capped_rows<3, false> problem{.rows = 2, .evaluations = 0};
    step_budget_solver solver{TestType{}, problem, inside, solver_options<>{}};
    for(const std::int32_t rows : {3, 4})
    {
        problem.rows = rows;
        const std::int32_t evaluations = problem.evaluations;
        CHECK(solver.reset(inside) == solver_status::invalid_problem);
        CHECK(solver.reset_clear(inside) == solver_status::invalid_problem);
        CHECK(solver.status() == solver_status::invalid_problem);
        CHECK(problem.evaluations == evaluations);
    }
    problem.rows = 2;
    CHECK(solver.reset(inside) == solver_status::running);
    CHECK(solver.solve(solver_options<>{}).status == solver_status::converged);
}

TEMPLATE_TEST_CASE("a line-search SQP policy reset keeps rejecting a problem over its declared constraint count", "[release-safety][sqp-shape]", (nw_sqp_policy<double, 2>), (filter_nw_sqp_policy<double, 2>))
{
    const capped_rows<1, false> problem{.rows = 3, .evaluations = 0};
    step_budget_solver solver{TestType{}, problem, inside, solver_options<>{}};
    const std::int32_t evaluations = problem.evaluations;
    CHECK(solver.reset(inside) == solver_status::invalid_problem);
    CHECK(solver.reset_clear(inside) == solver_status::invalid_problem);
    CHECK(problem.evaluations == evaluations);
}

TEMPLATE_TEST_CASE("a line-search SQP policy rejects more equality rows than variables", "[release-safety][sqp-shape]", nw_sqp_policy<>, filter_nw_sqp_policy<>, kraft_slsqp_policy<>, filter_slsqp_policy<>)
{
    check_rejected<TestType>(scaled_lines{.rows = 2, .evaluations = 0}, three());
}

TEMPLATE_TEST_CASE("a line-search SQP policy solves one equality row on one variable", "[release-safety][sqp-shape]", nw_sqp_policy<>, filter_nw_sqp_policy<>, kraft_slsqp_policy<>, filter_slsqp_policy<>)
{
    check_converges<TestType>(scaled_lines{.rows = 1, .evaluations = 0}, three(), Eigen::VectorXd::Ones(1));
}

TEMPLATE_TEST_CASE(
    "a line-search SQP policy reset rejects a changed equality row count until it is restored", "[release-safety][sqp-shape]", nw_sqp_policy<>, filter_nw_sqp_policy<>, kraft_slsqp_policy<>, filter_slsqp_policy<>)
{
    scaled_lines problem{.rows = 1, .evaluations = 0};
    step_budget_solver solver{TestType{}, problem, three(), solver_options<>{}};
    problem.rows = 2;
    const std::int32_t evaluations = problem.evaluations;
    CHECK(solver.reset(three()) == solver_status::invalid_problem);
    CHECK(solver.reset_clear(three()) == solver_status::invalid_problem);
    CHECK(problem.evaluations == evaluations);
    problem.rows = 1;
    CHECK(solver.reset_clear(three()) == solver_status::running);
    CHECK(solver.solve(solver_options<>{}).status == solver_status::converged);
}
