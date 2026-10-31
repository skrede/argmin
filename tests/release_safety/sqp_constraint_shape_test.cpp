#include "eigen_trap.h"
#include "sqp_shape_probe.h"

#include "argmin/solver/options.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"

#include "argmin/result/status.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <cstdint>

using namespace argmin;
using namespace argmin::release_safety;

namespace
{

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

TEMPLATE_TEST_CASE("a line-search SQP policy rejects more equality rows than variables", "[release-safety][sqp-shape]", nw_sqp_policy<>, filter_nw_sqp_policy<>, kraft_slsqp_policy<>, filter_slsqp_policy<>)
{
    check_rejected<TestType>(scaled_lines{.rows = 2, .evaluations = 0}, three());
}

TEMPLATE_TEST_CASE("a line-search SQP policy solves one equality row on one variable", "[release-safety][sqp-shape]", nw_sqp_policy<>, filter_nw_sqp_policy<>, kraft_slsqp_policy<>, filter_slsqp_policy<>)
{
    check_converges<TestType>(scaled_lines{.rows = 1, .evaluations = 0}, three(), Eigen::VectorXd::Ones(1));
}
