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

TEMPLATE_TEST_CASE("a line-search SQP policy reset rejects a changed inequality row count until it is restored", "[release-safety][sqp-shape]", (kraft_slsqp_policy<double, 2>), (filter_slsqp_policy<double, 2>))
{
    capped_rows<3, false> problem{.rows = 2, .evaluations = 0};
    step_budget_solver solver{TestType{}, problem, inside, solver_options<>{}};
    for(const std::int32_t rows : {3, 1})
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
