#include "rowed_box.h"

#include "argmin/solver/options.h"
#include "argmin/solver/lm_policy.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/multistart_policy.h"
#include "argmin/solver/restarting_policy.h"
#include "argmin/solver/byrd_lbfgsb_policy.h"
#include "argmin/solver/projected_gn_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/projected_gradient_gn_policy.h"

#include "argmin/solver/alternative/cmaes/repair_l2_penalty_policy.h"
#include "argmin/solver/alternative/cmaes/pwq_reparameterization_policy.h"
#include "argmin/solver/alternative/cmaes/no_repair_adaptive_penalty_policy.h"

#include "argmin/result/status.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <tuple>
#include <cstdint>

using namespace argmin;
using namespace argmin::rowed;

namespace
{

using pwq = alternative::cmaes::pwq_reparameterization_policy<double, 2>;
using repair_l2 = alternative::cmaes::repair_l2_penalty_policy<double, 2>;
using no_repair = alternative::cmaes::no_repair_adaptive_penalty_policy<double, 2>;
using lbfgsb = lbfgsb_policy<double, 2>;
using byrd_lbfgsb = byrd_lbfgsb_policy<double, 2>;
using bobyqa = bobyqa_policy<double, 2>;
using lm = lm_policy<double, 2>;
using projected_gn = projected_gn_policy<double, 2>;
using projected_gradient_gn = projected_gradient_gn_policy<double, 2>;
using box_only_families = std::tuple<pwq, repair_l2, no_repair, lbfgsb, byrd_lbfgsb, bobyqa, lm, projected_gn, projected_gradient_gn,
                                     restarting_policy<pwq>, restarting_policy<lbfgsb>, multistart_policy<bobyqa>>;

template <typename Policy>
void check_rejected_unevaluated(const rowed_box& problem)
{
    step_budget_solver solver{seeded<Policy>(), problem, start, solver_options<>{}};
    CHECK(solver.status() == solver_status::invalid_problem);
    CHECK(solver.solve().status == solver_status::invalid_problem);
    CHECK(solver.step().policy_status == solver_status::invalid_problem);
    CHECK(problem.evaluations == 0);
}

}

TEMPLATE_LIST_TEST_CASE("a box-only family rejects a problem with constraint rows as an invalid problem without evaluating it", "[invalid_problem]", box_only_families)
{
    SECTION("an inequality row")
    {
        check_rejected_unevaluated<TestType>(rowed_box{.n_eq = 0, .n_ineq = 1, .evaluations = 0});
    }
    SECTION("an equality row")
    {
        check_rejected_unevaluated<TestType>(rowed_box{.n_eq = 1, .n_ineq = 0, .evaluations = 0});
    }
}

TEMPLATE_LIST_TEST_CASE("a box-only family's reset rejects constraint rows its problem gained and runs once they are gone", "[invalid_problem]", box_only_families)
{
    rowed_box problem{.n_eq = 0, .n_ineq = 0, .evaluations = 0};
    step_budget_solver solver{seeded<TestType>(), problem, start, solver_options<>{}};
    REQUIRE(solver.status() == solver_status::running);
    problem.n_ineq = 1;
    CHECK(solver.reset(start) == solver_status::invalid_problem);
    CHECK(solver.reset_clear(start) == solver_status::invalid_problem);
    const std::int32_t evaluated = problem.evaluations;
    CHECK(solver.step().policy_status == solver_status::invalid_problem);
    CHECK(problem.evaluations == evaluated);
    problem.n_ineq = 0;
    CHECK(solver.reset(start) == solver_status::running);
    CHECK(solver.step().policy_status != solver_status::invalid_problem);
}

TEMPLATE_LIST_TEST_CASE("a box-only family solves a problem that declares zero constraint rows", "[solver]", box_only_families)
{
    const rowed_box problem{.n_eq = 0, .n_ineq = 0, .evaluations = 0};
    step_budget_solver solver{seeded<TestType>(), problem, start, solver_options<>{}};
    const auto result = solver.solve();
    CHECK(result.status != solver_status::invalid_problem);
    CHECK(result.objective_value < 1e-6);
}
