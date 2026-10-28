#include "oracle_tolerances.h"

#include "cases/case_options.h"
#include "cases/case_problems.h"

#include "oracle/result_oracle.h"

#include "argmin/solver/options.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/projected_gradient_gn_policy.h"

#include "argmin/result/solve_result.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;

namespace
{

oracle::verdict<double> judge_bounded_rosenbrock(const cases::bounded_rosenbrock_ls2& problem)
{
    const solver_options<> opts = cases::least_squares_options();
    step_budget_solver solver{projected_gradient_gn_policy<>{}, problem, problem.initial_point(), opts};
    const solve_result<double, 2> result = solver.solve(opts);
    return oracle::judge(problem, result, oracle::tolerances_of(opts));
}

}

TEST_CASE("projected_gradient_gn stays feasible on the bounded Rosenbrock least-squares case", "[least_squares]")
{
    const cases::bounded_rosenbrock_ls2 problem{};
    const oracle::verdict<double> v = judge_bounded_rosenbrock(problem);
    CHECK(v.legs.bound_violation == 0.0);
    CHECK(v.ok);
}

TEST_CASE("projected_gradient_gn makes an optimality claim the oracle accepts on the bounded Rosenbrock least-squares case", "[known-defect][!shouldfail]")
{
    const cases::bounded_rosenbrock_ls2 problem{};
    const oracle::verdict<double> v = judge_bounded_rosenbrock(problem);
    CAPTURE(v.legs.stationarity);
    CHECK(v.claimed != oracle::claim::none);
    CHECK(v.ok);
}
