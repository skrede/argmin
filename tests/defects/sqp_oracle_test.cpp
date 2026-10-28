#include "oracle_tolerances.h"

#include "cases/case_options.h"

#include "oracle/result_oracle.h"

#include "argmin/solver/options.h"
#include "argmin/solver/sqp_mode.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/test_functions/hock_schittkowski.h"

#include "argmin/types.h"

#include <catch2/catch_test_macros.hpp>

#include <initializer_list>

using namespace argmin;

namespace
{

template <sqp_mode Mode, typename Problem>
oracle::verdict<double> judge_tr_sqp(const Problem& problem)
{
    const solver_options<> opts = cases::gradient_options();
    step_budget_solver solver{tr_sqp_policy<double, dynamic_dimension, Mode>{}, problem, problem.initial_point(), opts};
    const auto result = solver.solve(opts);
    return oracle::judge(problem, result, oracle::tolerances_of(opts));
}

template <sqp_mode Mode, typename Problem>
void check_optimality_claim(const Problem& problem)
{
    const oracle::verdict<double> v = judge_tr_sqp<Mode>(problem);
    CAPTURE(v.legs.stationarity);
    CHECK(v.claimed != oracle::claim::none);
    CHECK(v.ok);
}

}

TEST_CASE("tr_sqp in fast mode stops at the hs076 optimum", "[sqp]")
{
    const oracle::verdict<double> v = judge_tr_sqp<sqp_mode::fast>(hs076<>{});
    CHECK(v.legs.stationarity <= oracle::tolerances_of(cases::gradient_options()).stationarity);
    CHECK(v.ok);
}

TEST_CASE("tr_sqp in fast mode makes an optimality claim the oracle accepts on hs076", "[known-defect][!shouldfail]")
{
    check_optimality_claim<sqp_mode::fast>(hs076<>{});
}

TEST_CASE("tr_sqp stays feasible on hs071 in both modes", "[sqp]")
{
    const double feasibility = oracle::tolerances_of(cases::gradient_options()).feasibility;
    for(const oracle::verdict<double>& v : {judge_tr_sqp<sqp_mode::fast>(hs071<>{}), judge_tr_sqp<sqp_mode::accurate>(hs071<>{})})
    {
        CHECK(v.legs.equality_violation <= feasibility);
        CHECK(v.legs.inequality_violation <= feasibility);
        CHECK(v.ok);
    }
}

TEST_CASE("tr_sqp in fast mode makes an optimality claim the oracle accepts on hs071", "[known-defect][!shouldfail]")
{
    check_optimality_claim<sqp_mode::fast>(hs071<>{});
}

TEST_CASE("tr_sqp in accurate mode makes an optimality claim the oracle accepts on hs071", "[known-defect][!shouldfail]")
{
    check_optimality_claim<sqp_mode::accurate>(hs071<>{});
}
