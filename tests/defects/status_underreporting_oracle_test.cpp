#include "oracle_tolerances.h"

#include "cases/case_options.h"

#include "oracle/result_oracle.h"

#include "argmin/solver/options.h"
#include "argmin/solver/sqp_mode.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_trsqp_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/solver/augmented_lagrangian_policy.h"

#include "argmin/test_functions/hock_schittkowski.h"

#include "argmin/types.h"

#include <catch2/catch_test_macros.hpp>

using namespace argmin;

namespace
{

template <typename Policy, typename Problem>
void check_optimality_claim(const Problem& problem)
{
    const solver_options<> opts = cases::gradient_options();
    step_budget_solver solver{Policy{}, problem, problem.initial_point(), opts};
    const oracle::verdict<double> v = oracle::judge(problem, solver.solve(opts), oracle::tolerances_of(opts));
    CAPTURE(v.legs.stationarity);
    CHECK((v.claimed == oracle::claim::optimality || v.claimed == oracle::claim::ftol_optimality));
    CHECK(v.ok);
}

template <sqp_mode Mode>
using tr_sqp = tr_sqp_policy<double, dynamic_dimension, Mode>;

template <sqp_mode Mode>
using filter_trsqp = filter_trsqp_policy<double, dynamic_dimension, Mode>;

using al_lbfgsb = augmented_lagrangian_policy<lbfgsb_policy<>>;

}

TEST_CASE("kraft_slsqp makes an optimality claim the oracle accepts on hs071", "[known-defect][!shouldfail]")
{
    check_optimality_claim<kraft_slsqp_policy<>>(hs071<>{});
}

TEST_CASE("filter_slsqp makes an optimality claim the oracle accepts on hs071", "[known-defect][!shouldfail]")
{
    check_optimality_claim<filter_slsqp_policy<>>(hs071<>{});
}

TEST_CASE("nw_sqp makes an optimality claim the oracle accepts on hs071", "[known-defect][!shouldfail]")
{
    check_optimality_claim<nw_sqp_policy<>>(hs071<>{});
}

TEST_CASE("filter_nw_sqp makes an optimality claim the oracle accepts on hs071", "[known-defect][!shouldfail]")
{
    check_optimality_claim<filter_nw_sqp_policy<>>(hs071<>{});
}

TEST_CASE("filter_trsqp in fast mode makes an optimality claim the oracle accepts on hs076", "[known-defect][!shouldfail]")
{
    check_optimality_claim<filter_trsqp<sqp_mode::fast>>(hs076<>{});
}

TEST_CASE("filter_trsqp in accurate mode makes an optimality claim the oracle accepts on hs076", "[known-defect][!shouldfail]")
{
    check_optimality_claim<filter_trsqp<sqp_mode::accurate>>(hs076<>{});
}

TEST_CASE("tr_sqp in accurate mode makes an optimality claim the oracle accepts on hs044", "[known-defect][!shouldfail]")
{
    check_optimality_claim<tr_sqp<sqp_mode::accurate>>(hs044<>{});
}

TEST_CASE("filter_trsqp in accurate mode makes an optimality claim the oracle accepts on hs044", "[known-defect][!shouldfail]")
{
    check_optimality_claim<filter_trsqp<sqp_mode::accurate>>(hs044<>{});
}

TEST_CASE("the augmented Lagrangian over lbfgsb makes an optimality claim the oracle accepts on hs071", "[known-defect][!shouldfail]")
{
    check_optimality_claim<al_lbfgsb>(hs071<>{});
}

TEST_CASE("the augmented Lagrangian over lbfgsb makes an optimality claim the oracle accepts on hs076", "[known-defect][!shouldfail]")
{
    check_optimality_claim<al_lbfgsb>(hs076<>{});
}
