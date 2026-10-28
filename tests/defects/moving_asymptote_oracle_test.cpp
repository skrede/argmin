#include "oracle_tolerances.h"

#include "cases/case_options.h"
#include "cases/case_problems.h"

#include "oracle/result_oracle.h"

#include "argmin/solver/options.h"
#include "argmin/solver/mma_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/solver/alternative/gcmma/move_limit_shrink_policy.h"

#include "argmin/test_functions/hock_schittkowski.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <initializer_list>

using namespace argmin;

namespace
{

constexpr std::int32_t n = hs076<>::problem_dimension;

template <typename Policy, typename Problem>
oracle::verdict<double> judge(const Problem& problem, const solver_options<>& opts)
{
    step_budget_solver solver{Policy{}, problem, problem.initial_point(), opts};
    const auto result = solver.solve(opts);
    return oracle::judge(problem, result, oracle::tolerances_of(opts));
}

void check_optimality_claim(const oracle::verdict<double>& v)
{
    CAPTURE(v.legs.stationarity);
    CHECK(v.claimed != oracle::claim::none);
    CHECK(v.ok);
}

// The case budget takes minutes on hs076 in a Debug build; the stall is as plain at 200 iterations, where
// the other variants have long converged.
solver_options<> move_limit_hs076_options()
{
    solver_options<> opts = cases::moving_asymptote_options();
    opts.max_iterations = 200;
    return opts;
}

using move_limit_shrink = alternative::gcmma::move_limit_shrink_policy<double, n>;

}

TEST_CASE("mma and gcmma move_limit_shrink stay feasible on the bound-only Wood problem", "[moving_asymptote]")
{
    const cases::no_constraint_rows<hs038<>> problem{};
    for(const oracle::verdict<double>& v : {judge<mma_policy<double, n>>(problem, cases::moving_asymptote_options()), judge<move_limit_shrink>(problem, cases::moving_asymptote_options())})
    {
        CHECK(v.legs.bound_violation == 0.0);
        CHECK(v.ok);
    }
}

TEST_CASE("mma makes an optimality claim the oracle accepts on the bound-only Wood problem", "[known-defect][!shouldfail]")
{
    check_optimality_claim(judge<mma_policy<double, n>>(cases::no_constraint_rows<hs038<>>{}, cases::moving_asymptote_options()));
}

TEST_CASE("gcmma move_limit_shrink makes an optimality claim the oracle accepts on the bound-only Wood problem", "[known-defect][!shouldfail]")
{
    check_optimality_claim(judge<move_limit_shrink>(cases::no_constraint_rows<hs038<>>{}, cases::moving_asymptote_options()));
}

TEST_CASE("gcmma move_limit_shrink stays feasible on hs076", "[moving_asymptote]")
{
    const oracle::verdict<double> v = judge<move_limit_shrink>(hs076<>{}, move_limit_hs076_options());
    CHECK(v.legs.inequality_violation <= oracle::tolerances_of(move_limit_hs076_options()).feasibility);
    CHECK(v.ok);
}

TEST_CASE("gcmma move_limit_shrink makes an optimality claim the oracle accepts on hs076", "[known-defect][!shouldfail]")
{
    check_optimality_claim(judge<move_limit_shrink>(hs076<>{}, move_limit_hs076_options()));
}
