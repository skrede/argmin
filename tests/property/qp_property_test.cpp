#include "generated_qp.h"
#include "property_support.h"

#include "oracle_tolerances.h"

#include "cases/qp_cases.h"

#include "oracle/qp_oracle.h"

#include <catch2/catch_test_macros.hpp>

#include <rapidcheck.h>

#include <string>
#include <cstdint>

using namespace argmin;

namespace
{

template <typename Solver, typename Options>
void judge_generated_qp(const Options& options)
{
    const cases::qp_case c = *property::convex_qp();
    auto solver = cases::make_qp_solver<Solver>(c);
    const auto result = cases::solve_qp(solver, c, options);
    if(!result.has_value())
        RC_FAIL("qp_error " + std::to_string(static_cast<std::int32_t>(result.error())));
    const oracle::qp_tolerances<double> tol = oracle::qp_tolerances_of(options);
    const qp_result<double> judged = property::plant_violation() ? property::planted_qp_result(c, *result, tol) : *result;
    const oracle::qp_verdict v = oracle::judge_qp(c.P, c.q, c.A, c.l, c.u, judged, tol);
    RC_LOG() << "status " << static_cast<std::int32_t>(result->status) << " after " << result->iterations << " iterations\n";
    RC_ASSERT(v.failing == std::string());
}

}

TEST_CASE("the oracle accepts dense ADMM results on generated convex QPs", "[property]")
{
    property::check("dense_admm_qp convex", [] { judge_generated_qp<dense_admm_qp_solver<double>>(dense_qp_options<double>{}); });
}

TEST_CASE("the oracle accepts sparse ADMM results on generated convex QPs", "[property]")
{
    property::check("sparse_admm_qp convex", [] { judge_generated_qp<sparse_admm_qp_solver<double>>(sparse_qp_options<double>{}); });
}

TEST_CASE("the oracle accepts dense active-set results on generated convex QPs", "[property]")
{
    property::check("dense_active_set_qp convex", [] { judge_generated_qp<dense_active_set_qp_solver<double>>(dense_active_set_qp_options<double>{}); });
}
