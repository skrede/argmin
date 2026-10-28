#include "generated_qp.h"
#include "property_support.h"

#include "oracle_tolerances.h"

#include "cases/qp_cases.h"

#include "oracle/qp_oracle.h"

#include <catch2/catch_test_macros.hpp>

#include <rapidcheck.h>

using namespace argmin;

// The property claims the oracle accepts a solution moved 100 times past its acceptance bound, which is
// false on every solved problem, so it must fail; an oracle weakened to accept anything turns it red.
TEST_CASE("the oracle rejects a solution moved past its acceptance bound", "[property][!shouldfail]")
{
    property::check("planted dense_admm_qp solution moved past the acceptance bound",
                    []
                    {
                        const cases::qp_case c = *property::convex_qp();
                        auto solver = cases::make_qp_solver<dense_admm_qp_solver<double>>(c);
                        const auto result = cases::solve_qp(solver, c, dense_qp_options<double>{});
                        RC_PRE(result.has_value() && result->status == qp_solve_status::solved);
                        const oracle::qp_tolerances<double> tol = oracle::qp_tolerances_of(dense_qp_options<double>{});
                        RC_ASSERT(oracle::judge_qp(c.P, c.q, c.A, c.l, c.u, property::planted_qp_result(c, *result, tol), tol).ok);
                    });
}
