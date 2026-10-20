// Direct, platform-independent tests of the shared polish accept rule.
//
// The rule went wrong in a way no solver-level test caught on the platforms it
// normally runs on: it demanded a *strict* decrease in both residuals, so a
// polish that drove the dual residual down by orders of magnitude while the
// primal residual merely tied at exactly zero -- the value an already-feasible
// iterate carries -- was rejected. The tie only materializes when a platform's
// rounding lands the primal residual on exactly zero, which is why it surfaced
// first under an instrumented libc++ and stayed invisible elsewhere. Testing
// the predicate on exact inputs pins the fixed behavior on every platform
// rather than on whichever one happens to produce the tie.

#include "argmin/qp/detail/polish_accept.h"

#include <catch2/catch_test_macros.hpp>

using argmin::detail::polish_is_accepted;

namespace
{
// A wide slack so these cases turn entirely on the residual logic, never on the
// objective guard; the guard has its own cases below.
constexpr double wide_slack = 1.0;
}

TEST_CASE("polish is accepted when one residual improves and the other ties at zero",
          "[qp][polish_accept]")
{
    // The regression case: primal already exactly feasible (0 == 0), dual
    // improves by eight orders of magnitude, objective unchanged. The strict-
    // both rule rejected this; the Pareto rule must accept it.
    CHECK(polish_is_accepted(/*polished_primal=*/0.0, /*polished_dual=*/1.4e-17,
                             /*current_primal=*/0.0, /*current_dual=*/1.6e-9,
                             /*polished_obj=*/-0.31, /*current_obj=*/-0.31,
                             wide_slack, /*has_constraints=*/true));
}

TEST_CASE("polish is accepted when the primal improves and the dual ties",
          "[qp][polish_accept]")
{
    CHECK(polish_is_accepted(1.0e-12, 2.0e-10, 3.0e-6, 2.0e-10, -0.31, -0.31,
                             wide_slack, true));
}

TEST_CASE("a no-op polish is rejected", "[qp][polish_accept]")
{
    // Both residuals tie exactly: nothing improved, so the iterate must not be
    // replaced. This is the behavior the strict-both rule got right and the
    // Pareto rule must preserve.
    CHECK_FALSE(polish_is_accepted(5.0e-9, 3.0e-9, 5.0e-9, 3.0e-9, -0.31, -0.31,
                                   wide_slack, true));
}

TEST_CASE("a polish that worsens either residual is rejected", "[qp][polish_accept]")
{
    // Dual improves but primal worsens: not a Pareto improvement.
    CHECK_FALSE(polish_is_accepted(9.0e-6, 1.0e-12, 3.0e-6, 1.0e-9, -0.31, -0.31,
                                   wide_slack, true));
    // Primal improves but dual worsens.
    CHECK_FALSE(polish_is_accepted(1.0e-12, 9.0e-6, 3.0e-6, 1.0e-9, -0.31, -0.31,
                                   wide_slack, true));
}

TEST_CASE("the unconstrained case ignores the primal residual", "[qp][polish_accept]")
{
    // With no constraints the primal residual is not a meaningful quantity, so
    // the decision rests on the dual residual alone. A dual improvement is
    // accepted whatever the primal inputs say...
    CHECK(polish_is_accepted(/*polished_primal=*/7.0, 1.0e-12,
                             /*current_primal=*/0.0, 1.0e-9, -0.31, -0.31,
                             wide_slack, /*has_constraints=*/false));
    // ...and a dual tie is a no-op that must be rejected, matching the strict
    // rule's unconstrained behavior exactly.
    CHECK_FALSE(polish_is_accepted(0.0, 1.0e-9, 7.0, 1.0e-9, -0.31, -0.31,
                                   wide_slack, false));
}

TEST_CASE("a residual-improving polish that worsens the objective past slack is rejected",
          "[qp][polish_accept]")
{
    // The objective guard is one-sided: it rejects a polish that drives the
    // objective the *wrong* way -- higher, for a minimization -- past the slack
    // that prices the residual violation. This is the failure direction of the
    // dense polish defect (polish pinning interior variables to bounds, making
    // the objective worse while both residuals improved), and the guard is what
    // catches it. A polish that lowers the objective is not the danger here and
    // is left to the residual logic.
    const double tight_slack = 1.0e-6;
    CHECK_FALSE(polish_is_accepted(1.0e-12, 1.0e-12, 3.0e-6, 1.0e-6,
                                   /*polished_obj=*/-0.30, /*current_obj=*/-0.31,
                                   tight_slack, true));
    // The same residual improvement with the objective within slack is accepted.
    CHECK(polish_is_accepted(1.0e-12, 1.0e-12, 3.0e-6, 1.0e-6,
                             /*polished_obj=*/-0.31 + 5.0e-7, /*current_obj=*/-0.31,
                             tight_slack, true));
}
