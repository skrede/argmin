#ifndef HPP_GUARD_ARGMIN_QP_DETAIL_POLISH_ACCEPT_H
#define HPP_GUARD_ARGMIN_QP_DETAIL_POLISH_ACCEPT_H

// The rule deciding whether a polished iterate replaces the ADMM iterate that
// produced it. Shared verbatim by the dense and sparse solvers: the two once
// carried hand-synced copies, and the copies drifting apart is exactly how the
// answer to "did polish help?" could diverge between the two solvers on the
// same problem.
//
// Accept on a Pareto improvement, not on strict improvement in both residuals.
// Requiring each residual to strictly improve discards a polish that improves
// one by orders of magnitude while the other merely ties -- and the tie is not
// hypothetical: an iterate that is already exactly feasible has a primal
// residual of exactly zero, which no polish can strictly beat, so the strict
// form throws the better iterate away. Requiring at least one strict
// improvement still rejects a genuine no-op polish, so the unconstrained
// (has_constraints == false) case and the both-residuals-tie case decide
// exactly as the strict-both rule did.
//
// The objective guard is unconditional: an accepted iterate is allowed to
// undercut the optimal objective only by the caller-supplied slack, which
// prices the residual violation the iterate is accepted while still carrying.
// A polish that beats the objective by more than that is a broken polish, not
// rounding, and is rejected however much it improves the residuals.

namespace argmin::detail
{

template <class Scalar>
bool polish_is_accepted(Scalar polished_primal_res,
                        Scalar polished_dual_res,
                        Scalar current_primal_res,
                        Scalar current_dual_res,
                        Scalar polished_objective,
                        Scalar current_objective,
                        Scalar objective_slack,
                        bool has_constraints)
{
    const bool dual_not_worse = polished_dual_res <= current_dual_res;
    const bool primal_not_worse =
        !has_constraints || polished_primal_res <= current_primal_res;
    const bool strictly_better =
        polished_dual_res < current_dual_res
        || (has_constraints && polished_primal_res < current_primal_res);
    return dual_not_worse && primal_not_worse && strictly_better
           && polished_objective <= current_objective + objective_slack;
}

}

#endif
