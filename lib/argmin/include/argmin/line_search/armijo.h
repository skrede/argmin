#ifndef HPP_GUARD_ARGMIN_LINE_SEARCH_ARMIJO_H
#define HPP_GUARD_ARGMIN_LINE_SEARCH_ARMIJO_H

#include "argmin/line_search/result.h"
#include "argmin/line_search/options.h"

#include <cmath>

namespace argmin
{

// Backtracking line search with Armijo (sufficient decrease) condition.
//
// Given a descent direction with directional derivative dphi0 < 0,
// finds alpha such that:
//
//   phi(alpha) <= phi0 + c1 * alpha * dphi0
//
// Starting from alpha = max_alpha, the step is shrunk by factor rho
// on each iteration until the condition is satisfied or the evaluation
// budget is exhausted.
//
// Reference: K&W Algorithm 4.3 (backtracking line search).
//            N&W Procedure 3.1, p. 37 (Armijo backtracking).

template <typename Phi, typename Scalar = double>
[[nodiscard]] line_search_result<Scalar> armijo(Phi&& phi,
                                                Scalar phi0,
                                                Scalar dphi0,
                                                const line_search_options& opts = {})
{
    line_search_result<Scalar> result;
    result.alpha = opts.max_alpha;
    result.value = phi0;

    Scalar alpha = opts.max_alpha;

    for(int i = 0; i < opts.max_iterations; ++i)
    {
        Scalar phi_alpha = phi(alpha);
        ++result.evaluations;

        // NaN/Inf recovery: treat non-finite phi as a backtrack trigger so
        // the Armijo comparison below stays defined. A non-finite return
        // value indicates either the objective or a constraint propagated
        // NaN/Inf on this trial iterate, or the iterate fell outside the
        // function's natural domain. Both fast and accurate modes enable
        // this gate — non-finite trial-point evaluations should never be
        // silently consumed.
        //
        // Reference: Ipopt IpIpoptCalculatedQuantities::f_or_grad_returned_nan
        //            (NaN detection model; argmin variant is Armijo-only).
        if(!std::isfinite(phi_alpha))
        {
            ++result.diagnostics.nan_eval_count;
            alpha *= opts.rho;
            continue;
        }

        result.alpha = alpha;
        result.value = phi_alpha;

        if(phi_alpha <= phi0 + opts.c1 * alpha * dphi0)
        {
            result.success = true;
            return result;
        }

        alpha *= opts.rho;
    }

    return result;
}

}

#endif
