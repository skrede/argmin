#ifndef HPP_GUARD_ARGMIN_LINE_SEARCH_ARMIJO_H
#define HPP_GUARD_ARGMIN_LINE_SEARCH_ARMIJO_H

#include "argmin/line_search/result.h"
#include "argmin/line_search/options.h"

#include <cmath>

// NaN/Inf gate semantics for the Armijo backtracker
//
// The backtracker tests `std::isfinite(phi_alpha)` before the Armijo
// sufficient-decrease comparison so a non-finite trial-iterate
// triggers a backtrack (alpha *= rho; continue) rather than crossing
// an IEEE-754 ordered comparison. The hand-rolled inline LS loops in
// the four line-search SQP policies mirror this pattern at their
// respective trial-iterate evaluation sites. The gate is enabled in
// both fast and accurate modes; NaN should never be silently
// consumed.
//
// Reference: Ipopt IpIpoptCalculatedQuantities::f_or_grad_returned_nan
//            (NaN detection model; argmin variant is Armijo-only —
//             QP and SOC sanitize layers are out of scope).

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

        // NaN/Inf gate: see header comment above for semantics and reference.
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
