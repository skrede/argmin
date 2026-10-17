#ifndef HPP_GUARD_ARGMIN_LINE_SEARCH_STRONG_WOLFE_H
#define HPP_GUARD_ARGMIN_LINE_SEARCH_STRONG_WOLFE_H

#include "argmin/line_search/result.h"
#include "argmin/line_search/options.h"
#include "argmin/line_search/approximate_wolfe.h"
#include "argmin/line_search/strong_wolfe_zoom.h"

#include <cmath>
#include <algorithm>

namespace argmin
{

// Strong Wolfe line search with bracketing and zoom phases.
//
// Finds alpha satisfying the strong Wolfe conditions:
//
//   (i)  phi(alpha) <= phi(0) + c1 * alpha * phi'(0)   (sufficient decrease)
//   (ii) |phi'(alpha)| <= c2 * |phi'(0)|               (curvature)
//
// The algorithm proceeds in two phases:
//   1. Bracketing (N&W Algorithm 3.5): starting from a unit initial trial
//      (alpha_1 = 1), grow geometrically -- capped at max_alpha -- until an
//      interval [lo, hi] that contains a point satisfying both conditions
//      is found.
//   2. Zoom (N&W Algorithm 3.6): bisect the bracket to find the point.
//
// NaN/Inf gate semantics mirror the Armijo backtracker (see
// argmin/line_search/armijo.h): every trial phi(alpha), and the slope
// phi'(alpha) where it is evaluated, is tested with std::isfinite before
// any ordered Armijo/value-increase/curvature comparison, so a non-finite
// trial is routed toward the zoom bracket rather than flowing NaN through
// an ordered comparison. Any return path that does not achieve success
// reports result.value = phi0 and result.success = false, matching the
// Armijo backtracker's failure contract.
//
// Reference: N&W Algorithms 3.5/3.6, pp. 60-62.
//            K&W Algorithm 4.4 (strong Wolfe line search).

template <typename Phi, typename DPhi, typename Scalar = double>
[[nodiscard]] line_search_result<Scalar> strong_wolfe(Phi&& phi,
                                                      DPhi&& dphi,
                                                      Scalar phi0,
                                                      Scalar dphi0,
                                                      const line_search_options<Scalar>& opts = {})
{
    line_search_result<Scalar> result;
    result.value = phi0;

    auto budget_ok = [&]() { return result.evaluations < opts.max_iterations; };

    // Any return path that has not achieved success reports value = phi0,
    // matching the Armijo backtracker's failure contract (see armijo.h).
    auto finalize = [&](line_search_result<Scalar> r) -> line_search_result<Scalar>
    {
        if(!r.success)
        {
            r.value = phi0;
            r.success = false;
        }
        return r;
    };

    // --- bracketing phase (N&W Algorithm 3.5) ---
    Scalar alpha_prev = Scalar(0);
    Scalar phi_prev = phi0;
    Scalar dphi_prev = dphi0;
    Scalar alpha = std::min(Scalar(1), opts.max_alpha);
    result.alpha = alpha;

    for(int i = 1; budget_ok(); ++i)
    {
        Scalar phi_alpha = phi(alpha);
        ++result.evaluations;

        bool phi_alpha_finite = std::isfinite(phi_alpha);
        if(!phi_alpha_finite)
            ++result.diagnostics.nan_eval_count;

        // Condition 1: non-finite trial, Armijo violated, or value
        // increased from previous. The isfinite gate short-circuits the
        // ordered comparisons that follow it.
        if(!phi_alpha_finite ||
           phi_alpha > phi0 + opts.c1 * alpha * dphi0 ||
           (phi_alpha >= phi_prev && i > 1))
        {
            if(detail::approximately_wolfe(dphi, alpha, phi_alpha, phi0, dphi0, opts, result))
                return result;
            return finalize(
                detail::strong_wolfe_zoom(phi, dphi, detail::wolfe_bracket<Scalar>{alpha_prev, alpha, phi_prev, dphi_prev, phi_alpha},
                                          phi0, dphi0, opts, result));
        }

        Scalar dphi_alpha = dphi(alpha);
        ++result.evaluations;

        bool dphi_alpha_finite = std::isfinite(dphi_alpha);
        if(!dphi_alpha_finite)
            ++result.diagnostics.nan_eval_count;

        // Condition 2: both Wolfe conditions satisfied
        if(dphi_alpha_finite && std::abs(dphi_alpha) <= opts.c2 * std::abs(dphi0))
        {
            result.alpha = alpha;
            result.value = phi_alpha;
            result.success = true;
            return result;
        }

        // Condition 3: non-finite slope, or positive slope -- bracket
        // found in reversed order.
        if(!dphi_alpha_finite || dphi_alpha >= Scalar(0))
        {
            return finalize(
                detail::strong_wolfe_zoom(phi, dphi, detail::wolfe_bracket<Scalar>{alpha, alpha_prev, phi_alpha, dphi_alpha, phi_prev},
                                          phi0, dphi0, opts, result));
        }

        // Expand step: geometric growth from the unit initial trial
        // (N&W Algorithm 3.5), capped at max_alpha.
        alpha_prev = alpha;
        phi_prev = phi_alpha;
        dphi_prev = dphi_alpha;
        alpha = std::min(Scalar(2) * alpha, opts.max_alpha);
        result.alpha = alpha;

        // If alpha cannot grow further, we are stuck at max_alpha.
        if(alpha == alpha_prev)
        {
            return finalize(result);
        }
    }

    return finalize(result);
}

}

#endif
