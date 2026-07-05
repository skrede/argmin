#ifndef HPP_GUARD_ARGMIN_LINE_SEARCH_STRONG_WOLFE_H
#define HPP_GUARD_ARGMIN_LINE_SEARCH_STRONG_WOLFE_H

#include "argmin/line_search/result.h"
#include "argmin/line_search/options.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

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
                                                      const line_search_options& opts = {})
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

    // Quadratic interpolation minimizer (N&W p. 59).
    // Given (a_lo, phi_lo, dphi_lo) and (a_hi, phi_hi), returns the
    // minimizer of the quadratic interpolant, or nullopt if degenerate.
    auto quadratic_minimize = [](Scalar a_lo, Scalar phi_lo, Scalar dphi_lo,
                                 Scalar a_hi, Scalar phi_hi) -> std::optional<Scalar>
    {
        Scalar d = a_hi - a_lo;
        if(std::abs(d) < std::numeric_limits<Scalar>::epsilon())
            return std::nullopt;

        Scalar denom = Scalar(2) * (phi_hi - phi_lo - dphi_lo * d);
        if(std::abs(denom) < Scalar(1e-12) * std::max(Scalar(1), std::abs(phi_hi - phi_lo)))
            return std::nullopt;

        Scalar alpha_min = a_lo - dphi_lo * d * d / denom;

        Scalar lo = std::min(a_lo, a_hi);
        Scalar hi = std::max(a_lo, a_hi);
        Scalar margin = Scalar(0.2) * (hi - lo);
        if(!std::isfinite(alpha_min) ||
           alpha_min <= lo + margin || alpha_min >= hi - margin)
            return std::nullopt;

        return alpha_min;
    };

    // --- zoom phase (N&W Algorithm 3.6) ---
    // Interpolation-based refinement within a bracket [alpha_lo, alpha_hi],
    // with bisection fallback when interpolation is degenerate.
    // Reference: N&W pp. 59-62.
    auto zoom = [&](Scalar alpha_lo, Scalar alpha_hi,
                    Scalar phi_lo, Scalar dphi_lo_val,
                    Scalar phi_hi) -> line_search_result<Scalar>
    {
        int zoom_iter = 0;
        while(budget_ok())
        {
            auto interp = (zoom_iter > 0)
                ? quadratic_minimize(alpha_lo, phi_lo, dphi_lo_val,
                                     alpha_hi, phi_hi)
                : std::optional<Scalar>{};
            Scalar alpha_j = interp ? *interp
                                    : (alpha_lo + alpha_hi) / Scalar(2);
            ++zoom_iter;

            Scalar phi_j = phi(alpha_j);
            ++result.evaluations;

            bool phi_j_finite = std::isfinite(phi_j);
            if(!phi_j_finite)
                ++result.diagnostics.nan_eval_count;

            // NaN/Inf gate: a non-finite trial is treated as an upper
            // bracket (short-circuits before the ordered comparisons).
            if(!phi_j_finite ||
               phi_j > phi0 + opts.c1 * alpha_j * dphi0 || phi_j >= phi_lo)
            {
                alpha_hi = alpha_j;
                phi_hi = phi_j;
            }
            else
            {
                Scalar dphi_j = dphi(alpha_j);
                ++result.evaluations;

                bool dphi_j_finite = std::isfinite(dphi_j);
                if(!dphi_j_finite)
                    ++result.diagnostics.nan_eval_count;

                if(dphi_j_finite && std::abs(dphi_j) <= opts.c2 * std::abs(dphi0))
                {
                    result.alpha = alpha_j;
                    result.value = phi_j;
                    result.success = true;
                    return result;
                }

                if(!dphi_j_finite || dphi_j * (alpha_hi - alpha_lo) >= Scalar(0))
                {
                    alpha_hi = alpha_lo;
                    phi_hi = phi_lo;
                }

                alpha_lo = alpha_j;
                phi_lo = phi_j;
                dphi_lo_val = dphi_j;
            }

            // Bracket too narrow -- return best found
            if(std::abs(alpha_hi - alpha_lo) <
               std::numeric_limits<Scalar>::epsilon() * std::max(Scalar(1), alpha_hi))
            {
                result.alpha = alpha_lo;
                result.value = phi_lo;
                return result;
            }
        }

        result.alpha = alpha_lo;
        result.value = phi_lo;
        return result;
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
            return finalize(zoom(alpha_prev, alpha, phi_prev, dphi_prev, phi_alpha));
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
            return finalize(zoom(alpha, alpha_prev, phi_alpha, dphi_alpha, phi_prev));
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
