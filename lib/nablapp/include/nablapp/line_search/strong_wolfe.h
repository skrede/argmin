#ifndef HPP_GUARD_NABLAPP_LINE_SEARCH_STRONG_WOLFE_H
#define HPP_GUARD_NABLAPP_LINE_SEARCH_STRONG_WOLFE_H

#include "nablapp/line_search/result.h"
#include "nablapp/line_search/options.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace nablapp
{

// Strong Wolfe line search with bracketing and zoom phases.
//
// Finds alpha satisfying the strong Wolfe conditions:
//
//   (i)  phi(alpha) <= phi(0) + c1 * alpha * phi'(0)   (sufficient decrease)
//   (ii) |phi'(alpha)| <= c2 * |phi'(0)|               (curvature)
//
// The algorithm proceeds in two phases:
//   1. Bracketing (N&W Algorithm 3.5): find an interval [lo, hi] that
//      contains a point satisfying both conditions.
//   2. Zoom (N&W Algorithm 3.6): bisect the bracket to find the point.
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

    auto budget_ok = [&]() { return result.evaluations < opts.max_iterations; };

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

            if(phi_j > phi0 + opts.c1 * alpha_j * dphi0 || phi_j >= phi_lo)
            {
                alpha_hi = alpha_j;
                phi_hi = phi_j;
            }
            else
            {
                Scalar dphi_j = dphi(alpha_j);
                ++result.evaluations;

                if(std::abs(dphi_j) <= opts.c2 * std::abs(dphi0))
                {
                    result.alpha = alpha_j;
                    result.value = phi_j;
                    result.success = true;
                    return result;
                }

                if(dphi_j * (alpha_hi - alpha_lo) >= Scalar(0))
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
    Scalar alpha = opts.max_alpha;

    for(int i = 1; budget_ok(); ++i)
    {
        Scalar phi_alpha = phi(alpha);
        ++result.evaluations;

        // Condition 1: Armijo violated or value increased from previous
        if(phi_alpha > phi0 + opts.c1 * alpha * dphi0 ||
           (phi_alpha >= phi_prev && i > 1))
        {
            return zoom(alpha_prev, alpha, phi_prev, dphi_prev, phi_alpha);
        }

        Scalar dphi_alpha = dphi(alpha);
        ++result.evaluations;

        // Condition 2: both Wolfe conditions satisfied
        if(std::abs(dphi_alpha) <= opts.c2 * std::abs(dphi0))
        {
            result.alpha = alpha;
            result.value = phi_alpha;
            result.success = true;
            return result;
        }

        // Condition 3: positive slope -- bracket found in reversed order
        if(dphi_alpha >= Scalar(0))
        {
            return zoom(alpha, alpha_prev, phi_alpha, dphi_alpha, phi_prev);
        }

        // Expand step: double alpha, capped at max_alpha
        alpha_prev = alpha;
        phi_prev = phi_alpha;
        dphi_prev = dphi_alpha;
        alpha = std::min(Scalar(2) * alpha, opts.max_alpha);

        // If alpha cannot grow further, we are stuck at max_alpha
        if(alpha == alpha_prev)
        {
            result.alpha = alpha;
            result.value = phi_alpha;
            return result;
        }
    }

    return result;
}

}

#endif
