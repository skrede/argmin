#ifndef HPP_GUARD_NABLAPP_LINE_SEARCH_STRONG_WOLFE_H
#define HPP_GUARD_NABLAPP_LINE_SEARCH_STRONG_WOLFE_H

#include "nablapp/line_search/result.h"
#include "nablapp/line_search/options.h"

#include <algorithm>
#include <cmath>

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
                                                      const line_search_options<Scalar>& opts = {})
{
    line_search_result<Scalar> result;

    auto budget_ok = [&]() { return result.evaluations < opts.max_iterations; };

    // --- zoom phase (N&W Algorithm 3.6) ---
    // Bisection-based refinement within a bracket [alpha_lo, alpha_hi].
    // phi_lo is the cached value of phi(alpha_lo) to avoid redundant evals.
    auto zoom = [&](Scalar alpha_lo, Scalar alpha_hi,
                    Scalar phi_lo) -> line_search_result<Scalar>
    {
        while(budget_ok())
        {
            Scalar alpha_j = (alpha_lo + alpha_hi) / Scalar(2);

            Scalar phi_j = phi(alpha_j);
            ++result.evaluations;

            if(phi_j > phi0 + opts.c1 * alpha_j * dphi0 || phi_j >= phi_lo)
            {
                alpha_hi = alpha_j;
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
                }

                alpha_lo = alpha_j;
                phi_lo = phi_j;
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
    Scalar alpha = opts.max_alpha;

    for(int i = 1; budget_ok(); ++i)
    {
        Scalar phi_alpha = phi(alpha);
        ++result.evaluations;

        // Condition 1: Armijo violated or value increased from previous
        if(phi_alpha > phi0 + opts.c1 * alpha * dphi0 ||
           (phi_alpha >= phi_prev && i > 1))
        {
            return zoom(alpha_prev, alpha, phi_prev);
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
            return zoom(alpha, alpha_prev, phi_alpha);
        }

        // Expand step: double alpha, capped at max_alpha
        alpha_prev = alpha;
        phi_prev = phi_alpha;
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
