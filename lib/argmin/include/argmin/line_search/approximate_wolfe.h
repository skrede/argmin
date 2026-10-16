#ifndef HPP_GUARD_ARGMIN_LINE_SEARCH_APPROXIMATE_WOLFE_H
#define HPP_GUARD_ARGMIN_LINE_SEARCH_APPROXIMATE_WOLFE_H

#include "argmin/line_search/result.h"
#include "argmin/line_search/options.h"

#include <cmath>

namespace argmin
{

namespace detail
{

// Approximate Wolfe acceptance: Hager and Zhang (2005), SIAM J. Optim. 16(1):170-192.
template <typename DPhi, typename Scalar>
bool approximately_wolfe(DPhi&& dphi, Scalar alpha, Scalar phi_alpha, Scalar phi0, Scalar dphi0,
                         const line_search_options<Scalar>& opts, line_search_result<Scalar>& result)
{
    const Scalar noise_band = opts.relative_value_noise * std::abs(phi0);
    if(!(noise_band > Scalar(0)) || !std::isfinite(phi_alpha) || phi_alpha > phi0 + noise_band)
        return false;
    Scalar dphi_alpha = dphi(alpha);
    ++result.evaluations;
    if(!std::isfinite(dphi_alpha) || std::abs(dphi_alpha) > opts.c2 * std::abs(dphi0))
        return false;
    result.alpha = alpha;
    result.value = phi_alpha;
    result.success = true;
    return true;
}

}

}

#endif
