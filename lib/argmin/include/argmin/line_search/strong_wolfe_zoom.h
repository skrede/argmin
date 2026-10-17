#ifndef HPP_GUARD_ARGMIN_LINE_SEARCH_STRONG_WOLFE_ZOOM_H
#define HPP_GUARD_ARGMIN_LINE_SEARCH_STRONG_WOLFE_ZOOM_H

#include "argmin/line_search/result.h"
#include "argmin/line_search/options.h"
#include "argmin/line_search/approximate_wolfe.h"

#include <cmath>
#include <limits>
#include <cstdint>
#include <optional>
#include <algorithm>

namespace argmin
{

namespace detail
{

template <scalar S>
struct strong_wolfe_constants;

template <>
struct strong_wolfe_constants<double>
{
    static constexpr double interpolation_floor = 1e-12;
};

template <>
struct strong_wolfe_constants<float>
{
    // A precision-class value keeps the double row's fraction of the available precision:
    // t_f = eps_f ^ (ln t_d / ln eps_d), rounded to one significant figure.
    static constexpr float interpolation_floor = 5e-6f;  // rule
};

template <typename Scalar>
struct wolfe_bracket
{
    Scalar alpha_lo;
    Scalar alpha_hi;
    Scalar phi_lo;
    Scalar dphi_lo;
    Scalar phi_hi;
};

// Minimizer of the quadratic interpolant through (a_lo, phi_lo, dphi_lo) and (a_hi, phi_hi),
// N&W p. 59; nullopt when degenerate or too close to either end of the bracket.
template <typename Scalar>
std::optional<Scalar> quadratic_minimize(Scalar a_lo, Scalar phi_lo, Scalar dphi_lo, Scalar a_hi, Scalar phi_hi)
{
    Scalar d = a_hi - a_lo;
    if(std::abs(d) < std::numeric_limits<Scalar>::epsilon())
        return std::nullopt;

    Scalar denom = Scalar(2) * (phi_hi - phi_lo - dphi_lo * d);
    if(std::abs(denom) < strong_wolfe_constants<Scalar>::interpolation_floor * std::max(Scalar(1), std::abs(phi_hi - phi_lo)))
        return std::nullopt;

    Scalar alpha_min = a_lo - dphi_lo * d * d / denom;

    Scalar lo = std::min(a_lo, a_hi);
    Scalar hi = std::max(a_lo, a_hi);
    Scalar margin = Scalar(0.2) * (hi - lo);
    if(!std::isfinite(alpha_min) || alpha_min <= lo + margin || alpha_min >= hi - margin)
        return std::nullopt;

    return alpha_min;
}

template <typename DPhi, typename Scalar>
bool zoom_lower_accepts(DPhi& dphi, wolfe_bracket<Scalar>& b, Scalar alpha_j, Scalar phi_j, Scalar dphi0,
                        const line_search_options<Scalar>& opts, line_search_result<Scalar>& result)
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
        return true;
    }
    if(!dphi_j_finite || dphi_j * (b.alpha_hi - b.alpha_lo) >= Scalar(0))
    {
        b.alpha_hi = b.alpha_lo;
        b.phi_hi = b.phi_lo;
    }
    b.alpha_lo = alpha_j;
    b.phi_lo = phi_j;
    b.dphi_lo = dphi_j;
    return false;
}

// A non-finite trial is treated as an upper bracket: the isfinite gate short-circuits the
// ordered comparisons that follow it.
template <typename DPhi, typename Scalar>
bool zoom_accepts(DPhi& dphi, wolfe_bracket<Scalar>& b, Scalar alpha_j, Scalar phi_j, Scalar phi0, Scalar dphi0,
                  const line_search_options<Scalar>& opts, line_search_result<Scalar>& result)
{
    if(!std::isfinite(phi_j) || phi_j > phi0 + opts.c1 * alpha_j * dphi0 || phi_j >= b.phi_lo)
    {
        if(approximately_wolfe(dphi, alpha_j, phi_j, phi0, dphi0, opts, result))
            return true;
        b.alpha_hi = alpha_j;
        b.phi_hi = phi_j;
        return false;
    }
    return zoom_lower_accepts(dphi, b, alpha_j, phi_j, dphi0, opts, result);
}

// Zoom phase, N&W Algorithm 3.6 (pp. 59-62): bisection first, quadratic interpolation after,
// with bisection again whenever the interpolant is degenerate.
template <typename Phi, typename DPhi, typename Scalar>
line_search_result<Scalar> strong_wolfe_zoom(Phi& phi, DPhi& dphi, wolfe_bracket<Scalar> b, Scalar phi0, Scalar dphi0,
                                             const line_search_options<Scalar>& opts, line_search_result<Scalar>& result)
{
    for(int32_t zoom_iter = 0; result.evaluations < opts.max_iterations; ++zoom_iter)
    {
        auto interp = (zoom_iter > 0) ? quadratic_minimize(b.alpha_lo, b.phi_lo, b.dphi_lo, b.alpha_hi, b.phi_hi)
                                      : std::optional<Scalar>{};
        Scalar alpha_j = interp ? *interp : (b.alpha_lo + b.alpha_hi) / Scalar(2);
        Scalar phi_j = phi(alpha_j);
        ++result.evaluations;
        if(!std::isfinite(phi_j))
            ++result.diagnostics.nan_eval_count;
        if(zoom_accepts(dphi, b, alpha_j, phi_j, phi0, dphi0, opts, result))
            return result;
        if(std::abs(b.alpha_hi - b.alpha_lo) < std::numeric_limits<Scalar>::epsilon() * std::max(Scalar(1), b.alpha_hi))
            break;
    }
    result.alpha = b.alpha_lo;
    result.value = b.phi_lo;
    return result;
}

}

}

#endif
