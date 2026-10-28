#ifndef HPP_GUARD_ARGMIN_TESTS_FUZZ_QP_OPTIONS_H
#define HPP_GUARD_ARGMIN_TESTS_FUZZ_QP_OPTIONS_H

#include "harness.h"
#include "qp_input.h"

#include <array>
#include <cmath>
#include <limits>
#include <cstdint>

namespace argmin::fuzzing
{

// One selector in sixteen leaves the tuning domain for a negative, zero, NaN or infinite value; the rest scale the
// default by a power of two, which can also cross a domain edge (alpha, adaptive_rho_tolerance).
inline double option_value(field_reader& fields, double nominal)
{
    const std::uint8_t selector = fields.byte();
    if(selector % 16 != 15)
        return std::ldexp(nominal, selector % 16 % 9 - 4);
    const std::array<double, 4> outside{-nominal, 0.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()};
    return outside[selector / 16 % outside.size()];
}

inline std::uint32_t iteration_budget(std::uint8_t selector)
{
    constexpr std::array<std::uint32_t, 8> budgets{0, 1, 7, 50, 400, 4000, 4000, std::numeric_limits<std::uint32_t>::max()};
    return budgets[selector % budgets.size()];
}

inline bool positive(double value)
{
    return std::isfinite(value) && value > 0.0;
}

inline bool budget_valid(std::uint32_t budget)
{
    return budget > 0 && budget <= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max());
}

// The documented option domains, stated independently of the solvers' own validation.
template <typename Options>
bool admm_options_valid(const Options& o)
{
    const bool penalties = positive(o.rho) && positive(o.sigma) && positive(o.delta);
    const bool relaxation = std::isfinite(o.alpha) && o.alpha > 0.0 && o.alpha < 2.0;
    const bool adaptive = std::isfinite(o.adaptive_rho_tolerance) && o.adaptive_rho_tolerance >= 1.0;
    const bool absolute = std::isfinite(o.eps_abs) && o.eps_abs >= 0.0;
    const bool relative = std::isfinite(o.eps_rel) && o.eps_rel >= 0.0;
    const bool convergence = absolute && relative && (o.eps_abs > 0.0 || o.eps_rel > 0.0);
    const bool certificates = positive(o.eps_prim_inf) && positive(o.eps_dual_inf);
    return penalties && relaxation && adaptive && convergence && certificates && budget_valid(o.max_iterations);
}

template <typename Options>
void decode_admm_tolerances(field_reader& fields, Options& o)
{
    o.rho = option_value(fields, o.rho);
    o.sigma = option_value(fields, o.sigma);
    o.alpha = option_value(fields, o.alpha);
    o.eps_abs = option_value(fields, o.eps_abs);
    o.eps_rel = option_value(fields, o.eps_rel);
    o.eps_prim_inf = option_value(fields, o.eps_prim_inf);
    o.eps_dual_inf = option_value(fields, o.eps_dual_inf);
    o.adaptive_rho_tolerance = option_value(fields, o.adaptive_rho_tolerance);
    o.delta = option_value(fields, o.delta);
}

// Convexity is the caller's obligation once validation is off, so validation stays on for a raw symmetric P.
template <typename Options>
Options admm_options(field_reader& fields, hessian_shape shape)
{
    Options o;
    if((fields.byte() & 1) == 0)
        return o;
    decode_admm_tolerances(fields, o);
    o.max_iterations = iteration_budget(fields.byte());
    const std::uint8_t flags = fields.byte();
    o.adaptive_rho = (flags & 1) == 0;
    o.polish = (flags & 2) == 0;
    o.warm_start = (flags & 4) == 0;
    o.validate_convexity = (flags & 8) == 0 || shape == hessian_shape::symmetric;
    o.check_termination = fields.byte();
    o.adaptive_rho_interval = fields.byte();
    o.scaling = static_cast<std::uint16_t>(fields.byte() % 16);
    o.polish_refine_iter = static_cast<std::uint16_t>(fields.byte() % 8);
    return o;
}

template <typename Options>
bool active_set_options_valid(const Options& o)
{
    return budget_valid(o.max_iterations) && positive(o.tolerance) && positive(o.equality_tolerance);
}

template <typename Options>
Options active_set_options(field_reader& fields)
{
    Options o;
    if((fields.byte() & 1) == 0)
        return o;
    o.max_iterations = iteration_budget(fields.byte());
    o.tolerance = option_value(fields, o.tolerance);
    o.equality_tolerance = option_value(fields, o.equality_tolerance);
    return o;
}

}

#endif
