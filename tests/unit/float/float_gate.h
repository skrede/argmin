#ifndef HPP_GUARD_ARGMIN_TESTS_UNIT_FLOAT_FLOAT_GATE_H
#define HPP_GUARD_ARGMIN_TESTS_UNIT_FLOAT_FLOAT_GATE_H

#include "argmin/types.h"

#include <cmath>
#include <limits>
#include <algorithm>

// A float solve passes when it is finite and lands within k * sqrt(eps_float), relative, of the
// double solve of the same problem; the double run is the oracle. The comparison runs in double.
namespace argmin::float_gate
{

inline constexpr double k_qp = 1.0;
inline constexpr double k_harness = 10.0;
inline constexpr double k_lbfgsb = 10.0;
inline constexpr double k_line_search_sqp = 10.0;
inline constexpr double k_trust_region_sqp = 10.0;
inline constexpr double k_least_squares = 10.0;
// Moving-asymptote solves stop on subproblem-level tolerances, so their float gap is wider.
inline constexpr double k_mma = 30.0;
// Derivative-free solves have a precision floor near sqrt(eps_float) * scale.
inline constexpr double k_derivative_free = 30.0;
// The float and double runs of a stochastic search share a seed but not a trajectory, since their
// samples round differently from the first generation on.
inline constexpr double k_stochastic = 100.0;
// A wrapper over a float inner policy inherits the inner policy's precision floor.
inline constexpr double k_wrapper = 10.0;

inline double tolerance(double k)
{
    return k * std::sqrt(static_cast<double>(std::numeric_limits<float>::epsilon()));
}

template <int N>
double relative_gap(const vector<float, N>& xf, const vector<double, N>& xd)
{
    return (xf.template cast<double>() - xd).norm() / std::max(1.0, xd.norm());
}

template <int N>
bool tracks(const vector<float, N>& xf, const vector<double, N>& xd, double k)
{
    return xf.allFinite() && relative_gap(xf, xd) <= tolerance(k);
}

inline bool objective_tracks(float ff, double fd, double k)
{
    const double gap = std::abs(static_cast<double>(ff) - fd) / std::max(1.0, std::abs(fd));
    return std::isfinite(ff) && gap <= tolerance(k);
}

}

#endif
