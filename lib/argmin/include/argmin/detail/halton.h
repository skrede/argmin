#ifndef HPP_GUARD_ARGMIN_DETAIL_HALTON_H
#define HPP_GUARD_ARGMIN_DETAIL_HALTON_H

// Halton quasi-random sequence for multi-start seeding.
//
// Generates low-discrepancy points in [0,1]^n using the van der Corput
// base-b construction with distinct prime bases per dimension.
//
// Reference: Halton, J. H. (1960), "On the efficiency of certain
//            quasi-random sequences of points in evaluating
//            multi-dimensional integrals", Numerische Mathematik, 2(1).

#include <Eigen/Core>

#include <array>
#include <cstdint>

namespace argmin::detail
{

// Van der Corput sequence value for index `i` in base `b`.
// Returns a value in (0, 1).
inline constexpr double van_der_corput(std::uint32_t i, std::uint32_t b)
{
    double result = 0.0;
    double f = 1.0 / static_cast<double>(b);
    std::uint32_t n = i;
    while(n > 0)
    {
        result += static_cast<double>(n % b) * f;
        n /= b;
        f /= static_cast<double>(b);
    }
    return result;
}

// First 10 prime bases for Halton sequences (supports up to 10 dimensions).
inline constexpr std::array<std::uint32_t, 10> halton_primes = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29
};

// Generate the i-th Halton point in [0, 1]^n.
// Each dimension uses a different prime base from halton_primes.
// Supports up to 10 dimensions; for n > 10, wraps back to base 2.
//
// Reference: Halton (1960).
inline Eigen::VectorXd halton_point(std::uint32_t index, int n)
{
    Eigen::VectorXd point(n);
    for(int d = 0; d < n; ++d)
    {
        std::uint32_t base = (d < 10)
            ? halton_primes[static_cast<std::size_t>(d)]
            : halton_primes[static_cast<std::size_t>(d % 10)];
        point[d] = van_der_corput(index + 1, base);  // +1: skip index 0 (returns 0.0)
    }
    return point;
}

// Map a Halton point from [0,1]^n to [lower, upper] bounds.
inline Eigen::VectorXd halton_to_bounds(
    std::uint32_t index, int n,
    const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper)
{
    auto h = halton_point(index, n);
    return (lower.array() + h.array() * (upper.array() - lower.array())).matrix();
}

}

#endif
