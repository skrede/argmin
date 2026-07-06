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

// First 64 prime bases for Halton sequences. Covers the common
// dimensionality range with distinct compile-time bases; higher
// dimensions extend the sequence at runtime via halton_base().
inline constexpr std::array<std::uint32_t, 64> halton_primes = {
      2,   3,   5,   7,  11,  13,  17,  19,  23,  29,
     31,  37,  41,  43,  47,  53,  59,  61,  67,  71,
     73,  79,  83,  89,  97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
    233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293, 307, 311
};

// Distinct prime base for coordinate `d` (0-indexed). Every dimension
// gets its OWN prime -- there is no modular reuse, so no two coordinates
// share a base (which would collapse those coordinates onto a diagonal
// and defeat multi-start coverage). For d within the compile-time table
// the base is a table lookup; beyond it, the next prime after the last
// table entry is found by trial division, keeping bases distinct and
// increasing for arbitrarily many dimensions.
//
// Reference: Halton (1960).
inline std::uint32_t halton_base(int d)
{
    if(d < static_cast<int>(halton_primes.size()))
        return halton_primes[static_cast<std::size_t>(d)];

    int count = static_cast<int>(halton_primes.size());
    std::uint32_t candidate = halton_primes.back();
    while(count <= d)
    {
        candidate += 2;  // all primes past 2 are odd; the table ends odd
        bool is_prime = true;
        for(std::uint32_t p = 3; p * p <= candidate; p += 2)
        {
            if(candidate % p == 0)
            {
                is_prime = false;
                break;
            }
        }
        if(is_prime)
            ++count;
    }
    return candidate;
}

// Generate the i-th Halton point in [0, 1]^n.
// Each dimension uses a distinct prime base (halton_base(d)); no two
// coordinates share a base for any n.
//
// Reference: Halton (1960).
inline Eigen::VectorXd halton_point(std::uint32_t index, int n)
{
    Eigen::VectorXd point(n);
    for(int d = 0; d < n; ++d)
    {
        const std::uint32_t base = halton_base(d);
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
