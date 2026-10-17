#ifndef HPP_GUARD_ARGMIN_DETAIL_SCALAR_DISTRIBUTIONS_H
#define HPP_GUARD_ARGMIN_DETAIL_SCALAR_DISTRIBUTIONS_H

// Uniform and Gaussian draws in a solver's scalar. The double draws keep the standard-library
// distributions that the double seeded trajectories were recorded with; the float draws are built
// from the raw engine bits, so a float stream does not depend on the standard library.

#include "argmin/detail/alternative/cmaes_sampling_marsaglia.h"

#include "argmin/types.h"

#include <random>

namespace argmin::detail
{

template <scalar Scalar>
class unit_uniform_distribution;

template <>
class unit_uniform_distribution<double>
{
public:
    unit_uniform_distribution()
        : distribution_(0.0, 1.0)
    {
    }

    template <typename RNG>
    double operator()(RNG& rng)
    {
        return distribution_(rng);
    }

private:
    std::uniform_real_distribution<double> distribution_;
};

template <>
class unit_uniform_distribution<float>
{
public:
    template <typename RNG>
    float operator()(RNG& rng)
    {
        return 0.5f * (alternative::marsaglia::canonical_uniform_pm1<float>(rng) + 1.0f);
    }
};

template <scalar Scalar>
class standard_normal_distribution;

template <>
class standard_normal_distribution<double>
{
public:
    standard_normal_distribution()
        : distribution_(0.0, 1.0)
    {
    }

    template <typename RNG>
    double operator()(RNG& rng)
    {
        return distribution_(rng);
    }

private:
    std::normal_distribution<double> distribution_;
};

template <>
class standard_normal_distribution<float>
{
public:
    template <typename RNG>
    float operator()(RNG& rng)
    {
        return alternative::marsaglia::marsaglia_normal<float>(rng);
    }
};

}

#endif
