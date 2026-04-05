#ifndef HPP_GUARD_NABLAPP_TEST_FUNCTIONS_RASTRIGIN_H
#define HPP_GUARD_NABLAPP_TEST_FUNCTIONS_RASTRIGIN_H

#include "nablapp/types.h"
#include "nablapp/test_functions/problem_class.h"

#include <Eigen/Core>

#include <cmath>
#include <numbers>

namespace nablapp
{

// n-dimensional Rastrigin function.
//
// Rastrigin 1974, "Systems of Extremal Control."
//
//   f(x) = 10*n + sum_{i=0}^{n-1} [ x_i^2 - 10*cos(2*pi*x_i) ]
//
// Highly multimodal with global minimum at x* = 0, f(x*) = 0.

template <typename Scalar = double, int N = dynamic_dimension>
struct rastrigin
{
    int n{N != dynamic_dimension ? N : 2};

    static constexpr int problem_dimension = N;
    static constexpr problem_class pclass = problem_class::global | problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return n; }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }

    [[nodiscard]] Eigen::Vector<Scalar, N> initial_point() const
    {
        return Eigen::Vector<Scalar, N>::Constant(n, Scalar(2.5));
    }

    [[nodiscard]] Eigen::Vector<Scalar, N> lower_bounds() const
    {
        return Eigen::Vector<Scalar, N>::Constant(n, Scalar(-5.12));
    }

    [[nodiscard]] Eigen::Vector<Scalar, N> upper_bounds() const
    {
        return Eigen::Vector<Scalar, N>::Constant(n, Scalar(5.12));
    }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, N>& x) const
    {
        constexpr Scalar two_pi = Scalar(2) * std::numbers::pi_v<Scalar>;
        Scalar f = Scalar(10) * Scalar(n);
        for(int i = 0; i < n; ++i)
        {
            f += x[i] * x[i] - Scalar(10) * std::cos(two_pi * x[i]);
        }
        return f;
    }

    void gradient(const Eigen::Vector<Scalar, N>& x, Eigen::Vector<Scalar, N>& g) const
    {
        constexpr Scalar two_pi = Scalar(2) * std::numbers::pi_v<Scalar>;
        g.resize(n);
        for(int i = 0; i < n; ++i)
        {
            g[i] = Scalar(2) * x[i] + Scalar(10) * two_pi * std::sin(two_pi * x[i]);
        }
    }
};

}

#endif
