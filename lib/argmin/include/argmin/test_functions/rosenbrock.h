#ifndef HPP_GUARD_ARGMIN_TEST_FUNCTIONS_ROSENBROCK_H
#define HPP_GUARD_ARGMIN_TEST_FUNCTIONS_ROSENBROCK_H

#include "argmin/types.h"
#include "argmin/test_functions/problem_class.h"

#include <Eigen/Core>

#include <cmath>

namespace argmin
{

// n-dimensional Rosenbrock function.
//
// K&W Appendix B.6, Eq. B.7/B.8.
//
//   f(x) = sum_{i=0}^{n-2} [ (a - x_i)^2 + b * (x_{i+1} - x_i^2)^2 ]
//
// Default parameters: a = 1, b = 5 (K&W convention).
// Global minimum: x* = (a, a, ..., a), f(x*) = 0.

template <typename Scalar = double, int N = dynamic_dimension>
struct rosenbrock
{
    Scalar a{1};
    Scalar b{5};
    int n{N != dynamic_dimension ? N : 2};

    static constexpr int problem_dimension = N;
    static constexpr problem_class pclass = problem_class::unconstrained;

    [[nodiscard]] int dimension() const { return n; }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }

    [[nodiscard]] Eigen::Vector<Scalar, N> initial_point() const
    {
        // Standard: alternating -1.2 and 1.0 (More, Garbow, Hillstrom 1981).
        Eigen::Vector<Scalar, N> x0(n);
        for(int i = 0; i < n; ++i)
            x0[i] = (i % 2 == 0) ? Scalar(-1.2) : Scalar(1);
        return x0;
    }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, N>& x) const
    {
        Scalar f{0};
        for(int i = 0; i < n - 1; ++i)
        {
            Scalar t1 = a - x[i];
            Scalar t2 = x[i + 1] - x[i] * x[i];
            f += t1 * t1 + b * t2 * t2;
        }
        return f;
    }

    void gradient(const Eigen::Vector<Scalar, N>& x, Eigen::Vector<Scalar, N>& g) const
    {
        g.resize(n);
        g.setZero();
        for(int i = 0; i < n - 1; ++i)
        {
            Scalar t1 = a - x[i];
            Scalar t2 = x[i + 1] - x[i] * x[i];
            g[i] += Scalar(-2) * t1 + Scalar(-4) * b * x[i] * t2;
            g[i + 1] += Scalar(2) * b * t2;
        }
    }
};

}

#endif
