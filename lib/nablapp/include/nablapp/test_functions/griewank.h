#ifndef HPP_GUARD_NABLAPP_TEST_FUNCTIONS_GRIEWANK_H
#define HPP_GUARD_NABLAPP_TEST_FUNCTIONS_GRIEWANK_H

// Griewank global optimization test function.
//
// Reference: Griewank, A.O., "Generalized Descent for Global Optimization",
//            Journal of Optimization Theory and Applications, Vol. 34,
//            No. 1, 1981, pp. 11-39.
//
// K&W Appendix B (global test functions).

#include "nablapp/types.h"
#include "nablapp/test_functions/problem_class.h"

#include <Eigen/Core>

#include <cmath>

namespace nablapp
{

// Variable-dimension Griewank function.
//
//   f(x) = 1 + sum_i [x_i^2 / 4000] - prod_i [cos(x_i / sqrt(i+1))]
//
// (0-indexed; sqrt(i+1) matches the standard 1-indexed definition.)
//
// Global minimum: x* = (0, ..., 0), f* = 0.
// Standard bounds: [-600, 600]^n.
// x0 = (100, ..., 100).
// Default n = 2.
//
// Reference: Griewank (1981).
template <typename Scalar = double, int N = dynamic_dimension>
struct griewank
{
    int n{N != dynamic_dimension ? N : 2};

    static constexpr int problem_dimension = N;
    static constexpr problem_class pclass = problem_class::global | problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return n; }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }

    [[nodiscard]] Eigen::Vector<Scalar, N> initial_point() const
    {
        return Eigen::Vector<Scalar, N>::Constant(n, Scalar(100));
    }

    [[nodiscard]] Eigen::Vector<Scalar, N> lower_bounds() const
    {
        return Eigen::Vector<Scalar, N>::Constant(n, Scalar(-600));
    }

    [[nodiscard]] Eigen::Vector<Scalar, N> upper_bounds() const
    {
        return Eigen::Vector<Scalar, N>::Constant(n, Scalar(600));
    }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, N>& x) const
    {
        using std::cos;
        using std::sqrt;
        Scalar sum{0};
        Scalar prod{1};
        for(int i = 0; i < n; ++i)
        {
            sum += x[i] * x[i] / Scalar(4000);
            prod *= cos(x[i] / sqrt(Scalar(i + 1)));
        }
        return Scalar(1) + sum - prod;
    }

    void gradient(const Eigen::Vector<Scalar, N>& x,
                  Eigen::Vector<Scalar, N>& g) const
    {
        using std::cos;
        using std::sin;
        using std::sqrt;
        // Product of all cos terms.
        Scalar prod{1};
        for(int i = 0; i < n; ++i)
            prod *= cos(x[i] / sqrt(Scalar(i + 1)));

        g.resize(n);
        for(int i = 0; i < n; ++i)
        {
            Scalar si = sqrt(Scalar(i + 1));
            Scalar ci = cos(x[i] / si);
            // d(prod)/dx_i = prod / cos(x_i/si) * (-sin(x_i/si)/si)
            Scalar dprod = (ci != Scalar(0))
                ? prod / ci * (-sin(x[i] / si) / si)
                : Scalar(0);
            g[i] = x[i] / Scalar(2000) - dprod;
        }
    }
};

}

#endif
