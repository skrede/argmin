#ifndef HPP_GUARD_NABLAPP_TEST_FUNCTIONS_SCHWEFEL_H
#define HPP_GUARD_NABLAPP_TEST_FUNCTIONS_SCHWEFEL_H

// Schwefel global optimization test function.
//
// Reference: Schwefel, H.-P., "Numerical Optimization of Computer Models",
//            Wiley, 1981.
//
// K&W Appendix B (global test functions).

#include "nablapp/types.h"
#include "nablapp/test_functions/problem_class.h"

#include <Eigen/Core>

#include <cmath>

namespace nablapp
{

// Variable-dimension Schwefel function.
//
//   f(x) = 418.9829*n - sum_i [x_i * sin(sqrt(|x_i|))]
//
// Global minimum: x* = (420.9687, ..., 420.9687), f* approx 0.
// Standard bounds: [-500, 500]^n.
// x0 = (-200, ..., -200).
// Default n = 2.
//
// Reference: Schwefel (1981).
template <typename Scalar = double, int N = dynamic_dimension>
struct schwefel
{
    int n{N != dynamic_dimension ? N : 2};

    static constexpr int problem_dimension = N;
    static constexpr problem_class pclass = problem_class::global | problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return n; }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }

    [[nodiscard]] Eigen::Vector<Scalar, N> initial_point() const
    {
        return Eigen::Vector<Scalar, N>::Constant(n, Scalar(-200));
    }

    [[nodiscard]] Eigen::Vector<Scalar, N> lower_bounds() const
    {
        return Eigen::Vector<Scalar, N>::Constant(n, Scalar(-500));
    }

    [[nodiscard]] Eigen::Vector<Scalar, N> upper_bounds() const
    {
        return Eigen::Vector<Scalar, N>::Constant(n, Scalar(500));
    }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, N>& x) const
    {
        using std::abs;
        using std::sin;
        using std::sqrt;
        Scalar s{0};
        for(int i = 0; i < n; ++i)
            s += x[i] * sin(sqrt(abs(x[i])));
        return Scalar(418.9829) * Scalar(n) - s;
    }

    void gradient(const Eigen::Vector<Scalar, N>& x,
                  Eigen::Vector<Scalar, N>& g) const
    {
        using std::abs;
        using std::cos;
        using std::sin;
        using std::sqrt;
        g.resize(n);
        for(int i = 0; i < n; ++i)
        {
            Scalar ax = abs(x[i]);
            Scalar sqax = sqrt(ax);
            Scalar sign = (x[i] >= Scalar(0)) ? Scalar(1) : Scalar(-1);
            // df/dx_i = -(sin(sqrt(|x|)) + x*cos(sqrt(|x|)) * sign/(2*sqrt(|x|)))
            //         = -(sin(sqax) + x*cos(sqax)/(2*sqax) * sign)  [when x != 0]
            if(ax < Scalar(1e-30))
            {
                g[i] = Scalar(0);
            }
            else
            {
                g[i] = -(sin(sqax)
                         + x[i] * cos(sqax) * sign / (Scalar(2) * sqax));
            }
        }
    }
};

}

#endif
