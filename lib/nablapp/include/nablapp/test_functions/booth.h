#ifndef HPP_GUARD_NABLAPP_TEST_FUNCTIONS_BOOTH_H
#define HPP_GUARD_NABLAPP_TEST_FUNCTIONS_BOOTH_H

#include "nablapp/test_functions/problem_class.h"

#include <Eigen/Core>

namespace nablapp
{

// Booth function.
//
// K&W Appendix B.2, Eq. B.2.
//
//   f(x) = (x1 + 2*x2 - 7)^2 + (2*x1 + x2 - 5)^2
//
// Dimension: 2 (fixed).
// Global minimum: x* = (1, 3), f(x*) = 0.

template <typename Scalar = double>
struct booth
{
    static constexpr int problem_dimension = 2;
    static constexpr problem_class pclass = problem_class::unconstrained;

    [[nodiscard]] int dimension() const { return 2; }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }

    [[nodiscard]] Eigen::Vector<Scalar, 2> initial_point() const
    {
        Eigen::Vector<Scalar, 2> x0;
        x0 << Scalar(0.5), Scalar(0.5);
        return x0;
    }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, 2>& x) const
    {
        Scalar t1 = x[0] + Scalar(2) * x[1] - Scalar(7);
        Scalar t2 = Scalar(2) * x[0] + x[1] - Scalar(5);
        return t1 * t1 + t2 * t2;
    }

    void gradient(const Eigen::Vector<Scalar, 2>& x, Eigen::Vector<Scalar, 2>& g) const
    {
        Scalar t1 = x[0] + Scalar(2) * x[1] - Scalar(7);
        Scalar t2 = Scalar(2) * x[0] + x[1] - Scalar(5);
        g[0] = Scalar(2) * t1 + Scalar(4) * t2;
        g[1] = Scalar(4) * t1 + Scalar(2) * t2;
    }
};

}

#endif
