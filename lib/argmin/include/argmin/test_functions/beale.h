#ifndef HPP_GUARD_ARGMIN_TEST_FUNCTIONS_BEALE_H
#define HPP_GUARD_ARGMIN_TEST_FUNCTIONS_BEALE_H

#include "argmin/test_functions/problem_class.h"

#include <Eigen/Core>

namespace argmin
{

// Beale function.
//
// Beale 1958, "On an iterative method for finding a local minimum of a
// function of more than one variable." Also K&W Appendix B.
//
//   f(x) = (1.5 - x1 + x1*x2)^2
//        + (2.25 - x1 + x1*x2^2)^2
//        + (2.625 - x1 + x1*x2^3)^2
//
// Dimension: 2 (fixed).
// Global minimum: x* = (3, 0.5), f(x*) = 0.

template <typename Scalar = double>
struct beale
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
        Scalar x1 = x[0];
        Scalar x2 = x[1];
        Scalar t1 = Scalar(1.5) - x1 + x1 * x2;
        Scalar t2 = Scalar(2.25) - x1 + x1 * x2 * x2;
        Scalar t3 = Scalar(2.625) - x1 + x1 * x2 * x2 * x2;
        return t1 * t1 + t2 * t2 + t3 * t3;
    }

    void gradient(const Eigen::Vector<Scalar, 2>& x, Eigen::Vector<Scalar, 2>& g) const
    {
        Scalar x1 = x[0];
        Scalar x2 = x[1];
        Scalar x2_2 = x2 * x2;
        Scalar x2_3 = x2_2 * x2;
        Scalar t1 = Scalar(1.5) - x1 + x1 * x2;
        Scalar t2 = Scalar(2.25) - x1 + x1 * x2_2;
        Scalar t3 = Scalar(2.625) - x1 + x1 * x2_3;
        // df/dx1: chain rule with d/dx1 of each term
        g[0] = Scalar(2) * t1 * (x2 - Scalar(1))
             + Scalar(2) * t2 * (x2_2 - Scalar(1))
             + Scalar(2) * t3 * (x2_3 - Scalar(1));
        // df/dx2: chain rule with d/dx2 of each term
        g[1] = Scalar(2) * t1 * x1
             + Scalar(2) * t2 * Scalar(2) * x1 * x2
             + Scalar(2) * t3 * Scalar(3) * x1 * x2_2;
    }
};

}

#endif
