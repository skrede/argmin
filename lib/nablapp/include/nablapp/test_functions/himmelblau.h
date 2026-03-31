#ifndef HPP_GUARD_NABLAPP_TEST_FUNCTIONS_HIMMELBLAU_H
#define HPP_GUARD_NABLAPP_TEST_FUNCTIONS_HIMMELBLAU_H

#include "nablapp/test_functions/problem_class.h"

#include <Eigen/Core>

namespace nablapp
{

// Himmelblau function.
//
// Himmelblau 1972, "Applied Nonlinear Programming."
//
//   f(x) = (x1^2 + x2 - 11)^2 + (x1 + x2^2 - 7)^2
//
// Dimension: 2 (fixed).
// Four global minima, all with f* = 0:
//   (3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)

template <typename Scalar = double>
struct himmelblau
{
    static constexpr problem_class pclass = problem_class::unconstrained;

    [[nodiscard]] int dimension() const { return 2; }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        // Start near (0, 0), which is not a minimum.
        Eigen::VectorX<Scalar> x0(2);
        x0 << Scalar(1), Scalar(1);
        return x0;
    }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        Scalar t1 = x[0] * x[0] + x[1] - Scalar(11);
        Scalar t2 = x[0] + x[1] * x[1] - Scalar(7);
        return t1 * t1 + t2 * t2;
    }

    void gradient(const Eigen::VectorX<Scalar>& x, Eigen::VectorX<Scalar>& g) const
    {
        g.resize(2);
        Scalar t1 = x[0] * x[0] + x[1] - Scalar(11);
        Scalar t2 = x[0] + x[1] * x[1] - Scalar(7);
        g[0] = Scalar(4) * x[0] * t1 + Scalar(2) * t2;
        g[1] = Scalar(2) * t1 + Scalar(4) * x[1] * t2;
    }
};

}

#endif
