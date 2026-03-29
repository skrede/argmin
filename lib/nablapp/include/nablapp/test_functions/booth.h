#ifndef HPP_GUARD_NABLAPP_TEST_FUNCTIONS_BOOTH_H
#define HPP_GUARD_NABLAPP_TEST_FUNCTIONS_BOOTH_H

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
    [[nodiscard]] int dimension() const { return 2; }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        Scalar t1 = x[0] + Scalar(2) * x[1] - Scalar(7);
        Scalar t2 = Scalar(2) * x[0] + x[1] - Scalar(5);
        return t1 * t1 + t2 * t2;
    }

    void gradient(const Eigen::VectorX<Scalar>& x, Eigen::VectorX<Scalar>& g) const
    {
        g.resize(2);
        Scalar t1 = x[0] + Scalar(2) * x[1] - Scalar(7);
        Scalar t2 = Scalar(2) * x[0] + x[1] - Scalar(5);
        g[0] = Scalar(2) * t1 + Scalar(4) * t2;
        g[1] = Scalar(4) * t1 + Scalar(2) * t2;
    }
};

}

#endif
