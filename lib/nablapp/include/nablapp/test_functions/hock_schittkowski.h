#ifndef HPP_GUARD_NABLAPP_TEST_FUNCTIONS_HOCK_SCHITTKOWSKI_H
#define HPP_GUARD_NABLAPP_TEST_FUNCTIONS_HOCK_SCHITTKOWSKI_H

// Hock-Schittkowski constrained optimization test problems.
//
// Reference: Hock & Schittkowski, "Test Examples for Nonlinear Programming
//            Codes", Lecture Notes in Economics and Mathematical Systems,
//            Vol. 187, Springer, 1981.
//
// All problems satisfy constrained + differentiable + bound_constrained.
// Inequality convention: c_ineq >= 0 (nablapp lower-bound form).
// Constraint layout: equality first (rows 0..n_eq-1), then inequality.

#include <Eigen/Core>

#include <cmath>
#include <limits>

namespace nablapp
{

// HS071: 4D, 1 equality, 1 inequality, box [1, 5].
//
// min  x0*x3*(x0 + x1 + x2) + x2
// s.t. x0^2 + x1^2 + x2^2 + x3^2 = 40       (equality)
//      x0*x1*x2*x3 >= 25                       (inequality, nablapp form)
//      1 <= xi <= 5
//
// x0 = (1, 5, 5, 1), f* = 17.0140173
//
// Reference: H&S Problem 71.
template <typename Scalar = double>
struct hs071
{
    [[nodiscard]] int dimension() const { return 4; }
    [[nodiscard]] int num_equality() const { return 1; }
    [[nodiscard]] int num_inequality() const { return 1; }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        g.resize(4);
        g[0] = x[3] * (Scalar(2) * x[0] + x[1] + x[2]);
        g[1] = x[0] * x[3];
        g[2] = x[0] * x[3] + Scalar(1);
        g[3] = x[0] * (x[0] + x[1] + x[2]);
    }

    void constraints(const Eigen::VectorX<Scalar>& x,
                     Eigen::VectorX<Scalar>& c) const
    {
        c.resize(2);
        // eq: x0^2 + x1^2 + x2^2 + x3^2 - 40 = 0
        c[0] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]
               - Scalar(40);
        // ineq: x0*x1*x2*x3 - 25 >= 0
        c[1] = x[0] * x[1] * x[2] * x[3] - Scalar(25);
    }

    void constraint_jacobian(const Eigen::VectorX<Scalar>& x,
                             Eigen::MatrixX<Scalar>& J) const
    {
        J.resize(2, 4);
        // d(eq)/dx
        J(0, 0) = Scalar(2) * x[0];
        J(0, 1) = Scalar(2) * x[1];
        J(0, 2) = Scalar(2) * x[2];
        J(0, 3) = Scalar(2) * x[3];
        // d(ineq)/dx
        J(1, 0) = x[1] * x[2] * x[3];
        J(1, 1) = x[0] * x[2] * x[3];
        J(1, 2) = x[0] * x[1] * x[3];
        J(1, 3) = x[0] * x[1] * x[2];
    }

    [[nodiscard]] Eigen::VectorX<Scalar> lower_bounds() const
    {
        return Eigen::VectorX<Scalar>::Constant(4, Scalar(1));
    }

    [[nodiscard]] Eigen::VectorX<Scalar> upper_bounds() const
    {
        return Eigen::VectorX<Scalar>::Constant(4, Scalar(5));
    }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        Eigen::VectorX<Scalar> x0(4);
        x0 << Scalar(1), Scalar(5), Scalar(5), Scalar(1);
        return x0;
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(17.0140173); }
};

// HS076: 4D, 0 equality, 3 inequality, x >= 0.
//
// min  x0^2 + 0.5*x1^2 + x2^2 + 0.5*x3^2
//      - x0*x2 + x2*x3 - x0 - 3*x1 + x2 - x3
// s.t. c0 = 5 - (x0 + 2*x1 + x2 + x3) >= 0
//      c1 = 4 - (3*x0 + x1 + 2*x2 - x3) >= 0
//      c2 = x1 + 4*x2 - 1.5 >= 0
//      xi >= 0
//
// x0 = (0.5, 0.5, 0.5, 0.5), f* = -4.6818181818...
//
// Reference: H&S Problem 76.
template <typename Scalar = double>
struct hs076
{
    [[nodiscard]] int dimension() const { return 4; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 3; }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        return x[0] * x[0] + Scalar(0.5) * x[1] * x[1]
               + x[2] * x[2] + Scalar(0.5) * x[3] * x[3]
               - x[0] * x[2] + x[2] * x[3]
               - x[0] - Scalar(3) * x[1] + x[2] - x[3];
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        g.resize(4);
        g[0] = Scalar(2) * x[0] - x[2] - Scalar(1);
        g[1] = x[1] - Scalar(3);
        g[2] = Scalar(2) * x[2] - x[0] + x[3] + Scalar(1);
        g[3] = x[3] + x[2] - Scalar(1);
    }

    void constraints(const Eigen::VectorX<Scalar>& x,
                     Eigen::VectorX<Scalar>& c) const
    {
        c.resize(3);
        c[0] = Scalar(5) - (x[0] + Scalar(2) * x[1] + x[2] + x[3]);
        c[1] = Scalar(4) - (Scalar(3) * x[0] + x[1] + Scalar(2) * x[2] - x[3]);
        c[2] = x[1] + Scalar(4) * x[2] - Scalar(1.5);
    }

    void constraint_jacobian(const Eigen::VectorX<Scalar>& /*x*/,
                             Eigen::MatrixX<Scalar>& J) const
    {
        J.resize(3, 4);
        J << Scalar(-1), Scalar(-2), Scalar(-1), Scalar(-1),
             Scalar(-3), Scalar(-1), Scalar(-2), Scalar(1),
             Scalar(0),  Scalar(1),  Scalar(4),  Scalar(0);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> lower_bounds() const
    {
        return Eigen::VectorX<Scalar>::Zero(4);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(4, inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        return Eigen::VectorX<Scalar>::Constant(4, Scalar(0.5));
    }

    [[nodiscard]] Scalar optimal_value() const
    {
        return Scalar(-4.6818181818181818);
    }
};

// HS024: 2D, 0 equality, 3 inequality, x >= 0.
//
// min  1/(27*sqrt(3)) * ((x0 - 3)^2 - 9) * x1^3
// s.t. c0 = x0/sqrt(3) - x1 >= 0
//      c1 = x0 + sqrt(3)*x1 >= 0
//      c2 = 6 - x0 - sqrt(3)*x1 >= 0
//      xi >= 0
//
// x0 = (1, 0.5), f* = -1.0
//
// Reference: H&S Problem 24.
template <typename Scalar = double>
struct hs024
{
    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 3; }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        using std::sqrt;
        Scalar k = Scalar(1) / (Scalar(27) * sqrt(Scalar(3)));
        Scalar t = (x[0] - Scalar(3)) * (x[0] - Scalar(3)) - Scalar(9);
        return k * t * x[1] * x[1] * x[1];
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        using std::sqrt;
        Scalar k = Scalar(1) / (Scalar(27) * sqrt(Scalar(3)));
        Scalar t = (x[0] - Scalar(3)) * (x[0] - Scalar(3)) - Scalar(9);
        Scalar x1_cubed = x[1] * x[1] * x[1];
        g.resize(2);
        // df/dx0 = k * 2*(x0-3) * x1^3
        g[0] = k * Scalar(2) * (x[0] - Scalar(3)) * x1_cubed;
        // df/dx1 = k * t * 3*x1^2
        g[1] = k * t * Scalar(3) * x[1] * x[1];
    }

    void constraints(const Eigen::VectorX<Scalar>& x,
                     Eigen::VectorX<Scalar>& c) const
    {
        using std::sqrt;
        Scalar s3 = sqrt(Scalar(3));
        c.resize(3);
        c[0] = x[0] / s3 - x[1];
        c[1] = x[0] + s3 * x[1];
        c[2] = Scalar(6) - x[0] - s3 * x[1];
    }

    void constraint_jacobian(const Eigen::VectorX<Scalar>& /*x*/,
                             Eigen::MatrixX<Scalar>& J) const
    {
        using std::sqrt;
        Scalar s3 = sqrt(Scalar(3));
        J.resize(3, 2);
        J(0, 0) = Scalar(1) / s3;  J(0, 1) = Scalar(-1);
        J(1, 0) = Scalar(1);       J(1, 1) = s3;
        J(2, 0) = Scalar(-1);      J(2, 1) = -s3;
    }

    [[nodiscard]] Eigen::VectorX<Scalar> lower_bounds() const
    {
        return Eigen::VectorX<Scalar>::Zero(2);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(2, inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        Eigen::VectorX<Scalar> x0(2);
        x0 << Scalar(1), Scalar(0.5);
        return x0;
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(-1); }
};

// HS035: 3D, 0 equality, 1 inequality, x >= 0.
//
// min  9 - 8*x0 - 6*x1 - 4*x2
//      + 2*x0^2 + 2*x1^2 + x2^2 + 2*x0*x1 + 2*x0*x2
// s.t. c0 = 3 - (x0 + x1 + 2*x2) >= 0
//      xi >= 0
//
// x0 = (0.5, 0.5, 0.5), f* = 1/9 = 0.1111111...
//
// Reference: H&S Problem 35.
template <typename Scalar = double>
struct hs035
{
    [[nodiscard]] int dimension() const { return 3; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 1; }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        return Scalar(9) - Scalar(8) * x[0] - Scalar(6) * x[1]
               - Scalar(4) * x[2] + Scalar(2) * x[0] * x[0]
               + Scalar(2) * x[1] * x[1] + x[2] * x[2]
               + Scalar(2) * x[0] * x[1] + Scalar(2) * x[0] * x[2];
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        g.resize(3);
        g[0] = Scalar(-8) + Scalar(4) * x[0] + Scalar(2) * x[1]
               + Scalar(2) * x[2];
        g[1] = Scalar(-6) + Scalar(2) * x[0] + Scalar(4) * x[1];
        g[2] = Scalar(-4) + Scalar(2) * x[0] + Scalar(2) * x[2];
    }

    void constraints(const Eigen::VectorX<Scalar>& x,
                     Eigen::VectorX<Scalar>& c) const
    {
        c.resize(1);
        c[0] = Scalar(3) - (x[0] + x[1] + Scalar(2) * x[2]);
    }

    void constraint_jacobian(const Eigen::VectorX<Scalar>& /*x*/,
                             Eigen::MatrixX<Scalar>& J) const
    {
        J.resize(1, 3);
        J(0, 0) = Scalar(-1);
        J(0, 1) = Scalar(-1);
        J(0, 2) = Scalar(-2);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> lower_bounds() const
    {
        return Eigen::VectorX<Scalar>::Zero(3);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(3, inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        return Eigen::VectorX<Scalar>::Constant(3, Scalar(0.5));
    }

    [[nodiscard]] Scalar optimal_value() const
    {
        return Scalar(1) / Scalar(9);
    }
};

}

#endif
