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

#include "nablapp/types.h"
#include "nablapp/test_functions/problem_class.h"

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <numbers>

namespace nablapp
{

// HS001: 2D, bound-constrained Rosenbrock variant.
//
// min  100*(x1 - x0^2)^2 + (1 - x0)^2
//      lb = (-inf, -1.5)
//
// x0 = (-2, 1), f* = 0 at (1, 1).
//
// Reference: H&S Problem 1.
template <typename Scalar = double>
struct hs001
{
    static constexpr int problem_dimension = dynamic_dimension;
    static constexpr problem_class pclass = problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        Scalar t1 = x[1] - x[0] * x[0];
        Scalar t2 = Scalar(1) - x[0];
        return Scalar(100) * t1 * t1 + t2 * t2;
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        g.resize(2);
        Scalar t = x[1] - x[0] * x[0];
        g[0] = Scalar(-400) * x[0] * t - Scalar(2) * (Scalar(1) - x[0]);
        g[1] = Scalar(200) * t;
    }

    [[nodiscard]] Eigen::VectorX<Scalar> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        Eigen::VectorX<Scalar> lb(2);
        lb << -inf, Scalar(-1.5);
        return lb;
    }

    [[nodiscard]] Eigen::VectorX<Scalar> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(2, inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        Eigen::VectorX<Scalar> x0(2);
        x0 << Scalar(-2), Scalar(1);
        return x0;
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }
};

// HS002: 2D, bound-constrained Rosenbrock variant.
//
// min  100*(x1 - x0^2)^2 + (1 - x0)^2
//      lb = (-inf, 1.5)
//
// x0 = (-2, 1), f* = 0.0504261879 at (1.2247, 1.5).
//
// Reference: H&S Problem 2.
template <typename Scalar = double>
struct hs002
{
    static constexpr int problem_dimension = dynamic_dimension;
    static constexpr problem_class pclass = problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        Scalar t1 = x[1] - x[0] * x[0];
        Scalar t2 = Scalar(1) - x[0];
        return Scalar(100) * t1 * t1 + t2 * t2;
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        g.resize(2);
        Scalar t = x[1] - x[0] * x[0];
        g[0] = Scalar(-400) * x[0] * t - Scalar(2) * (Scalar(1) - x[0]);
        g[1] = Scalar(200) * t;
    }

    [[nodiscard]] Eigen::VectorX<Scalar> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        Eigen::VectorX<Scalar> lb(2);
        lb << -inf, Scalar(1.5);
        return lb;
    }

    [[nodiscard]] Eigen::VectorX<Scalar> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(2, inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        Eigen::VectorX<Scalar> x0(2);
        x0 << Scalar(-2), Scalar(1);
        return x0;
    }

    [[nodiscard]] Scalar optimal_value() const
    {
        return Scalar(0.0504261879);
    }
};

// HS005: 2D, bound-constrained trigonometric.
//
// min  sin(x0 + x1) + (x0 - x1)^2 - 1.5*x0 + 2.5*x1 + 1
//      -1.5 <= x0 <= 4, -3 <= x1 <= 3
//
// x0 = (0, 0), f* = -(sqrt(3)/2 + pi/3) approx -1.91322295.
//
// Reference: H&S Problem 5.
template <typename Scalar = double>
struct hs005
{
    static constexpr int problem_dimension = dynamic_dimension;
    static constexpr problem_class pclass = problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        using std::sin;
        Scalar d = x[0] - x[1];
        return sin(x[0] + x[1]) + d * d
               - Scalar(1.5) * x[0] + Scalar(2.5) * x[1] + Scalar(1);
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        using std::cos;
        Scalar c = cos(x[0] + x[1]);
        Scalar d = x[0] - x[1];
        g.resize(2);
        g[0] = c + Scalar(2) * d - Scalar(1.5);
        g[1] = c - Scalar(2) * d + Scalar(2.5);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> lower_bounds() const
    {
        Eigen::VectorX<Scalar> lb(2);
        lb << Scalar(-1.5), Scalar(-3);
        return lb;
    }

    [[nodiscard]] Eigen::VectorX<Scalar> upper_bounds() const
    {
        Eigen::VectorX<Scalar> ub(2);
        ub << Scalar(4), Scalar(3);
        return ub;
    }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        return Eigen::VectorX<Scalar>::Zero(2);
    }

    [[nodiscard]] Scalar optimal_value() const
    {
        using std::sqrt;
        return -(sqrt(Scalar(3)) / Scalar(2)
                 + std::numbers::pi_v<Scalar> / Scalar(3));
    }
};

// HS006: 2D, 1 equality constraint.
//
// min  (1 - x0)^2
// s.t. 10*(x1 - x0^2) = 0
//
// x0 = (-1.2, 1), f* = 0 at (1, 1).
//
// Reference: H&S Problem 6.
template <typename Scalar = double>
struct hs006
{
    static constexpr int problem_dimension = dynamic_dimension;
    static constexpr problem_class pclass = problem_class::equality;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 1; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        Scalar t = Scalar(1) - x[0];
        return t * t;
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        g.resize(2);
        g[0] = Scalar(-2) * (Scalar(1) - x[0]);
        g[1] = Scalar(0);
    }

    void constraints(const Eigen::VectorX<Scalar>& x,
                     Eigen::VectorX<Scalar>& c) const
    {
        c.resize(1);
        c[0] = Scalar(10) * (x[1] - x[0] * x[0]);
    }

    void constraint_jacobian(const Eigen::VectorX<Scalar>& x,
                             Eigen::MatrixX<Scalar>& J) const
    {
        J.resize(1, 2);
        J(0, 0) = Scalar(-20) * x[0];
        J(0, 1) = Scalar(10);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(2, -inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(2, inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        Eigen::VectorX<Scalar> x0(2);
        x0 << Scalar(-1.2), Scalar(1);
        return x0;
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }
};

// HS007: 2D, 1 equality constraint.
//
// min  ln(1 + x0^2) - x1
// s.t. (1 + x0^2)^2 + x1^2 - 4 = 0
//
// x0 = (2, 2), f* = -sqrt(3) approx -1.7320508.
//
// Reference: H&S Problem 7.
template <typename Scalar = double>
struct hs007
{
    static constexpr int problem_dimension = dynamic_dimension;
    static constexpr problem_class pclass = problem_class::equality;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 1; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        using std::log;
        return log(Scalar(1) + x[0] * x[0]) - x[1];
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        g.resize(2);
        g[0] = Scalar(2) * x[0] / (Scalar(1) + x[0] * x[0]);
        g[1] = Scalar(-1);
    }

    void constraints(const Eigen::VectorX<Scalar>& x,
                     Eigen::VectorX<Scalar>& c) const
    {
        c.resize(1);
        Scalar t = Scalar(1) + x[0] * x[0];
        c[0] = t * t + x[1] * x[1] - Scalar(4);
    }

    void constraint_jacobian(const Eigen::VectorX<Scalar>& x,
                             Eigen::MatrixX<Scalar>& J) const
    {
        J.resize(1, 2);
        Scalar t = Scalar(1) + x[0] * x[0];
        J(0, 0) = Scalar(4) * x[0] * t;
        J(0, 1) = Scalar(2) * x[1];
    }

    [[nodiscard]] Eigen::VectorX<Scalar> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(2, -inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(2, inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        return Eigen::VectorX<Scalar>::Constant(2, Scalar(2));
    }

    [[nodiscard]] Scalar optimal_value() const
    {
        using std::sqrt;
        return -sqrt(Scalar(3));
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
    static constexpr int problem_dimension = dynamic_dimension;
    static constexpr problem_class pclass = problem_class::inequality | problem_class::bound_constrained;

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

// HS026: 3D, 1 equality constraint.
//
// min  (x0 - x1)^2 + (x1 - x2)^4
// s.t. (1 + x1^2)*x0 + x2^4 - 3 = 0
//
// x0 = (-2.6, 2, 2), f* = 0 at (1, 1, 1).
//
// Reference: H&S Problem 26.
template <typename Scalar = double>
struct hs026
{
    static constexpr int problem_dimension = dynamic_dimension;
    static constexpr problem_class pclass = problem_class::equality;

    [[nodiscard]] int dimension() const { return 3; }
    [[nodiscard]] int num_equality() const { return 1; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        Scalar d01 = x[0] - x[1];
        Scalar d12 = x[1] - x[2];
        return d01 * d01 + d12 * d12 * d12 * d12;
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        Scalar d01 = x[0] - x[1];
        Scalar d12 = x[1] - x[2];
        Scalar d12_cubed = d12 * d12 * d12;
        g.resize(3);
        g[0] = Scalar(2) * d01;
        g[1] = Scalar(-2) * d01 + Scalar(4) * d12_cubed;
        g[2] = Scalar(-4) * d12_cubed;
    }

    void constraints(const Eigen::VectorX<Scalar>& x,
                     Eigen::VectorX<Scalar>& c) const
    {
        c.resize(1);
        Scalar x2_sq = x[2] * x[2];
        c[0] = (Scalar(1) + x[1] * x[1]) * x[0] + x2_sq * x2_sq - Scalar(3);
    }

    void constraint_jacobian(const Eigen::VectorX<Scalar>& x,
                             Eigen::MatrixX<Scalar>& J) const
    {
        J.resize(1, 3);
        J(0, 0) = Scalar(1) + x[1] * x[1];
        J(0, 1) = Scalar(2) * x[0] * x[1];
        J(0, 2) = Scalar(4) * x[2] * x[2] * x[2];
    }

    [[nodiscard]] Eigen::VectorX<Scalar> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(3, -inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(3, inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        Eigen::VectorX<Scalar> x0(3);
        x0 << Scalar(-2.6), Scalar(2), Scalar(2);
        return x0;
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }
};

// HS028: 3D, 1 equality constraint.
//
// min  (x0 + x1)^2 + (x1 + x2)^2
// s.t. x0 + 2*x1 + 3*x2 - 1 = 0
//
// x0 = (-4, 1, 1), f* = 0 at (0.5, -0.5, 0.5).
//
// Reference: H&S Problem 28.
template <typename Scalar = double>
struct hs028
{
    static constexpr int problem_dimension = dynamic_dimension;
    static constexpr problem_class pclass = problem_class::equality;

    [[nodiscard]] int dimension() const { return 3; }
    [[nodiscard]] int num_equality() const { return 1; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        Scalar s01 = x[0] + x[1];
        Scalar s12 = x[1] + x[2];
        return s01 * s01 + s12 * s12;
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        Scalar s01 = x[0] + x[1];
        Scalar s12 = x[1] + x[2];
        g.resize(3);
        g[0] = Scalar(2) * s01;
        g[1] = Scalar(2) * s01 + Scalar(2) * s12;
        g[2] = Scalar(2) * s12;
    }

    void constraints(const Eigen::VectorX<Scalar>& x,
                     Eigen::VectorX<Scalar>& c) const
    {
        c.resize(1);
        c[0] = x[0] + Scalar(2) * x[1] + Scalar(3) * x[2] - Scalar(1);
    }

    void constraint_jacobian(const Eigen::VectorX<Scalar>& /*x*/,
                             Eigen::MatrixX<Scalar>& J) const
    {
        J.resize(1, 3);
        J(0, 0) = Scalar(1);
        J(0, 1) = Scalar(2);
        J(0, 2) = Scalar(3);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(3, -inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(3, inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        Eigen::VectorX<Scalar> x0(3);
        x0 << Scalar(-4), Scalar(1), Scalar(1);
        return x0;
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }
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
    static constexpr int problem_dimension = dynamic_dimension;
    static constexpr problem_class pclass = problem_class::inequality | problem_class::bound_constrained;

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

// HS039: 4D, 2 equality constraints.
//
// min  -x0
// s.t. x1 - x0^3 - x2^2 = 0
//      x0^2 - x1 - x3^2 = 0
//
// x0 = (2, 2, 2, 2), f* = -1 at (1, 1, 0, 0).
//
// Reference: H&S Problem 39.
template <typename Scalar = double>
struct hs039
{
    static constexpr int problem_dimension = dynamic_dimension;
    static constexpr problem_class pclass = problem_class::equality;

    [[nodiscard]] int dimension() const { return 4; }
    [[nodiscard]] int num_equality() const { return 2; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        return -x[0];
    }

    void gradient(const Eigen::VectorX<Scalar>& /*x*/,
                  Eigen::VectorX<Scalar>& g) const
    {
        g.resize(4);
        g[0] = Scalar(-1);
        g[1] = Scalar(0);
        g[2] = Scalar(0);
        g[3] = Scalar(0);
    }

    void constraints(const Eigen::VectorX<Scalar>& x,
                     Eigen::VectorX<Scalar>& c) const
    {
        c.resize(2);
        c[0] = x[1] - x[0] * x[0] * x[0] - x[2] * x[2];
        c[1] = x[0] * x[0] - x[1] - x[3] * x[3];
    }

    void constraint_jacobian(const Eigen::VectorX<Scalar>& x,
                             Eigen::MatrixX<Scalar>& J) const
    {
        J.resize(2, 4);
        J(0, 0) = Scalar(-3) * x[0] * x[0];
        J(0, 1) = Scalar(1);
        J(0, 2) = Scalar(-2) * x[2];
        J(0, 3) = Scalar(0);
        J(1, 0) = Scalar(2) * x[0];
        J(1, 1) = Scalar(-1);
        J(1, 2) = Scalar(0);
        J(1, 3) = Scalar(-2) * x[3];
    }

    [[nodiscard]] Eigen::VectorX<Scalar> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(4, -inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(4, inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        return Eigen::VectorX<Scalar>::Constant(4, Scalar(2));
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(-1); }
};

// HS040: 4D, 3 equality constraints.
//
// min  -x0*x1*x2*x3
// s.t. x0^3 + x1^2 - 1 = 0
//      x0^2*x3 - x2 = 0
//      x3^2 - x1 = 0
//
// x0 = (0.8, 0.8, 0.8, 0.8), f* = -0.25.
//
// Reference: H&S Problem 40.
template <typename Scalar = double>
struct hs040
{
    static constexpr int problem_dimension = dynamic_dimension;
    static constexpr problem_class pclass = problem_class::equality;

    [[nodiscard]] int dimension() const { return 4; }
    [[nodiscard]] int num_equality() const { return 3; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        return -x[0] * x[1] * x[2] * x[3];
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        g.resize(4);
        g[0] = -x[1] * x[2] * x[3];
        g[1] = -x[0] * x[2] * x[3];
        g[2] = -x[0] * x[1] * x[3];
        g[3] = -x[0] * x[1] * x[2];
    }

    void constraints(const Eigen::VectorX<Scalar>& x,
                     Eigen::VectorX<Scalar>& c) const
    {
        c.resize(3);
        c[0] = x[0] * x[0] * x[0] + x[1] * x[1] - Scalar(1);
        c[1] = x[0] * x[0] * x[3] - x[2];
        c[2] = x[3] * x[3] - x[1];
    }

    void constraint_jacobian(const Eigen::VectorX<Scalar>& x,
                             Eigen::MatrixX<Scalar>& J) const
    {
        J.resize(3, 4);
        J(0, 0) = Scalar(3) * x[0] * x[0];
        J(0, 1) = Scalar(2) * x[1];
        J(0, 2) = Scalar(0);
        J(0, 3) = Scalar(0);
        J(1, 0) = Scalar(2) * x[0] * x[3];
        J(1, 1) = Scalar(0);
        J(1, 2) = Scalar(-1);
        J(1, 3) = x[0] * x[0];
        J(2, 0) = Scalar(0);
        J(2, 1) = Scalar(-1);
        J(2, 2) = Scalar(0);
        J(2, 3) = Scalar(2) * x[3];
    }

    [[nodiscard]] Eigen::VectorX<Scalar> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(4, -inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(4, inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        return Eigen::VectorX<Scalar>::Constant(4, Scalar(0.8));
    }

    [[nodiscard]] Scalar optimal_value() const
    {
        return Scalar(-0.25);
    }
};

// HS043: 4D, 3 inequality constraints.
//
// min  x0^2 + x1^2 + 2*x2^2 + x3^2
//      - 5*x0 - 5*x1 - 21*x2 + 7*x3
// s.t. c0 = 8 - (x0^2 + x1^2 + x2^2 + x3^2 + x0 - x1 + x2 - x3) >= 0
//      c1 = 10 - (x0^2 + 2*x1^2 + x2^2 + 2*x3^2 - x0 - x3) >= 0
//      c2 = 5 - (2*x0^2 + x1^2 + x2^2 + 2*x0 - x1 - x3) >= 0
//
// x0 = (0, 0, 0, 0), f* = -44 at (0, -1, 2, -3).
//
// Reference: H&S Problem 43.
template <typename Scalar = double>
struct hs043
{
    static constexpr int problem_dimension = dynamic_dimension;
    static constexpr problem_class pclass = problem_class::inequality;

    [[nodiscard]] int dimension() const { return 4; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 3; }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        return x[0] * x[0] + x[1] * x[1] + Scalar(2) * x[2] * x[2]
               + x[3] * x[3] - Scalar(5) * x[0] - Scalar(5) * x[1]
               - Scalar(21) * x[2] + Scalar(7) * x[3];
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        g.resize(4);
        g[0] = Scalar(2) * x[0] - Scalar(5);
        g[1] = Scalar(2) * x[1] - Scalar(5);
        g[2] = Scalar(4) * x[2] - Scalar(21);
        g[3] = Scalar(2) * x[3] + Scalar(7);
    }

    void constraints(const Eigen::VectorX<Scalar>& x,
                     Eigen::VectorX<Scalar>& c) const
    {
        c.resize(3);
        c[0] = Scalar(8) - (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
               + x[3] * x[3] + x[0] - x[1] + x[2] - x[3]);
        c[1] = Scalar(10) - (x[0] * x[0] + Scalar(2) * x[1] * x[1]
               + x[2] * x[2] + Scalar(2) * x[3] * x[3] - x[0] - x[3]);
        c[2] = Scalar(5) - (Scalar(2) * x[0] * x[0] + x[1] * x[1]
               + x[2] * x[2] + Scalar(2) * x[0] - x[1] - x[3]);
    }

    void constraint_jacobian(const Eigen::VectorX<Scalar>& x,
                             Eigen::MatrixX<Scalar>& J) const
    {
        J.resize(3, 4);
        J(0, 0) = -(Scalar(2) * x[0] + Scalar(1));
        J(0, 1) = -(Scalar(2) * x[1] - Scalar(1));
        J(0, 2) = -(Scalar(2) * x[2] + Scalar(1));
        J(0, 3) = -(Scalar(2) * x[3] - Scalar(1));
        J(1, 0) = -(Scalar(2) * x[0] - Scalar(1));
        J(1, 1) = -Scalar(4) * x[1];
        J(1, 2) = -Scalar(2) * x[2];
        J(1, 3) = -(Scalar(4) * x[3] - Scalar(1));
        J(2, 0) = -(Scalar(4) * x[0] + Scalar(2));
        J(2, 1) = -(Scalar(2) * x[1] - Scalar(1));
        J(2, 2) = -Scalar(2) * x[2];
        J(2, 3) = Scalar(1);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(4, -inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(4, inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        return Eigen::VectorX<Scalar>::Zero(4);
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(-44); }
};

// HS048: 5D, 2 equality constraints.
//
// min  (x0 - 1)^2 + (x1 - x2)^2 + (x3 - x4)^2
// s.t. x0 + x1 + x2 + x3 + x4 - 5 = 0
//      x2 - 2*(x3 + x4) + 3 = 0
//
// x0 = (3, 5, -3, 2, -2), f* = 0 at (1, 1, 1, 1, 1).
//
// Reference: H&S Problem 48.
template <typename Scalar = double>
struct hs048
{
    static constexpr int problem_dimension = dynamic_dimension;
    static constexpr problem_class pclass = problem_class::equality;

    [[nodiscard]] int dimension() const { return 5; }
    [[nodiscard]] int num_equality() const { return 2; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        Scalar t0 = x[0] - Scalar(1);
        Scalar t1 = x[1] - x[2];
        Scalar t2 = x[3] - x[4];
        return t0 * t0 + t1 * t1 + t2 * t2;
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        g.resize(5);
        g[0] = Scalar(2) * (x[0] - Scalar(1));
        g[1] = Scalar(2) * (x[1] - x[2]);
        g[2] = Scalar(-2) * (x[1] - x[2]);
        g[3] = Scalar(2) * (x[3] - x[4]);
        g[4] = Scalar(-2) * (x[3] - x[4]);
    }

    void constraints(const Eigen::VectorX<Scalar>& x,
                     Eigen::VectorX<Scalar>& c) const
    {
        c.resize(2);
        c[0] = x[0] + x[1] + x[2] + x[3] + x[4] - Scalar(5);
        c[1] = x[2] - Scalar(2) * (x[3] + x[4]) + Scalar(3);
    }

    void constraint_jacobian(const Eigen::VectorX<Scalar>& /*x*/,
                             Eigen::MatrixX<Scalar>& J) const
    {
        J.resize(2, 5);
        J(0, 0) = Scalar(1);
        J(0, 1) = Scalar(1);
        J(0, 2) = Scalar(1);
        J(0, 3) = Scalar(1);
        J(0, 4) = Scalar(1);
        J(1, 0) = Scalar(0);
        J(1, 1) = Scalar(0);
        J(1, 2) = Scalar(1);
        J(1, 3) = Scalar(-2);
        J(1, 4) = Scalar(-2);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(5, -inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(5, inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        Eigen::VectorX<Scalar> x0(5);
        x0 << Scalar(3), Scalar(5), Scalar(-3), Scalar(2), Scalar(-2);
        return x0;
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }
};

// HS050: 5D, 3 equality constraints.
//
// min  (x0 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^4 + (x3 - x4)^2
// s.t. x0 + 2*x1 + 3*x2 - 6 = 0
//      x1 + 2*x2 + 3*x3 - 6 = 0
//      x2 + 2*x3 + 3*x4 - 6 = 0
//
// x0 = (35, -31, 11, 5, -5), f* = 0 at (1, 1, 1, 1, 1).
//
// Reference: H&S Problem 50.
template <typename Scalar = double>
struct hs050
{
    static constexpr int problem_dimension = dynamic_dimension;
    static constexpr problem_class pclass = problem_class::equality;

    [[nodiscard]] int dimension() const { return 5; }
    [[nodiscard]] int num_equality() const { return 3; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        Scalar d01 = x[0] - x[1];
        Scalar d12 = x[1] - x[2];
        Scalar d23 = x[2] - x[3];
        Scalar d34 = x[3] - x[4];
        return d01 * d01 + d12 * d12 + d23 * d23 * d23 * d23 + d34 * d34;
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        Scalar d01 = x[0] - x[1];
        Scalar d12 = x[1] - x[2];
        Scalar d23 = x[2] - x[3];
        Scalar d23_cubed = d23 * d23 * d23;
        Scalar d34 = x[3] - x[4];
        g.resize(5);
        g[0] = Scalar(2) * d01;
        g[1] = Scalar(-2) * d01 + Scalar(2) * d12;
        g[2] = Scalar(-2) * d12 + Scalar(4) * d23_cubed;
        g[3] = Scalar(-4) * d23_cubed + Scalar(2) * d34;
        g[4] = Scalar(-2) * d34;
    }

    void constraints(const Eigen::VectorX<Scalar>& x,
                     Eigen::VectorX<Scalar>& c) const
    {
        c.resize(3);
        c[0] = x[0] + Scalar(2) * x[1] + Scalar(3) * x[2] - Scalar(6);
        c[1] = x[1] + Scalar(2) * x[2] + Scalar(3) * x[3] - Scalar(6);
        c[2] = x[2] + Scalar(2) * x[3] + Scalar(3) * x[4] - Scalar(6);
    }

    void constraint_jacobian(const Eigen::VectorX<Scalar>& /*x*/,
                             Eigen::MatrixX<Scalar>& J) const
    {
        J.resize(3, 5);
        J.setZero();
        J(0, 0) = Scalar(1);
        J(0, 1) = Scalar(2);
        J(0, 2) = Scalar(3);
        J(1, 1) = Scalar(1);
        J(1, 2) = Scalar(2);
        J(1, 3) = Scalar(3);
        J(2, 2) = Scalar(1);
        J(2, 3) = Scalar(2);
        J(2, 4) = Scalar(3);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(5, -inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(5, inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        Eigen::VectorX<Scalar> x0(5);
        x0 << Scalar(35), Scalar(-31), Scalar(11), Scalar(5), Scalar(-5);
        return x0;
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }
};

// HS051: 5D, 3 equality constraints.
//
// min  (x0 - x1)^2 + (x1 + x2 - 2)^2 + (x3 - 1)^2 + (x4 - 1)^2
// s.t. x0 + 3*x1 - 4 = 0
//      x2 + x3 - 2*x4 = 0
//      x1 - x4 = 0
//
// x0 = (2.5, 0.5, 2, -1, 0.5), f* = 0 at (1, 1, 1, 1, 1).
//
// Reference: H&S Problem 51.
template <typename Scalar = double>
struct hs051
{
    static constexpr int problem_dimension = dynamic_dimension;
    static constexpr problem_class pclass = problem_class::equality;

    [[nodiscard]] int dimension() const { return 5; }
    [[nodiscard]] int num_equality() const { return 3; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        Scalar t0 = x[0] - x[1];
        Scalar t1 = x[1] + x[2] - Scalar(2);
        Scalar t2 = x[3] - Scalar(1);
        Scalar t3 = x[4] - Scalar(1);
        return t0 * t0 + t1 * t1 + t2 * t2 + t3 * t3;
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        g.resize(5);
        g[0] = Scalar(2) * (x[0] - x[1]);
        g[1] = Scalar(-2) * (x[0] - x[1])
               + Scalar(2) * (x[1] + x[2] - Scalar(2));
        g[2] = Scalar(2) * (x[1] + x[2] - Scalar(2));
        g[3] = Scalar(2) * (x[3] - Scalar(1));
        g[4] = Scalar(2) * (x[4] - Scalar(1));
    }

    void constraints(const Eigen::VectorX<Scalar>& x,
                     Eigen::VectorX<Scalar>& c) const
    {
        c.resize(3);
        c[0] = x[0] + Scalar(3) * x[1] - Scalar(4);
        c[1] = x[2] + x[3] - Scalar(2) * x[4];
        c[2] = x[1] - x[4];
    }

    void constraint_jacobian(const Eigen::VectorX<Scalar>& /*x*/,
                             Eigen::MatrixX<Scalar>& J) const
    {
        J.resize(3, 5);
        J.setZero();
        J(0, 0) = Scalar(1);
        J(0, 1) = Scalar(3);
        J(1, 2) = Scalar(1);
        J(1, 3) = Scalar(1);
        J(1, 4) = Scalar(-2);
        J(2, 1) = Scalar(1);
        J(2, 4) = Scalar(-1);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(5, -inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::VectorX<Scalar>::Constant(5, inf);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        Eigen::VectorX<Scalar> x0(5);
        x0 << Scalar(2.5), Scalar(0.5), Scalar(2), Scalar(-1), Scalar(0.5);
        return x0;
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }
};

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
    static constexpr int problem_dimension = dynamic_dimension;
    static constexpr problem_class pclass = problem_class::mixed;

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
    static constexpr int problem_dimension = dynamic_dimension;
    static constexpr problem_class pclass = problem_class::inequality | problem_class::bound_constrained;

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

}

#endif
