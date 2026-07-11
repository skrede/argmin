#ifndef HPP_GUARD_ARGMIN_TEST_FUNCTIONS_HOCK_SCHITTKOWSKI_H
#define HPP_GUARD_ARGMIN_TEST_FUNCTIONS_HOCK_SCHITTKOWSKI_H

// Hock-Schittkowski constrained optimization test problems.
//
// Reference: Hock & Schittkowski, "Test Examples for Nonlinear Programming
//            Codes", Lecture Notes in Economics and Mathematical Systems,
//            Vol. 187, Springer, 1981.
//
// All problems satisfy constrained + differentiable + bound_constrained.
// Inequality convention: c_ineq >= 0 (argmin lower-bound form).
// Constraint layout: equality first (rows 0..n_eq-1), then inequality.

#include "argmin/test_functions/problem_class.h"

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <numbers>

namespace argmin
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
    static constexpr int problem_dimension = 2;
    static constexpr problem_class pclass = problem_class::bound_constrained;
    static constexpr int constraint_count = 0;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        Scalar t1 = x[1] - x[0] * x[0];
        Scalar t2 = Scalar(1) - x[0];
        return Scalar(100) * t1 * t1 + t2 * t2;
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        Scalar t = x[1] - x[0] * x[0];
        g[0] = Scalar(-400) * x[0] * t - Scalar(2) * (Scalar(1) - x[0]);
        g[1] = Scalar(200) * t;
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        Eigen::Vector<Scalar, problem_dimension> lb;
        lb << -inf, Scalar(-1.5);
        return lb;
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(2, inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        Eigen::Vector<Scalar, problem_dimension> x0;
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
    static constexpr int problem_dimension = 2;
    static constexpr problem_class pclass = problem_class::bound_constrained;
    static constexpr int constraint_count = 0;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        Scalar t1 = x[1] - x[0] * x[0];
        Scalar t2 = Scalar(1) - x[0];
        return Scalar(100) * t1 * t1 + t2 * t2;
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        Scalar t = x[1] - x[0] * x[0];
        g[0] = Scalar(-400) * x[0] * t - Scalar(2) * (Scalar(1) - x[0]);
        g[1] = Scalar(200) * t;
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        Eigen::Vector<Scalar, problem_dimension> lb;
        lb << -inf, Scalar(1.5);
        return lb;
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(2, inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        Eigen::Vector<Scalar, problem_dimension> x0;
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
    static constexpr int problem_dimension = 2;
    static constexpr problem_class pclass = problem_class::bound_constrained;
    static constexpr int constraint_count = 0;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        using std::sin;
        Scalar d = x[0] - x[1];
        return sin(x[0] + x[1]) + d * d
               - Scalar(1.5) * x[0] + Scalar(2.5) * x[1] + Scalar(1);
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        using std::cos;
        Scalar c = cos(x[0] + x[1]);
        Scalar d = x[0] - x[1];
        g[0] = c + Scalar(2) * d - Scalar(1.5);
        g[1] = c - Scalar(2) * d + Scalar(2.5);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        Eigen::Vector<Scalar, problem_dimension> lb;
        lb << Scalar(-1.5), Scalar(-3);
        return lb;
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        Eigen::Vector<Scalar, problem_dimension> ub;
        ub << Scalar(4), Scalar(3);
        return ub;
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Zero(2);
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
    static constexpr int problem_dimension = 2;
    static constexpr problem_class pclass = problem_class::equality;
    static constexpr int constraint_count = 1;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 1; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        Scalar t = Scalar(1) - x[0];
        return t * t;
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        g[0] = Scalar(-2) * (Scalar(1) - x[0]);
        g[1] = Scalar(0);
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        c[0] = Scalar(10) * (x[1] - x[0] * x[0]);
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& x, auto& J) const
    {
        J(0, 0) = Scalar(-20) * x[0];
        J(0, 1) = Scalar(10);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(2, -inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(2, inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        Eigen::Vector<Scalar, problem_dimension> x0;
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
    static constexpr int problem_dimension = 2;
    static constexpr problem_class pclass = problem_class::equality;
    static constexpr int constraint_count = 1;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 1; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        using std::log;
        return log(Scalar(1) + x[0] * x[0]) - x[1];
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        g[0] = Scalar(2) * x[0] / (Scalar(1) + x[0] * x[0]);
        g[1] = Scalar(-1);
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        Scalar t = Scalar(1) + x[0] * x[0];
        c[0] = t * t + x[1] * x[1] - Scalar(4);
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& x, auto& J) const
    {
        Scalar t = Scalar(1) + x[0] * x[0];
        J(0, 0) = Scalar(4) * x[0] * t;
        J(0, 1) = Scalar(2) * x[1];
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(2, -inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(2, inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Constant(2, Scalar(2));
    }

    [[nodiscard]] Scalar optimal_value() const
    {
        using std::sqrt;
        return -sqrt(Scalar(3));
    }
};

// HS021: 2D, 0 equality, 1 inequality, box bounds.
//
// min  0.01*x0^2 + x1^2 - 100
// s.t. c0 = 10*x0 - x1 - 10 >= 0
//      2 <= x0 <= 50, -50 <= x1 <= 50
//
// x0 = (-1, -1), f* = -99.96 at (2, 0).
//
// Reference: H&S Problem 21.
template <typename Scalar = double>
struct hs021
{
    static constexpr int problem_dimension = 2;
    static constexpr problem_class pclass = problem_class::inequality | problem_class::bound_constrained;
    static constexpr int constraint_count = 1;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 1; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        return Scalar(0.01) * x[0] * x[0] + x[1] * x[1] - Scalar(100);
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        g[0] = Scalar(0.02) * x[0];
        g[1] = Scalar(2) * x[1];
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        c[0] = Scalar(10) * x[0] - x[1] - Scalar(10);
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& /*x*/, auto& J) const
    {
        J(0, 0) = Scalar(10);
        J(0, 1) = Scalar(-1);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        Eigen::Vector<Scalar, problem_dimension> lb;
        lb << Scalar(2), Scalar(-50);
        return lb;
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        Eigen::Vector<Scalar, problem_dimension> ub;
        ub << Scalar(50), Scalar(50);
        return ub;
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Constant(2, Scalar(-1));
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(-99.96); }
};

// HS023: 2D, 0 equality, 5 inequality, box bounds.
//
// min  x0^2 + x1^2
// s.t. c0 = x0 + x1 - 1 >= 0
//      c1 = x0^2 + x1^2 - 1 >= 0
//      c2 = 9*x0^2 + x1^2 - 9 >= 0
//      c3 = x0^2 - x1 >= 0
//      c4 = x1^2 - x0 >= 0
//      -50 <= xi <= 50
//
// x0 = (3, 1), f* = 2 at (1, 1).
//
// Reference: H&S Problem 23.
template <typename Scalar = double>
struct hs023
{
    static constexpr int problem_dimension = 2;
    static constexpr problem_class pclass = problem_class::inequality | problem_class::bound_constrained;
    static constexpr int constraint_count = 5;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 5; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        return x[0] * x[0] + x[1] * x[1];
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        g[0] = Scalar(2) * x[0];
        g[1] = Scalar(2) * x[1];
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        c[0] = x[0] + x[1] - Scalar(1);
        c[1] = x[0] * x[0] + x[1] * x[1] - Scalar(1);
        c[2] = Scalar(9) * x[0] * x[0] + x[1] * x[1] - Scalar(9);
        c[3] = x[0] * x[0] - x[1];
        c[4] = x[1] * x[1] - x[0];
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& x, auto& J) const
    {
        J(0, 0) = Scalar(1);
        J(0, 1) = Scalar(1);
        J(1, 0) = Scalar(2) * x[0];
        J(1, 1) = Scalar(2) * x[1];
        J(2, 0) = Scalar(18) * x[0];
        J(2, 1) = Scalar(2) * x[1];
        J(3, 0) = Scalar(2) * x[0];
        J(3, 1) = Scalar(-1);
        J(4, 0) = Scalar(-1);
        J(4, 1) = Scalar(2) * x[1];
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Constant(2, Scalar(-50));
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Constant(2, Scalar(50));
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        Eigen::Vector<Scalar, problem_dimension> x0;
        x0 << Scalar(3), Scalar(1);
        return x0;
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(2); }
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
    static constexpr int problem_dimension = 2;
    static constexpr problem_class pclass = problem_class::inequality | problem_class::bound_constrained;
    static constexpr int constraint_count = 3;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 3; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        using std::sqrt;
        Scalar k = Scalar(1) / (Scalar(27) * sqrt(Scalar(3)));
        Scalar t = (x[0] - Scalar(3)) * (x[0] - Scalar(3)) - Scalar(9);
        return k * t * x[1] * x[1] * x[1];
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        using std::sqrt;
        Scalar k = Scalar(1) / (Scalar(27) * sqrt(Scalar(3)));
        Scalar t = (x[0] - Scalar(3)) * (x[0] - Scalar(3)) - Scalar(9);
        Scalar x1_cubed = x[1] * x[1] * x[1];
        // df/dx0 = k * 2*(x0-3) * x1^3
        g[0] = k * Scalar(2) * (x[0] - Scalar(3)) * x1_cubed;
        // df/dx1 = k * t * 3*x1^2
        g[1] = k * t * Scalar(3) * x[1] * x[1];
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        using std::sqrt;
        Scalar s3 = sqrt(Scalar(3));
        c[0] = x[0] / s3 - x[1];
        c[1] = x[0] + s3 * x[1];
        c[2] = Scalar(6) - x[0] - s3 * x[1];
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& /*x*/, auto& J) const
    {
        using std::sqrt;
        Scalar s3 = sqrt(Scalar(3));
        J(0, 0) = Scalar(1) / s3;  J(0, 1) = Scalar(-1);
        J(1, 0) = Scalar(1);       J(1, 1) = s3;
        J(2, 0) = Scalar(-1);      J(2, 1) = -s3;
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Zero(2);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(2, inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        Eigen::Vector<Scalar, problem_dimension> x0;
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
    static constexpr int problem_dimension = 3;
    static constexpr problem_class pclass = problem_class::equality;
    static constexpr int constraint_count = 1;

    [[nodiscard]] int dimension() const { return 3; }
    [[nodiscard]] int num_equality() const { return 1; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        Scalar d01 = x[0] - x[1];
        Scalar d12 = x[1] - x[2];
        return d01 * d01 + d12 * d12 * d12 * d12;
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        Scalar d01 = x[0] - x[1];
        Scalar d12 = x[1] - x[2];
        Scalar d12_cubed = d12 * d12 * d12;
        g[0] = Scalar(2) * d01;
        g[1] = Scalar(-2) * d01 + Scalar(4) * d12_cubed;
        g[2] = Scalar(-4) * d12_cubed;
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        Scalar x2_sq = x[2] * x[2];
        c[0] = (Scalar(1) + x[1] * x[1]) * x[0] + x2_sq * x2_sq - Scalar(3);
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& x, auto& J) const
    {
        J(0, 0) = Scalar(1) + x[1] * x[1];
        J(0, 1) = Scalar(2) * x[0] * x[1];
        J(0, 2) = Scalar(4) * x[2] * x[2] * x[2];
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(3, -inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(3, inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        Eigen::Vector<Scalar, problem_dimension> x0;
        x0 << Scalar(-2.6), Scalar(2), Scalar(2);
        return x0;
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }
};

// HS027: 3D, 1 equality constraint.
//
// min  0.01*(1 - x0)^2 + (x1 - x0^2)^2
// s.t. x0 + x2^2 + 1 = 0
//
// x0 = (2, 2, 2), f* = 0.04 at (-1, 1, 0).
//
// Reference: H&S Problem 27.
template <typename Scalar = double>
struct hs027
{
    static constexpr int problem_dimension = 3;
    static constexpr problem_class pclass = problem_class::equality;
    static constexpr int constraint_count = 1;

    [[nodiscard]] int dimension() const { return 3; }
    [[nodiscard]] int num_equality() const { return 1; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        Scalar t0 = Scalar(1) - x[0];
        Scalar t1 = x[1] - x[0] * x[0];
        return Scalar(0.01) * t0 * t0 + t1 * t1;
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        Scalar t1 = x[1] - x[0] * x[0];
        g[0] = Scalar(-0.02) * (Scalar(1) - x[0]) - Scalar(4) * x[0] * t1;
        g[1] = Scalar(2) * t1;
        g[2] = Scalar(0);
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        c[0] = x[0] + x[2] * x[2] + Scalar(1);
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& x, auto& J) const
    {
        J(0, 0) = Scalar(1);
        J(0, 1) = Scalar(0);
        J(0, 2) = Scalar(2) * x[2];
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(3, -inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(3, inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Constant(3, Scalar(2));
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0.04); }
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
    static constexpr int problem_dimension = 3;
    static constexpr problem_class pclass = problem_class::equality;
    static constexpr int constraint_count = 1;

    [[nodiscard]] int dimension() const { return 3; }
    [[nodiscard]] int num_equality() const { return 1; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        Scalar s01 = x[0] + x[1];
        Scalar s12 = x[1] + x[2];
        return s01 * s01 + s12 * s12;
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        Scalar s01 = x[0] + x[1];
        Scalar s12 = x[1] + x[2];
        g[0] = Scalar(2) * s01;
        g[1] = Scalar(2) * s01 + Scalar(2) * s12;
        g[2] = Scalar(2) * s12;
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        c[0] = x[0] + Scalar(2) * x[1] + Scalar(3) * x[2] - Scalar(1);
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& /*x*/, auto& J) const
    {
        J(0, 0) = Scalar(1);
        J(0, 1) = Scalar(2);
        J(0, 2) = Scalar(3);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(3, -inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(3, inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        Eigen::Vector<Scalar, problem_dimension> x0;
        x0 << Scalar(-4), Scalar(1), Scalar(1);
        return x0;
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }
};

// HS029: 3D, 0 equality, 1 inequality.
//
// min  -x0*x1*x2
// s.t. c0 = 48 - x0^2 - 2*x1^2 - 4*x2^2 >= 0
//
// x0 = (1, 1, 1), f* = -16*sqrt(2) at (4, 2*sqrt(2), 2).
//
// Reference: H&S Problem 29.
template <typename Scalar = double>
struct hs029
{
    static constexpr int problem_dimension = 3;
    static constexpr problem_class pclass = problem_class::inequality;
    static constexpr int constraint_count = 1;

    [[nodiscard]] int dimension() const { return 3; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 1; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        return -x[0] * x[1] * x[2];
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        g[0] = -x[1] * x[2];
        g[1] = -x[0] * x[2];
        g[2] = -x[0] * x[1];
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        c[0] = Scalar(48) - x[0] * x[0] - Scalar(2) * x[1] * x[1]
               - Scalar(4) * x[2] * x[2];
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& x, auto& J) const
    {
        J(0, 0) = Scalar(-2) * x[0];
        J(0, 1) = Scalar(-4) * x[1];
        J(0, 2) = Scalar(-8) * x[2];
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(3, -inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(3, inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Ones(3);
    }

    [[nodiscard]] Scalar optimal_value() const
    {
        using std::sqrt;
        return Scalar(-16) * sqrt(Scalar(2));
    }
};

// HS030: 3D, 0 equality, 1 inequality, box bounds.
//
// min  x0^2 + x1^2 + x2^2
// s.t. c0 = x0^2 + x1^2 - 1 >= 0
//      1 <= x0 <= 10, -10 <= x1,x2 <= 10
//
// x0 = (1, 1, 1), f* = 1 at (1, 0, 0).
//
// Reference: H&S Problem 30.
template <typename Scalar = double>
struct hs030
{
    static constexpr int problem_dimension = 3;
    static constexpr problem_class pclass = problem_class::inequality | problem_class::bound_constrained;
    static constexpr int constraint_count = 1;

    [[nodiscard]] int dimension() const { return 3; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 1; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        return x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        g[0] = Scalar(2) * x[0];
        g[1] = Scalar(2) * x[1];
        g[2] = Scalar(2) * x[2];
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        c[0] = x[0] * x[0] + x[1] * x[1] - Scalar(1);
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& x, auto& J) const
    {
        J(0, 0) = Scalar(2) * x[0];
        J(0, 1) = Scalar(2) * x[1];
        J(0, 2) = Scalar(0);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        Eigen::Vector<Scalar, problem_dimension> lb;
        lb << Scalar(1), Scalar(-10), Scalar(-10);
        return lb;
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Constant(3, Scalar(10));
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Ones(3);
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(1); }
};

// HS031: 3D, 0 equality, 1 inequality, box bounds.
//
// min  9*x0^2 + x1^2 + 9*x2^2
// s.t. c0 = x0*x1 - 1 >= 0
//      -10 <= x0 <= 10, 1 <= x1 <= 10, -10 <= x2 <= 1
//
// x0 = (1, 1, 1), f* = 6 at (1/sqrt(3), sqrt(3), 0).
//
// Reference: H&S Problem 31.
template <typename Scalar = double>
struct hs031
{
    static constexpr int problem_dimension = 3;
    static constexpr problem_class pclass = problem_class::inequality | problem_class::bound_constrained;
    static constexpr int constraint_count = 1;

    [[nodiscard]] int dimension() const { return 3; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 1; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        return Scalar(9) * x[0] * x[0] + x[1] * x[1] + Scalar(9) * x[2] * x[2];
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        g[0] = Scalar(18) * x[0];
        g[1] = Scalar(2) * x[1];
        g[2] = Scalar(18) * x[2];
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        c[0] = x[0] * x[1] - Scalar(1);
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& x, auto& J) const
    {
        J(0, 0) = x[1];
        J(0, 1) = x[0];
        J(0, 2) = Scalar(0);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        Eigen::Vector<Scalar, problem_dimension> lb;
        lb << Scalar(-10), Scalar(1), Scalar(-10);
        return lb;
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        Eigen::Vector<Scalar, problem_dimension> ub;
        ub << Scalar(10), Scalar(10), Scalar(1);
        return ub;
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Ones(3);
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(6); }
};

// HS034: 3D, 0 equality, 2 inequality, x >= 0 with upper bounds.
//
// min  -x0
// s.t. c0 = x1 - exp(x0) >= 0
//      c1 = x2 - exp(x1) >= 0
//      0 <= x0,x1 <= 100, 0 <= x2 <= 10
//
// x0 = (0, 1.05, 2.9), f* = -log(log(10)).
//
// Reference: H&S Problem 34.
template <typename Scalar = double>
struct hs034
{
    static constexpr int problem_dimension = 3;
    static constexpr problem_class pclass = problem_class::inequality | problem_class::bound_constrained;
    static constexpr int constraint_count = 2;

    [[nodiscard]] int dimension() const { return 3; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 2; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const { return -x[0]; }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& /*x*/,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        g[0] = Scalar(-1);
        g[1] = Scalar(0);
        g[2] = Scalar(0);
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        using std::exp;
        c[0] = x[1] - exp(x[0]);
        c[1] = x[2] - exp(x[1]);
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& x, auto& J) const
    {
        using std::exp;
        J(0, 0) = -exp(x[0]);
        J(0, 1) = Scalar(1);
        J(0, 2) = Scalar(0);
        J(1, 0) = Scalar(0);
        J(1, 1) = -exp(x[1]);
        J(1, 2) = Scalar(1);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Zero(3);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        Eigen::Vector<Scalar, problem_dimension> ub;
        ub << Scalar(100), Scalar(100), Scalar(10);
        return ub;
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        Eigen::Vector<Scalar, problem_dimension> x0;
        x0 << Scalar(0), Scalar(1.05), Scalar(2.9);
        return x0;
    }

    [[nodiscard]] Scalar optimal_value() const
    {
        using std::log;
        return -log(log(Scalar(10)));
    }
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
    static constexpr int problem_dimension = 3;
    static constexpr problem_class pclass = problem_class::inequality | problem_class::bound_constrained;
    static constexpr int constraint_count = 1;

    [[nodiscard]] int dimension() const { return 3; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 1; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        return Scalar(9) - Scalar(8) * x[0] - Scalar(6) * x[1]
               - Scalar(4) * x[2] + Scalar(2) * x[0] * x[0]
               + Scalar(2) * x[1] * x[1] + x[2] * x[2]
               + Scalar(2) * x[0] * x[1] + Scalar(2) * x[0] * x[2];
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        g[0] = Scalar(-8) + Scalar(4) * x[0] + Scalar(2) * x[1]
               + Scalar(2) * x[2];
        g[1] = Scalar(-6) + Scalar(2) * x[0] + Scalar(4) * x[1];
        g[2] = Scalar(-4) + Scalar(2) * x[0] + Scalar(2) * x[2];
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        c[0] = Scalar(3) - (x[0] + x[1] + Scalar(2) * x[2]);
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& /*x*/, auto& J) const
    {
        J(0, 0) = Scalar(-1);
        J(0, 1) = Scalar(-1);
        J(0, 2) = Scalar(-2);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Zero(3);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(3, inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Constant(3, Scalar(0.5));
    }

    [[nodiscard]] Scalar optimal_value() const
    {
        return Scalar(1) / Scalar(9);
    }
};

// HS036: 3D, 0 equality, 1 inequality, x >= 0 with upper bounds.
//
// min  -x0*x1*x2
// s.t. c0 = 72 - x0 - 2*x1 - 2*x2 >= 0
//      0 <= x0 <= 20, 0 <= x1 <= 11, 0 <= x2 <= 42
//
// x0 = (10, 10, 10), f* = -3300 at (20, 11, 15).
//
// Reference: H&S Problem 36.
template <typename Scalar = double>
struct hs036
{
    static constexpr int problem_dimension = 3;
    static constexpr problem_class pclass = problem_class::inequality | problem_class::bound_constrained;
    static constexpr int constraint_count = 1;

    [[nodiscard]] int dimension() const { return 3; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 1; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const { return -x[0] * x[1] * x[2]; }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        g[0] = -x[1] * x[2];
        g[1] = -x[0] * x[2];
        g[2] = -x[0] * x[1];
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        c[0] = Scalar(72) - x[0] - Scalar(2) * x[1] - Scalar(2) * x[2];
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& /*x*/, auto& J) const
    {
        J(0, 0) = Scalar(-1);
        J(0, 1) = Scalar(-2);
        J(0, 2) = Scalar(-2);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Zero(3);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        Eigen::Vector<Scalar, problem_dimension> ub;
        ub << Scalar(20), Scalar(11), Scalar(42);
        return ub;
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Constant(3, Scalar(10));
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(-3300); }
};

// HS037: 3D, 0 equality, 2 inequality, x >= 0 with upper bounds.
//
// min  -x0*x1*x2
// s.t. c0 = 72 - x0 - 2*x1 - 2*x2 >= 0
//      c1 = x0 + 2*x1 + 2*x2 >= 0
//      0 <= xi <= 42
//
// x0 = (10, 10, 10), f* = -3456 at (24, 12, 12).
//
// Reference: H&S Problem 37.
template <typename Scalar = double>
struct hs037
{
    static constexpr int problem_dimension = 3;
    static constexpr problem_class pclass = problem_class::inequality | problem_class::bound_constrained;
    static constexpr int constraint_count = 2;

    [[nodiscard]] int dimension() const { return 3; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 2; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const { return -x[0] * x[1] * x[2]; }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        g[0] = -x[1] * x[2];
        g[1] = -x[0] * x[2];
        g[2] = -x[0] * x[1];
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        c[0] = Scalar(72) - x[0] - Scalar(2) * x[1] - Scalar(2) * x[2];
        c[1] = x[0] + Scalar(2) * x[1] + Scalar(2) * x[2];
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& /*x*/, auto& J) const
    {
        J(0, 0) = Scalar(-1);
        J(0, 1) = Scalar(-2);
        J(0, 2) = Scalar(-2);
        J(1, 0) = Scalar(1);
        J(1, 1) = Scalar(2);
        J(1, 2) = Scalar(2);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Zero(3);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Constant(3, Scalar(42));
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Constant(3, Scalar(10));
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(-3456); }
};

// HS038: 4D, bound-constrained Colville function.
//
// min  100*(x1 - x0^2)^2 + (x0 - 1)^2
//      + 90*(x3 - x2^2)^2 + (x2 - 1)^2
//      + 10.1*((x1 - 1)^2 + (x3 - 1)^2)
//      + 19.8*(x1 - 1)*(x3 - 1)
//      -10 <= xi <= 10
//
// x0 = (-3, -1, -3, -1), f* = 0 at (1, 1, 1, 1).
//
// Reference: H&S Problem 38.
template <typename Scalar = double>
struct hs038
{
    static constexpr int problem_dimension = 4;
    static constexpr problem_class pclass = problem_class::bound_constrained;
    static constexpr int constraint_count = 0;

    [[nodiscard]] int dimension() const { return 4; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        Scalar t01 = x[1] - x[0] * x[0];
        Scalar t23 = x[3] - x[2] * x[2];
        Scalar d0 = x[0] - Scalar(1);
        Scalar d1 = x[1] - Scalar(1);
        Scalar d2 = x[2] - Scalar(1);
        Scalar d3 = x[3] - Scalar(1);
        return Scalar(100) * t01 * t01 + d0 * d0 + Scalar(90) * t23 * t23
               + d2 * d2 + Scalar(10.1) * (d1 * d1 + d3 * d3)
               + Scalar(19.8) * d1 * d3;
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        Scalar t01 = x[1] - x[0] * x[0];
        Scalar t23 = x[3] - x[2] * x[2];
        g[0] = Scalar(-400) * x[0] * t01 + Scalar(2) * (x[0] - Scalar(1));
        g[1] = Scalar(200) * t01 + Scalar(20.2) * (x[1] - Scalar(1)) + Scalar(19.8) * (x[3] - Scalar(1));
        g[2] = Scalar(-360) * x[2] * t23 + Scalar(2) * (x[2] - Scalar(1));
        g[3] = Scalar(180) * t23 + Scalar(20.2) * (x[3] - Scalar(1)) + Scalar(19.8) * (x[1] - Scalar(1));
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Constant(4, Scalar(-10));
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Constant(4, Scalar(10));
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        Eigen::Vector<Scalar, problem_dimension> x0;
        x0 << Scalar(-3), Scalar(-1), Scalar(-3), Scalar(-1);
        return x0;
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }
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
    static constexpr int problem_dimension = 4;
    static constexpr problem_class pclass = problem_class::equality;
    static constexpr int constraint_count = 2;

    [[nodiscard]] int dimension() const { return 4; }
    [[nodiscard]] int num_equality() const { return 2; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        return -x[0];
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& /*x*/,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        g[0] = Scalar(-1);
        g[1] = Scalar(0);
        g[2] = Scalar(0);
        g[3] = Scalar(0);
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        c[0] = x[1] - x[0] * x[0] * x[0] - x[2] * x[2];
        c[1] = x[0] * x[0] - x[1] - x[3] * x[3];
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& x, auto& J) const
    {
        J(0, 0) = Scalar(-3) * x[0] * x[0];
        J(0, 1) = Scalar(1);
        J(0, 2) = Scalar(-2) * x[2];
        J(0, 3) = Scalar(0);
        J(1, 0) = Scalar(2) * x[0];
        J(1, 1) = Scalar(-1);
        J(1, 2) = Scalar(0);
        J(1, 3) = Scalar(-2) * x[3];
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(4, -inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(4, inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Constant(4, Scalar(2));
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
    static constexpr int problem_dimension = 4;
    static constexpr problem_class pclass = problem_class::equality;
    static constexpr int constraint_count = 3;

    [[nodiscard]] int dimension() const { return 4; }
    [[nodiscard]] int num_equality() const { return 3; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        return -x[0] * x[1] * x[2] * x[3];
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        g[0] = -x[1] * x[2] * x[3];
        g[1] = -x[0] * x[2] * x[3];
        g[2] = -x[0] * x[1] * x[3];
        g[3] = -x[0] * x[1] * x[2];
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        c[0] = x[0] * x[0] * x[0] + x[1] * x[1] - Scalar(1);
        c[1] = x[0] * x[0] * x[3] - x[2];
        c[2] = x[3] * x[3] - x[1];
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& x, auto& J) const
    {
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

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(4, -inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(4, inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Constant(4, Scalar(0.8));
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
    static constexpr int problem_dimension = 4;
    static constexpr problem_class pclass = problem_class::inequality;
    static constexpr int constraint_count = 3;

    [[nodiscard]] int dimension() const { return 4; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 3; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        return x[0] * x[0] + x[1] * x[1] + Scalar(2) * x[2] * x[2]
               + x[3] * x[3] - Scalar(5) * x[0] - Scalar(5) * x[1]
               - Scalar(21) * x[2] + Scalar(7) * x[3];
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        g[0] = Scalar(2) * x[0] - Scalar(5);
        g[1] = Scalar(2) * x[1] - Scalar(5);
        g[2] = Scalar(4) * x[2] - Scalar(21);
        g[3] = Scalar(2) * x[3] + Scalar(7);
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        c[0] = Scalar(8) - (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
               + x[3] * x[3] + x[0] - x[1] + x[2] - x[3]);
        c[1] = Scalar(10) - (x[0] * x[0] + Scalar(2) * x[1] * x[1]
               + x[2] * x[2] + Scalar(2) * x[3] * x[3] - x[0] - x[3]);
        c[2] = Scalar(5) - (Scalar(2) * x[0] * x[0] + x[1] * x[1]
               + x[2] * x[2] + Scalar(2) * x[0] - x[1] - x[3]);
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& x, auto& J) const
    {
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

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(4, -inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(4, inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Zero(4);
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(-44); }
};

// HS044: 4D, 0 equality, 6 inequality, x >= 0.
//
// min  x0 - x1 - x2 - x0*x2 + x0*x3 - x1*x3 + x1*x2
// s.t. c0 = 8 - x0 - 2*x1 >= 0
//      c1 = 12 - 4*x0 - x1 >= 0
//      c2 = 12 - 3*x0 - 4*x1 >= 0
//      c3 = 8 - 2*x2 - x3 >= 0
//      c4 = 8 - x2 - 2*x3 >= 0
//      c5 = 5 - x2 - x3 >= 0
//      xi >= 0
//
// x0 = (0, 0, 0, 0), f* = -15 at (0, 3, 0, 4).
//
// Reference: H&S Problem 44.
template <typename Scalar = double>
struct hs044
{
    static constexpr int problem_dimension = 4;
    static constexpr problem_class pclass = problem_class::inequality | problem_class::bound_constrained;
    static constexpr int constraint_count = 6;

    [[nodiscard]] int dimension() const { return 4; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 6; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        return x[0] - x[1] - x[2] - x[0] * x[2] + x[0] * x[3]
               - x[1] * x[3] + x[1] * x[2];
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        g[0] = Scalar(1) - x[2] + x[3];
        g[1] = Scalar(-1) - x[3] + x[2];
        g[2] = Scalar(-1) - x[0] + x[1];
        g[3] = x[0] - x[1];
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        c[0] = Scalar(8) - x[0] - Scalar(2) * x[1];
        c[1] = Scalar(12) - Scalar(4) * x[0] - x[1];
        c[2] = Scalar(12) - Scalar(3) * x[0] - Scalar(4) * x[1];
        c[3] = Scalar(8) - Scalar(2) * x[2] - x[3];
        c[4] = Scalar(8) - x[2] - Scalar(2) * x[3];
        c[5] = Scalar(5) - x[2] - x[3];
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& /*x*/, auto& J) const
    {
        J(0, 0) = Scalar(-1);
        J(0, 1) = Scalar(-2);
        J(0, 2) = Scalar(0);
        J(0, 3) = Scalar(0);
        J(1, 0) = Scalar(-4);
        J(1, 1) = Scalar(-1);
        J(1, 2) = Scalar(0);
        J(1, 3) = Scalar(0);
        J(2, 0) = Scalar(-3);
        J(2, 1) = Scalar(-4);
        J(2, 2) = Scalar(0);
        J(2, 3) = Scalar(0);
        J(3, 0) = Scalar(0);
        J(3, 1) = Scalar(0);
        J(3, 2) = Scalar(-2);
        J(3, 3) = Scalar(-1);
        J(4, 0) = Scalar(0);
        J(4, 1) = Scalar(0);
        J(4, 2) = Scalar(-1);
        J(4, 3) = Scalar(-2);
        J(5, 0) = Scalar(0);
        J(5, 1) = Scalar(0);
        J(5, 2) = Scalar(-1);
        J(5, 3) = Scalar(-1);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Zero(4);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(4, inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Zero(4);
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(-15); }
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
    static constexpr int problem_dimension = 5;
    static constexpr problem_class pclass = problem_class::equality;
    static constexpr int constraint_count = 2;

    [[nodiscard]] int dimension() const { return 5; }
    [[nodiscard]] int num_equality() const { return 2; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        Scalar t0 = x[0] - Scalar(1);
        Scalar t1 = x[1] - x[2];
        Scalar t2 = x[3] - x[4];
        return t0 * t0 + t1 * t1 + t2 * t2;
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        g[0] = Scalar(2) * (x[0] - Scalar(1));
        g[1] = Scalar(2) * (x[1] - x[2]);
        g[2] = Scalar(-2) * (x[1] - x[2]);
        g[3] = Scalar(2) * (x[3] - x[4]);
        g[4] = Scalar(-2) * (x[3] - x[4]);
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        c[0] = x[0] + x[1] + x[2] + x[3] + x[4] - Scalar(5);
        c[1] = x[2] - Scalar(2) * (x[3] + x[4]) + Scalar(3);
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& /*x*/, auto& J) const
    {
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

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(5, -inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(5, inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        Eigen::Vector<Scalar, problem_dimension> x0;
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
    static constexpr int problem_dimension = 5;
    static constexpr problem_class pclass = problem_class::equality;
    static constexpr int constraint_count = 3;

    [[nodiscard]] int dimension() const { return 5; }
    [[nodiscard]] int num_equality() const { return 3; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        Scalar d01 = x[0] - x[1];
        Scalar d12 = x[1] - x[2];
        Scalar d23 = x[2] - x[3];
        Scalar d34 = x[3] - x[4];
        return d01 * d01 + d12 * d12 + d23 * d23 * d23 * d23 + d34 * d34;
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        Scalar d01 = x[0] - x[1];
        Scalar d12 = x[1] - x[2];
        Scalar d23 = x[2] - x[3];
        Scalar d23_cubed = d23 * d23 * d23;
        Scalar d34 = x[3] - x[4];
        g[0] = Scalar(2) * d01;
        g[1] = Scalar(-2) * d01 + Scalar(2) * d12;
        g[2] = Scalar(-2) * d12 + Scalar(4) * d23_cubed;
        g[3] = Scalar(-4) * d23_cubed + Scalar(2) * d34;
        g[4] = Scalar(-2) * d34;
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        c[0] = x[0] + Scalar(2) * x[1] + Scalar(3) * x[2] - Scalar(6);
        c[1] = x[1] + Scalar(2) * x[2] + Scalar(3) * x[3] - Scalar(6);
        c[2] = x[2] + Scalar(2) * x[3] + Scalar(3) * x[4] - Scalar(6);
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& /*x*/, auto& J) const
    {
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

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(5, -inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(5, inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        Eigen::Vector<Scalar, problem_dimension> x0;
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
    static constexpr int problem_dimension = 5;
    static constexpr problem_class pclass = problem_class::equality;
    static constexpr int constraint_count = 3;

    [[nodiscard]] int dimension() const { return 5; }
    [[nodiscard]] int num_equality() const { return 3; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        Scalar t0 = x[0] - x[1];
        Scalar t1 = x[1] + x[2] - Scalar(2);
        Scalar t2 = x[3] - Scalar(1);
        Scalar t3 = x[4] - Scalar(1);
        return t0 * t0 + t1 * t1 + t2 * t2 + t3 * t3;
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        g[0] = Scalar(2) * (x[0] - x[1]);
        g[1] = Scalar(-2) * (x[0] - x[1])
               + Scalar(2) * (x[1] + x[2] - Scalar(2));
        g[2] = Scalar(2) * (x[1] + x[2] - Scalar(2));
        g[3] = Scalar(2) * (x[3] - Scalar(1));
        g[4] = Scalar(2) * (x[4] - Scalar(1));
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        c[0] = x[0] + Scalar(3) * x[1] - Scalar(4);
        c[1] = x[2] + x[3] - Scalar(2) * x[4];
        c[2] = x[1] - x[4];
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& /*x*/, auto& J) const
    {
        J.setZero();
        J(0, 0) = Scalar(1);
        J(0, 1) = Scalar(3);
        J(1, 2) = Scalar(1);
        J(1, 3) = Scalar(1);
        J(1, 4) = Scalar(-2);
        J(2, 1) = Scalar(1);
        J(2, 4) = Scalar(-1);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(5, -inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(5, inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        Eigen::Vector<Scalar, problem_dimension> x0;
        x0 << Scalar(2.5), Scalar(0.5), Scalar(2), Scalar(-1), Scalar(0.5);
        return x0;
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }
};

// HS052: 5D, 3 equality constraints.
//
// min  (4*x0 - x1)^2 + (x1 + x2 - 2)^2 + (x3 - 1)^2 + (x4 - 1)^2
// s.t. x0 + 3*x1 = 0
//      x2 + x3 - 2*x4 = 0
//      x1 - x4 = 0
//
// x0 = (2, 2, 2, 2, 2), f* = 1859/349.
//
// Reference: H&S Problem 52.
template <typename Scalar = double>
struct hs052
{
    static constexpr int problem_dimension = 5;
    static constexpr problem_class pclass = problem_class::equality;
    static constexpr int constraint_count = 3;

    [[nodiscard]] int dimension() const { return 5; }
    [[nodiscard]] int num_equality() const { return 3; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        Scalar t0 = Scalar(4) * x[0] - x[1];
        Scalar t1 = x[1] + x[2] - Scalar(2);
        Scalar t2 = x[3] - Scalar(1);
        Scalar t3 = x[4] - Scalar(1);
        return t0 * t0 + t1 * t1 + t2 * t2 + t3 * t3;
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        Scalar t0 = Scalar(4) * x[0] - x[1];
        Scalar t1 = x[1] + x[2] - Scalar(2);
        g[0] = Scalar(8) * t0;
        g[1] = Scalar(-2) * t0 + Scalar(2) * t1;
        g[2] = Scalar(2) * t1;
        g[3] = Scalar(2) * (x[3] - Scalar(1));
        g[4] = Scalar(2) * (x[4] - Scalar(1));
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        c[0] = x[0] + Scalar(3) * x[1];
        c[1] = x[2] + x[3] - Scalar(2) * x[4];
        c[2] = x[1] - x[4];
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& /*x*/, auto& J) const
    {
        J.setZero();
        J(0, 0) = Scalar(1);
        J(0, 1) = Scalar(3);
        J(1, 2) = Scalar(1);
        J(1, 3) = Scalar(1);
        J(1, 4) = Scalar(-2);
        J(2, 1) = Scalar(1);
        J(2, 4) = Scalar(-1);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(5, -inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(5, inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Constant(5, Scalar(2));
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(1859) / Scalar(349); }
};

// HS071: 4D, 1 equality, 1 inequality, box [1, 5].
//
// min  x0*x3*(x0 + x1 + x2) + x2
// s.t. x0^2 + x1^2 + x2^2 + x3^2 = 40       (equality)
//      x0*x1*x2*x3 >= 25                       (inequality, argmin form)
//      1 <= xi <= 5
//
// x0 = (1, 5, 5, 1), f* = 17.014017289134664
//
// Reference: H&S Problem 71.
template <typename Scalar = double>
struct hs071
{
    static constexpr int problem_dimension = 4;
    static constexpr problem_class pclass = problem_class::mixed;
    static constexpr int constraint_count = 2;

    [[nodiscard]] int dimension() const { return 4; }
    [[nodiscard]] int num_equality() const { return 1; }
    [[nodiscard]] int num_inequality() const { return 1; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        g[0] = x[3] * (Scalar(2) * x[0] + x[1] + x[2]);
        g[1] = x[0] * x[3];
        g[2] = x[0] * x[3] + Scalar(1);
        g[3] = x[0] * (x[0] + x[1] + x[2]);
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        // eq: x0^2 + x1^2 + x2^2 + x3^2 - 40 = 0
        c[0] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]
               - Scalar(40);
        // ineq: x0*x1*x2*x3 - 25 >= 0
        c[1] = x[0] * x[1] * x[2] * x[3] - Scalar(25);
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& x, auto& J) const
    {
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

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Constant(4, Scalar(1));
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Constant(4, Scalar(5));
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        Eigen::Vector<Scalar, problem_dimension> x0;
        x0 << Scalar(1), Scalar(5), Scalar(5), Scalar(1);
        return x0;
    }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(17.014017289134664); }
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
    static constexpr int problem_dimension = 4;
    static constexpr problem_class pclass = problem_class::inequality | problem_class::bound_constrained;
    static constexpr int constraint_count = 3;

    [[nodiscard]] int dimension() const { return 4; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 3; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& x) const
    {
        return x[0] * x[0] + Scalar(0.5) * x[1] * x[1]
               + x[2] * x[2] + Scalar(0.5) * x[3] * x[3]
               - x[0] * x[2] + x[2] * x[3]
               - x[0] - Scalar(3) * x[1] + x[2] - x[3];
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& x,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        g[0] = Scalar(2) * x[0] - x[2] - Scalar(1);
        g[1] = x[1] - Scalar(3);
        g[2] = Scalar(2) * x[2] - x[0] + x[3] + Scalar(1);
        g[3] = x[3] + x[2] - Scalar(1);
    }

    void constraints(const Eigen::Vector<Scalar, problem_dimension>& x, auto& c) const
    {
        c[0] = Scalar(5) - (x[0] + Scalar(2) * x[1] + x[2] + x[3]);
        c[1] = Scalar(4) - (Scalar(3) * x[0] + x[1] + Scalar(2) * x[2] - x[3]);
        c[2] = x[1] + Scalar(4) * x[2] - Scalar(1.5);
    }

    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& /*x*/, auto& J) const
    {
        J << Scalar(-1), Scalar(-2), Scalar(-1), Scalar(-1),
             Scalar(-3), Scalar(-1), Scalar(-2), Scalar(1),
             Scalar(0),  Scalar(1),  Scalar(4),  Scalar(0);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Zero(4);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        return Eigen::Vector<Scalar, problem_dimension>::Constant(4, inf);
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        return Eigen::Vector<Scalar, problem_dimension>::Constant(4, Scalar(0.5));
    }

    [[nodiscard]] Scalar optimal_value() const
    {
        return Scalar(-4.6818181818181818);
    }
};

}

#endif
