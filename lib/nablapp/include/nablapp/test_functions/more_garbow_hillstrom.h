#ifndef HPP_GUARD_NABLAPP_TEST_FUNCTIONS_MORE_GARBOW_HILLSTROM_H
#define HPP_GUARD_NABLAPP_TEST_FUNCTIONS_MORE_GARBOW_HILLSTROM_H

// More-Garbow-Hillstrom unconstrained optimization test problems.
//
// Reference: More, Garbow, Hillstrom, "Testing Unconstrained Optimization
//            Software", ACM Transactions on Mathematical Software, Vol. 7,
//            No. 1, March 1981, pp. 17-41.
//
// All problems are unconstrained with analytic gradients.

#include "nablapp/test_functions/problem_class.h"

#include <Eigen/Core>

#include <cmath>
#include <numbers>

namespace nablapp
{

// Powell singular function, 4D.
//
//   f(x) = (x0 + 10*x1)^2 + 5*(x2 - x3)^2
//          + (x1 - 2*x2)^4 + 10*(x0 - x3)^4
//
// x0 = (3, -1, 0, 1), f* = 0 at origin.
//
// Reference: More, Garbow, Hillstrom (1981), Problem 13.
template <typename Scalar = double>
struct powell_singular
{
    static constexpr problem_class pclass = problem_class::unconstrained;

    [[nodiscard]] int dimension() const { return 4; }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        Eigen::VectorX<Scalar> x0(4);
        x0 << Scalar(3), Scalar(-1), Scalar(0), Scalar(1);
        return x0;
    }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        Scalar t1 = x[0] + Scalar(10) * x[1];
        Scalar t2 = x[2] - x[3];
        Scalar t3 = x[1] - Scalar(2) * x[2];
        Scalar t4 = x[0] - x[3];
        return t1 * t1 + Scalar(5) * t2 * t2
               + t3 * t3 * t3 * t3 + Scalar(10) * t4 * t4 * t4 * t4;
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        Scalar t1 = x[0] + Scalar(10) * x[1];
        Scalar t2 = x[2] - x[3];
        Scalar t3 = x[1] - Scalar(2) * x[2];
        Scalar t4 = x[0] - x[3];
        Scalar t3_cubed = t3 * t3 * t3;
        Scalar t4_cubed = t4 * t4 * t4;
        g.resize(4);
        g[0] = Scalar(2) * t1 + Scalar(40) * t4_cubed;
        g[1] = Scalar(20) * t1 + Scalar(4) * t3_cubed;
        g[2] = Scalar(10) * t2 - Scalar(8) * t3_cubed;
        g[3] = -Scalar(10) * t2 - Scalar(40) * t4_cubed;
    }
};

// Brown badly scaled function, 2D.
//
//   f(x) = (x0 - 1e6)^2 + (x1 - 2e-6)^2 + (x0*x1 - 2)^2
//
// x0 = (1, 1), f* = 0 at (1e6, 2e-6).
//
// Reference: More, Garbow, Hillstrom (1981), Problem 4.
template <typename Scalar = double>
struct brown_badly_scaled
{
    static constexpr problem_class pclass = problem_class::unconstrained;

    [[nodiscard]] int dimension() const { return 2; }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        return Eigen::VectorX<Scalar>::Ones(2);
    }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        Scalar t1 = x[0] - Scalar(1e6);
        Scalar t2 = x[1] - Scalar(2e-6);
        Scalar t3 = x[0] * x[1] - Scalar(2);
        return t1 * t1 + t2 * t2 + t3 * t3;
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        Scalar t1 = x[0] - Scalar(1e6);
        Scalar t2 = x[1] - Scalar(2e-6);
        Scalar t3 = x[0] * x[1] - Scalar(2);
        g.resize(2);
        g[0] = Scalar(2) * t1 + Scalar(2) * t3 * x[1];
        g[1] = Scalar(2) * t2 + Scalar(2) * t3 * x[0];
    }
};

// Trigonometric function, variable-dimension.
//
//   f(x) = sum_i [ (n - sum_j cos(x_j) + i*(1 - cos(x_i)) - sin(x_i))^2 ]
//
// (1-indexed: i = 1..n in the formula; 0-indexed in code.)
//
// x0 = (1/n, ..., 1/n), f* = 0. Default n = 5.
//
// Reference: More, Garbow, Hillstrom (1981), Problem 26.
template <typename Scalar = double>
struct trigonometric
{
    int n{5};

    static constexpr problem_class pclass = problem_class::unconstrained;

    [[nodiscard]] int dimension() const { return n; }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        return Eigen::VectorX<Scalar>::Constant(n, Scalar(1) / Scalar(n));
    }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        using std::cos;
        using std::sin;
        Scalar sum_cos{0};
        for(int j = 0; j < n; ++j)
            sum_cos += cos(x[j]);

        Scalar f{0};
        for(int i = 0; i < n; ++i)
        {
            Scalar ri = Scalar(n) - sum_cos
                        + Scalar(i + 1) * (Scalar(1) - cos(x[i]))
                        - sin(x[i]);
            f += ri * ri;
        }
        return f;
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        using std::cos;
        using std::sin;
        Scalar sum_cos{0};
        for(int j = 0; j < n; ++j)
            sum_cos += cos(x[j]);

        // Precompute residuals.
        Eigen::VectorX<Scalar> r(n);
        for(int i = 0; i < n; ++i)
        {
            r[i] = Scalar(n) - sum_cos
                   + Scalar(i + 1) * (Scalar(1) - cos(x[i]))
                   - sin(x[i]);
        }

        // sum of all residuals (for the shared cos derivative).
        Scalar r_sum{0};
        for(int i = 0; i < n; ++i)
            r_sum += r[i];

        g.resize(n);
        for(int i = 0; i < n; ++i)
        {
            Scalar si = sin(x[i]);
            Scalar ci = cos(x[i]);
            g[i] = Scalar(2) * (r_sum * si
                   + r[i] * (Scalar(i + 1) * si - ci));
        }
    }
};

// Wood function, 4D.
//
//   f(x) = 100*(x1 - x0^2)^2 + (1 - x0)^2 + 90*(x3 - x2^2)^2
//          + (1 - x2)^2 + 10.1*((x1 - 1)^2 + (x3 - 1)^2)
//          + 19.8*(x1 - 1)*(x3 - 1)
//
// x0 = (-3, -1, -3, -1), f* = 0 at (1, 1, 1, 1).
//
// Reference: More, Garbow, Hillstrom (1981), Problem 14.
template <typename Scalar = double>
struct wood
{
    static constexpr problem_class pclass = problem_class::unconstrained;

    [[nodiscard]] int dimension() const { return 4; }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        Eigen::VectorX<Scalar> x0(4);
        x0 << Scalar(-3), Scalar(-1), Scalar(-3), Scalar(-1);
        return x0;
    }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        Scalar t1 = x[1] - x[0] * x[0];
        Scalar t2 = Scalar(1) - x[0];
        Scalar t3 = x[3] - x[2] * x[2];
        Scalar t4 = Scalar(1) - x[2];
        Scalar t5 = x[1] - Scalar(1);
        Scalar t6 = x[3] - Scalar(1);
        return Scalar(100) * t1 * t1 + t2 * t2
               + Scalar(90) * t3 * t3 + t4 * t4
               + Scalar(10.1) * (t5 * t5 + t6 * t6)
               + Scalar(19.8) * t5 * t6;
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        Scalar t1 = x[1] - x[0] * x[0];
        Scalar t3 = x[3] - x[2] * x[2];
        Scalar t5 = x[1] - Scalar(1);
        Scalar t6 = x[3] - Scalar(1);
        g.resize(4);
        g[0] = Scalar(-400) * x[0] * t1 - Scalar(2) * (Scalar(1) - x[0]);
        g[1] = Scalar(200) * t1 + Scalar(20.2) * t5 + Scalar(19.8) * t6;
        g[2] = Scalar(-360) * x[2] * t3 - Scalar(2) * (Scalar(1) - x[2]);
        g[3] = Scalar(180) * t3 + Scalar(20.2) * t6 + Scalar(19.8) * t5;
    }
};

// Helical valley function, 3D.
//
//   f(x) = 100*((x2 - 10*theta(x0,x1))^2 + (sqrt(x0^2+x1^2) - 1)^2) + x2^2
//
// where theta = (1/(2*pi))*atan2(x1, x0).
//
// x0 = (-1, 0, 0), f* = 0 at (1, 0, 0).
//
// Reference: More, Garbow, Hillstrom (1981), Problem 7.
template <typename Scalar = double>
struct helical_valley
{
    static constexpr problem_class pclass = problem_class::unconstrained;

    [[nodiscard]] int dimension() const { return 3; }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        Eigen::VectorX<Scalar> x0(3);
        x0 << Scalar(-1), Scalar(0), Scalar(0);
        return x0;
    }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        using std::atan2;
        using std::sqrt;
        Scalar theta = atan2(x[1], x[0])
                       / (Scalar(2) * std::numbers::pi_v<Scalar>);
        Scalar r = sqrt(x[0] * x[0] + x[1] * x[1]);
        Scalar t1 = x[2] - Scalar(10) * theta;
        Scalar t2 = r - Scalar(1);
        return Scalar(100) * (t1 * t1 + t2 * t2) + x[2] * x[2];
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        using std::atan2;
        using std::sqrt;
        Scalar pi2 = Scalar(2) * std::numbers::pi_v<Scalar>;
        Scalar theta = atan2(x[1], x[0]) / pi2;
        Scalar r2 = x[0] * x[0] + x[1] * x[1];
        Scalar r = sqrt(r2);
        Scalar t1 = x[2] - Scalar(10) * theta;
        Scalar t2 = r - Scalar(1);
        // dtheta/dx0 = -x1 / (2*pi*r2), dtheta/dx1 = x0 / (2*pi*r2)
        Scalar k = Scalar(10) / (pi2 * r2);
        g.resize(3);
        g[0] = Scalar(200) * (t1 * k * x[1] + t2 * x[0] / r);
        g[1] = Scalar(200) * (-t1 * k * x[0] + t2 * x[1] / r);
        g[2] = Scalar(200) * t1 + Scalar(2) * x[2];
    }
};

// Penalty I function, variable-dimension.
//
//   f(x) = sum_i [sqrt(a)*(x_i - 1)]^2 + [sum_i x_i^2 - 0.25]^2
//
// where a = 1e-5.
//
// x0 = (1, 2, ..., n). For n = 4, f* approx 2.24997e-5.
// Default n = 4.
//
// Reference: More, Garbow, Hillstrom (1981), Problem 23.
template <typename Scalar = double>
struct penalty_i
{
    int n{4};

    static constexpr problem_class pclass = problem_class::unconstrained;

    [[nodiscard]] int dimension() const { return n; }

    [[nodiscard]] Scalar optimal_value() const
    {
        return Scalar(2.24997e-5);
    }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        Eigen::VectorX<Scalar> x0(n);
        for(int i = 0; i < n; ++i)
            x0[i] = Scalar(i + 1);
        return x0;
    }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        using std::sqrt;
        constexpr Scalar a = Scalar(1e-5);
        Scalar sqrt_a = sqrt(a);
        Scalar f{0};
        Scalar sum_sq{0};
        for(int i = 0; i < n; ++i)
        {
            Scalar ti = sqrt_a * (x[i] - Scalar(1));
            f += ti * ti;
            sum_sq += x[i] * x[i];
        }
        Scalar pen = sum_sq - Scalar(0.25);
        return f + pen * pen;
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        constexpr Scalar a = Scalar(1e-5);
        Scalar sum_sq{0};
        for(int i = 0; i < n; ++i)
            sum_sq += x[i] * x[i];
        Scalar pen = sum_sq - Scalar(0.25);

        g.resize(n);
        for(int i = 0; i < n; ++i)
        {
            g[i] = Scalar(2) * a * (x[i] - Scalar(1))
                   + Scalar(4) * pen * x[i];
        }
    }
};

// Variably dimensioned function.
//
//   f(x) = sum_i (x_i - 1)^2 + [sum_i i*(x_i - 1)]^2
//          + [sum_i i*(x_i - 1)]^4
//
// (1-indexed i = 1..n in formula, 0-indexed in code uses i+1.)
//
// x0_i = 1 - (i+1)/n, f* = 0 at (1, ..., 1). Default n = 5.
//
// Reference: More, Garbow, Hillstrom (1981), Problem 25.
template <typename Scalar = double>
struct variably_dimensioned
{
    int n{5};

    static constexpr problem_class pclass = problem_class::unconstrained;

    [[nodiscard]] int dimension() const { return n; }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        Eigen::VectorX<Scalar> x0(n);
        for(int i = 0; i < n; ++i)
            x0[i] = Scalar(1) - Scalar(i + 1) / Scalar(n);
        return x0;
    }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        Scalar sum_sq{0};
        Scalar s{0};
        for(int i = 0; i < n; ++i)
        {
            Scalar di = x[i] - Scalar(1);
            sum_sq += di * di;
            s += Scalar(i + 1) * di;
        }
        return sum_sq + s * s + s * s * s * s;
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        Scalar s{0};
        for(int i = 0; i < n; ++i)
            s += Scalar(i + 1) * (x[i] - Scalar(1));

        Scalar ds = Scalar(2) * s + Scalar(4) * s * s * s;
        g.resize(n);
        for(int i = 0; i < n; ++i)
        {
            g[i] = Scalar(2) * (x[i] - Scalar(1))
                   + Scalar(i + 1) * ds;
        }
    }
};

// Extended Rosenbrock function, variable-dimension (n must be even).
//
//   f(x) = sum_{i=0,2,4,...} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
//
// x0 = (-1.2, 1, -1.2, 1, ...), f* = 0 at (1, ..., 1). Default n = 4.
//
// Reference: More, Garbow, Hillstrom (1981), Problem 21.
template <typename Scalar = double>
struct extended_rosenbrock
{
    int n{4};

    static constexpr problem_class pclass = problem_class::unconstrained;

    [[nodiscard]] int dimension() const { return n; }

    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }

    [[nodiscard]] Eigen::VectorX<Scalar> initial_point() const
    {
        Eigen::VectorX<Scalar> x0(n);
        for(int i = 0; i < n; ++i)
            x0[i] = (i % 2 == 0) ? Scalar(-1.2) : Scalar(1);
        return x0;
    }

    [[nodiscard]] Scalar value(const Eigen::VectorX<Scalar>& x) const
    {
        Scalar f{0};
        for(int i = 0; i < n - 1; i += 2)
        {
            Scalar t1 = x[i + 1] - x[i] * x[i];
            Scalar t2 = Scalar(1) - x[i];
            f += Scalar(100) * t1 * t1 + t2 * t2;
        }
        return f;
    }

    void gradient(const Eigen::VectorX<Scalar>& x,
                  Eigen::VectorX<Scalar>& g) const
    {
        g.setZero(n);
        for(int i = 0; i < n - 1; i += 2)
        {
            Scalar t = x[i + 1] - x[i] * x[i];
            g[i] = Scalar(-400) * x[i] * t
                   - Scalar(2) * (Scalar(1) - x[i]);
            g[i + 1] = Scalar(200) * t;
        }
    }
};

}

#endif
