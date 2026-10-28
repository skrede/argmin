#ifndef HPP_GUARD_ARGMIN_TESTS_DEFECTS_BOUND_LS_AL_PROBLEMS_H
#define HPP_GUARD_ARGMIN_TESTS_DEFECTS_BOUND_LS_AL_PROBLEMS_H

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <cstdint>
#include <optional>

namespace argmin::defects
{

inline constexpr double unbounded = std::numeric_limits<double>::infinity();

// 0.5 |x - center|^2 on the box [lower, upper]; records the first point value() is called at.
struct box_bowl
{
    static constexpr std::int32_t problem_dimension = 2;

    std::int32_t dimension() const { return 2; }

    double value(const Eigen::Vector2d& x) const
    {
        if(!first_value_at)
            first_value_at = x;
        return 0.5 * (x - center).squaredNorm();
    }

    void gradient(const Eigen::Vector2d& x, Eigen::Vector2d& g) const { g = x - center; }

    Eigen::Vector2d lower_bounds() const { return lower; }

    Eigen::Vector2d upper_bounds() const { return upper; }

    Eigen::Vector2d center;
    Eigen::Vector2d lower;
    Eigen::Vector2d upper;
    mutable std::optional<Eigen::Vector2d> first_value_at;
};

// Rosenbrock; counts gradient calls made at the same point as the gradient call before them.
struct repeat_counting_rosenbrock
{
    static constexpr std::int32_t problem_dimension = 2;

    std::int32_t dimension() const { return 2; }

    double value(const Eigen::Vector2d& x) const
    {
        const double a = x[1] - x[0] * x[0];
        return 100.0 * a * a + (1.0 - x[0]) * (1.0 - x[0]);
    }

    void gradient(const Eigen::Vector2d& x, Eigen::Vector2d& g) const
    {
        if(last_gradient_at && *last_gradient_at == x)
            ++repeated_gradient_calls;
        last_gradient_at = x;
        g[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
        g[1] = 200.0 * (x[1] - x[0] * x[0]);
    }

    mutable std::optional<Eigen::Vector2d> last_gradient_at;
    mutable std::int32_t repeated_gradient_calls;
};

// r(x) = sqrt(x) - 0.1: the first undamped Levenberg-Marquardt trial from x = 1 lands at x < 0, an exact NaN.
struct sqrt_residual
{
    static constexpr std::int32_t problem_dimension = 1;

    std::int32_t dimension() const { return 1; }

    std::int32_t num_residuals() const { return 1; }

    double value(const Eigen::Vector<double, 1>& x) const
    {
        const double r = std::sqrt(x[0]) - 0.1;
        return 0.5 * r * r;
    }

    void residuals(const Eigen::Vector<double, 1>& x, Eigen::VectorXd& r) const
    {
        r.resize(1);
        r[0] = std::sqrt(x[0]) - 0.1;
    }

    void jacobian(const Eigen::Vector<double, 1>& x, Eigen::MatrixXd& J) const
    {
        J.resize(1, 1);
        J(0, 0) = 0.5 / std::sqrt(x[0]);
    }
};

// r(x) = J x + (0.1, 0.8) with J = [[2, 1], [1, 0]] on x0 >= 0; counts residual calls.
struct counting_linear_residual
{
    static constexpr std::int32_t problem_dimension = 2;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_residuals() const { return 2; }

    static Eigen::Matrix2d J() { return (Eigen::Matrix2d() << 2.0, 1.0, 1.0, 0.0).finished(); }

    double value(const Eigen::Vector2d& x) const { return 0.5 * (J() * x + Eigen::Vector2d(0.1, 0.8)).squaredNorm(); }

    void residuals(const Eigen::Vector2d& x, Eigen::VectorXd& r) const
    {
        ++residual_calls;
        r = J() * x + Eigen::Vector2d(0.1, 0.8);
    }

    void jacobian(const Eigen::Vector2d&, Eigen::MatrixXd& out) const { out = J(); }

    Eigen::Vector2d lower_bounds() const { return {0.0, -unbounded}; }

    Eigen::Vector2d upper_bounds() const { return Eigen::Vector2d::Constant(unbounded); }

    mutable std::int32_t residual_calls;
};

// r(x) = (sqrt(x0) + 1, x1 - 0.5) on [0, 2] x [-10, 10] with no jacobian(): the finite-difference leg at x0 = 0
// evaluates sqrt of a negative number.
struct sqrt_box_residual
{
    static constexpr std::int32_t problem_dimension = 2;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_residuals() const { return 2; }

    double value(const Eigen::Vector2d& x) const
    {
        const double a = std::sqrt(x[0]) + 1.0;
        const double b = x[1] - 0.5;
        return 0.5 * (a * a + b * b);
    }

    void residuals(const Eigen::Vector2d& x, Eigen::VectorXd& r) const
    {
        r.resize(2);
        r[0] = std::sqrt(x[0]) + 1.0;
        r[1] = x[1] - 0.5;
    }

    Eigen::Vector2d lower_bounds() const { return {0.0, -10.0}; }

    Eigen::Vector2d upper_bounds() const { return {2.0, 10.0}; }
};

}

#endif
