#ifndef HPP_GUARD_ARGMIN_TESTS_DEFECTS_MOVING_ASYMPTOTE_PROBLEMS_H
#define HPP_GUARD_ARGMIN_TESTS_DEFECTS_MOVING_ASYMPTOTE_PROBLEMS_H

#include "argmin/types.h"

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <cstdint>

namespace argmin::defects
{

// scale (x - center)^2 on [lower, upper] with `rows` copies of the row offset - x >= 0. f is NaN for x > nan_beyond.
struct scalar_bowl
{
    using vector = Eigen::Matrix<double, 1, 1>;

    static constexpr std::int32_t problem_dimension = 1;

    std::int32_t dimension() const { return 1; }

    std::int32_t num_equality() const { return 0; }

    std::int32_t num_inequality() const { return rows; }

    double value(const vector& x) const
    {
        if(x[0] > nan_beyond)
            return std::numeric_limits<double>::quiet_NaN();
        return scale * (x[0] - center) * (x[0] - center);
    }

    void gradient(const vector& x, vector& g) const { g[0] = 2.0 * scale * (x[0] - center); }

    template <typename Values>
    void constraints(const vector& x, Values& out) const
    {
        out.setConstant(offset - x[0]);
    }

    template <typename Jacobian>
    void constraint_jacobian(const vector&, Jacobian& J) const
    {
        J.setConstant(-1.0);
    }

    vector lower_bounds() const { return vector::Constant(lower); }

    vector upper_bounds() const { return vector::Constant(upper); }

    double scale;
    double center;
    std::int32_t rows;
    double offset;
    double lower;
    double upper;
    double nan_beyond;
};

// 10 (x1 - x0^2)^2 + (1 - x0)^2 on [0, 10]^2 with the active row 2.5 - x0 - x1 >= 0.
struct rosenbrock_in_box
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_equality() const { return 0; }

    std::int32_t num_inequality() const { return 1; }

    double value(const Eigen::VectorXd& x) const
    {
        const double a = x[1] - x[0] * x[0];
        return 10.0 * a * a + (1.0 - x[0]) * (1.0 - x[0]);
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g.resize(2);
        g[0] = -40.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
        g[1] = 20.0 * (x[1] - x[0] * x[0]);
    }

    template <typename Values>
    void constraints(const Eigen::VectorXd& x, Values& out) const
    {
        out[0] = 2.5 - x[0] - x[1];
    }

    template <typename Jacobian>
    void constraint_jacobian(const Eigen::VectorXd&, Jacobian& J) const
    {
        J(0, 0) = -1.0;
        J(0, 1) = -1.0;
    }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd::Zero(2); }

    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd::Constant(2, 10.0); }
};

// (x0 - 1)^2 + 3 (x1 - 2)^4 on [0, 3]^2 with the row 2 - x0 - x1 >= 0, posed in y = k x.
struct scaled_twin
{
    static constexpr std::int32_t problem_dimension = 2;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_equality() const { return 0; }

    std::int32_t num_inequality() const { return 1; }

    double value(const Eigen::Vector2d& y) const { return (y[0] / k - 1.0) * (y[0] / k - 1.0) + 3.0 * std::pow(y[1] / k - 2.0, 4); }

    void gradient(const Eigen::Vector2d& y, Eigen::Vector2d& g) const
    {
        g[0] = 2.0 * (y[0] / k - 1.0) / k;
        g[1] = 12.0 * std::pow(y[1] / k - 2.0, 3) / k;
    }

    template <typename Values>
    void constraints(const Eigen::Vector2d& y, Values& out) const
    {
        out[0] = 2.0 - (y[0] + y[1]) / k;
    }

    template <typename Jacobian>
    void constraint_jacobian(const Eigen::Vector2d&, Jacobian& J) const
    {
        J.setConstant(-1.0 / k);
    }

    Eigen::Vector2d lower_bounds() const { return Eigen::Vector2d::Zero(); }

    Eigen::Vector2d upper_bounds() const { return Eigen::Vector2d::Constant(3.0 * k); }

    double k;
};

}

#endif
