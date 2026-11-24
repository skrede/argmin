#ifndef HPP_GUARD_ARGMIN_TESTS_DEFECTS_NON_FINITE_LIMIT_PROBLEMS_H
#define HPP_GUARD_ARGMIN_TESTS_DEFECTS_NON_FINITE_LIMIT_PROBLEMS_H

#include "moving_asymptote_problems.h"

#include <Eigen/Core>

#include <limits>
#include <cstdint>

namespace argmin::defects
{

// r(x) = x - 2 on (-1, 1], NaN for x > 1 and x + 3 - 1e-17 for x <= -1, with J = 1 on [-10, 10]: every Gauss-Newton
// trial from x = 1 heads into the NaN region, and the Gauss-Newton step at x = -3 (1e-17) lies below x's resolution.
struct stepped_wall_residual
{
    static constexpr std::int32_t problem_dimension = 1;

    static double residual(double x) { return x > 1.0 ? std::numeric_limits<double>::quiet_NaN() : x > -1.0 ? x - 2.0 : x + 3.0 - 1e-17; }

    std::int32_t dimension() const { return 1; }

    std::int32_t num_residuals() const { return 1; }

    double value(const Eigen::Vector<double, 1>& x) const { return 0.5 * residual(x[0]) * residual(x[0]); }

    void residuals(const Eigen::Vector<double, 1>& x, Eigen::VectorXd& r) const { r.setConstant(1, residual(x[0])); }

    void jacobian(const Eigen::Vector<double, 1>&, Eigen::MatrixXd& J) const { J = Eigen::MatrixXd::Ones(1, 1); }

    Eigen::Vector<double, 1> lower_bounds() const { return Eigen::Vector<double, 1>::Constant(-10.0); }

    Eigen::Vector<double, 1> upper_bounds() const { return Eigen::Vector<double, 1>::Constant(10.0); }
};

// r(x) = (x0 + 2 x1 + 1, x1 + 1, x1 + 1) on [0, 10]^2, the first residual NaN for x0 > 0. The origin is stationary on
// the box (g = (1, 4)), yet the lightly damped Gauss-Newton step (1, -1) projects into the NaN region; from a damping
// of 1/3 the projected direction is zero.
struct blocked_corner_residual
{
    static constexpr std::int32_t problem_dimension = 2;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_residuals() const { return 3; }

    double value(const Eigen::Vector2d& x) const
    {
        Eigen::VectorXd r(3);
        residuals(x, r);
        return 0.5 * r.squaredNorm();
    }

    void residuals(const Eigen::Vector2d& x, Eigen::VectorXd& r) const
    {
        r.resize(3);
        r << (x[0] > 0.0 ? std::numeric_limits<double>::quiet_NaN() : x[0] + 2.0 * x[1] + 1.0), x[1] + 1.0, x[1] + 1.0;
    }

    void jacobian(const Eigen::Vector2d&, Eigen::MatrixXd& J) const
    {
        J.resize(3, 2);
        J << 1.0, 2.0, 0.0, 1.0, 0.0, 1.0;
    }

    Eigen::Vector2d lower_bounds() const { return Eigen::Vector2d::Zero(); }

    Eigen::Vector2d upper_bounds() const { return Eigen::Vector2d::Constant(10.0); }
};

// cliff_bowl whose gradient is NaN beyond nan_beyond too: a trial there improves on f yet fails conservativity.
struct gradient_cliff_bowl : cliff_bowl
{
    void gradient(const vector& x, vector& g) const
    {
        scalar_bowl::gradient(x, g);
        if(x[0] > nan_beyond)
            g[0] = std::numeric_limits<double>::quiet_NaN();
    }
};

// scalar_bowl whose gradient is NaN on its glitch_call-th evaluation only.
struct glitch_bowl : scalar_bowl
{
    void gradient(const vector& x, vector& g) const
    {
        scalar_bowl::gradient(x, g);
        if(++gradient_calls == glitch_call)
            g[0] = std::numeric_limits<double>::quiet_NaN();
    }

    std::int32_t glitch_call;
    mutable std::int32_t gradient_calls;
};

}

#endif
