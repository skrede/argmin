#ifndef HPP_GUARD_ARGMIN_TESTS_DEFECTS_DERIVATIVE_FREE_PROBLEMS_H
#define HPP_GUARD_ARGMIN_TESTS_DEFECTS_DERIVATIVE_FREE_PROBLEMS_H

#include "argmin/types.h"

#include <Eigen/Core>

#include <cstdint>

namespace argmin::defects
{

// scale * |x - (2, 1)|^2 under scale * (1 - x0 - x1) >= 0 and scale * (x0 - x1^2) >= 0 on [-10, 10]^2.
struct scaled_two_row
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_equality() const { return 0; }

    std::int32_t num_inequality() const { return 2; }

    double value(const Eigen::VectorXd& x) const { return scale * ((x[0] - 2.0) * (x[0] - 2.0) + (x[1] - 1.0) * (x[1] - 1.0)); }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(2);
        c[0] = scale * (1.0 - x[0] - x[1]);
        c[1] = scale * (x[0] - x[1] * x[1]);
    }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd::Constant(2, -10.0); }

    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd::Constant(2, 10.0); }

    double scale;
};

// a (x0 - c0)^2 + b (x1 - c1)^2 + cross (x0 - c0)(x1 - c1) on [lower, upper]^2.
struct box_quadratic
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        const double u = x[0] - center[0];
        const double v = x[1] - center[1];
        return a * u * u + b * v * v + cross * u * v;
    }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd::Constant(2, lower); }

    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd::Constant(2, upper); }

    Eigen::Vector2d center;
    double a;
    double b;
    double cross;
    double lower;
    double upper;
};

}

#endif
