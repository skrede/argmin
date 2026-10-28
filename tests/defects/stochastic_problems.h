#ifndef HPP_GUARD_ARGMIN_TESTS_DEFECTS_STOCHASTIC_PROBLEMS_H
#define HPP_GUARD_ARGMIN_TESTS_DEFECTS_STOCHASTIC_PROBLEMS_H

#include "argmin/types.h"

#include <Eigen/Core>

#include <limits>
#include <cstdint>
#include <algorithm>

namespace argmin::defects
{

// |x|^2 on [lower, upper]; counts every objective call and every call outside the box.
struct counting_box
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t dimension() const { return static_cast<std::int32_t>(lower.size()); }

    double value(const Eigen::VectorXd& x) const
    {
        ++calls;
        if((x.array() < lower.array()).any() || (x.array() > upper.array()).any())
            ++calls_outside_box;
        return x.squaredNorm();
    }

    Eigen::VectorXd lower_bounds() const { return lower; }

    Eigen::VectorXd upper_bounds() const { return upper; }

    Eigen::VectorXd lower;
    Eigen::VectorXd upper;
    mutable std::int64_t calls;
    mutable std::int64_t calls_outside_box;
};

// f(x) = x0 on [0, 1]; counts every call outside the box.
struct linear_unit_interval
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t dimension() const { return 1; }

    double value(const Eigen::VectorXd& x) const
    {
        if(x[0] < 0.0 || x[0] > 1.0)
            ++calls_outside_box;
        return x[0];
    }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd::Zero(1); }

    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd::Ones(1); }

    mutable std::int64_t calls_outside_box;
};

// NaN at every point of [-1, 1]^2: every pair of values is incomparable, so ranking them is still well defined.
struct nan_everywhere
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t dimension() const { return 2; }

    double value(const Eigen::VectorXd&) const { return std::numeric_limits<double>::quiet_NaN(); }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd::Constant(2, -1.0); }

    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd::Constant(2, 1.0); }
};

// |x|^2 on [-10, 10]^2 with the row 1 - x0 - x1 >= 0.
struct sphere_under_row
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_equality() const { return 0; }

    std::int32_t num_inequality() const { return 1; }

    double value(const Eigen::VectorXd& x) const { return x.squaredNorm(); }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = 1.0 - x[0] - x[1];
    }

    double violation(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd c;
        constraints(x, c);
        const double slack = std::max(0.0, -c[0]);
        return slack * slack;
    }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd::Constant(2, -10.0); }

    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd::Constant(2, 10.0); }
};

}

#endif
