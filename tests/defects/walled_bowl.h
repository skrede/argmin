#ifndef HPP_GUARD_ARGMIN_TESTS_DEFECTS_WALLED_BOWL_H
#define HPP_GUARD_ARGMIN_TESTS_DEFECTS_WALLED_BOWL_H

#include "argmin/types.h"

#include <Eigen/Core>

#include <array>
#include <limits>
#include <cstddef>
#include <cstdint>

namespace argmin::defects
{

enum class bowl_callback : std::uint8_t
{
    none,
    value,
    gradient,
    constraints,
    constraint_jacobian
};

// (x0 - 3)^2 + (x1 - 3)^2 on [-5, 5]^2 with `rows` copies of the inactive row 10 - x0 - x1 >= 0. Past x0 = wall the
// callback named by walled is NaN, or every callback when it names none. The callback named by glitch returns bad from
// its bad_from-th call on, or at that call only when once is set.
struct walled_bowl
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_equality() const { return 0; }

    std::int32_t num_inequality() const { return rows; }

    double value(const Eigen::VectorXd& x) const
    {
        const double f = (x[0] - 3.0) * (x[0] - 3.0) + (x[1] - 3.0) * (x[1] - 3.0);
        return spoiled(bowl_callback::value, x) ? bad_or_nan(bowl_callback::value) : f;
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g.resize(2);
        g << 2.0 * (x[0] - 3.0), 2.0 * (x[1] - 3.0);
        if(spoiled(bowl_callback::gradient, x))
            g.setConstant(bad_or_nan(bowl_callback::gradient));
    }

    template <typename Values>
    void constraints(const Eigen::VectorXd& x, Values& c) const
    {
        c.resize(rows);
        c.setConstant(10.0 - x[0] - x[1]);
        if(spoiled(bowl_callback::constraints, x))
            c.setConstant(bad_or_nan(bowl_callback::constraints));
    }

    template <typename Jacobian>
    void constraint_jacobian(const Eigen::VectorXd& x, Jacobian& J) const
    {
        J.resize(rows, 2);
        J.setConstant(-1.0);
        if(spoiled(bowl_callback::constraint_jacobian, x))
            J.setConstant(bad_or_nan(bowl_callback::constraint_jacobian));
    }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd::Constant(2, -5.0); }

    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd::Constant(2, 5.0); }

    std::int32_t rows;
    double wall;
    bowl_callback walled;
    bowl_callback glitch;
    std::int32_t bad_from;
    bool once;
    double bad;
    mutable std::array<std::int32_t, 5> calls;

private:
    bool spoiled(bowl_callback target, const Eigen::VectorXd& x) const
    {
        const std::int32_t call = ++calls[static_cast<std::size_t>(target)];
        if(target == glitch && (once ? call == bad_from : call >= bad_from))
            return true;
        return x[0] > wall && (walled == bowl_callback::none || walled == target);
    }

    double bad_or_nan(bowl_callback target) const { return target == glitch ? bad : std::numeric_limits<double>::quiet_NaN(); }
};

inline walled_bowl wall_at(double wall, std::int32_t rows, bowl_callback walled = bowl_callback::none)
{
    return walled_bowl{rows, wall, walled, bowl_callback::none, 0, false, 0.0, {}};
}

inline walled_bowl glitching(bowl_callback glitch, std::int32_t bad_from, bool once, double bad, std::int32_t rows)
{
    return walled_bowl{rows, std::numeric_limits<double>::infinity(), bowl_callback::none, glitch, bad_from, once, bad, {}};
}

}

#endif
