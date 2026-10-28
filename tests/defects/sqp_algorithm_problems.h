#ifndef HPP_GUARD_ARGMIN_TESTS_DEFECTS_SQP_ALGORITHM_PROBLEMS_H
#define HPP_GUARD_ARGMIN_TESTS_DEFECTS_SQP_ALGORITHM_PROBLEMS_H

#include "sqp_safety_problems.h"

#include "argmin/types.h"

#include <Eigen/Core>

#include <cstdint>

namespace argmin::defects
{

struct boxed_row_quadratic : row_quadratic
{
    Eigen::VectorXd lower_bounds() const { return lower; }

    Eigen::VectorXd upper_bounds() const { return upper; }

    Eigen::VectorXd lower;
    Eigen::VectorXd upper;
};

// Reports the negated gradient, so every descent direction the solver computes is an ascent direction.
struct flipped_gradient : row_quadratic
{
    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        row_quadratic::gradient(x, g);
        g = -g;
    }
};

// Rosenbrock with the inactive linear row 10 - x0 - x1 >= 0.
struct rosenbrock_inside_row
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_equality() const { return 0; }

    std::int32_t num_inequality() const { return 1; }

    double value(const Eigen::VectorXd& x) const
    {
        const double a = x[1] - x[0] * x[0];
        return 100.0 * a * a + (1.0 - x[0]) * (1.0 - x[0]);
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g.resize(2);
        g[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
        g[1] = 200.0 * (x[1] - x[0] * x[0]);
    }

    template <typename Values>
    void constraints(const Eigen::VectorXd& x, Values& out) const
    {
        out[0] = 10.0 - x[0] - x[1];
    }

    template <typename Jacobian>
    void constraint_jacobian(const Eigen::VectorXd&, Jacobian& J) const
    {
        J(0, 0) = -1.0;
        J(0, 1) = -1.0;
    }
};

}

#endif
