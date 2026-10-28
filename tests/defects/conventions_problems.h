#ifndef HPP_GUARD_ARGMIN_TESTS_DEFECTS_CONVENTIONS_PROBLEMS_H
#define HPP_GUARD_ARGMIN_TESTS_DEFECTS_CONVENTIONS_PROBLEMS_H

#include "argmin/types.h"

#include <Eigen/Core>

#include <cstdint>

namespace argmin::defects
{

// |x - (1, 2)|^2 on the box [-5, 0.5] x [-5, 5]; counts every objective evaluation.
struct counting_box_bowl
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        ++calls;
        return (x - Eigen::Vector2d(1.0, 2.0)).squaredNorm();
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const { g = 2.0 * (x - Eigen::Vector2d(1.0, 2.0)); }

    Eigen::VectorXd lower_bounds() const { return Eigen::Vector2d(-5.0, -5.0); }

    Eigen::VectorXd upper_bounds() const { return Eigen::Vector2d(0.5, 5.0); }

    mutable std::int64_t calls;
};

// |x - (1, 2)|^2 under the row 2 - x0 - x1 >= 0 in the box [-5, 5]^2; counts every objective evaluation.
struct counting_row_bowl
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_equality() const { return 0; }

    std::int32_t num_inequality() const { return 1; }

    double value(const Eigen::VectorXd& x) const
    {
        ++calls;
        return (x - Eigen::Vector2d(1.0, 2.0)).squaredNorm();
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const { g = 2.0 * (x - Eigen::Vector2d(1.0, 2.0)); }

    template <typename Values>
    void constraints(const Eigen::VectorXd& x, Values& out) const
    {
        out[0] = 2.0 - x[0] - x[1];
    }

    template <typename Jacobian>
    void constraint_jacobian(const Eigen::VectorXd&, Jacobian& J) const
    {
        J(0, 0) = -1.0;
        J(0, 1) = -1.0;
    }

    Eigen::VectorXd lower_bounds() const { return Eigen::Vector2d::Constant(-5.0); }

    Eigen::VectorXd upper_bounds() const { return Eigen::Vector2d::Constant(5.0); }

    mutable std::int64_t calls;
};

// r(x) = x - (1, 2) on the box [-5, 0.5] x [-5, 5]; counts every residual and objective evaluation.
struct counting_box_residual
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_residuals() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        ++calls;
        return 0.5 * (x - Eigen::Vector2d(1.0, 2.0)).squaredNorm();
    }

    void residuals(const Eigen::VectorXd& x, Eigen::VectorXd& r) const
    {
        ++calls;
        r = x - Eigen::Vector2d(1.0, 2.0);
    }

    void jacobian(const Eigen::VectorXd&, Eigen::MatrixXd& J) const { J = Eigen::Matrix2d::Identity(); }

    Eigen::VectorXd lower_bounds() const { return Eigen::Vector2d(-5.0, -5.0); }

    Eigen::VectorXd upper_bounds() const { return Eigen::Vector2d(0.5, 5.0); }

    mutable std::int64_t calls;
};

}

#endif
