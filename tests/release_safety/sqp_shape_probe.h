#ifndef HPP_GUARD_ARGMIN_TESTS_RELEASE_SAFETY_SQP_SHAPE_PROBE_H
#define HPP_GUARD_ARGMIN_TESTS_RELEASE_SAFETY_SQP_SHAPE_PROBE_H

#include "eigen_trap.h"

#include "argmin/types.h"

#include <Eigen/Core>

#include <cstdint>

namespace argmin::release_safety
{

using vector2 = Eigen::Vector2d;

// f = 0.5 |x - (1, 2)|^2 on the first `rows` of x0 + 10 >= 0, x1 + 10 >= 0 and 10 - x0 - x1 >= 0. Every row and the
// optional [-5, 5]^2 box are inactive at the minimizer, so no converging solve sits on a degenerate vertex.
template <std::int32_t Cap, bool Boxed>
struct capped_rows
{
    static constexpr std::int32_t problem_dimension = 2;
    static constexpr std::int32_t constraint_count = Cap;

    std::int32_t rows;
    mutable std::int32_t evaluations;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_equality() const { return 0; }

    std::int32_t num_inequality() const { return rows; }

    double value(const vector2& x) const
    {
        ++evaluations;
        return 0.5 * (x - vector2(1.0, 2.0)).squaredNorm();
    }

    void gradient(const vector2& x, vector2& g) const
    {
        ++evaluations;
        g = x - vector2(1.0, 2.0);
    }

    template <typename Values>
    void constraints(const vector2& x, Values& c) const
    {
        ++evaluations;
        c = normals().topRows(rows) * x + Eigen::Vector3d::Constant(10.0).head(rows);
    }

    template <typename Jacobian>
    void constraint_jacobian(const vector2&, Jacobian& J) const
    {
        ++evaluations;
        J = normals().topRows(rows);
    }

    vector2 lower_bounds() const
        requires Boxed
    {
        return vector2::Constant(-5.0);
    }

    vector2 upper_bounds() const
        requires Boxed
    {
        return vector2::Constant(5.0);
    }

    static Eigen::Matrix<double, 3, 2> normals() { return (Eigen::Matrix<double, 3, 2>() << 1.0, 0.0, 0.0, 1.0, -1.0, -1.0).finished(); }
};

// f = 0.5 x^2 on one variable with the equality rows k (x - 1) = 0 for k = 1 .. rows, all satisfied at x = 1.
struct scaled_lines
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t rows;
    mutable std::int32_t evaluations;

    std::int32_t dimension() const { return 1; }

    std::int32_t num_equality() const { return rows; }

    std::int32_t num_inequality() const { return 0; }

    double value(const Eigen::VectorXd& x) const
    {
        ++evaluations;
        return 0.5 * x.squaredNorm();
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        ++evaluations;
        g = x;
    }

    template <typename Values>
    void constraints(const Eigen::VectorXd& x, Values& c) const
    {
        ++evaluations;
        c = slopes() * (x[0] - 1.0);
    }

    template <typename Jacobian>
    void constraint_jacobian(const Eigen::VectorXd&, Jacobian& J) const
    {
        ++evaluations;
        J = slopes();
    }

    Eigen::VectorXd slopes() const { return Eigen::VectorXd::LinSpaced(rows, 1.0, static_cast<double>(rows)); }
};

inline const vector2 inside(0.5, 0.5);

inline Eigen::VectorXd three()
{
    return Eigen::VectorXd::Constant(1, 3.0);
}

}

#endif
