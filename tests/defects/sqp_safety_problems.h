#ifndef HPP_GUARD_ARGMIN_TESTS_DEFECTS_SQP_SAFETY_PROBLEMS_H
#define HPP_GUARD_ARGMIN_TESTS_DEFECTS_SQP_SAFETY_PROBLEMS_H

#include "argmin/types.h"

#include <Eigen/Core>

#include <cstdint>
#include <limits>

namespace argmin::defects
{

// f = 0.5 x^T Q x + c^T x on rows E x - e = 0 and G x - h >= 0, with no box. f is NaN for x0 > nan_beyond while
// the gradient keeps its analytic value, so only the objective comparison sees the NaN.
struct row_quadratic
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t dimension() const { return static_cast<std::int32_t>(c.size()); }

    std::int32_t num_equality() const { return static_cast<std::int32_t>(E.rows()); }

    std::int32_t num_inequality() const { return static_cast<std::int32_t>(G.rows()); }

    double value(const Eigen::VectorXd& x) const
    {
        if(x[0] > nan_beyond)
            return std::numeric_limits<double>::quiet_NaN();
        return 0.5 * x.dot(Q * x) + c.dot(x);
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const { g = Q * x + c; }

    template <typename Values>
    void constraints(const Eigen::VectorXd& x, Values& out) const
    {
        out.head(E.rows()) = E * x - e;
        out.tail(G.rows()) = G * x - h;
    }

    template <typename Jacobian>
    void constraint_jacobian(const Eigen::VectorXd&, Jacobian& J) const
    {
        J.topRows(E.rows()) = E;
        J.bottomRows(G.rows()) = G;
    }

    Eigen::MatrixXd Q;
    Eigen::VectorXd c;
    Eigen::MatrixXd E;
    Eigen::VectorXd e;
    Eigen::MatrixXd G;
    Eigen::VectorXd h;
    double nan_beyond;
};

// (x0 - 3)^2 + x1^2 up to a constant, minimized at (3, 0).
inline row_quadratic shifted_bowl(const Eigen::MatrixXd& E, const Eigen::VectorXd& e, const Eigen::MatrixXd& G, const Eigen::VectorXd& h, double nan_beyond)
{
    return {2.0 * Eigen::MatrixXd::Identity(2, 2), Eigen::Vector2d(-6.0, 0.0), E, e, G, h, nan_beyond};
}

inline constexpr double nowhere = std::numeric_limits<double>::infinity();

}

#endif
