#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_LINEAR_QUADRATIC_PROBLEMS_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_LINEAR_QUADRATIC_PROBLEMS_H

#include "argmin/types.h"

#include <Eigen/Core>

#include <cstdint>

namespace argmin::cases
{

// f = 0.5 x^T Q x + c^T x with Q symmetric positive definite.
struct convex_quadratic
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t dimension() const { return static_cast<std::int32_t>(c.size()); }

    double value(const Eigen::VectorXd& x) const { return 0.5 * x.dot(Q * x) + c.dot(x); }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const { g = Q * x + c; }

    Eigen::VectorXd initial_point() const { return start; }

    Eigen::MatrixXd Q;
    Eigen::VectorXd c;
    Eigen::VectorXd start;
};

struct boxed_quadratic : convex_quadratic
{
    Eigen::VectorXd lower_bounds() const { return lower; }

    Eigen::VectorXd upper_bounds() const { return upper; }

    Eigen::VectorXd lower;
    Eigen::VectorXd upper;
};

// Rows are E x = e followed by G x - h >= 0. A problem with zero rows is the bound class of the policies
// that require the constrained interface.
struct linearly_constrained_quadratic : boxed_quadratic
{
    std::int32_t num_equality() const { return static_cast<std::int32_t>(E.rows()); }

    std::int32_t num_inequality() const { return static_cast<std::int32_t>(G.rows()); }

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

    Eigen::MatrixXd E;
    Eigen::VectorXd e;
    Eigen::MatrixXd G;
    Eigen::VectorXd h;
};

// r = M x - b, with a box that is infinite when the problem is unbounded.
struct linear_least_squares
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t dimension() const { return static_cast<std::int32_t>(M.cols()); }

    std::int32_t num_residuals() const { return static_cast<std::int32_t>(M.rows()); }

    double value(const Eigen::VectorXd& x) const { return 0.5 * (M * x - b).squaredNorm(); }

    void residuals(const Eigen::VectorXd& x, Eigen::VectorXd& r) const { r = M * x - b; }

    void jacobian(const Eigen::VectorXd&, Eigen::MatrixXd& J) const { J = M; }

    Eigen::VectorXd initial_point() const { return start; }

    Eigen::VectorXd lower_bounds() const { return lower; }

    Eigen::VectorXd upper_bounds() const { return upper; }

    Eigen::MatrixXd M;
    Eigen::VectorXd b;
    Eigen::VectorXd start;
    Eigen::VectorXd lower;
    Eigen::VectorXd upper;
};

// Levenberg-Marquardt never reads a box, so its problems hide the base's.
struct unboxed_least_squares
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t dimension() const { return p.dimension(); }

    std::int32_t num_residuals() const { return p.num_residuals(); }

    double value(const Eigen::VectorXd& x) const { return p.value(x); }

    void residuals(const Eigen::VectorXd& x, Eigen::VectorXd& r) const { p.residuals(x, r); }

    void jacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const { p.jacobian(x, J); }

    Eigen::VectorXd initial_point() const { return p.start; }

    linear_least_squares p;
};

}

#endif
