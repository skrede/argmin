#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_CASE_PROBLEMS_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_CASE_PROBLEMS_H

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <cstdint>

namespace argmin::cases
{

struct rosenbrock_ls2
{
    static constexpr std::int32_t problem_dimension = 2;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_residuals() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        Eigen::VectorXd r(2);
        residuals(x, r);
        return 0.5 * r.squaredNorm();
    }

    void residuals(const Eigen::Vector<double, 2>& x, Eigen::VectorXd& r) const
    {
        r[0] = 1.0 - x[0];
        r[1] = std::sqrt(5.0) * (x[1] - x[0] * x[0]);
    }

    void jacobian(const Eigen::Vector<double, 2>& x, Eigen::MatrixXd& J) const
    {
        J(0, 0) = -1.0;
        J(0, 1) = 0.0;
        J(1, 0) = -2.0 * std::sqrt(5.0) * x[0];
        J(1, 1) = std::sqrt(5.0);
    }

    Eigen::Vector<double, 2> initial_point() const { return Eigen::Vector<double, 2>(-1.0, 1.0); }
};

// x0 <= 0.5 cuts off the minimizer (1, 1), so the solution (0.5, 0.25) holds the upper bound on x0
// active with a strictly positive multiplier.
struct bounded_rosenbrock_ls2 : rosenbrock_ls2
{
    Eigen::Vector<double, 2> lower_bounds() const { return Eigen::Vector<double, 2>(-2.0, -2.0); }

    Eigen::Vector<double, 2> upper_bounds() const { return Eigen::Vector<double, 2>(0.5, 2.0); }
};

// The projected Gauss-Newton policies read the box unconditionally, so an unconstrained case hands
// them an infinite one.
template <typename Problem>
struct infinite_box : Problem
{
    using vector = Eigen::Vector<double, Problem::problem_dimension>;

    vector lower_bounds() const { return vector::Constant(-std::numeric_limits<double>::infinity()); }

    vector upper_bounds() const { return vector::Constant(std::numeric_limits<double>::infinity()); }
};

// The policies that require the constrained interface take a bound-only case as one with zero
// constraint rows. The count is left dynamic because a fixed zero row count does not compile for the
// moving-asymptote family.
template <typename Problem>
struct no_constraint_rows : Problem
{
    using vector = Eigen::Vector<double, Problem::problem_dimension>;

    static constexpr std::int32_t constraint_count = Eigen::Dynamic;

    std::int32_t num_equality() const { return 0; }

    std::int32_t num_inequality() const { return 0; }

    void constraints(const vector&, Eigen::VectorXd&) const {}

    void constraint_jacobian(const vector&, Eigen::MatrixXd&) const {}
};

}

#endif
