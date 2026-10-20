#ifndef HPP_GUARD_ARGMIN_TESTS_UNIT_FLOAT_FLOAT_FIXTURES_H
#define HPP_GUARD_ARGMIN_TESTS_UNIT_FLOAT_FLOAT_FIXTURES_H

#include "argmin/types.h"

#include <Eigen/Core>

#include <cmath>
#include <limits>

namespace argmin::float_fixtures
{

// Rosenbrock (b = 100) on a box whose upper bound on x0 cuts off the unconstrained minimizer (1, 1),
// so the constrained minimizer (0.5, 0.25) has x0 on that bound.
template <scalar S>
struct active_box_rosenbrock
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }

    S value(const vector<S, 2>& x) const
    {
        const S t1 = S(1) - x[0];
        const S t2 = x[1] - x[0] * x[0];
        return t1 * t1 + S(100) * t2 * t2;
    }

    void gradient(const vector<S, 2>& x, vector<S, 2>& g) const
    {
        const S t2 = x[1] - x[0] * x[0];
        g[0] = S(-2) * (S(1) - x[0]) - S(400) * x[0] * t2;
        g[1] = S(200) * t2;
    }

    vector<S, 2> lower_bounds() const { return vector<S, 2>(S(-2), S(-2)); }

    vector<S, 2> upper_bounds() const { return vector<S, 2>(S(0.5), S(2)); }

    vector<S, 2> initial_point() const { return vector<S, 2>(S(-1), S(1)); }
};

// Zero-residual Rosenbrock (b = 5) as least squares: r = (1 - x0, sqrt(5) (x1 - x0^2)), minimizer (1, 1).
template <scalar S>
struct rosenbrock_ls
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }

    int num_residuals() const { return 2; }

    S value(const vector<S, 2>& x) const
    {
        const S r0 = S(1) - x[0];
        const S r1 = std::sqrt(S(5)) * (x[1] - x[0] * x[0]);
        return S(0.5) * (r0 * r0 + r1 * r1);
    }

    void residuals(const vector<S, 2>& x, Eigen::VectorX<S>& r) const
    {
        r[0] = S(1) - x[0];
        r[1] = std::sqrt(S(5)) * (x[1] - x[0] * x[0]);
    }

    void jacobian(const vector<S, 2>& x, Eigen::MatrixX<S>& J) const
    {
        J(0, 0) = S(-1);
        J(0, 1) = S(0);
        J(1, 0) = S(-2) * std::sqrt(S(5)) * x[0];
        J(1, 1) = std::sqrt(S(5));
    }

    vector<S, 2> initial_point() const { return vector<S, 2>(S(-1), S(1)); }
};

// The least-squares form of active_box_rosenbrock: r = (1 - x0, 10 (x1 - x0^2)), minimizer (0.5, 0.25).
template <scalar S>
struct active_box_rosenbrock_ls
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }

    int num_residuals() const { return 2; }

    S value(const vector<S, 2>& x) const
    {
        const S r0 = S(1) - x[0];
        const S r1 = S(10) * (x[1] - x[0] * x[0]);
        return S(0.5) * (r0 * r0 + r1 * r1);
    }

    void residuals(const vector<S, 2>& x, Eigen::VectorX<S>& r) const
    {
        r[0] = S(1) - x[0];
        r[1] = S(10) * (x[1] - x[0] * x[0]);
    }

    void jacobian(const vector<S, 2>& x, Eigen::MatrixX<S>& J) const
    {
        J(0, 0) = S(-1);
        J(0, 1) = S(0);
        J(1, 0) = S(-20) * x[0];
        J(1, 1) = S(10);
    }

    vector<S, 2> lower_bounds() const { return vector<S, 2>(S(-2), S(-2)); }

    vector<S, 2> upper_bounds() const { return vector<S, 2>(S(0.5), S(2)); }

    vector<S, 2> initial_point() const { return vector<S, 2>(S(-1), S(1)); }
};

// Infinite bounds for a box-constrained policy on an unconstrained least-squares problem.
template <typename Problem, scalar S>
struct unbounded : Problem
{
    vector<S, Problem::problem_dimension> lower_bounds() const
    {
        return vector<S, Problem::problem_dimension>::Constant(this->dimension(), -std::numeric_limits<S>::infinity());
    }

    vector<S, Problem::problem_dimension> upper_bounds() const
    {
        return vector<S, Problem::problem_dimension>::Constant(this->dimension(), std::numeric_limits<S>::infinity());
    }
};

}

#endif
