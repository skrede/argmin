#ifndef HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_NATIVE_PROBLEMS_H
#define HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_NATIVE_PROBLEMS_H

#include "argmin/types.h"
#include "argmin/test_functions/rosenbrock.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <limits>

namespace argmin::python
{

inline constexpr double unbounded = std::numeric_limits<double>::infinity();

// Compiled, dynamically dimensioned face of the library's own banana-shaped
// valley. The box is exposed so a caller can pose the identical bounded
// problem to the interpreter-side methods, which is the only way the parity
// comparison holds every input fixed but the boundary.
struct rosenbrock
{
    static constexpr int problem_dimension = dynamic_dimension;

    argmin::rosenbrock<double> shape{};
    vector<double> lower{};
    vector<double> upper{};

    [[nodiscard]] int dimension() const { return shape.n; }
    [[nodiscard]] double optimal_value() const { return shape.optimal_value(); }
    [[nodiscard]] vector<double> initial_point() const { return shape.initial_point(); }
    [[nodiscard]] double value(const vector<double>& x) const { return shape.value(x); }

    void gradient(const vector<double>& x, vector<double>& g) const { shape.gradient(x, g); }

    [[nodiscard]] vector<double> lower_bounds() const { return lower; }
    [[nodiscard]] vector<double> upper_bounds() const { return upper; }
};

// The equality-constrained problem is published at a fixed size; this face
// re-expresses it at dynamic dimension so a native result carries the same
// vector type the wrapped methods return.
struct constrained_test_problem
{
    static constexpr int problem_dimension = dynamic_dimension;

    hs006<double> shape{};

    [[nodiscard]] int dimension() const { return shape.dimension(); }
    [[nodiscard]] int num_equality() const { return shape.num_equality(); }
    [[nodiscard]] int num_inequality() const { return shape.num_inequality(); }
    [[nodiscard]] double optimal_value() const { return shape.optimal_value(); }

    [[nodiscard]] vector<double> initial_point() const
    {
        return vector<double>(shape.initial_point());
    }

    [[nodiscard]] double value(const vector<double>& x) const
    {
        return shape.value(Eigen::Vector2d(x));
    }

    void gradient(const vector<double>& x, vector<double>& g) const
    {
        Eigen::Vector2d fixed = Eigen::Vector2d(x);
        Eigen::Vector2d produced;
        shape.gradient(fixed, produced);
        g = produced;
    }

    void constraints(const vector<double>& x, vector<double>& c) const
    {
        c.resize(1);
        shape.constraints(Eigen::Vector2d(x), c);
    }

    void constraint_jacobian(const vector<double>& x, matrix<double>& J) const
    {
        J.resize(1, 2);
        shape.constraint_jacobian(Eigen::Vector2d(x), J);
    }

    [[nodiscard]] vector<double> lower_bounds() const
    {
        return vector<double>::Constant(2, -unbounded);
    }

    [[nodiscard]] vector<double> upper_bounds() const
    {
        return vector<double>::Constant(2, unbounded);
    }
};

}

#endif
