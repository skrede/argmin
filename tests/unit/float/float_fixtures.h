#ifndef HPP_GUARD_ARGMIN_TESTS_UNIT_FLOAT_FLOAT_FIXTURES_H
#define HPP_GUARD_ARGMIN_TESTS_UNIT_FLOAT_FLOAT_FIXTURES_H

#include "argmin/types.h"

#include <Eigen/Core>

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

}

#endif
