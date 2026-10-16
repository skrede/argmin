#ifndef HPP_GUARD_ARGMIN_TESTS_UNIT_LINE_SEARCH_FAILURE_PROBLEMS_H
#define HPP_GUARD_ARGMIN_TESTS_UNIT_LINE_SEARCH_FAILURE_PROBLEMS_H

#include <Eigen/Core>

#include <cmath>
#include <limits>

namespace argmin::line_search_failure
{

using vector2 = Eigen::Vector<double, 2>;

// A strictly convex coupled quadratic: every curvature pair is accepted, and the quasi-Newton
// direction is not parallel to the gradient.
inline double coupled_quadratic(const vector2& x)
{
    const double u = x[0] - 1.0;
    const double v = x[1] + 0.5;
    return u * u + 10.0 * v * v + 3.0 * u * v;
}

inline void coupled_quadratic_gradient(const vector2& x, vector2& g)
{
    const double u = x[0] - 1.0;
    const double v = x[1] + 0.5;
    g[0] = 2.0 * u + 3.0 * v;
    g[1] = 20.0 * v + 3.0 * u;
}

// Once poisoned, the objective is NaN everywhere except at the anchor.
struct poisoned_everywhere
{
    static constexpr int problem_dimension = 2;

    poisoned_everywhere()
        : poisoned(false)
        , anchor(vector2::Zero())
    {}

    int dimension() const { return problem_dimension; }

    double value(const vector2& x) const
    {
        if(poisoned && x != anchor)
            return std::numeric_limits<double>::quiet_NaN();
        return coupled_quadratic(x);
    }

    void gradient(const vector2& x, vector2& g) const { coupled_quadratic_gradient(x, g); }

    bool poisoned;
    vector2 anchor;
};

// Once poisoned, the objective is finite only at the anchor and along the steepest-descent ray
// from it, so only a step along minus the anchor gradient can be taken.
struct poisoned_off_steepest_descent
{
    static constexpr int problem_dimension = 2;

    poisoned_off_steepest_descent()
        : poisoned(false)
        , anchor(vector2::Zero())
        , anchor_gradient(vector2::Zero())
    {}

    int dimension() const { return problem_dimension; }

    double value(const vector2& x) const
    {
        if(poisoned && x != anchor && !on_descent_ray(x))
            return std::numeric_limits<double>::quiet_NaN();
        return coupled_quadratic(x);
    }

    void gradient(const vector2& x, vector2& g) const { coupled_quadratic_gradient(x, g); }

    bool on_descent_ray(const vector2& x) const
    {
        const vector2 step = x - anchor;
        const double cross = step[0] * anchor_gradient[1] - step[1] * anchor_gradient[0];
        const double scale = step.norm() * anchor_gradient.norm();
        return step.dot(anchor_gradient) < 0.0 && std::abs(cross) <= 1e-9 * scale;
    }

    bool poisoned;
    vector2 anchor;
    vector2 anchor_gradient;
};

// The minimizer (100, 100) lies far outside the unit box, so a strong-Wolfe search from the
// interior stops at the box with sufficient decrease but without the curvature condition.
struct far_minimizer_box
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return problem_dimension; }

    double value(const vector2& x) const { return (x - vector2::Constant(100.0)).squaredNorm(); }

    void gradient(const vector2& x, vector2& g) const { g = 2.0 * (x - vector2::Constant(100.0)); }

    vector2 lower_bounds() const { return vector2::Constant(-1.0); }

    vector2 upper_bounds() const { return vector2::Constant(1.0); }
};

}

#endif
