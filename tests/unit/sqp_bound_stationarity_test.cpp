#include "argmin/solver/options.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

namespace
{

using vector2 = Eigen::Vector2d;

// min ||x - target||^2 on a box; the declared constraint capacity is either absent or Capacity with no rows.
template <std::int32_t Capacity>
struct box_quadratic
{
    static constexpr std::int32_t problem_dimension = 2;
    static constexpr std::int32_t constraint_count = Capacity;

    vector2 target;
    vector2 lower;
    vector2 upper;

    std::int32_t dimension() const { return 2; }
    std::int32_t num_equality() const { return 0; }
    std::int32_t num_inequality() const { return 0; }
    double value(const vector2& x) const { return (x - target).squaredNorm(); }
    void gradient(const vector2& x, vector2& g) const { g = 2.0 * (x - target); }
    template <typename V> void constraints(const vector2&, V& c) const { c.resize(0); }
    template <typename J> void constraint_jacobian(const vector2&, J& j) const { j.resize(0, 2); }
    vector2 lower_bounds() const { return lower; }
    vector2 upper_bounds() const { return upper; }
};

struct undeclared_box_quadratic
{
    static constexpr std::int32_t problem_dimension = 2;

    vector2 target;
    vector2 lower;
    vector2 upper;

    std::int32_t dimension() const { return 2; }
    std::int32_t num_equality() const { return 0; }
    std::int32_t num_inequality() const { return 0; }
    double value(const vector2& x) const { return (x - target).squaredNorm(); }
    void gradient(const vector2& x, vector2& g) const { g = 2.0 * (x - target); }
    template <typename V> void constraints(const vector2&, V& c) const { c.resize(0); }
    template <typename J> void constraint_jacobian(const vector2&, J& j) const { j.resize(0, 2); }
    vector2 lower_bounds() const { return lower; }
    vector2 upper_bounds() const { return upper; }
};

template <typename Problem>
Problem vertex_problem()
{
    return Problem{vector2(-1.0, -1.0), vector2::Zero(), vector2::Constant(5.0)};
}

template <typename Problem>
Problem face_problem()
{
    return Problem{vector2(-1.0, 1.0), vector2(0.0, -5.0), vector2::Constant(5.0)};
}

template <template <typename, int> typename Policy, typename Problem>
void check_converges_on_bound(const Problem& problem, const vector2& minimizer)
{
    argmin::step_budget_solver solver{Policy<double, 2>{}, problem, vector2(0.5, 0.5), argmin::solver_options<>{}};
    const auto r = solver.solve(argmin::solver_options<>{});
    CHECK(r.status == argmin::solver_status::converged);
    CHECK(r.iterations < 10);
    CHECK((r.x - minimizer).cwiseAbs().maxCoeff() < 1e-12);
    CHECK(r.gradient_norm < 1e-12);
}

template <template <typename, int> typename Policy>
void check_every_bound_shape()
{
    check_converges_on_bound<Policy>(vertex_problem<box_quadratic<0>>(), vector2::Zero());
    check_converges_on_bound<Policy>(vertex_problem<box_quadratic<1>>(), vector2::Zero());
    check_converges_on_bound<Policy>(vertex_problem<undeclared_box_quadratic>(), vector2::Zero());
    check_converges_on_bound<Policy>(face_problem<box_quadratic<0>>(), vector2(0.0, 1.0));
}

// The minimizer and every iterate sit farther from the box than the gradient is long, so no bound is reached.
template <template <typename, int> typename Policy>
void check_interior_gradient_norm_is_raw()
{
    const auto problem = box_quadratic<0>{vector2(1.0, 2.0), vector2::Constant(-10.0), vector2::Constant(10.0)};
    argmin::step_budget_solver solver{Policy<double, 2>{}, problem, vector2(0.5, 0.5), argmin::solver_options<>{}};
    for(std::int32_t i = 0; i < 3 && solver.status() == argmin::solver_status::running; ++i)
    {
        const auto step = solver.step();
        vector2 g;
        problem.gradient(solver.state().x, g);
        CHECK(step.gradient_norm == g.norm());
    }
}

}

TEST_CASE("nw_sqp converges on a minimizer resting on its bounds", "[sqp][bounds]")
{
    check_every_bound_shape<argmin::nw_sqp_policy>();
}

TEST_CASE("filter_nw_sqp converges on a minimizer resting on its bounds", "[sqp][bounds]")
{
    check_every_bound_shape<argmin::filter_nw_sqp_policy>();
}

TEST_CASE("filter_slsqp converges on a minimizer resting on its bounds", "[sqp][bounds]")
{
    check_every_bound_shape<argmin::filter_slsqp_policy>();
}

TEST_CASE("kraft_slsqp converges on a minimizer resting on its bounds", "[sqp][bounds]")
{
    check_every_bound_shape<argmin::kraft_slsqp_policy>();
}

TEST_CASE("a line-search SQP iterate clear of its bounds reports the raw gradient norm", "[sqp][bounds]")
{
    check_interior_gradient_norm_is_raw<argmin::nw_sqp_policy>();
    check_interior_gradient_norm_is_raw<argmin::filter_nw_sqp_policy>();
    check_interior_gradient_norm_is_raw<argmin::filter_slsqp_policy>();
    check_interior_gradient_norm_is_raw<argmin::kraft_slsqp_policy>();
}
