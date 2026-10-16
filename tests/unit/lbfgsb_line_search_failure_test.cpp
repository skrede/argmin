#include "line_search_failure_problems.h"

#include "argmin/detail/bound_projection.h"

#include "argmin/line_search/armijo.h"
#include "argmin/line_search/strong_wolfe.h"

#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/byrd_lbfgsb_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <cstddef>
#include <algorithm>

using argmin::solver_status;
using argmin::lbfgsb_line_search;
using argmin::line_search_failure::vector2;
using argmin::line_search_failure::far_minimizer_box;
using argmin::line_search_failure::poisoned_everywhere;
using argmin::line_search_failure::poisoned_off_steepest_descent;

namespace
{

constexpr std::array<lbfgsb_line_search, 2> line_searches{lbfgsb_line_search::strong_wolfe,
                                                          lbfgsb_line_search::armijo};

const vector2 start{3.0, 2.0};

using lbfgsb_2 = argmin::lbfgsb_policy<double, 2>;
using byrd_lbfgsb_2 = argmin::byrd_lbfgsb_policy<double, 2>;

template <typename Policy, typename Problem>
using solver_for = argmin::step_budget_solver<Policy, 2, Problem>;

template <typename Policy, typename Problem>
solver_for<Policy, Problem> make_solver(const Problem& problem, const vector2& x0, lbfgsb_line_search kind)
{
    typename Policy::options_type options;
    options.line_search_type = kind;
    return solver_for<Policy, Problem>{Policy{}, problem, x0, argmin::solver_options<>{}, options};
}

}

TEMPLATE_TEST_CASE("L-BFGS-B line-search failure: a rejected step leaves the iterate in place", "[lbfgsb]",
                   lbfgsb_2, byrd_lbfgsb_2)
{
    for(lbfgsb_line_search kind : line_searches)
    {
        poisoned_everywhere problem;
        solver_for<TestType, poisoned_everywhere> solver = make_solver<TestType>(problem, start, kind);
        solver.step();
        solver.step();
        problem.anchor = solver.state().x;
        const double held = solver.state().objective_value;
        problem.poisoned = true;
        const argmin::step_result<double> r = solver.step();
        CHECK((solver.state().x == problem.anchor));
        CHECK(solver.state().objective_value == held);
        CHECK(r.policy_status == solver_status::line_search_failed);
        CHECK(r.diagnostics.bfgs_reset_count == std::size_t{1});
        CHECK(solver.state().B.size() == 0);
    }
}

TEMPLATE_TEST_CASE("L-BFGS-B line-search failure: an empty memory terminates without a retry", "[lbfgsb]",
                   lbfgsb_2, byrd_lbfgsb_2)
{
    for(lbfgsb_line_search kind : line_searches)
    {
        poisoned_everywhere problem;
        problem.anchor = start;
        problem.poisoned = true;
        solver_for<TestType, poisoned_everywhere> solver = make_solver<TestType>(problem, start, kind);
        const argmin::step_result<double> r = solver.step();
        CHECK((solver.state().x == start));
        CHECK(r.policy_status == solver_status::line_search_failed);
        CHECK(r.diagnostics.bfgs_reset_count == std::size_t{0});
        CHECK(solver.status() == solver_status::line_search_failed);
    }
}

TEMPLATE_TEST_CASE("L-BFGS-B line-search failure: the reset retry takes a steepest-descent step", "[lbfgsb]",
                   lbfgsb_2, byrd_lbfgsb_2)
{
    for(lbfgsb_line_search kind : line_searches)
    {
        poisoned_off_steepest_descent problem;
        solver_for<TestType, poisoned_off_steepest_descent> solver = make_solver<TestType>(problem, start, kind);
        solver.step();
        solver.step();
        problem.anchor = solver.state().x;
        problem.anchor_gradient = solver.state().g;
        problem.poisoned = true;
        const argmin::step_result<double> r = solver.step();
        CHECK(((solver.state().x - problem.anchor).norm() > 0.0));
        CHECK(problem.on_descent_ray(solver.state().x));
        CHECK(solver.state().B.size() == 1);
        CHECK(r.diagnostics.bfgs_reset_count == std::size_t{1});
        CHECK_FALSE(r.policy_status.has_value());
    }
}

TEMPLATE_TEST_CASE("L-BFGS-B line-search failure: an all-NaN Armijo search does not move the iterate", "[lbfgsb]",
                   lbfgsb_2, byrd_lbfgsb_2)
{
    argmin::line_search_options<double> opts;
    const argmin::line_search_result<double> kernel = argmin::armijo(
        [](double) { return std::numeric_limits<double>::quiet_NaN(); }, 1.0, -1.0, opts);
    REQUIRE_FALSE(kernel.success);
    REQUIRE(kernel.alpha == opts.max_alpha);

    for(lbfgsb_line_search kind : line_searches)
    {
        poisoned_everywhere problem;
        solver_for<TestType, poisoned_everywhere> solver = make_solver<TestType>(problem, start, kind);
        solver.step();
        problem.anchor = solver.state().x;
        problem.poisoned = true;
        const argmin::step_result<double> r = solver.step();
        CHECK((solver.state().x == problem.anchor));
        CHECK(solver.state().x.allFinite());
        CHECK(std::isfinite(solver.state().objective_value));
        CHECK(r.policy_status == solver_status::line_search_failed);
    }
}

TEMPLATE_TEST_CASE("L-BFGS-B line-search failure: a capped step with sufficient decrease is still accepted",
                   "[lbfgsb]", lbfgsb_2, byrd_lbfgsb_2)
{
    far_minimizer_box problem;
    const vector2 x0 = vector2::Zero();
    vector2 g0;
    problem.gradient(x0, g0);
    const vector2 d = -g0;
    const double cap = argmin::detail::compute_alpha_max<double, 2>(
        x0, d, problem.lower_bounds(), problem.upper_bounds());
    argmin::line_search_options<double> opts;
    opts.max_alpha = std::min(opts.max_alpha, cap);
    auto phi = [&](double a) { return problem.value((x0 + a * d).eval()); };
    auto dphi = [&](double a) {
        vector2 g;
        problem.gradient((x0 + a * d).eval(), g);
        return g.dot(d);
    };
    const argmin::line_search_result<double> slice = argmin::strong_wolfe(phi, dphi, problem.value(x0), g0.dot(d), opts);
    REQUIRE_FALSE(slice.success);
    REQUIRE(slice.alpha == cap);

    for(lbfgsb_line_search kind : line_searches)
    {
        solver_for<TestType, far_minimizer_box> solver = make_solver<TestType>(problem, x0, kind);
        const argmin::step_result<double> r = solver.step();
        CHECK((solver.state().x == problem.upper_bounds()));
        CHECK(r.diagnostics.bfgs_reset_count == std::size_t{0});
        CHECK_FALSE(r.policy_status.has_value());
        CHECK(solver.state().B.size() == 1);
    }
}

TEMPLATE_TEST_CASE("L-BFGS-B line-search failure: a second consecutive failure ends the solve non-converged",
                   "[lbfgsb]", lbfgsb_2, byrd_lbfgsb_2)
{
    for(lbfgsb_line_search kind : line_searches)
    {
        poisoned_everywhere problem;
        solver_for<TestType, poisoned_everywhere> solver = make_solver<TestType>(problem, start, kind);
        solver.step();
        solver.step();
        problem.anchor = solver.state().x;
        problem.poisoned = true;
        const argmin::solve_result<double, 2> result = solver.solve();
        CHECK(result.status == solver_status::line_search_failed);
        CHECK(result.status != solver_status::converged);
        CHECK((result.x == problem.anchor));
    }
}
