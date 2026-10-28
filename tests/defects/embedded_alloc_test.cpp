// The counting eigen_assert must be defined before the first Eigen include.
#include "argmin/detail/diagnostics/alloc_counter.h"
#include "argmin/detail/diagnostics/steady_state_driver.h"

#include "embedded_problems.h"

#include "argmin/solver/options.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>

using namespace argmin;

namespace
{

namespace sensor = argmin::detail::bench;

sensor::steady_state_result measure_equality_rich_kraft()
{
    solver_options<> opts;
    opts.max_iterations = 200;
    return sensor::measure_steady(kraft_slsqp_policy<double, 48>{}, defects::coupled_quadratic<48, 48, 1>{}, opts, 10);
}

template <typename Body>
std::size_t eigen_mallocs_in(Body&& body)
{
    sensor::reset_alloc_count();
    sensor::arm_alloc_trace();
    body();
    sensor::disarm_alloc_trace();
    return sensor::read_eigen_malloc_count();
}

}

TEST_CASE("kraft_slsqp with 48 equality rows is still stepping across the measured window", "[embedded]")
{
    CHECK_FALSE(measure_equality_rich_kraft().terminated_early);
}

TEST_CASE("kraft_slsqp steps without heap allocation at 48 equality rows", "[known-defect][!shouldfail]")
{
    CHECK(measure_equality_rich_kraft().eigen_malloc == 0);
}

TEST_CASE("constructing a fixed-dimension solver from a fixed start allocates no more than the policy's init", "[known-defect][!shouldfail]")
{
    const defects::coupled_quadratic<4, 0, 0> problem;
    const Eigen::Vector<double, 4> x0 = problem.initial_point();
    const solver_options<> opts;
    const std::size_t init = eigen_mallocs_in([&] { static_cast<void>(lbfgsb_policy<double, 4>{}.init(problem, x0, opts)); });
    const std::size_t construct = eigen_mallocs_in([&] { step_budget_solver solver{lbfgsb_policy<double, 4>{}, problem, x0, opts}; });
    CHECK(construct == init);
}
