#include "moving_asymptote_problems.h"

#include "argmin/solver/options.h"
#include "argmin/solver/mma_policy.h"
#include "argmin/solver/gcmma_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/result/step_result.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <cstdint>

using namespace argmin;
using defects::scalar_bowl;

namespace
{

constexpr double nowhere = std::numeric_limits<double>::infinity();

// After 60 steps from (3, 3) the asymptotes have contracted around (1, 1); the hot reset moves far outside them.
template <typename Policy>
bool asymptotes_bracket_hot_reset_point()
{
    const defects::rosenbrock_in_box problem;
    step_budget_solver solver{Policy{}, problem, Eigen::Vector2d(3.0, 3.0).eval(), solver_options<>{}};
    for(std::int32_t k = 0; k < 60; ++k)
        solver.step();
    const Eigen::Vector2d target(9.5, 9.5);
    solver.reset(target);
    return (solver.state().L.array() < target.array()).all() && (target.array() < solver.state().U.array()).all();
}

// The change in U - L across the second of two consecutive null steps, or -1 if no such pair occurs.
double asymptote_growth_across_null_step()
{
    const scalar_bowl problem{1.0, 2.0, 1, 10.0, 0.0, 3.0, 1.5};
    step_budget_solver solver{gcmma_policy<double, 1>{}, problem, scalar_bowl::vector::Zero(), solver_options<>{}};
    bool previous_null = false;
    for(std::int32_t k = 0; k < 40; ++k)
    {
        const double width = solver.state().U[0] - solver.state().L[0];
        const bool null_step = solver.step().is_null_step;
        if(null_step && previous_null)
            return solver.state().U[0] - solver.state().L[0] - width;
        previous_null = null_step;
    }
    return -1.0;
}

using rho_wval = alternative::gcmma::rho_wval_policy<double, 1>;

// x^2 on [-10, 10] from -5 with one inner iteration: the fresh approximation overshoots to about 4.
rho_wval::state_type<scalar_bowl> first_rho_wval_step_on_wide_quadratic()
{
    static const scalar_bowl problem{1.0, 0.0, 0, 0.0, -10.0, 10.0, nowhere};
    rho_wval policy;
    rho_wval::options_type popts;
    popts.rho_init = 1e-5;
    popts.rho_min = 1e-5;
    popts.max_inner_iterations = 1;
    popts.rho_decay = 1.0;
    popts.raai = 0.0;
    rho_wval::state_type<scalar_bowl> state = policy.init(problem, scalar_bowl::vector::Constant(-5.0), solver_options<>{}, popts);
    policy.step(state);
    return state;
}

}

TEST_CASE("mma re-anchors its asymptotes around the point of a hot reset", "[known-defect][!shouldfail]")
{
    CHECK(asymptotes_bracket_hot_reset_point<mma_policy<>>());
}

TEST_CASE("gcmma re-anchors its asymptotes around the point of a hot reset", "[known-defect][!shouldfail]")
{
    CHECK(asymptotes_bracket_hot_reset_point<gcmma_policy<>>());
}

TEST_CASE("mma keeps a finite objective after a trial into a NaN region", "[known-defect][!shouldfail]")
{
    const scalar_bowl problem{1.0, 2.0, 1, 10.0, 0.0, 3.0, 1.2};
    step_budget_solver solver{mma_policy<double, 1>{}, problem, scalar_bowl::vector::Constant(1.0), solver_options<>{}};
    solver.step();
    CHECK(std::isfinite(solver.state().f));
}

TEST_CASE("gcmma takes two consecutive null steps at a NaN wall", "[moving_asymptote]")
{
    CHECK(asymptote_growth_across_null_step() >= 0.0);
}

TEST_CASE("gcmma leaves its asymptote width unchanged across a repeated null step", "[known-defect][!shouldfail]")
{
    CHECK(asymptote_growth_across_null_step() == 0.0);
}

TEST_CASE("the first trial of gcmma rho_wval on a wide quadratic fails its conservativity test", "[moving_asymptote]")
{
    CHECK(first_rho_wval_step_on_wide_quadratic().rho_obj > 1e-5);
}

TEST_CASE("gcmma rho_wval commits no trial that failed its conservativity test", "[known-defect][!shouldfail]")
{
    CHECK(first_rho_wval_step_on_wide_quadratic().x[0] == -5.0);
}
