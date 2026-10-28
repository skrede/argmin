#include "moving_asymptote_problems.h"

#include "argmin/solver/options.h"
#include "argmin/solver/mma_policy.h"
#include "argmin/solver/gcmma_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/ccsa_quadratic_policy.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"
#include "argmin/result/solve_result.h"

#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <cstdint>

using namespace argmin;
using defects::scalar_bowl;

namespace
{

double last_dual_kkt = 0.0;

// A dual solver whose line search has no evaluation budget, so every dual step fails.
template <scalar S, int M>
struct failing_dual : lbfgsb_policy<S, M>
{
    template <typename State>
    step_result<S> step(State& s)
    {
        this->options.line_search.max_iterations = 0;
        const step_result<S> r = lbfgsb_policy<S, M>::step(s);
        last_dual_kkt = r.kkt_residual.value_or(r.gradient_norm);
        return r;
    }
};

step_result<double> mma_step_on_failed_dual()
{
    const hs044<> problem;
    step_budget_solver solver{mma_policy<double, 4, failing_dual>{}, problem, problem.initial_point(), solver_options<>{}};
    return solver.step();
}

// s (x - 2)^2 on [0, 3] with 1 - x >= 0 from 0.5: the multiplier is 2 s.
template <typename Policy>
double scaled_bowl_endpoint(double scale)
{
    const scalar_bowl problem{scale, 2.0, 1, 1.0, 0.0, 3.0, std::numeric_limits<double>::infinity()};
    solver_options<> opts;
    opts.max_iterations = 500;
    step_budget_solver solver{Policy{}, problem, scalar_bowl::vector::Constant(0.5), opts};
    return solver.solve(opts).x[0];
}

struct termination
{
    double objective_error;
    double kkt;
};

template <typename Policy>
termination hs044_termination()
{
    const hs044<> problem;
    solver_options<> opts;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-6);
    opts.set_step_threshold(1e-6);
    opts.set_stationarity_threshold(1e-8);
    step_budget_solver solver{Policy{}, problem, problem.initial_point(), opts};
    double kkt = std::numeric_limits<double>::infinity();
    for(std::int32_t k = 0; k < 500 && solver.status() == solver_status::running; ++k)
        kkt = solver.step().kkt_residual.value_or(kkt);
    return {std::abs(solver.state().f - problem.optimal_value()), kkt};
}

// The point after three steps, mapped back from y = k x.
Eigen::Vector2d scaled_twin_after_three_steps(double k)
{
    const defects::scaled_twin problem{k};
    step_budget_solver solver{gcmma_policy<double, 2>{}, problem, Eigen::Vector2d(0.25 * k, 0.25 * k), solver_options<>{}};
    for(std::int32_t i = 0; i < 3; ++i)
        solver.step();
    return solver.state().x / k;
}

}

TEST_CASE("a zero line-search budget leaves the mma dual unsolved on hs044", "[moving_asymptote]")
{
    mma_step_on_failed_dual();
    CHECK(last_dual_kkt >= 1e-9);
}

TEST_CASE("mma does not commit the subproblem point of a failed dual solve", "[known-defect][!shouldfail]")
{
    const step_result<double> r = mma_step_on_failed_dual();
    CHECK((r.is_null_step || r.policy_status.has_value()));
}

TEST_CASE("mma, gcmma and ccsa_quadratic solve the unit-scaled bowl with an active row", "[moving_asymptote]")
{
    CHECK(std::abs(scaled_bowl_endpoint<mma_policy<double, 1>>(1.0) - 1.0) < 1e-4);
    CHECK(std::abs(scaled_bowl_endpoint<gcmma_policy<double, 1>>(1.0) - 1.0) < 1e-4);
    CHECK(std::abs(scaled_bowl_endpoint<ccsa_quadratic_policy<double, 1>>(1.0) - 1.0) < 1e-4);
}

TEST_CASE("mma reaches the active row of a bowl scaled by 1e3", "[known-defect][!shouldfail]")
{
    CHECK(std::abs(scaled_bowl_endpoint<mma_policy<double, 1>>(1e3) - 1.0) < 1e-4);
}

TEST_CASE("gcmma reaches the active row of a bowl scaled by 1e3", "[known-defect][!shouldfail]")
{
    CHECK(std::abs(scaled_bowl_endpoint<gcmma_policy<double, 1>>(1e3) - 1.0) < 1e-4);
}

TEST_CASE("ccsa_quadratic reaches the active row of a bowl scaled by 1e3", "[known-defect][!shouldfail]")
{
    CHECK(std::abs(scaled_bowl_endpoint<ccsa_quadratic_policy<double, 1>>(1e3) - 1.0) < 1e-4);
}

TEST_CASE("mma and ccsa_quadratic end within 1e-8 of the hs044 optimum under tight tolerances", "[moving_asymptote]")
{
    CHECK(hs044_termination<mma_policy<double, 4>>().objective_error < 1e-8);
    CHECK(hs044_termination<ccsa_quadratic_policy<double, 4>>().objective_error < 1e-8);
}

TEST_CASE("mma reports a bound-aware KKT residual at the hs044 optimum", "[known-defect][!shouldfail]")
{
    CHECK(hs044_termination<mma_policy<double, 4>>().kkt <= 1e-6);
}

TEST_CASE("ccsa_quadratic reports a bound-aware KKT residual at the hs044 optimum", "[known-defect][!shouldfail]")
{
    CHECK(hs044_termination<ccsa_quadratic_policy<double, 4>>().kkt <= 1e-6);
}

TEST_CASE("gcmma takes the same steps on a problem posed in rescaled variables", "[known-defect][!shouldfail]")
{
    CHECK((scaled_twin_after_three_steps(1.0) - scaled_twin_after_three_steps(1024.0)).cwiseAbs().maxCoeff() <= 1e-8);
}
