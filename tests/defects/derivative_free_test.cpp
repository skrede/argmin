#include "derivative_free_problems.h"

#include "argmin/solver/options.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <cstdint>
#include <algorithm>

using namespace argmin;
using defects::box_quadratic;

namespace
{

using bobyqa_state = bobyqa_policy<>::state_type<box_quadratic>;

struct rescue_outcome
{
    double objective_before;
    double objective_after;
    double incumbent_distance;
    bool rescued;
};

// The incumbent of the near-corner quadratic sits within rho of two bounds after 10 steps from (0.6, 0.6).
bobyqa_state near_corner_state(const box_quadratic& problem, bobyqa_policy<>& policy)
{
    bobyqa_state s = policy.init(problem, Eigen::Vector2d(0.6, 0.6), solver_options<>{});
    for(std::int32_t k = 0; k < 10; ++k)
        if(policy.step(s).policy_status.has_value())
            break;
    return s;
}

double distance_to_interpolation_set(const bobyqa_state& s, const Eigen::VectorXd& point)
{
    double nearest = std::numeric_limits<double>::infinity();
    for(std::int32_t k = 0; k < s.sys.m_points; ++k)
        nearest = std::min(nearest, (s.sys.xbase + s.sys.xpt.col(k).head(2) - point).norm());
    return nearest;
}

// Collapse the factored system so the next step has to rebuild it through a rescue.
void collapse_for_rescue(bobyqa_state& s)
{
    s.rho = s.delta = 0.1;
    s.sys.bmat.setZero();
    s.sys.zmat.setZero();
    s.sys.gopt = Eigen::VectorXd::Constant(2, -1.0);
    s.ntrits = 0;
    s.nresc = s.nevals - 1;
}

rescue_outcome rescue_near_corner()
{
    const box_quadratic problem{Eigen::Vector2d(0.03, 0.05), 1.0, 3.0, 1.5, 0.0, 1.0};
    bobyqa_policy<> policy;
    bobyqa_state s = near_corner_state(problem, policy);
    const Eigen::VectorXd incumbent = s.sys.xbase + s.sys.xopt;
    const double before = s.objective_value;
    collapse_for_rescue(s);
    const auto rescues_before = s.nresc;
    policy.step(s);
    return {before, s.objective_value, distance_to_interpolation_set(s, incumbent), s.nresc != rescues_before};
}

}

TEST_CASE("cobyla stops roundoff_limited instead of evaluating a step its linear program failed to produce", "[known-defect][!shouldfail]")
{
    const defects::scaled_two_row problem{1e-200};
    solver_options<> options;
    options.max_iterations = 500;
    step_budget_solver solver{cobyla_policy<>{}, problem, Eigen::Vector2d(0.3, 0.2), options};
    CHECK(solver.solve().status == solver_status::roundoff_limited);
}

TEST_CASE("bobyqa reports roundoff_limited when every update denominator collapses and no rescue is left", "[known-defect][!shouldfail]")
{
    const box_quadratic problem{Eigen::Vector2d(0.3, -0.2), 1.0, 2.0, 0.5, -1.0, 1.0};
    bobyqa_policy<> policy;
    bobyqa_state s = policy.init(problem, Eigen::Vector2d(0.9, 0.9), solver_options<>{});
    s.sys.bmat.setZero();
    s.sys.zmat.setZero();
    s.nresc = s.nevals;
    CHECK(policy.step(s).policy_status == solver_status::roundoff_limited);
}

TEST_CASE("bobyqa rescues the collapsed system of the near-corner quadratic", "[derivative-free]")
{
    CHECK(rescue_near_corner().rescued);
}

TEST_CASE("bobyqa keeps its incumbent as an interpolation point through a rescue", "[known-defect][!shouldfail]")
{
    CHECK(rescue_near_corner().incumbent_distance < 1e-12);
}

TEST_CASE("bobyqa does not raise its objective value through a rescue", "[known-defect][!shouldfail]")
{
    const rescue_outcome outcome = rescue_near_corner();
    CHECK(outcome.objective_after <= outcome.objective_before);
}
