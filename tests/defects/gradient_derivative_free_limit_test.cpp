#include "walled_bowl.h"

#include "argmin/solver/options.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/byrd_lbfgsb_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/augmented_lagrangian_policy.h"

#include "argmin/result/status.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <cstdint>

using namespace argmin;
using defects::bowl_callback;

namespace
{

constexpr double not_a_number = std::numeric_limits<double>::quiet_NaN();
constexpr double infinity = std::numeric_limits<double>::infinity();

Eigen::VectorXd point(double x0, double x1)
{
    Eigen::VectorXd x(2);
    x << x0, x1;
    return x;
}

// Every value is finite and above the moderated extreme barrier.
struct towering_bowl
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t dimension() const { return 2; }

    std::int32_t num_equality() const { return 0; }

    std::int32_t num_inequality() const { return 0; }

    template <typename Values>
    void constraints(const Eigen::VectorXd&, Values& c) const
    {
        c.resize(0);
    }

    double value(const Eigen::VectorXd& x) const { return 1e31 * ((x[0] - 3.0) * (x[0] - 3.0) + (x[1] - 3.0) * (x[1] - 3.0) + 1.0); }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd::Constant(2, -5.0); }

    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd::Constant(2, 5.0); }
};

solver_options<> options(std::uint32_t max_iterations)
{
    solver_options<> opts;
    opts.max_iterations = max_iterations;
    opts.set_gradient_threshold(1e-6);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);
    return opts;
}

template <typename Policy, typename Problem>
solve_result<double> solve_from(const Policy& policy, const Problem& problem, const Eigen::VectorXd& x0, std::uint32_t max_iterations)
{
    const solver_options<> opts = options(max_iterations);
    step_budget_solver solver{policy, problem, x0, opts};
    return solver.solve(opts);
}

template <typename Policy>
solver_status step_with_a_finite_gradient(const Policy& policy, const defects::walled_bowl& problem)
{
    const solver_options<> opts = options(200);
    step_budget_solver solver{policy, problem, point(0.0, 0.0), opts};
    while(solver.status() == solver_status::running)
    {
        solver.step();
        REQUIRE(solver.state().g.allFinite());
    }
    return solver.status();
}

}

TEST_CASE("lbfgsb and byrd_lbfgsb report non_finite_evaluation for a non-finite gradient at the start", "[gradient]")
{
    const defects::walled_bowl problem = defects::glitching(bowl_callback::gradient, 1, true, not_a_number, 0);
    CHECK(solve_from(lbfgsb_policy<>{}, problem, point(0.0, 0.0), 200).status == solver_status::non_finite_evaluation);
    const defects::walled_bowl other = defects::glitching(bowl_callback::gradient, 1, true, not_a_number, 0);
    CHECK(solve_from(byrd_lbfgsb_policy<>{}, other, point(0.0, 0.0), 200).status == solver_status::non_finite_evaluation);
}

TEST_CASE("lbfgsb ends non_finite_evaluation at its line-search limit against a NaN wall", "[gradient]")
{
    for(const double start : {1.0, 0.0})
    {
        const solve_result<double> r = solve_from(lbfgsb_policy<>{}, defects::wall_at(1.0, 0), point(start, 0.0), 200);
        CHECK(r.status == solver_status::non_finite_evaluation);
        CHECK(r.x[0] <= 1.0);
    }
}

// Gradient call 1 is the start, call 2 the first step's trial, call 3 the second step's refresh and call 4 its trial.
TEST_CASE("lbfgsb keeps the commit's finite gradient when the step-start gradient refresh evaluates non-finite", "[gradient]")
{
    CHECK(step_with_a_finite_gradient(lbfgsb_policy<>{}, defects::glitching(bowl_callback::gradient, 3, true, not_a_number, 0)) == solver_status::converged);
}

TEST_CASE("lbfgsb and byrd_lbfgsb retry instead of committing an accepted trial whose gradient evaluates non-finite", "[gradient]")
{
    CHECK(step_with_a_finite_gradient(lbfgsb_policy<>{}, defects::glitching(bowl_callback::gradient, 4, true, not_a_number, 0)) == solver_status::converged);
    CHECK(step_with_a_finite_gradient(byrd_lbfgsb_policy<>{}, defects::glitching(bowl_callback::gradient, 4, true, not_a_number, 0)) == solver_status::converged);
}

TEST_CASE("bobyqa reports non_finite_evaluation for a non-finite objective at the start", "[derivative_free]")
{
    const solve_result<double> r = solve_from(bobyqa_policy<>{}, defects::glitching(bowl_callback::value, 1, true, not_a_number, 0), point(0.0, 0.0), 2000);
    CHECK(r.status == solver_status::non_finite_evaluation);
}

TEST_CASE("bobyqa ends non_finite_evaluation at its final radius when it approaches a NaN wall", "[derivative_free]")
{
    const solve_result<double> r = solve_from(bobyqa_policy<>{}, defects::wall_at(1.0, 0), point(0.0, 0.0), 2000);
    CHECK(r.status == solver_status::non_finite_evaluation);
    CHECK(r.x[0] <= 1.0);
}

TEST_CASE("bobyqa and cobyla solve a finite objective whose values exceed the moderated barrier", "[derivative_free]")
{
    const solve_result<double> b = solve_from(bobyqa_policy<>{}, towering_bowl{}, point(0.0, 0.0), 2000);
    CHECK(b.status == solver_status::converged);
    CHECK((b.x - point(3.0, 3.0)).norm() < 1e-6);
    const solve_result<double> c = solve_from(cobyla_policy<>{}, towering_bowl{}, point(0.0, 0.0), 2000);
    // With fused multiply-add contraction (such as under -march=native) the solver's own criteria stop it first.
    CHECK((c.status == solver_status::xtol_reached || c.status == solver_status::converged));
    CHECK((c.x - point(3.0, 3.0)).norm() < 1e-6);
}

TEST_CASE("cobyla reports non_finite_evaluation for a non-finite row at the start", "[derivative_free]")
{
    const solve_result<double> r = solve_from(cobyla_policy<>{}, defects::glitching(bowl_callback::constraints, 1, true, not_a_number, 1), point(0.0, 0.0), 2000);
    CHECK(r.status == solver_status::non_finite_evaluation);
}

TEST_CASE("cobyla ends non_finite_evaluation at its final radius against a NaN wall", "[derivative_free]")
{
    for(const double start : {1.0, 0.0})
    {
        const solve_result<double> r = solve_from(cobyla_policy<>{}, defects::wall_at(1.0, 1), point(start, 0.0), 2000);
        CHECK(r.status == solver_status::non_finite_evaluation);
        CHECK(r.x[0] <= 1.0);
    }
}

TEST_CASE("cobyla never returns a vertex whose rows evaluated non-finite", "[derivative_free]")
{
    const solve_result<double> r = solve_from(cobyla_policy<>{}, defects::glitching(bowl_callback::constraints, 2, false, -infinity, 1), point(0.0, 0.0), 2000);
    CHECK(r.status == solver_status::non_finite_evaluation);
    CHECK(r.x == point(0.0, 0.0));
}

TEST_CASE("the augmented Lagrangian reports non_finite_evaluation for a non-finite row Jacobian at the start", "[augmented_lagrangian]")
{
    const defects::walled_bowl problem = defects::glitching(bowl_callback::constraint_jacobian, 1, true, not_a_number, 1);
    CHECK(solve_from(augmented_lagrangian_policy<lbfgsb_policy<>>{}, problem, point(0.0, 0.0), 200).status == solver_status::non_finite_evaluation);
}

TEST_CASE("the augmented Lagrangian ends non_finite_evaluation at its penalty floor against a NaN wall", "[augmented_lagrangian]")
{
    for(const double start : {1.0, 0.0})
    {
        const solve_result<double> r = solve_from(augmented_lagrangian_policy<lbfgsb_policy<>>{}, defects::wall_at(1.0, 1), point(start, 0.0), 200);
        CHECK(r.status == solver_status::non_finite_evaluation);
        CHECK(r.x[0] <= 1.0);
    }
}

TEST_CASE("the augmented Lagrangian ends non_finite_evaluation at its penalty floor when every outer trial's rows are infinite", "[augmented_lagrangian]")
{
    const defects::walled_bowl problem = defects::glitching(bowl_callback::constraints, 2, false, infinity, 1);
    const solve_result<double> r = solve_from(augmented_lagrangian_policy<lbfgsb_policy<>>{}, problem, point(0.0, 0.0), 200);
    CHECK(r.status == solver_status::non_finite_evaluation);
    CHECK(r.x == point(0.0, 0.0));
}
