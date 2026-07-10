// COBYLA solver policy unit tests.
//
// Validates concept satisfaction, basic convergence on a simple constrained
// problem, HS024 convergence, and basic_solver_group compatibility.

#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/schedule/basic_solver_group.h"
#include "argmin/schedule/round_robin_schedule.h"
#include "argmin/test_functions/hock_schittkowski.h"
#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <map>
#include <cmath>
#include <string>
#include <vector>
#include <cstddef>
#include <fstream>
#include <sstream>

using Catch::Approx;
using namespace argmin;

namespace
{

// Minimal oracle CSV reader (mirrors the CMA-ES / MMA / parmu-pin readers):
// '#' lines are comments, data lines are "name,v1,v2,...".
std::map<std::string, std::vector<double>> load_cobyla_oracle(const std::string& path)
{
    std::map<std::string, std::vector<double>> rows;
    std::ifstream in(path);
    if(!in.is_open())
        return rows;
    std::string line;
    while(std::getline(in, line))
    {
        if(line.empty() || line.front() == '#')
            continue;
        std::istringstream ss(line);
        std::string name;
        std::getline(ss, name, ',');
        std::vector<double> vals;
        std::string tok;
        while(std::getline(ss, tok, ','))
            vals.push_back(std::stod(tok));
        rows.emplace(name, std::move(vals));
    }
    return rows;
}

}

namespace
{

// Simple constrained problem: minimise x0^2 + x1^2
// subject to x0 + x1 >= 1 (argmin: c >= 0)
// with box bounds [-10, 10]^2.
//
// Satisfies constrained_values && bound_constrained but NOT constrained
// (no constraint_jacobian), proving COBYLA works without gradients.
//
// Optimal: x* = (0.5, 0.5), f* = 0.5.
struct simple_constrained
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        return x[0] * x[0] + x[1] * x[1];
    }

    void constraints(const Eigen::Vector<double, 2>& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = x[0] + x[1] - 1.0;
    }

    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    Eigen::Vector<double, 2> lower_bounds() const
    {
        Eigen::Vector<double, 2> lb;
        lb << -10.0, -10.0;
        return lb;
    }

    Eigen::Vector<double, 2> upper_bounds() const
    {
        Eigen::Vector<double, 2> ub;
        ub << 10.0, 10.0;
        return ub;
    }
};

}

// Concept satisfaction checks.
static_assert(constrained_values<simple_constrained>);
static_assert(!constrained<simple_constrained>);
static_assert(bound_constrained<simple_constrained>);

static_assert(constrained_values<hs071<>>);
static_assert(constrained<hs071<>>);

static_assert(constrained_values<hs024<>>);
static_assert(constrained<hs024<>>);

TEST_CASE("cobyla_policy: simple constrained 2D", "[cobyla]")
{
    simple_constrained problem;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    step_budget_solver solver{cobyla_policy{}, problem, x0, opts};
    auto result = solver.solve();

    CHECK(result.objective_value < 0.6);
    CHECK(solver.constraint_violation() < 1e-3);
}

TEST_CASE("cobyla_policy: HS024", "[cobyla][hs]")
{
    // HS024 in 2D, f* = -1.0 at x* = (3, sqrt(3)). The pre-rewrite
    // penalized-subgradient driver stalled at f = -0.30 (wrong optimum) and,
    // after a partial hotfix, at f = -0.652. The Powell-faithful two-stage
    // TRSTLP driver reaches the true optimum, so the bar is now tight and
    // pinned against the checked-in NLopt reference optimum.
    const auto oracle = load_cobyla_oracle("oracles/cobyla_convergence.csv");
    REQUIRE(oracle.contains("hs024_fopt"));
    REQUIRE(oracle.contains("hs024_xopt"));

    hs024 problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 2000;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    step_budget_solver solver{cobyla_policy{}, problem, x0, opts};
    auto result = solver.solve();

    INFO("HS024 objective: " << result.objective_value);
    INFO("HS024 cv:        " << solver.constraint_violation());
    CHECK(result.objective_value == Approx(oracle.at("hs024_fopt")[0]).margin(1e-4));
    CHECK(result.objective_value == Approx(-1.0).margin(1e-4));
    CHECK(result.x(0) == Approx(oracle.at("hs024_xopt")[0]).margin(1e-3));
    CHECK(result.x(1) == Approx(oracle.at("hs024_xopt")[1]).margin(1e-3));
    CHECK(solver.constraint_violation() < 1e-6);
}

// Equality-constrained convergence: HS048/050/051 all have f* = 0 at the
// all-ones minimizer. The pre-rewrite driver reached wrong optima on these
// (the L1-sum merit and penalized-subgradient stand-in); the Powell-faithful
// max-violation merit + two-stage TRSTLP close them. Pinned against the
// checked-in NLopt (f2c'd Powell) reference optima.
namespace
{

template <typename Problem>
void check_cobyla_optimum(const char* name, double f_oracle,
                          const std::vector<double>& x_oracle)
{
    Problem problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 2000;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    step_budget_solver solver{cobyla_policy{}, problem, x0, opts};
    auto result = solver.solve();

    INFO(name << " objective: " << result.objective_value);
    INFO(name << " cv:        " << solver.constraint_violation());
    // Reaches f* = 0 (the oracle fopt is ~1e-20; our rhoend = 1e-8 stops a
    // little sooner, so pin to the shared true optimum with a small margin).
    CHECK(result.objective_value == Approx(0.0).margin(1e-6));
    CHECK(std::abs(result.objective_value - f_oracle) < 1e-6);
    for(int i = 0; i < static_cast<int>(x_oracle.size()); ++i)
        CHECK(result.x(i) == Approx(x_oracle[static_cast<std::size_t>(i)]).margin(1e-3));
    CHECK(solver.constraint_violation() < 1e-6);
}

}

TEST_CASE("cobyla_policy: HS048/050/051 equality-constrained optima", "[cobyla][hs]")
{
    const auto oracle = load_cobyla_oracle("oracles/cobyla_convergence.csv");
    REQUIRE(oracle.contains("hs048_fopt"));
    REQUIRE(oracle.contains("hs050_fopt"));
    REQUIRE(oracle.contains("hs051_fopt"));

    SECTION("HS048")
    {
        check_cobyla_optimum<hs048<>>("HS048", oracle.at("hs048_fopt")[0],
                                      oracle.at("hs048_xopt"));
    }
    SECTION("HS050")
    {
        check_cobyla_optimum<hs050<>>("HS050", oracle.at("hs050_fopt")[0],
                                      oracle.at("hs050_xopt"));
    }
    SECTION("HS051")
    {
        check_cobyla_optimum<hs051<>>("HS051", oracle.at("hs051_fopt")[0],
                                      oracle.at("hs051_xopt"));
    }
}

TEST_CASE("cobyla_policy: basic_solver_group compatibility", "[cobyla][solver_group]")
{
    simple_constrained problem;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    basic_solver_group<round_robin_schedule, dynamic_dimension, simple_constrained, cobyla_policy> group{
        problem, x0, opts};
    auto result = group.solve();

    CHECK(std::isfinite(result.objective_value));
}

// Verify that COBYLA returns normal (non-denormalized) objective values on all
// HS constrained problems. A denormalized value (nonzero but smaller than
// std::numeric_limits<double>::min() ~ 2.2e-308) indicates uninitialized memory
// or memory corruption rather than a legitimate computation result.
//
// Reference: IEEE 754-2019, Section 3.4 (subnormal numbers).
namespace
{

bool is_normal_or_zero(double v)
{
    return v == 0.0 || std::abs(v) >= std::numeric_limits<double>::min();
}

template <typename Problem>
void run_cobyla_denorm_check(const char* name)
{
    Problem problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 2000;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    step_budget_solver solver{cobyla_policy{}, problem, x0, opts};
    auto result = solver.solve();

    INFO(name << ": objective_value = " << result.objective_value);
    CHECK(std::isfinite(result.objective_value));
    CHECK(is_normal_or_zero(result.objective_value));
}

}

TEST_CASE("cobyla_policy: objective values are not denormalized on HS problems", "[cobyla][hs]")
{
    SECTION("HS006") { run_cobyla_denorm_check<hs006<>>("HS006"); }
    SECTION("HS007") { run_cobyla_denorm_check<hs007<>>("HS007"); }
    SECTION("HS024") { run_cobyla_denorm_check<hs024<>>("HS024"); }
    SECTION("HS026") { run_cobyla_denorm_check<hs026<>>("HS026"); }
    SECTION("HS028") { run_cobyla_denorm_check<hs028<>>("HS028"); }
    SECTION("HS035") { run_cobyla_denorm_check<hs035<>>("HS035"); }
    SECTION("HS039") { run_cobyla_denorm_check<hs039<>>("HS039"); }
    SECTION("HS040") { run_cobyla_denorm_check<hs040<>>("HS040"); }
    SECTION("HS043") { run_cobyla_denorm_check<hs043<>>("HS043"); }
    SECTION("HS048") { run_cobyla_denorm_check<hs048<>>("HS048"); }
    SECTION("HS050") { run_cobyla_denorm_check<hs050<>>("HS050"); }
    SECTION("HS051") { run_cobyla_denorm_check<hs051<>>("HS051"); }
    SECTION("HS071") { run_cobyla_denorm_check<hs071<>>("HS071"); }
    SECTION("HS076") { run_cobyla_denorm_check<hs076<>>("HS076"); }
}
