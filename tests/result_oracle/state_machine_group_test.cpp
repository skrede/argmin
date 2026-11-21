#include "oracle/state_machine.h"

#include "argmin/schedule/basic_solver_group.h"
#include "argmin/schedule/round_robin_schedule.h"

#include "argmin/solver/options.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

using namespace argmin;

namespace
{

struct bowl
{
    std::int32_t dimension() const { return 2; }
    double value(const Eigen::VectorXd& x) const { return 0.5 * x.squaredNorm(); }
};

// Every step a member takes is counted here, so a test sees which members the group stepped.
std::array<std::uint32_t, 2> member_steps{};

// Halves x on each step and reports End on its Last-th step.
template <std::size_t Id, std::uint32_t Last, solver_status End>
struct ending_policy
{
    using scalar_type = double;

    struct state_type
    {
        Eigen::VectorXd x;
        double objective_value;
        std::uint32_t steps;
    };

    template <typename Problem, typename Convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0, const solver_options<Convergence>&)
    {
        return {x0, 0.5 * x0.squaredNorm(), 0};
    }

    step_result<double> step(state_type& s)
    {
        ++member_steps[Id];
        const double previous = s.objective_value;
        s.x *= 0.5;
        s.objective_value = 0.5 * s.x.squaredNorm();
        const std::optional<solver_status> status = ++s.steps == Last ? std::optional{End} : std::nullopt;
        return {.objective_value = s.objective_value, .gradient_norm = s.x.norm(), .step_size = 0.5, .objective_change = s.objective_value - previous, .improved = true, .x_norm = s.x.norm(), .policy_status = status, .evaluations = 1};
    }

    solver_status reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s = {x0, 0.5 * x0.squaredNorm(), 0};
        return solver_status::running;
    }

    solver_status reset_clear(state_type& s, const Eigen::VectorXd& x0) { return reset(s, x0); }
};

using failing_pair = basic_solver_group<round_robin_schedule, dynamic_dimension, void, ending_policy<0, 1, solver_status::line_search_failed>, ending_policy<1, 1, solver_status::line_search_failed>>;
using mixed_pair = basic_solver_group<round_robin_schedule, dynamic_dimension, void, ending_policy<0, 1, solver_status::line_search_failed>, ending_policy<1, 3, solver_status::converged>>;

solver_options<> fifty_iterations()
{
    solver_options<> opts;
    opts.max_iterations = 50;
    return opts;
}

const Eigen::VectorXd start{{3.0, 2.0}};

}

TEST_CASE("a group whose members all end line_search_failed stops after one step each", "[oracle][state-machine][solver_group]")
{
    member_steps = {};
    failing_pair group{bowl{}, start, fifty_iterations()};
    const solve_result<double> result = group.step_n(50);
    CHECK(result.iterations == 2);
    CHECK(member_steps == std::array<std::uint32_t, 2>{1, 1});
    CHECK(result.status == solver_status::line_search_failed);
    CHECK(group.results()[0].status == solver_status::line_search_failed);
    CHECK(group.results()[1].status == solver_status::line_search_failed);
}

TEST_CASE("a retired group member is never stepped again and the group reports the returned member's status", "[oracle][state-machine][solver_group]")
{
    member_steps = {};
    mixed_pair group{bowl{}, start, fifty_iterations()};
    const solve_result<double> result = group.step_n(50);
    CHECK(result.iterations == 4);
    CHECK(member_steps == std::array<std::uint32_t, 2>{1, 3});
    CHECK(result.status == solver_status::converged);
    CHECK(group.results()[0].status == solver_status::line_search_failed);
    CHECK(group.results()[1].status == solver_status::converged);
}

TEST_CASE("a group step after every member retired returns the latched status and steps no member", "[oracle][state-machine][solver_group]")
{
    member_steps = {};
    mixed_pair group{bowl{}, start, fifty_iterations()};
    const solve_result<double> result = group.step_n(50);
    const step_result<double> r = group.step();
    CHECK(oracle::unevaluated_failure(r).empty());
    CHECK(r.policy_status == result.status);
    CHECK(group.step_n(10).iterations == 0);
    CHECK(member_steps == std::array<std::uint32_t, 2>{1, 3});
}
