#include "argmin/schedule/basic_solver_group.h"
#include "argmin/schedule/round_robin_schedule.h"

#include "argmin/solver/options.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <optional>

using namespace argmin;

namespace
{

struct line
{
    std::int32_t dimension() const { return 1; }
    double value(const Eigen::VectorXd& x) const { return x(0); }
};

struct boxed_line
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t dimension() const { return 1; }
    double value(const Eigen::VectorXd& x) const { return x(0); }
    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd::Constant(1, -1.0); }
    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd::Constant(1, 1.0); }
};

// Holds the point x = Position, with two equality rows of residual RowResidual each, at objective Objective and reports
// Claim as its policy status on every step (running: none).
template <std::int32_t Objective, solver_status Claim, std::int32_t Position = 0, double RowResidual = 0.0>
struct parked_member_policy
{
    using scalar_type = double;

    struct state_type
    {
        Eigen::VectorXd x;
        Eigen::VectorXd c_eq;
        double objective_value;
    };

    template <typename Problem, typename Convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0, const solver_options<Convergence>&)
    {
        return state_type{Eigen::VectorXd::Constant(x0.size(), static_cast<double>(Position)), Eigen::VectorXd::Constant(2, RowResidual), static_cast<double>(Objective)};
    }

    step_result<double> step(state_type& s)
    {
        const std::optional<solver_status> claim = Claim == solver_status::running ? std::nullopt : std::optional<solver_status>{Claim};
        return step_result<double>{.objective_value = s.objective_value, .gradient_norm = 1.0, .step_size = 1.0, .objective_change = 1.0, .x_norm = 1.0, .policy_status = claim};
    }

    solver_status reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
        return solver_status::running;
    }

    solver_status reset_clear(state_type& s, const Eigen::VectorXd& x0) { return reset(s, x0); }
};

template <typename First, typename Second>
double chosen_objective()
{
    basic_solver_group<round_robin_schedule, dynamic_dimension, void, First, Second> group{line{}, Eigen::VectorXd::Constant(1, 0.0)};
    return group.step_n(4).objective_value;
}

using lower_running = parked_member_policy<1, solver_status::running>;
using higher_running = parked_member_policy<2, solver_status::running>;
using lower_converged = parked_member_policy<1, solver_status::converged>;
using higher_converged = parked_member_policy<2, solver_status::converged>;
using lower_outside_box = parked_member_policy<1, solver_status::running, 5>;
// Two rows of 0.6e-6: their maximum is within the 1e-6 default feasibility tolerance, their sum is not.
using lower_rows_within_tolerance = parked_member_policy<1, solver_status::running, 0, 0.6e-6>;

template <typename First, typename Second>
double chosen_objective_in_box()
{
    basic_solver_group<round_robin_schedule, dynamic_dimension, boxed_line, First, Second> group{boxed_line{}, Eigen::VectorXd::Constant(1, 0.0)};
    return group.step_n(4).objective_value;
}

}

TEST_CASE("a solver group ranks its members as the solve loop ranks iterates", "[solver_group][feasibility]")
{
    SECTION("a certified member beats a lower-objective member that makes no claim, in either order")
    {
        CHECK(chosen_objective<lower_running, higher_converged>() == 2.0);
        CHECK(chosen_objective<higher_converged, lower_running>() == 2.0);
    }

    SECTION("members that agree on certification rank by objective")
    {
        CHECK(chosen_objective<higher_running, lower_running>() == 1.0);
        CHECK(chosen_objective<higher_converged, lower_converged>() == 1.0);
    }
}

TEST_CASE("a solver group ranks a member outside the box below one inside it", "[solver_group][feasibility]")
{
    CHECK(chosen_objective_in_box<lower_outside_box, higher_running>() == 2.0);
    CHECK(chosen_objective_in_box<higher_running, lower_outside_box>() == 2.0);
}

TEST_CASE("a solver group judges a member's rows on their largest residual, as a solve does", "[solver_group][feasibility]")
{
    basic_solver_group<round_robin_schedule, dynamic_dimension, void, lower_rows_within_tolerance, higher_running> group{line{}, Eigen::VectorXd::Constant(1, 0.0)};
    const solve_result<double> r = group.step_n(4);
    CHECK(r.objective_value == 1.0);
    CHECK(r.constraint_violation == 0.6e-6);
}
