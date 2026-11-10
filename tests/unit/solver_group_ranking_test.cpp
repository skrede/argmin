#include "argmin/schedule/basic_solver_group.h"
#include "argmin/schedule/round_robin_schedule.h"

#include "argmin/solver/options.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"

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

// Holds a feasible point of objective Objective and reports Claim as its policy status on every step (running: none).
template <std::int32_t Objective, solver_status Claim>
struct parked_member_policy
{
    using scalar_type = double;

    struct state_type
    {
        Eigen::VectorXd x;
        double objective_value;
    };

    template <typename Problem, typename Convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0, const solver_options<Convergence>&)
    {
        return state_type{x0, static_cast<double>(Objective)};
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
