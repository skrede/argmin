#include "argmin/solver/options.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/filter_trsqp_policy.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"

#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <optional>

using namespace argmin;

namespace
{

struct line
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t dimension() const { return 1; }
    double value(const Eigen::VectorXd& x) const { return x(0); }
};

// Step 1 lowers the objective by leaning on a violation inside the default 1e-6 tolerance; step 2 is
// exactly feasible at a higher objective and either reports `final_status` or trips the step criterion.
struct leaning_then_final_policy
{
    using scalar_type = double;

    struct state_type
    {
        Eigen::VectorXd x;
        std::int32_t step_count;
    };

    std::optional<solver_status> final_status;

    template <typename Problem, typename Convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0, const solver_options<Convergence>&)
    {
        return state_type{x0, 0};
    }

    step_result<double> step(state_type& s)
    {
        ++s.step_count;
        s.x(0) = s.step_count == 1 ? 1.0 : 2.0;
        if(s.step_count == 1)
            return step_result<double>{.objective_value = 1.0, .gradient_norm = 1.0, .step_size = 1.0, .constraint_violation = 5e-7, .x_norm = 1.0};
        return step_result<double>{.objective_value = 2.0, .gradient_norm = 1.0, .step_size = 0.0, .constraint_violation = 0.0, .x_norm = 2.0, .policy_status = final_status};
    }

    solver_status reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s = state_type{x0, 0};
        return solver_status::running;
    }

    solver_status reset_clear(state_type& s, const Eigen::VectorXd& x0) { return reset(s, x0); }
};

solve_result<double, dynamic_dimension> solve_leaning(std::optional<solver_status> final_status)
{
    solver_options<> opts;
    opts.max_iterations = 3;
    step_budget_solver<leaning_then_final_policy, dynamic_dimension, line> solver{leaning_then_final_policy{final_status}, line{}, Eigen::VectorXd::Constant(1, 3.0), opts};
    return solver.solve();
}

}

TEST_CASE("a solve whose status certifies its final iterate returns that iterate", "[solver]")
{
    SECTION("certified by the convergence policy")
    {
        const auto result = solve_leaning(std::nullopt);
        CHECK(result.status == solver_status::stalled);
        CHECK(result.x(0) == 2.0);
        CHECK(result.objective_value == 2.0);
    }

    SECTION("certified by the policy")
    {
        const auto result = solve_leaning(solver_status::converged);
        CHECK(result.status == solver_status::converged);
        CHECK(result.x(0) == 2.0);
        CHECK(result.constraint_violation == 0.0);
    }

    SECTION("a claimless failure keeps the best iterate seen")
    {
        const auto result = solve_leaning(solver_status::line_search_failed);
        CHECK(result.status == solver_status::line_search_failed);
        CHECK(result.x(0) == 1.0);
    }
}

TEST_CASE("filter trust-region SQP returns the hs071 iterate its stalled status describes", "[solver]")
{
    const hs071<> problem;
    solver_options<> opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-6);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);
    step_budget_solver solver{filter_trsqp_policy<>{}, problem, problem.initial_point(), opts};
    const auto result = solver.solve();

    REQUIRE(result.status == solver_status::stalled);
    CHECK(result.x == solver.state().x);
    CHECK(result.objective_value == problem.value(solver.state().x));
}
