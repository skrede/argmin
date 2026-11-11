#include "scripted_claim_policy.h"

#include "argmin/solver/options.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/result/status.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <cstdint>
#include <optional>

using namespace argmin;
using namespace argmin::test;

namespace
{

template <typename S>
struct boxed_line
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    explicit boxed_line(S upper) : upper_{upper} {}

    std::int32_t dimension() const { return 1; }
    S value(const Eigen::VectorX<S>& x) const { return x(0); }
    Eigen::VectorX<S> lower_bounds() const { return Eigen::VectorX<S>::Constant(1, S(-10)); }
    Eigen::VectorX<S> upper_bounds() const { return Eigen::VectorX<S>::Constant(1, upper_); }

private:
    S upper_;
};

// The scripted iterate after step k is x = k, so the box decides whether every point the script claims is feasible.
template <typename S>
struct box_holding_policy : scripted_policy<S>
{
    struct state_type : scripted_policy<S>::state_type
    {
        Eigen::VectorX<S> lower;
        Eigen::VectorX<S> upper;
    };

    template <typename Problem, typename Convergence>
    state_type init(const Problem& problem, const Eigen::VectorX<S>& x0, const solver_options<Convergence>& opts)
    {
        return state_type{scripted_policy<S>::init(problem, x0, opts), problem.lower_bounds(), problem.upper_bounds()};
    }
};

template <typename Policy, typename S>
solve_result<S> solve_in_box(S upper, const scripted_step<S>& step)
{
    const boxed_line<S> problem{upper};
    Policy policy{};
    policy.script = {step, step};
    solver_options<default_convergence<S>> opts;
    opts.max_iterations = 3;
    step_budget_solver<Policy, dynamic_dimension, boxed_line<S>, default_convergence<S>> solver{policy, problem, Eigen::VectorX<S>::Constant(1, S(-1)), opts};
    return solver.solve();
}

template <typename S>
constexpr scripted_step<S> policy_stalled{feas_tol<S> / S(2), std::nullopt, solver_status::stalled};

template <typename Policy, typename S>
void check_box_gates_claims()
{
    SECTION("a converged claim at a point outside the box is no claim and the solve goes on")
    {
        const auto result = solve_in_box<Policy>(S(0.5), feasible<S>);
        CHECK(result.status == solver_status::max_iterations);
        CHECK(result.iterations == 3);
    }

    SECTION("a policy's own claim at a point outside the box ends the solve without the claim")
    {
        CHECK(solve_in_box<Policy>(S(0.5), policy_stalled<S>).status == solver_status::objective_stalled);
    }

    SECTION("the same claims inside the box are admitted")
    {
        CHECK(solve_in_box<Policy>(S(10), feasible<S>).status == solver_status::converged);
        CHECK(solve_in_box<Policy>(S(10), policy_stalled<S>).status == solver_status::stalled);
    }
}

}

TEMPLATE_TEST_CASE("a claim requires the iterate within the box for a family that keeps the box", "[solver][convergence]", double, float)
{
    check_box_gates_claims<box_holding_policy<TestType>, TestType>();
}

TEMPLATE_TEST_CASE("a claim requires the iterate within the box for a family that ignores the box", "[solver][convergence]", double, float)
{
    check_box_gates_claims<scripted_policy<TestType>, TestType>();
}
