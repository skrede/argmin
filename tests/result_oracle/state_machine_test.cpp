#include "oracle/state_machine.h"

#include "argmin/solver/options.h"
#include "argmin/solver/stepper.h"
#include "argmin/solver/lbfgsb_policy.h"

#include "argmin/test_functions/rosenbrock.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <cstdint>

using namespace argmin;

namespace
{

using rosenbrock2 = rosenbrock<double, 2>;
using rosenbrock_stepper = stepper<lbfgsb_policy<double, 2>, 2, rosenbrock2>;

struct diverging_policy
{
    using scalar_type = double;

    struct state_type
    {
        Eigen::Vector2d x;
    };

    template <typename Problem, typename Convergence>
    state_type init(const Problem&, const Eigen::Vector2d& x0, const solver_options<Convergence>&)
    {
        return {x0};
    }

    step_result<double> step(state_type& s) { return {.objective_value = 1.0, .x_norm = s.x.norm(), .policy_status = solver_status::diverged}; }

    solver_status reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
        return solver_status::running;
    }

    solver_status reset_clear(state_type& s, const Eigen::VectorXd& x0) { return reset(s, x0); }
};

rosenbrock_stepper rosenbrock_at_start(const rosenbrock2& problem)
{
    return {lbfgsb_policy<double, 2>{}, problem, problem.initial_point(), solver_options<>{}, lbfgsb_policy<double, 2>::options_type{}};
}

template <typename Stepper>
void step_until_finished(Stepper& st)
{
    for(std::uint32_t i = 0; i < 1000 && !st.finished(); ++i)
        st.step();
    REQUIRE(st.finished());
}

template <typename Stepper>
std::string sticky_step_failure(Stepper& st)
{
    const oracle::snapshot before = oracle::take_snapshot(st);
    const step_result<double> r = st.step();
    return oracle::sticky_failure(before, oracle::take_snapshot(st), r);
}

}

TEST_CASE("an abort through step() ends the stepper aborted without running the policy", "[oracle][state-machine]")
{
    const rosenbrock2 problem{};
    rosenbrock_stepper st = rosenbrock_at_start(problem);
    st.step();
    st.step();
    REQUIRE(st.status() == solver_status::running);
    const oracle::snapshot before = oracle::take_snapshot(st);
    st.abort();
    const step_result<double> r = st.step();
    CHECK(oracle::abort_failure(before, oracle::take_snapshot(st), r).empty());
}

TEST_CASE("a stepper stays aborted on every later step", "[oracle][state-machine]")
{
    const rosenbrock2 problem{};
    rosenbrock_stepper st = rosenbrock_at_start(problem);
    st.abort();
    st.step();
    CHECK(sticky_step_failure(st).empty());
}

TEST_CASE("a stepper keeps its claim status on a step after it", "[oracle][state-machine]")
{
    const rosenbrock2 problem{};
    rosenbrock_stepper st = rosenbrock_at_start(problem);
    step_until_finished(st);
    REQUIRE(oracle::is_claim(st.status()));
    CHECK(sticky_step_failure(st).empty());
}

TEST_CASE("a stepper keeps a policy-reported divergence on a step after it", "[oracle][state-machine]")
{
    const rosenbrock2 problem{};
    stepper<diverging_policy, 2, rosenbrock2> st{problem, problem.initial_point()};
    st.step();
    REQUIRE(st.status() == solver_status::diverged);
    CHECK(sticky_step_failure(st).empty());
}

TEST_CASE("stepper converged() and finished() follow the oracle's claim table", "[oracle][state-machine]")
{
    const rosenbrock2 problem{};
    rosenbrock_stepper claimed = rosenbrock_at_start(problem);
    CHECK(oracle::claim_query_failure(oracle::take_snapshot(claimed)).empty());
    step_until_finished(claimed);
    CHECK(oracle::claim_query_failure(oracle::take_snapshot(claimed)).empty());
    stepper<diverging_policy, 2, rosenbrock2> diverged{problem, problem.initial_point()};
    diverged.step();
    CHECK(oracle::claim_query_failure(oracle::take_snapshot(diverged)).empty());
    const rosenbrock<double> dynamic_problem{};
    stepper rejected{lbfgsb_policy<>{}, dynamic_problem, Eigen::VectorXd::Zero(3)};
    REQUIRE(rejected.status() == solver_status::invalid_problem);
    CHECK(oracle::claim_query_failure(oracle::take_snapshot(rejected)).empty());
}

TEST_CASE("a stepper reset clears its terminal status and steps again", "[oracle][state-machine]")
{
    const rosenbrock2 problem{};
    rosenbrock_stepper st = rosenbrock_at_start(problem);
    step_until_finished(st);
    st.reset(problem.initial_point());
    CHECK(oracle::reset_failure(oracle::take_snapshot(st), problem.initial_point()).empty());
    const step_result<double> r = st.step();
    CHECK(r.evaluations > 0);
}
