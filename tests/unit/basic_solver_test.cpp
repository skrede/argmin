#include "mock_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/convergence.h"
#include "argmin/result/status.h"
#include "argmin/test_functions/rosenbrock.h"

#include <cmath>
#include <type_traits>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using namespace argmin;

// Dummy problem type -- mock_policy ignores it, but basic_solver needs one.
struct quadratic
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }
    double value(const Eigen::Vector<double, 2>& x) const { return 0.5 * x.squaredNorm(); }
};

// Gradient-bearing quadratic for the CTAD-with-custom-Convergence test below
// (lbfgsb_policy needs an analytic gradient; mock_policy has no rebind<N>
// so it cannot participate in bare CTAD).
struct quadratic_with_gradient
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }
    double value(const Eigen::Vector<double, 2>& x) const { return 0.5 * x.squaredNorm(); }
    void gradient(const Eigen::Vector<double, 2>& x, Eigen::Vector<double, 2>& g) const { g = x; }
};

// Scripted two-step policy for the best-seen-feasibility tolerance test
// below: step 1 reports a small constraint violation (5e-4) with a lower
// objective (5.0); step 2 reports exact feasibility (cv=0) with a higher
// objective (10.0). Under the default feasibility_tolerance (1e-6) step 1
// is infeasible and step 2 wins; under a widened tolerance (1e-3) step 1
// becomes feasible and wins on the lower objective. Deliberately has no
// c_eq/c_ineq state members, so seed_best_cv() reports the unconstrained
// fallback (0.0) and the constructed Problem is void (single-arg
// basic_solver<Policy> spelling), so seed_best_f() reports +infinity --
// both steps must out-compete that seed on their own terms.
struct scripted_feasibility_policy
{
    using scalar_type = double;

    struct state_type
    {
        Eigen::VectorXd x;
        double objective_value{};
        int step_count{0};
    };

    template <typename Problem, typename Convergence = argmin::default_convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0,
                    const argmin::solver_options<Convergence>&)
    {
        return state_type{.x = x0, .objective_value = 100.0};
    }

    argmin::step_result<double> step(state_type& s)
    {
        ++s.step_count;
        // gradient_norm/step_size/objective_change are deliberately set
        // well above every default_convergence criterion's direct-value
        // literature default (see convergence.h) so this scripted budget
        // runs to completion instead of tripping an unrelated criterion
        // early -- this test isolates the best-seen feasibility-tolerance
        // comparator, not the convergence policy.
        if(s.step_count == 1)
        {
            s.objective_value = 5.0;
            return argmin::step_result<double>{
                .objective_value = 5.0,
                .gradient_norm = 1.0,
                .step_size = 1.0,
                .objective_change = -95.0,
                .constraint_violation = 5e-4,
            };
        }
        s.objective_value = 10.0;
        return argmin::step_result<double>{
            .objective_value = 10.0,
            .gradient_norm = 1.0,
            .step_size = 1.0,
            .objective_change = 5.0,
            .constraint_violation = 0.0,
        };
    }

    void reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
        s.objective_value = 100.0;
        s.step_count = 0;
    }

    void reset_clear(state_type& s, const Eigen::VectorXd& x0)
    {
        reset(s, x0);
    }
};

// Policy that performs a fixed number of objective evaluations per step
// (mimicking a line search) and never converges by criterion, so step_n's
// iteration count equals the budget while its evaluation count is a genuine
// multiple of it. Pins that solve_result::function_evaluations accumulates
// the real per-step count instead of aliasing the iteration count.
struct multi_eval_policy
{
    using scalar_type = double;

    struct state_type
    {
        Eigen::VectorXd x;
        double objective_value{};
    };

    template <typename Problem, typename Convergence = argmin::default_convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0,
                    const argmin::solver_options<Convergence>&)
    {
        return state_type{.x = x0, .objective_value = 100.0};
    }

    static constexpr std::uint32_t evals_per_step = 3;

    argmin::step_result<double> step(state_type&)
    {
        return argmin::step_result<double>{
            .objective_value = 100.0,
            .gradient_norm = 10.0,
            .step_size = 1.0,
            .objective_change = -1.0,
            .improved = true,
            .evaluations = evals_per_step,
        };
    }

    void reset(state_type& s, const Eigen::VectorXd& x0) { s.x = x0; }
    void reset_clear(state_type& s, const Eigen::VectorXd& x0) { reset(s, x0); }
};

TEST_CASE("basic_solver step", "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    solver_options opts;

    basic_solver<test::mock_policy> solver{prob, x0, opts};

    SECTION("single step returns valid step_result")
    {
        auto sr = solver.step();
        CHECK(sr.objective_value < 0.5 * x0.squaredNorm());
        CHECK(sr.gradient_norm > 0.0);
        CHECK(sr.improved);
    }

    SECTION("step modifies state")
    {
        solver.step();
        CHECK(solver.state().x.norm() < x0.norm());
    }
}

TEST_CASE("basic_solver solve converges", "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options<> opts;
    std::get<gradient_tolerance_criterion>(opts.convergence.criteria).threshold = 1e-4;

    basic_solver<test::mock_policy> solver{prob, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.status == solver_status::converged);
    CHECK(result.iterations > 0);
    CHECK(result.gradient_norm < 1e-4);
    CHECK(result.x.norm() < 1e-3);
}

TEST_CASE("basic_solver solve max_iterations", "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{1.0, 1.0}};
    solver_options<> opts;
    opts.max_iterations = 10;

    basic_solver<test::non_converging_policy> solver{prob, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.status == solver_status::max_iterations);
    CHECK(result.iterations == 10);
}

TEST_CASE("basic_solver step_n budget", "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{10.0, 10.0}};
    solver_options<> opts;
    opts.max_iterations = 1000;

    basic_solver<test::non_converging_policy> solver{prob, x0, opts};
    auto result = solver.step_n(3, opts);

    CHECK(result.status == solver_status::budget_exhausted);
    CHECK(result.iterations == 3);
}

TEST_CASE("basic_solver state access", "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{5.0, -3.0}};
    solver_options opts;

    basic_solver<test::mock_policy> solver{prob, x0, opts};

    CHECK(solver.state().x.isApprox(x0));
    CHECK(solver.state().objective_value == Approx(0.5 * x0.squaredNorm()));
}

TEST_CASE("basic_solver convergence on objective_change", "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{0.001, 0.001}};
    solver_options<> opts;
    // Loosen gradient/step so the objective (ftol) criterion is isolated
    // as the one under test: gradient_tolerance_criterion now carries a
    // direct-value literature default (1e-5, see convergence.h) and would
    // otherwise fire first on this geometrically-shrinking mock trajectory.
    std::get<gradient_tolerance_criterion>(opts.convergence.criteria).threshold = 1e-15;
    std::get<step_tolerance_criterion>(opts.convergence.criteria).threshold = 1e-15;
    std::get<objective_tolerance_criterion>(opts.convergence.criteria).threshold = 1e-8;

    basic_solver<test::mock_policy> solver{prob, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.status == solver_status::ftol_reached);
}

TEST_CASE("basic_solver solve_uses_stored_convergence", "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    solver_options<> opts;
    std::get<gradient_tolerance_criterion>(opts.convergence.criteria).threshold = 1e-4;

    basic_solver<test::mock_policy> solver{prob, x0, opts};
    auto result = solver.solve();

    CHECK(result.status == solver_status::converged);
    CHECK(result.iterations < 500);
    CHECK(result.gradient_norm < 1e-4);
}

TEST_CASE("basic_solver step_n_no_opts_uses_stored_convergence", "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    solver_options<> opts;
    std::get<gradient_tolerance_criterion>(opts.convergence.criteria).threshold = 1e-4;

    basic_solver<test::mock_policy> solver{prob, x0, opts};
    auto result = solver.step_n(1000);

    CHECK(result.status == solver_status::converged);
    CHECK(result.iterations < 500);
    CHECK(result.gradient_norm < 1e-4);
}

TEST_CASE("basic_solver step_n(budget, opts) back-copies last_check_results",
          "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    solver_options<> opts;
    std::get<gradient_tolerance_criterion>(opts.convergence.criteria).threshold = 1e-4;

    basic_solver<test::mock_policy> solver{prob, x0, opts};

    // Before any step: all entries are nullopt.
    const auto& before = solver.convergence().last_check_results();
    CHECK(!before[0].has_value());
    CHECK(!before[1].has_value());
    CHECK(!before[2].has_value());
    CHECK(!before[3].has_value());

    // Explicit-opts step_n: this is the cartan-side usage pattern that
    // previously did not back-copy into stored_convergence_.
    auto result = solver.step_n(1000, opts);
    REQUIRE(result.status == solver_status::converged);

    // After: solver.convergence().last_check_results() mirrors what was
    // written into opts.convergence through the const-ref check() calls.
    const auto& after = solver.convergence().last_check_results();
    REQUIRE(after[0].has_value());
    CHECK(*after[0] == solver_status::converged);

    // Same array on both sides -- this is what cartan wanted for the
    // accessor symmetry between the no-opts and explicit-opts paths.
    CHECK(after[0] == opts.convergence.last_check_results()[0]);
}

TEST_CASE("basic_solver step_n(budget, opts) back-copy is gated on Convergence type",
          "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    solver_options<slsqp_compatible_convergence> alias_opts;
    std::get<0>(alias_opts.convergence.criteria).threshold = 1e-10;
    std::get<0>(alias_opts.convergence.criteria).stationarity_threshold = 1.0;
    std::get<1>(alias_opts.convergence.criteria).threshold = 1e-10;

    basic_solver<test::non_converging_policy> solver{prob, x0, solver_options<>{}};

    // step_n with non-default Convergence compiles and runs through the
    // same code path, but the if-constexpr guard skips the back-copy into
    // stored_convergence_ because default_convergence has four criteria
    // and slsqp_compatible_convergence has three. Consumers read from
    // their own alias_opts.convergence in this case.
    auto result = solver.step_n(5, alias_opts);
    CHECK(result.iterations == 5);

    // stored_convergence_ is untouched -- still all nullopt.
    const auto& stored = solver.convergence().last_check_results();
    CHECK(!stored[0].has_value());
    CHECK(!stored[1].has_value());
    CHECK(!stored[2].has_value());
    CHECK(!stored[3].has_value());
}

TEST_CASE("basic_solver reset preserves convergence ability", "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{10.0, 10.0}};
    solver_options<> opts;
    std::get<gradient_tolerance_criterion>(opts.convergence.criteria).threshold = 1e-4;

    basic_solver<test::mock_policy> solver{prob, x0, opts};

    SECTION("reset to new x0 and solve again")
    {
        auto result1 = solver.solve(opts);
        CHECK(result1.status == solver_status::converged);

        Eigen::VectorXd new_x0{{5.0, 5.0}};
        solver.reset(new_x0);

        CHECK(solver.state().x.isApprox(new_x0));

        auto result2 = solver.solve(opts);
        CHECK(result2.status == solver_status::converged);
    }

    SECTION("reset_clear to new x0 and solve again")
    {
        auto result1 = solver.solve(opts);
        CHECK(result1.status == solver_status::converged);

        Eigen::VectorXd new_x0{{5.0, 5.0}};
        solver.reset_clear(new_x0);

        CHECK(solver.state().x.isApprox(new_x0));

        auto result2 = solver.solve(opts);
        CHECK(result2.status == solver_status::converged);
    }
}

// ---------------------------------------------------------------------------
// User-Convergence storage without lossy remap.
// ---------------------------------------------------------------------------

TEST_CASE("basic_solver stores a non-default explicit Convergence and honors it on the defaulted solve() path",
          "[solver]")
{
    // mock_policy has no rebind<N>, so it cannot participate in bare CTAD;
    // the Convergence template parameter is spelled out explicitly here
    // (basic_solver<Policy, N, Problem, Convergence>) instead.
    using rel_conv = convergence_policy<objective_tolerance_rel_criterion>;

    quadratic prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    solver_options<rel_conv> opts;
    opts.max_iterations = 1000;
    opts.convergence.criteria = {objective_tolerance_rel_criterion{.threshold = 1e-2}};

    basic_solver<test::mock_policy, argmin::dynamic_dimension, void, rel_conv> solver{prob, x0, opts};

    // Defaulted call path (no opts argument): must use the solver's own
    // stored rel_conv, not a coerced default_convergence.
    auto result = solver.solve();

    CHECK(result.status == solver_status::ftol_reached);
    CHECK(result.iterations < 1000);
}

TEST_CASE("basic_solver CTAD deduces a non-default Convergence and honors it on the defaulted solve() path",
          "[solver]")
{
    // lbfgsb_policy supports rebind<N>, so Convergence flows through the
    // "Policy + Problem + x0 + opts" deduction guide via bare CTAD -- no
    // template arguments are spelled out at all.
    using rel_conv = convergence_policy<step_tolerance_rel_criterion>;

    quadratic_with_gradient prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    solver_options<rel_conv> opts;
    opts.max_iterations = 1000;
    std::get<0>(opts.convergence.criteria).threshold = 0.5;

    basic_solver solver{lbfgsb_policy<>{}, prob, x0, opts};
    auto result = solver.solve();

    // Stops on the relative step criterion (xtol_reached), not on a
    // coerced absolute one -- default_convergence has no
    // step_tolerance_rel_criterion slot at all, so this can only pass if
    // the solver's stored Convergence really is rel_conv.
    CHECK(result.status == solver_status::xtol_reached);
    CHECK(result.iterations < 1000);
}

// ---------------------------------------------------------------------------
// Construction-time constraint/feasibility tolerance honored on defaulted
// call paths (solve() / step_n(budget), no opts argument).
// ---------------------------------------------------------------------------

TEST_CASE("basic_solver honors a widened construction-time feasibility_tolerance on step_n(budget)",
          "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{1.0, 1.0}};

    SECTION("default feasibility_tolerance: the exactly-feasible, higher-objective step wins")
    {
        solver_options<> opts;  // feasibility_tolerance left at its 1e-6 default
        basic_solver<scripted_feasibility_policy> solver{prob, x0, opts};
        auto result = solver.step_n(2);

        CHECK(result.objective_value == Approx(10.0));
        CHECK(result.constraint_violation == Approx(0.0));
    }

    SECTION("widened construction-time feasibility_tolerance: the lower-objective step wins")
    {
        solver_options<> opts;
        opts.feasibility_tolerance = 1e-3;  // configured at construction

        basic_solver<scripted_feasibility_policy> solver{prob, x0, opts};
        auto result = solver.step_n(2);

        CHECK(result.objective_value == Approx(5.0));
        CHECK(result.constraint_violation == Approx(5e-4));
    }
}

TEST_CASE("basic_solver honors a widened construction-time feasibility_tolerance on solve()",
          "[solver]")
{
    quadratic prob;
    Eigen::VectorXd x0{{1.0, 1.0}};

    solver_options<> opts;
    opts.max_iterations = 2;
    opts.feasibility_tolerance = 1e-3;

    basic_solver<scripted_feasibility_policy> solver{prob, x0, opts};
    auto result = solver.solve();

    CHECK(result.objective_value == Approx(5.0));
    CHECK(result.constraint_violation == Approx(5e-4));
}

// ---------------------------------------------------------------------------
// CTAD rebind preserves configured policy options and accepts lvalue policies.
// ---------------------------------------------------------------------------

TEST_CASE("basic_solver CTAD rebind preserves configured policy options",
          "[solver][ctad]")
{
    // quadratic_with_gradient::problem_dimension == 2, so the deduction guide
    // rebinds lbfgsb_policy<> (dynamic) to lbfgsb_policy<2>: the converting
    // constructor -- not the plain same-type constructor -- runs. The
    // configured option must survive the rebind rather than being reset by a
    // default-constructed rebound policy.
    quadratic_with_gradient prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    solver_options<> opts;

    SECTION("lvalue policy is accepted and its options survive the rebind")
    {
        lbfgsb_policy<> configured;
        configured.options.history_depth = 7;  // non-default configured option

        basic_solver solver{configured, prob, x0, opts};

        // The converting ctor really ran: the stored policy is the rebound
        // fixed-dimension type, not the dynamic source type.
        static_assert(std::is_same_v<
            std::remove_cvref_t<decltype(solver.policy())>, lbfgsb_policy<2>>);

        REQUIRE(solver.policy().options.history_depth.has_value());
        CHECK(*solver.policy().options.history_depth == 7);
    }

    SECTION("rvalue policy carrying configured options preserves them")
    {
        lbfgsb_policy<> configured;
        configured.options.history_depth = 9;

        basic_solver solver{std::move(configured), prob, x0, opts};

        REQUIRE(solver.policy().options.history_depth.has_value());
        CHECK(*solver.policy().options.history_depth == 9);
    }
}

// ---------------------------------------------------------------------------
// step() consults convergence and updates status; solve() re-entrancy.
// ---------------------------------------------------------------------------

TEST_CASE("basic_solver step() consults convergence and reports a by-criterion status",
          "[solver][step]")
{
    quadratic_with_gradient prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    solver_options<> opts;

    basic_solver solver{lbfgsb_policy<>{}, prob, x0, opts};

    // No step yet: the solver is running.
    CHECK(solver.status() == solver_status::running);

    // Pure step() tick loop (no step_n): the stored default convergence must
    // flip status() to a by-criterion terminal state at the stationary
    // iterate, with no criterion re-implemented by the caller.
    bool terminal = false;
    for(int i = 0; i < 200; ++i)
    {
        solver.step();
        if(solver.status() != solver_status::running)
        {
            terminal = true;
            break;
        }
    }
    REQUIRE(terminal);
    CHECK(solver.status() == solver_status::converged);
}

TEST_CASE("basic_solver solve() re-entrancy is a continuation; reset() restarts",
          "[solver][reentrancy]")
{
    quadratic prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    solver_options<> opts;
    std::get<gradient_tolerance_criterion>(opts.convergence.criteria).threshold = 1e-4;

    basic_solver<test::mock_policy> solver{prob, x0, opts};

    auto r1 = solver.solve(opts);
    REQUIRE(r1.status == solver_status::converged);
    const auto n1 = r1.iterations;
    REQUIRE(n1 > 0);

    // Second solve() WITHOUT reset is a continuation: the iteration counter
    // accumulates, so the solver resumes from the converged iterate and
    // reports a strictly larger cumulative count.
    auto r2 = solver.solve(opts);
    CHECK(r2.iterations > n1);

    // reset() restarts the counter and the iterate; the next solve() reports
    // the same fresh count as the first, strictly below the continuation.
    solver.reset(x0);
    auto r3 = solver.solve(opts);
    CHECK(r3.iterations == n1);
    CHECK(r3.iterations < r2.iterations);
}

// ---------------------------------------------------------------------------
// Truthful telemetry: function_evaluations counts real evaluations.
// ---------------------------------------------------------------------------

TEST_CASE("basic_solver reports genuine function_evaluations, not an iteration alias",
          "[solver][telemetry]")
{
    quadratic prob;
    Eigen::VectorXd x0{{1.0, 1.0}};
    solver_options<> opts;

    basic_solver<multi_eval_policy> solver{prob, x0, opts};
    auto result = solver.step_n(5);

    CHECK(result.iterations == 5);
    // Three evaluations per iteration, genuinely accumulated -- impossible
    // under the old function_evaluations = iterations_ alias.
    CHECK(result.function_evaluations == 5 * multi_eval_policy::evals_per_step);
    CHECK(result.function_evaluations > result.iterations);
}

TEST_CASE("basic_solver function_evaluations counts real line-search evaluations",
          "[solver][telemetry]")
{
    // Rosenbrock's curved valley forces the strong-Wolfe line search to make
    // several trial evaluations per iteration, so the accumulated evaluation
    // count must exceed the iteration count.
    rosenbrock<> prob{.n = 2};
    Eigen::VectorXd x0{{-1.2, 1.0}};
    solver_options<> opts;
    opts.max_iterations = 200;

    basic_solver solver{lbfgsb_policy<>{}, prob, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.iterations > 0);
    CHECK(result.function_evaluations > result.iterations);
}
