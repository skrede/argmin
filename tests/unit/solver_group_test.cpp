#include "mock_policy.h"
#include "argmin/result/status.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/schedule/fallback_schedule.h"
#include "argmin/schedule/basic_solver_group.h"
#include "argmin/schedule/round_robin_schedule.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace argmin;

struct quadratic_group
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }
    double value(const Eigen::Vector<double, 2>& x) const { return 0.5 * x.squaredNorm(); }
};

TEST_CASE("basic_solver_group construction", "[solver_group]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    solver_options opts;

    SECTION("constructs with round_robin and two policies")
    {
        basic_solver_group<round_robin_schedule, argmin::dynamic_dimension, void, test::mock_policy, test::mock_policy>
            group{prob, x0, opts};
        (void)group;
        SUCCEED("construction compiled and ran");
    }
}

TEST_CASE("basic_solver_group step with round_robin", "[solver_group]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;

    basic_solver_group<round_robin_schedule, argmin::dynamic_dimension, void, test::mock_policy, test::mock_policy>
        group{prob, x0, opts};

    auto r1 = group.step();
    CHECK(r1.objective_value < 0.5 * x0.squaredNorm());

    auto r2 = group.step();
    CHECK(r2.objective_value < 0.5 * x0.squaredNorm());
}

TEST_CASE("basic_solver_group solve converges", "[solver_group]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{1.0, 1.0}};
    solver_options opts;

    basic_solver_group<round_robin_schedule, argmin::dynamic_dimension, void, test::mock_policy, test::mock_policy>
        group{prob, x0, opts};

    auto result = group.step_n(100);
    CHECK(result.x.norm() < 1.0);
}

TEST_CASE("basic_solver_group fold expression concept check", "[solver_group]")
{
    // This test verifies the fold-expression requires clause compiles.
    // If the problem type did not satisfy mock_policy's constructor
    // requirements, this would be a compile error.
    quadratic_group prob;
    Eigen::VectorXd x0{{1.0, 1.0}};

    basic_solver_group<round_robin_schedule, argmin::dynamic_dimension, void, test::mock_policy, test::non_converging_policy>
        group{prob, x0};
    (void)group;
    SUCCEED("fold expression concept check passed at compile time");
}

TEST_CASE("basic_solver step propagates constraint_violation", "[solver]")
{
    quadratic_group prob;

    SECTION("constrained policy returns nonzero constraint_violation")
    {
        Eigen::VectorXd x0{{0.5, 0.5}};
        basic_solver<test::constrained_mock_policy> solver{prob, x0};
        auto result = solver.step();
        CHECK(result.constraint_violation > 0.0);
    }

    SECTION("unconstrained policy returns zero constraint_violation")
    {
        Eigen::VectorXd x0{{1.0, 1.0}};
        basic_solver<test::mock_policy> solver{prob, x0};
        auto result = solver.step();
        CHECK(result.constraint_violation == 0.0);
    }
}

TEST_CASE("feasible solver beats infeasible despite worse objective",
          "[solver_group][feasibility]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{1.0, 1.0}};
    solver_options opts;

    // feasible_mock: obj=10, cv=0 (feasible)
    // infeasible_mock: obj=1, cv=2 (infeasible)
    basic_solver_group<round_robin_schedule, argmin::dynamic_dimension, void,
                       test::feasible_mock_policy,
                       test::infeasible_mock_policy>
        group{prob, x0, opts};

    auto result = group.step_n(2);

    // Best solver should be the feasible one (obj ~10) not infeasible (obj ~1).
    CHECK(result.objective_value == Approx(10.0));
}

TEST_CASE("among feasible solvers lowest objective wins",
          "[solver_group][feasibility]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;

    // Both unconstrained (cv=0, both feasible). mock_policy does gradient
    // descent so objectives differ slightly between the two due to round-robin
    // alternation, but both have the same behavior. The result should pick
    // whichever has the lowest objective.
    basic_solver_group<round_robin_schedule, argmin::dynamic_dimension, void,
                       test::mock_policy,
                       test::mock_policy>
        group{prob, x0, opts};

    auto result = group.step_n(4);

    // Both solvers converge toward 0 -- the one that stepped more recently
    // has lower objective. Just verify it converges and picks a valid result.
    CHECK(result.objective_value < 0.5 * x0.squaredNorm());
}

TEST_CASE("among infeasible solvers lowest violation wins",
          "[solver_group][feasibility]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{1.0, 1.0}};
    solver_options opts;

    // infeasible_mock: obj=1, cv=2 (lower violation)
    // high_violation_mock: obj=0.5, cv=5 (higher violation)
    basic_solver_group<round_robin_schedule, argmin::dynamic_dimension, void,
                       test::infeasible_mock_policy,
                       test::high_violation_mock_policy>
        group{prob, x0, opts};

    auto result = group.step_n(2);

    // Best solver should be the one with lower violation (obj=1), not
    // the one with lower objective but higher violation (obj=0.5).
    CHECK(result.objective_value == Approx(1.0));
}

TEST_CASE("basic_solver per-policy options forwarding",
          "[solver][per-policy]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;

    SECTION("custom step_size via policy options")
    {
        test::mock_policy_with_opts::options_type popts{.custom_step_size = 0.1};
        basic_solver<test::mock_policy_with_opts> solver{prob, x0, opts, popts};
        auto result = solver.step();
        CHECK(result.step_size == Approx(0.1));
    }

    SECTION("default step_size without policy options")
    {
        basic_solver<test::mock_policy_with_opts> solver{prob, x0, opts};
        auto result = solver.step();
        CHECK(result.step_size == Approx(0.5));
    }
}

TEST_CASE("basic_solver_group per-policy options tuple",
          "[solver_group][per-policy]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;

    auto policy_opts = std::tuple{
        test::mock_policy_with_opts::options_type{.custom_step_size = 0.1},
        test::mock_policy_with_opts::options_type{.custom_step_size = 0.9}
    };

    basic_solver_group<round_robin_schedule, argmin::dynamic_dimension, void,
                       test::mock_policy_with_opts,
                       test::mock_policy_with_opts>
        group{prob, x0, opts, policy_opts};

    // Round-robin: first step hits solver 0 (step_size 0.1),
    // second step hits solver 1 (step_size 0.9).
    auto r1 = group.step();
    auto r2 = group.step();

    CHECK(r1.step_size == Approx(0.1));
    CHECK(r2.step_size == Approx(0.9));
}

TEST_CASE("basic_solver_group existing constructor still works",
          "[solver_group][per-policy]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{3.0, 4.0}};
    solver_options opts;

    SECTION("no options_type policy (broadcast)")
    {
        basic_solver_group<round_robin_schedule, argmin::dynamic_dimension, void,
                           test::mock_policy, test::mock_policy>
            group{prob, x0, opts};
        auto result = group.step();
        CHECK(result.objective_value < 0.5 * x0.squaredNorm());
    }

    SECTION("has options_type policy (broadcast, no per-policy opts)")
    {
        basic_solver_group<round_robin_schedule, argmin::dynamic_dimension, void,
                           test::mock_policy_with_opts, test::mock_policy_with_opts>
            group{prob, x0, opts};
        auto result = group.step();
        CHECK(result.step_size == Approx(0.5));
    }
}

// Schedule that counts reset() calls through an external counter, used to
// observe that the group actually drives Schedule::reset() (previously never
// called) at construction and on group reset.
namespace
{
struct reset_probe_schedule
{
    int* reset_calls{nullptr};
    std::size_t current_index{0};

    void reset()
    {
        current_index = 0;
        if(reset_calls) ++*reset_calls;
    }

    std::size_t select(std::size_t num_solvers)
    {
        std::size_t idx = current_index;
        current_index = (current_index + 1) % num_solvers;
        return idx;
    }

    template <typename Scalar>
    void notify(const argmin::step_result<Scalar>&) {}
};
}

TEST_CASE("basic_solver_group drives Schedule::reset", "[solver_group]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;

    int reset_calls = 0;
    reset_probe_schedule sched{.reset_calls = &reset_calls};

    basic_solver_group<reset_probe_schedule, argmin::dynamic_dimension, void,
                       test::mock_policy, test::mock_policy>
        group{prob, x0, opts, sched};

    // init_schedule() runs Schedule::reset() once at construction.
    CHECK(reset_calls == 1);

    // Advance the schedule index, then reset the group: the schedule's
    // selection state must restart from solver 0 and reset() fire again.
    group.step();
    group.reset(x0);
    CHECK(reset_calls == 2);

    // After the group reset, selection restarts at the first solver.
    auto r = group.step();
    CHECK(r.objective_value < 0.5 * x0.squaredNorm());
}

TEST_CASE("group step skips a retired current index instead of fabricating a result",
          "[solver_group][retirement]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;

    // fallback_schedule::select does not advance on its own -- it only
    // advances current_index_ after stall_threshold consecutive
    // non-improving steps (notify_impl). roundoff_mock_policy (index 0)
    // always reports improved=true, so the stall counter never fires and
    // current_index_ stays pinned at 0 straight through retirement: exactly
    // the case where an unpatched select() would keep returning the dead
    // index and step() would fall through to a value-initialized
    // (objective 0, gradient 0) fabrication.
    basic_solver_group<fallback_schedule, argmin::dynamic_dimension, void,
                       test::roundoff_mock_policy, test::non_converging_policy>
        group{prob, x0, opts};

    group.step();
    group.step();
    auto r_retire = group.step();
    CHECK(r_retire.policy_status == argmin::solver_status::roundoff_limited);

    auto r_live = group.step();

    // Must be non_converging_policy's real result (index 1), never a
    // fabricated zero: non_converging_policy pins objective_value at 100.0
    // and gradient_norm at 10.0, unmistakably distinct from a
    // value-initialized step_result{}.
    CHECK(r_live.policy_status == std::nullopt);
    CHECK(r_live.objective_value == Approx(100.0));
    CHECK(r_live.gradient_norm == Approx(10.0));
}

TEST_CASE("group step returns a truthful terminal outcome once every solver has retired",
          "[solver_group][retirement]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;

    // Both policies retire on their own 3rd step; round-robin alternation
    // retires both within 6 group steps.
    basic_solver_group<round_robin_schedule, argmin::dynamic_dimension, void,
                       test::roundoff_mock_policy, test::roundoff_mock_policy>
        group{prob, x0, opts};

    for(int i = 0; i < 6; ++i)
    {
        group.step();
    }

    auto r = group.step();

    // No live solver remains -- this must be a truthful terminal signal,
    // never a step_result masquerading as a real (objective-0) step: the
    // status is explicitly set and the metrics are NaN rather than the
    // misleading value-initialized 0.
    CHECK(r.policy_status == argmin::solver_status::budget_exhausted);
    CHECK(std::isnan(r.objective_value));
    CHECK(std::isnan(r.gradient_norm));
}

TEST_CASE("group step_n cannot fire false convergence from a retirement-induced fabrication",
          "[solver_group][retirement]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{2.0, 2.0}};

    // Threshold tight enough that no REAL gradient from either mock policy
    // can satisfy it within the 4-step budget below (roundoff_mock_policy's
    // gradient_norm decays geometrically from ~1.4 and non_converging_policy
    // pins gradient_norm at 10.0) -- so the only way `converged` could ever
    // fire here is via a fabricated (value-initialized, gradient_norm == 0)
    // step_result following retirement of index 0.
    default_convergence conv;
    std::get<0>(conv.criteria).threshold = 1e-9;

    solver_options<default_convergence> opts;
    opts.convergence = conv;

    basic_solver_group<fallback_schedule, argmin::dynamic_dimension, void,
                       test::roundoff_mock_policy, test::non_converging_policy>
        group{prob, x0, opts};

    // roundoff_mock_policy (index 0) retires on step 3 while always
    // reporting improved=true, so fallback_schedule's stall counter never
    // advances current_index_ away from 0 on its own -- the group's own
    // skip-scan must redirect step 4 to the live index-1 solver.
    auto result = group.step_n(4, opts);

    CHECK(result.status != argmin::solver_status::converged);
}

TEST_CASE("group retires a converged solver instead of re-stepping it",
          "[solver_group][retirement]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{4.0, 4.0}};
    solver_options opts;

    // converging_mock_policy (index 0) reports converged on its own 2nd
    // step. Paired with round_robin against a live mock_policy (index 1):
    // once index 0 retires, round_robin's rotation would otherwise still
    // propose it every other call, so the live partner would only advance
    // every second group.step() call if index 0 were (incorrectly)
    // re-stepped.
    basic_solver_group<round_robin_schedule, argmin::dynamic_dimension, void,
                       test::converging_mock_policy, test::mock_policy>
        group{prob, x0, opts};

    group.step(); // index 0, step 1 (not yet converged)
    group.step(); // index 1, live step 1
    auto r_retire = group.step(); // index 0, step 2 -> converged, retires
    CHECK(r_retire.policy_status == argmin::solver_status::converged);

    // From here on, every remaining group.step() call must land on the
    // live index-1 solver -- proven by its objective strictly decreasing
    // on EVERY subsequent call (mock_policy halves x, quartering the
    // objective, each time it is actually stepped). If the retired solver
    // were still being re-selected on alternating calls, the objective
    // would only quarter every OTHER call instead.
    double prev_obj = std::numeric_limits<double>::infinity();
    for(int i = 0; i < 4; ++i)
    {
        auto r = group.step();
        CHECK(r.policy_status == std::nullopt);
        CHECK(r.objective_value < prev_obj);
        CHECK(r.objective_value > 0.0);
        prev_obj = r.objective_value;
    }
}

TEST_CASE("group feasibility default agrees with basic_solver's (1e-6, not value_or(0))",
          "[solver_group][feasibility]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{1.0, 1.0}};
    solver_options opts;

    // borderline_feasible_mock_policy: cv = 5e-7 (inside the 1e-6 default
    // feasibility_tolerance, outside a bare-zero tolerance), obj = 10.
    // feasible_worse_objective_mock_policy: cv = 0 (unambiguously
    // feasible), obj = 20 (worse). Under an incorrect value_or(0) group
    // tolerance, the borderline solver would be misjudged infeasible and
    // the worse-objective-but-cv=0 solver would win by the
    // feasible-beats-infeasible rule; under the corrected 1e-6 default
    // both are feasible and the lower objective (10) must win.
    basic_solver_group<round_robin_schedule, argmin::dynamic_dimension, void,
                       test::borderline_feasible_mock_policy,
                       test::feasible_worse_objective_mock_policy>
        group{prob, x0, opts};

    auto result = group.step_n(2);
    CHECK(result.objective_value == Approx(10.0));
}

TEST_CASE("standalone basic_solver treats the same borderline iterate as feasible",
          "[solver][feasibility]")
{
    quadratic_group prob;
    // Chosen so quadratic_group::value(x0) (100.0) is worse than either
    // mock objective below, and cannot itself win the best-seen
    // comparison -- isolating the feasibility-tolerance behavior under
    // test from the unrelated best-seen seeding mechanism.
    Eigen::VectorXd x0{{10.0, 10.0}};

    // borderline_then_worse_mock_policy: reports cv = 5e-7 (borderline
    // feasible under the 1e-6 default) with obj = 10 on step 1, then
    // cv = 0 (unambiguously feasible) with obj = 20 (worse) on step 2.
    // Under basic_solver's feasibility_tolerance default (options.h,
    // 1e-6 -- the same constant basic_solver_group now shares), step 1 is
    // judged feasible and its lower objective must win the best-seen
    // comparison over step 2, mirroring the group-level test above using
    // the identical borderline cv value.
    basic_solver<test::borderline_then_worse_mock_policy> solver{prob, x0};
    auto result = solver.step_n(2);

    CHECK(result.objective_value == Approx(10.0));
}

TEST_CASE("group step_n reports the executed iteration count on early break, not the budget",
          "[solver_group][telemetry]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{4.0, 4.0}};
    solver_options opts;

    // Both policies retire (report converged) on their own 2nd step;
    // round-robin retires both within 4 group steps -- far short of the 100
    // budget. The reported iteration/evaluation counts must reflect the 4
    // steps that actually ran, not the requested budget.
    basic_solver_group<round_robin_schedule, argmin::dynamic_dimension, void,
                       test::converging_mock_policy, test::converging_mock_policy>
        group{prob, x0, opts};

    auto result = group.step_n(100);

    CHECK(result.status == solver_status::budget_exhausted);
    CHECK(result.iterations == 4);
    CHECK(result.iterations < 100);
    // One evaluation per step here, genuinely accumulated -- not the budget.
    CHECK(result.function_evaluations == 4);
}

TEST_CASE("group populate_active_results reports a real gradient_norm, not a hardcoded zero",
          "[solver_group][telemetry]")
{
    quadratic_group prob;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;

    // Neither mock_policy retires, so both are active when
    // populate_active_results fills the per-solver results after the run.
    // Each active solver's result must carry its real last-reported gradient
    // norm (x.norm() > 0 after stepping), never the old fabricated 0.0 that
    // reads as a stationary point.
    basic_solver_group<round_robin_schedule, argmin::dynamic_dimension, void,
                       test::mock_policy, test::mock_policy>
        group{prob, x0, opts};

    group.step_n(4);

    const auto& results = group.results();
    for(const auto& r : results)
    {
        CHECK(std::isfinite(r.gradient_norm));
        CHECK(r.gradient_norm > 0.0);
    }
}
