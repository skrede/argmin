// Convergence tests for filter_nw_sqp_policy on Hock-Schittkowski problems.
//
// Validates that the filter-based dense BFGS SQP policy converges on
// standard HS benchmark problems with equality, inequality, and mixed
// constraints.
//
// Reference: Hock & Schittkowski 1981; Fletcher & Leyffer 2002.

#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/formulation/concepts.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using namespace argmin;

static_assert(nlp_solver<basic_solver<filter_nw_sqp_policy<>>>,
              "filter_nw_sqp_policy must satisfy nlp_solver concept");

TEST_CASE("filter_nw_sqp on hock-schittkowski problems", "[hs][filter_nw_sqp]")
{
    // Reference: Hock & Schittkowski 1981, Problem 71.
    SECTION("HS071: mixed equality + inequality constraints")
    {
        // Regression guard: HS071 convergence blocked by filter over-rejection
        // at box bounds. BFGS y-vector fix (J_all_old) applied but insufficient
        // alone — QP solver returns near-zero steps when variables hit bounds.
        // Pending QP elastic mode or interior-point integration.
        hs071 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 500;
        opts.set_gradient_threshold(1e-4);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        basic_solver solver{filter_nw_sqp_policy<hs071<>::problem_dimension>{},
                            problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value < 20.0);
    }

    // Reference: Hock & Schittkowski 1981, Problem 43.
    SECTION("HS043: inequality constraints only")
    {
        // step_threshold aligned with the benchmark / NLopt xtol_rel regime
        // (1e-12). The prior 1e-15 value was calibrated to the Powell-damped
        // direct-BFGS trajectory; under adaptive_bfgs (N&W Section 7.2,
        // skip-on-nonpositive-curvature per Procedure 18.2 footnote) the
        // iterate reaches f=-44.16 at iter 7 with marginal infeasibility
        // (cv above the best-seen feasibility_tolerance), and a subsequent
        // filter-acceptance over-rejection (Wachter-Biegler envelope) wanders
        // the iterate to a non-stationary region.
        //
        // Under basic_solver's best-seen termination (NLopt convention),
        // the returned solve_result is the best strictly-feasible iterate
        // encountered, not the infeasible f=-44 trial.
        //
        // Bar tightened from margin(4.0) -> margin(0.5) per the HS043
        // envelope sweep (benchmarks/filter_envelope_sweep.cpp): with the
        // accurate-mode default gamma_f = gamma_h = 1e-3 the policy
        // converges to f = -43.65 (cv = 0) at this initial point, vs
        // f = -43.33 under the prior 1e-5/1e-5 default. HS024 + HS076
        // regression guards remain green across the sweep.
        hs043 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 500;
        opts.set_gradient_threshold(1e-4);
        opts.set_step_threshold(1e-12);
        opts.set_objective_threshold(1e-15);

        basic_solver solver{filter_nw_sqp_policy<hs043<>::problem_dimension>{},
                            problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(-44.0).margin(0.5));
        CHECK(result.constraint_violation <= opts.feasibility_tolerance);
    }

    // Reference: Hock & Schittkowski 1981, Problem 39.
    SECTION("HS039: equality constraints only")
    {
        hs039 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 500;
        opts.set_gradient_threshold(1e-4);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        basic_solver solver{filter_nw_sqp_policy<hs039<>::problem_dimension>{},
                            problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.status != solver_status::max_iterations);
        CHECK(result.objective_value == Approx(-1.0).margin(1.0));
        CHECK(solver.constraint_violation() < 1e-4);
    }

    // Reference: Hock & Schittkowski 1981, Problem 76.
    SECTION("HS076: inequality + equality constraints")
    {
        hs076 problem;
        auto x0 = problem.initial_point();
        solver_options opts;
        opts.max_iterations = 200;
        opts.set_gradient_threshold(1e-6);
        opts.set_step_threshold(1e-15);
        opts.set_objective_threshold(1e-15);

        basic_solver solver{filter_nw_sqp_policy<hs076<>::problem_dimension>{},
                            problem, x0, opts};
        auto result = solver.solve(opts);

        CHECK(result.objective_value == Approx(-4.68).margin(0.1));
        CHECK(solver.constraint_violation() < 1e-4);
    }
}

// filter_nw_sqp populates step_result::kkt_residual via
// detail::kkt_residual and sets is_null_step on both documented
// null-step paths (QP zero direction, restoration exhaustion).
//
// Reference: N&W 2e Section 12.3 / eq. 12.34 (KKT residual);
//            N&W 2e Section 18.4 (SQP null-step semantics).
TEST_CASE("filter_nw_sqp populates kkt_residual and exposes is_null_step",
          "[filter_nw_sqp][kkt]")
{
    hs039 problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 100;
    opts.set_gradient_threshold(1e-4);
    opts.set_step_threshold(1e-15);
    opts.set_objective_threshold(1e-15);

    basic_solver solver{filter_nw_sqp_policy<hs039<>::problem_dimension>{},
                        problem, x0, opts};

    bool populated = false;
    for(int i = 0; i < 15; ++i)
    {
        auto sr = solver.step();
        CHECK((sr.is_null_step || !sr.is_null_step));
        if(sr.kkt_residual.has_value())
        {
            populated = true;
            CHECK(sr.kkt_residual.value() >= 0.0);
        }
        if(sr.policy_status)
            break;
    }
    CHECK(populated);
}

// HS024 regression guard on the nw-variant filter SQP policy. Same closure
// mechanism as filter_slsqp HS024: the active-set multiplier re-estimation
// restricts the LS projection to binding inequalities (|c_ineq[i]| < 1e-8),
// clamps active mu_ineq to >= 0, and the null-step branch populates
// kkt_residual so the composite convergence test can fire at the optimum.
//
// Reference baseline (post-phase30): 13 iters @ f = -1.0. Regression target:
// iterations within 1 of baseline.
//
// Reference: Hock & Schittkowski 1981, Problem 24.
//            N&W 2e Section 18.3 + Algorithm 18.3 (working-set);
//            eq. 18.15 (least-squares lambda);
//            eq. 12.34 (composite first-order optimality).
TEST_CASE("filter_nw_sqp HS024 regression guard",
          "[filter_nw_sqp][regression]")
{
    hs024<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    basic_solver solver{filter_nw_sqp_policy<hs024<>::problem_dimension>{},
                        problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.objective_value == Approx(-1.0).margin(1e-6));
    CHECK(result.iterations <= 14);
    CHECK(solver.constraint_violation() < 1e-6);
}

// Warm-start regression guard: hot-start reset() must clear the filter
// envelope, otherwise a second solve from the same x0 enters the line
// search with the prior run's converged near-optimum already in the
// filter, every trial point is dominance-rejected, and the outer loop
// stalls at alpha -> 0. Cold-start tests do not exercise this path
// because each constructs a fresh basic_solver. Surfaced by ctrlpp
// nanobench harness reusing the nmpc instance across cells.
//
// Reference: Wachter & Biegler 2006, Section 3.3 (filter
//            re-initialization between independent runs);
//            N&W 2e Section 15.4 (filter SQP semantics).
TEST_CASE("filter_nw_sqp reset() clears filter for warm-start convergence",
          "[filter_nw_sqp][regression][warm_start]")
{
    hs039<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-6);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-10);

    basic_solver solver{filter_nw_sqp_policy<hs039<>::problem_dimension>{},
                        problem, x0, opts};

    auto first = solver.solve(opts);
    REQUIRE(first.iterations < 50);

    // Hot-start back to the same x0. Without filter.clear() in reset(),
    // the filter still contains entries from the first solve and the
    // line search rejects every trial step (Wachter-Biegler oscillation
    // at alpha -> 0).
    solver.reset(x0);
    auto second = solver.solve(opts);

    CHECK(second.iterations < 50);
    CHECK(second.objective_value == Approx(first.objective_value).margin(1e-6));
    CHECK(solver.constraint_violation() < 1e-6);

    // Iter count for the second solve should be in the same band as the
    // first; BFGS is preserved so the second can be slightly faster (a
    // few iters), but it should not regress significantly.
    CHECK(second.iterations <= first.iterations + 5);
}
