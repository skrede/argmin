#include "scripted_claim_policy.h"

#include "argmin/solver/options.h"
#include "argmin/solver/convergence.h"

#include "argmin/result/status.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

using namespace argmin;
using namespace argmin::test;

TEMPLATE_TEST_CASE("an optimality claim requires feasibility within the tolerance in force", "[solver][convergence]", double, float)
{
    using S = TestType;

    SECTION("a residual under the gradient threshold at an infeasible iterate does not converge")
    {
        STATIC_REQUIRE(passing_kkt<S> > feas_tol<S>);
        auto solver = make_solver<S>({infeasible<S>, infeasible<S>});
        const auto result = solver.solve();
        CHECK(result.status == solver_status::max_iterations);
        CHECK(result.iterations == 3);
    }

    SECTION("the claim is made at the first feasible iterate")
    {
        auto solver = make_solver<S>({infeasible<S>, feasible<S>});
        const auto result = solver.solve();
        CHECK(result.status == solver_status::converged);
        CHECK(result.iterations == 2);
        CHECK(result.constraint_violation <= feas_tol<S>);
    }

    SECTION("a policy-reported convergence at an infeasible iterate ends the solve without the claim")
    {
        const scripted_step<S> claimed{S(2) * feas_tol<S>, std::nullopt, solver_status::converged};
        auto solver = make_solver<S>({claimed, feasible<S>});
        const auto result = solver.solve();
        CHECK(result.status == solver_status::objective_stalled);
        CHECK(result.iterations == 1);
    }

    SECTION("an objective-tolerance claim at an infeasible iterate is no claim")
    {
        using ftol_only = convergence_policy<objective_tolerance_criterion<S>>;
        const scripted_step<S> stationary{S(2) * feas_tol<S>, S(0), std::nullopt};
        auto solver = make_solver<S, ftol_only>({stationary, stationary});
        CHECK(solver.solve().status == solver_status::max_iterations);
    }

    SECTION("the public step reports no claim at an infeasible iterate")
    {
        auto solver = make_solver<S>({infeasible<S>, feasible<S>});
        solver.step();
        CHECK(solver.status() == solver_status::running);
        solver.step();
        CHECK(solver.status() == solver_status::converged);
    }

    SECTION("a caller constraint tolerance is the tolerance in force")
    {
        solver_options<default_convergence<S>> opts;
        opts.constraint_tolerance = S(4) * feas_tol<S>;
        auto solver = make_solver<S>({infeasible<S>, infeasible<S>}, opts);
        const auto result = solver.solve();
        CHECK(result.status == solver_status::converged);
        CHECK(result.iterations == 1);
    }
}
