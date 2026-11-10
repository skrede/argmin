#include "scripted_claim_policy.h"

#include "argmin/solver/options.h"
#include "argmin/solver/convergence.h"

#include "argmin/result/status.h"

#include "argmin/options/convergence_defaults.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <tuple>
#include <optional>

using namespace argmin;
using namespace argmin::test;

namespace
{

template <typename S>
constexpr S tiny_step = convergence_defaults<S>::step_threshold / S(2);

template <typename S>
constexpr scripted_step<S> infeasible_tiny_step{S(2) * feas_tol<S>, std::nullopt, std::nullopt, tiny_step<S>};

template <typename S>
constexpr scripted_step<S> feasible_tiny_step{feas_tol<S> / S(2), std::nullopt, std::nullopt, tiny_step<S>};

template <typename S>
using step_only = convergence_policy<step_tolerance_criterion<S>>;

template <typename S>
solver_options<step_only<S>> step_only_options(S constraint_tolerance)
{
    solver_options<step_only<S>> opts;
    opts.constraint_tolerance = constraint_tolerance;
    return opts;
}

}

TEMPLATE_TEST_CASE("a feasibility claim requires feasibility within the tolerance in force", "[solver][convergence]", double, float)
{
    using S = TestType;

    SECTION("a step-size stall at an infeasible iterate is no claim and the solve goes on")
    {
        auto solver = make_solver<S, step_only<S>>({infeasible_tiny_step<S>, infeasible_tiny_step<S>});
        const auto result = solver.solve();
        CHECK(result.status == solver_status::max_iterations);
        CHECK(result.iterations == 3);
    }

    SECTION("the step criterion admits a stall within a looser caller constraint tolerance")
    {
        auto solver = make_solver<S, step_only<S>>({infeasible_tiny_step<S>, infeasible_tiny_step<S>}, step_only_options(S(4) * feas_tol<S>));
        CHECK(solver.solve().status == solver_status::stalled);
    }

    SECTION("the step criterion makes no claim above a tighter caller constraint tolerance")
    {
        auto solver = make_solver<S, step_only<S>>({feasible_tiny_step<S>, feasible_tiny_step<S>}, step_only_options(feas_tol<S> / S(4)));
        CHECK(solver.solve().status == solver_status::max_iterations);
    }

    SECTION("the relative step criterion makes no claim above a tighter caller constraint tolerance")
    {
        solver_options<convergence_policy<step_tolerance_rel_criterion<S>>> opts;
        opts.constraint_tolerance = feas_tol<S> / S(4);
        std::get<0>(opts.convergence.criteria).threshold = convergence_defaults<S>::step_threshold;
        auto solver = make_solver<S>({feasible_tiny_step<S>, feasible_tiny_step<S>}, opts);
        CHECK(solver.solve().status == solver_status::max_iterations);
    }

    SECTION("a stall of the objective at an infeasible iterate is no claim")
    {
        solver_options<convergence_policy<stall_tolerance_criterion<S>>> opts;
        opts.set_stall_threshold(S(1e-3));
        opts.set_stall_window(2);
        auto solver = make_solver<S>({infeasible<S>, infeasible<S>}, opts);
        const auto result = solver.solve();
        CHECK(result.status == solver_status::max_iterations);
        CHECK(result.iterations == 3);
    }

    SECTION("a policy-reported feasibility claim at an infeasible iterate ends the solve without the claim")
    {
        for(const solver_status claimed : {solver_status::stalled, solver_status::xtol_reached})
        {
            auto solver = make_solver<S>({scripted_step<S>{S(2) * feas_tol<S>, std::nullopt, claimed}, feasible<S>});
            const auto result = solver.solve();
            CHECK(result.status == solver_status::objective_stalled);
            CHECK(result.iterations == 1);
        }
    }

    SECTION("the public step reports the same resolution")
    {
        auto solver = make_solver<S>({scripted_step<S>{S(2) * feas_tol<S>, std::nullopt, solver_status::stalled}, feasible<S>});
        solver.step();
        CHECK(solver.status() == solver_status::objective_stalled);
    }

    SECTION("a policy-reported feasibility claim within the tolerance stands")
    {
        auto solver = make_solver<S>({scripted_step<S>{feas_tol<S> / S(2), std::nullopt, solver_status::xtol_reached}, feasible<S>});
        CHECK(solver.solve().status == solver_status::xtol_reached);
    }
}
