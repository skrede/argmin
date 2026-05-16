// Filter trust-region SQP HS-suite smoke unit tests.
//
// Reference: Fletcher and Leyffer (2002) Math. Programming 91:239-269
//            Section 2.1 (filter dominance) and Section 3 (radius rule
//            decoupling);
//            Wachter and Biegler (2006) Math. Programming 106:25-57
//            Section 2.3 eq. 6 (filter envelope; gamma_f, gamma_h);
//            Nocedal and Wright 2e Section 18.5 Algorithm 18.4
//            (Byrd-Omojokun composite step);
//            Hock and Schittkowski (1981) Test Examples for Nonlinear
//            Programming Codes (HS026, HS028).
//
// argmin variant: HS026 and HS028 are the wiring-correctness smoke
//                 cells for filter_trsqp_policy. Both are drawn from
//                 the tr_sqp_policy reference set (same constraint
//                 structure, same |f - f*| / cv bars) so the cells
//                 directly compare filter acceptance against the
//                 L2-merit baseline on problems where the baseline is
//                 known to close. The cells exercise the slack-
//                 augmented joint constraint formulation, the
//                 multiplier-reestimation cadence, the sqp_mode NTTP
//                 dispatch, and the filter_set lifecycle (envelope
//                 retune, initial add, accept-on-dominance,
//                 reject-shrink). Both modes must converge to the
//                 standard filter-SQP publication bars (accurate
//                 |f - f*| absolute or relative < 1e-2 and
//                 constraint_violation < 1e-4; fast bar relaxed to
//                 < 5e-2 / 1e-2). This is the cheapest test that fires
//                 when the filter primitive fails to integrate
//                 cleanly, when the Byrd-Omojokun helper refactor
//                 regresses, when mode dispatch picks the wrong
//                 constexpr defaults, or when the h-coordinate
//                 aggregate computes on the wrong constraint view.

#include "argmin/solver/filter_trsqp_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/solver/sqp_mode.h"
#include "argmin/solver/options.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace argmin;

TEMPLATE_TEST_CASE_SIG(
    "filter_trsqp HS026 (parametric on mode)",
    "[sqp][filter_trsqp][regression][mode]",
    ((typename Policy), Policy),
    filter_trsqp_policy_accurate<hs026<>::problem_dimension>,
    filter_trsqp_policy_fast<hs026<>::problem_dimension>)
{
    using policy_t = Policy;

    hs026<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    basic_solver solver{policy_t{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // HS026 optimum: f* = 0 at (1, 1, 1). Absolute bar because f* = 0
    // makes the |f - f*| / |f*| ratio ill-posed. Bars mirror the
    // tr_sqp HS026 reference cell.
    if constexpr(policy_t::mode_ == sqp_mode::fast)
    {
        CHECK(result.objective_value == Approx(0.0).margin(0.05));
        CHECK(solver.constraint_violation() < 1e-2);
    }
    else
    {
        CHECK(result.objective_value == Approx(0.0).margin(0.01));
        CHECK(solver.constraint_violation() < 1e-4);
    }
}

TEMPLATE_TEST_CASE_SIG(
    "filter_trsqp HS028 (parametric on mode)",
    "[sqp][filter_trsqp][regression][mode]",
    ((typename Policy), Policy),
    filter_trsqp_policy_accurate<hs028<>::problem_dimension>,
    filter_trsqp_policy_fast<hs028<>::problem_dimension>)
{
    using policy_t = Policy;

    hs028<> problem;
    auto x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    basic_solver solver{policy_t{}, problem, x0, opts};
    auto result = solver.solve(opts);

    // HS028 optimum: f* = 0 at (0.5, -0.5, 0.5). Bars mirror the
    // tr_sqp HS028 reference cell -- accurate mode uses an absolute
    // 1e-6 margin (the quadratic objective with linear equality
    // collapses to a residual-zero solution on filter acceptance just
    // as it does on the L2-merit baseline); fast mode uses the
    // relaxed 1e-2 margin sized to the fast-mode gradient tolerance.
    if constexpr(policy_t::mode_ == sqp_mode::fast)
    {
        CHECK(result.objective_value == Approx(0.0).margin(1e-2));
        CHECK(solver.constraint_violation() < 1e-2);
    }
    else
    {
        CHECK(result.objective_value == Approx(0.0).margin(1e-6));
        CHECK(solver.constraint_violation() < 1e-4);
        CHECK(result.gradient_norm < 1e-4);
    }
}
