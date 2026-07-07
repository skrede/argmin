// Warm-start reset boundary: a reset handed a plain vector must not allocate
// through the Eigen allocator, at both fixed and dynamic N. This translation
// unit is compiled with EIGEN_RUNTIME_NO_MALLOC (set target-side) so that any
// in-window Eigen heap allocation trips Eigen's internal is-malloc-allowed
// assertion. That assertion is redirected to a throw below so the harness
// observes it as a catchable exception instead of an abort.
//
// This is the Eigen-side sensor for the boundary claim (re-measuring the
// conversion-temporary assumption for the Eigen::Ref reset variant). The full
// dual-sensor, malloc-interposed proof lives in the allocation-gate benches;
// the armed fixed-N reset there passes through this same boundary.
//
// The file also pins the two reset() defect fixes: the filter_slsqp filter
// re-seed at the new point, and the projected Gauss-Newton damping reading the
// configured tau. Each pin fails against the pre-fix code (demonstrated once
// during development, noted in the plan summary).

#include <stdexcept>

#ifndef eigen_assert
#define eigen_assert(cond)                                                    \
    do                                                                        \
    {                                                                         \
        if(!(cond))                                                           \
            throw std::runtime_error("eigen_assert: " #cond);                 \
    } while(false)
#endif

#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/projected_gn_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/detail/lagrangian.h"
#include "argmin/test_functions/hock_schittkowski.h"
#include "argmin/test_functions/rosenbrock.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <algorithm>

using Catch::Approx;
using namespace argmin;

namespace
{

// RAII arming of Eigen's runtime no-malloc gate. On destruction the gate is
// released even if the guarded region threw (the redirected eigen_assert).
struct eigen_malloc_gate
{
    eigen_malloc_gate() { Eigen::internal::set_is_malloc_allowed(false); }
    ~eigen_malloc_gate() { Eigen::internal::set_is_malloc_allowed(true); }
};

// Minimal bounded least-squares problem for the projected_gn tau pin
// (bounded Rosenbrock in residual form, b = 5).
struct bounded_rosenbrock_ls
{
    static constexpr int problem_dimension = dynamic_dimension;

    int dimension() const { return 2; }
    int num_residuals() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        double r0 = 1.0 - x(0);
        double r1 = std::sqrt(5.0) * (x(1) - x(0) * x(0));
        return 0.5 * (r0 * r0 + r1 * r1);
    }

    void residuals(const Eigen::VectorXd& x, Eigen::VectorXd& r) const
    {
        r(0) = 1.0 - x(0);
        r(1) = std::sqrt(5.0) * (x(1) - x(0) * x(0));
    }

    void jacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const
    {
        J(0, 0) = -1.0;
        J(0, 1) = 0.0;
        J(1, 0) = -2.0 * std::sqrt(5.0) * x(0);
        J(1, 1) = std::sqrt(5.0);
    }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd{{-2.0, -2.0}}; }
    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd{{2.0, 2.0}}; }
};

}

// Fixed-N constrained reset (kraft_slsqp on HS071, n = 4): the plain
// Eigen::Vector<double, 4> caller buffer binds straight into the policy Ref, so
// the warm reset touches only pre-sized state storage.
TEST_CASE("warm reset is Eigen-allocation-free at fixed N",
          "[warm_start][embed]")
{
    hs071<> problem;
    Eigen::Vector<double, hs071<>::problem_dimension> x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 50;

    step_budget_solver solver{kraft_slsqp_policy<hs071<>::problem_dimension>{},
                              problem, x0, opts};
    solver.step();
    solver.step();
    solver.reset(x0); // settle any first-reset lazy sizing before arming

    bool threw = false;
    {
        eigen_malloc_gate gate;
        try
        {
            solver.reset(x0);
        }
        catch(const std::exception&)
        {
            threw = true;
        }
    }
    CHECK_FALSE(threw);
}

// Dynamic-N reset (kraft_slsqp<> on a dynamic Rosenbrock): the VectorXd caller
// buffer binds into the dynamic policy Ref with no conversion temporary -- the
// assumption re-measured for the Ref migration.
TEST_CASE("warm reset is Eigen-allocation-free at dynamic N",
          "[warm_start][embed]")
{
    rosenbrock<> problem{.n = 4};
    Eigen::VectorXd x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 50;

    step_budget_solver solver{kraft_slsqp_policy<>{}, problem, x0, opts};
    solver.step();
    solver.step();
    solver.reset(x0); // settle

    bool threw = false;
    {
        eigen_malloc_gate gate;
        try
        {
            solver.reset(x0);
        }
        catch(const std::exception&)
        {
            threw = true;
        }
    }
    CHECK_FALSE(threw);
}

// Reset defect pin: a warm reset from a high-violation point must re-seed the
// filter ceiling at the new h_0. Pre-fix reset() only cleared the filter,
// leaving the default h_max = 1e4; post-fix it re-seeds h_max = 1e4 * max(1,
// h_0), which for a high-violation restart is strictly larger.
TEST_CASE("filter_slsqp warm reset re-seeds the filter ceiling at the new point",
          "[warm_start][filter_slsqp]")
{
    hs071<> problem;
    Eigen::Vector<double, hs071<>::problem_dimension> x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-6);

    step_budget_solver solver{filter_slsqp_policy<hs071<>::problem_dimension>{},
                              problem, x0, opts};
    solver.solve(opts); // converge; the filter accumulates entries

    // Warm reset from a deliberately high-violation point (all coordinates at
    // the upper bound: the equality residual alone is 4*25 - 40 = 60).
    Eigen::Vector<double, hs071<>::problem_dimension> x_hi;
    x_hi << 5.0, 5.0, 5.0, 5.0;
    solver.reset(x_hi);

    const auto& s = solver.state();
    const double h0 = detail::constraint_violation(s.c_eq, s.c_ineq);
    REQUIRE(h0 > 1.0); // the regime that distinguishes a re-seed from a clear()
    CHECK(solver.state().filter.h_max() == Approx(1e4 * std::max(1.0, h0)));
    CHECK(solver.state().filter.h_max() > 1e4 + 1.0);
}

// Reset defect pin: the projected Gauss-Newton warm reset must read the
// configured tau for the Nielsen damping. Two solvers reset from the same point
// (identical J, identical max diag) differ in lambda exactly by their tau ratio;
// pre-fix reset() hardcoded 1e-3 for both, collapsing the ratio to 1.
TEST_CASE("projected_gn warm reset honors options.tau",
          "[warm_start][projected_gn]")
{
    bounded_rosenbrock_ls problem;
    Eigen::VectorXd x0{{-1.0, 1.0}};
    solver_options opts;
    opts.max_iterations = 10;

    projected_gn_policy<>::options_type popts_default{}; // tau = 1e-3
    projected_gn_policy<>::options_type popts_tau{};
    popts_tau.tau = 5e-2;

    step_budget_solver s_default{projected_gn_policy<>{}, problem, x0, opts,
                                 popts_default};
    step_budget_solver s_tau{projected_gn_policy<>{}, problem, x0, opts, popts_tau};

    s_default.reset(x0);
    s_tau.reset(x0);

    const double lam_default = s_default.state().lambda;
    const double lam_tau = s_tau.state().lambda;

    REQUIRE(lam_default > 0.0);
    CHECK(lam_tau == Approx(50.0 * lam_default));
}
