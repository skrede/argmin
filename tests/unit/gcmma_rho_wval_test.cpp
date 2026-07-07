// Tests for the rho/wval GCMMA variant
// (argmin::alternative::gcmma::rho_wval_policy).
//
// NLopt-mma.c-style globalization: rho-augmented MMA approximation
// with wval-based rho-growth on non-conservative trials. Numerical
// per-component primal (Newton) replaces the closed-form analytic
// minimum used by plain MMA / move-limit-shrink.
//
// Expected behavior: should reach optimum on HS024/HS035/HS076 at
// the cost of higher per-iter wall time (Newton per j per dual
// evaluation) compared to move-limit-shrink. The empirical
// comparison in the published benchmark suite captures the
// quantitative trade-off.

#include "argmin/solver/alternative/gcmma/rho_wval_policy.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/options.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace argmin;

namespace
{

// Construction-counting wrapper over lbfgsb_policy: counts inner dual-solver
// constructions (init) and cold restarts (reset_clear) so a test can assert
// the persistent inner solver is built once per outer iteration and reused
// (reset_clear) across the conservativity loop, rather than re-allocated on
// every inner iteration.
inline int g_dual_init_count = 0;
inline int g_dual_reset_clear_count = 0;

template <int M>
struct counting_lbfgsb
{
    lbfgsb_policy<M> inner_;

    template <typename Problem, typename Convergence>
    auto init(const Problem& p, const Eigen::Vector<double, M>& x0,
              const solver_options<Convergence>& opts)
    {
        ++g_dual_init_count;
        return inner_.init(p, x0, opts);
    }

    template <typename State>
    auto step(State& s) { return inner_.step(s); }

    template <typename State>
    void reset_clear(State& s, const Eigen::Vector<double, M>& x0)
    {
        ++g_dual_reset_clear_count;
        inner_.reset_clear(s, x0);
    }
};

// Unconstrained convex quadratic on a wide box. With a small initial rho
// the fresh reciprocal approximation is non-conservative near a steep
// left flank: its minimizer overshoots the true minimum. Used to exercise
// the conservativity loop's null-step-and-retry on a non-conservative,
// non-improving trial.
struct wide_quadratic
{
    static constexpr int problem_dimension = dynamic_dimension;

    int dimension() const { return 1; }
    int num_equality() const { return 0; }
    int num_inequality() const { return 0; }

    double value(const Eigen::VectorXd& x) const { return x[0] * x[0]; }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g.resize(1);
        g[0] = 2.0 * x[0];
    }

    void constraints(const Eigen::VectorXd&, Eigen::VectorXd& c) const
    {
        c.resize(0);
    }

    void constraint_jacobian(const Eigen::VectorXd&, Eigen::MatrixXd& J) const
    {
        J.resize(0, 1);
    }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd{{-10.0}}; }
    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd{{10.0}}; }
};

}

TEST_CASE("gcmma rho-wval converges on HS024", "[gcmma_rho_wval]")
{
    hs024 problem;
    Eigen::VectorXd x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-5);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);

    step_budget_solver solver{
        alternative::gcmma::rho_wval_policy<
            hs024<>::problem_dimension>{},
        problem, x0, opts};
    const auto result = solver.solve(opts);

    CHECK(result.objective_value == Approx(-1.0).margin(0.05));
}

TEST_CASE("gcmma rho-wval converges on HS035", "[gcmma_rho_wval]")
{
    hs035 problem;
    Eigen::VectorXd x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-5);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);

    step_budget_solver solver{
        alternative::gcmma::rho_wval_policy<
            hs035<>::problem_dimension>{},
        problem, x0, opts};
    const auto result = solver.solve(opts);

    CHECK(result.objective_value == Approx(1.0 / 9.0).margin(0.05));
}

TEST_CASE("gcmma rho-wval converges on HS076", "[gcmma_rho_wval]")
{
    hs076 problem;
    Eigen::VectorXd x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-5);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);

    step_budget_solver solver{
        alternative::gcmma::rho_wval_policy<
            hs076<>::problem_dimension>{},
        problem, x0, opts};
    const auto result = solver.solve(opts);

    CHECK(result.objective_value == Approx(-4.6818).margin(0.05));
}

// Conservativity loop: a non-conservative trial that would move to a
// strictly worse point is never committed. With rho_init at its floor and
// a single inner iteration (no room to grow rho back to conservativity),
// the reciprocal approximation at x0 = -3 overshoots to x ~ 6 where the
// true objective (36) exceeds f(x0) = 9. The step must be a null step:
// x is left unchanged, no improvement is claimed, and the grown rho is
// retained (not decayed) for the next outer iteration.
TEST_CASE("gcmma rho-wval null-steps on a non-conservative worse trial",
          "[gcmma_rho_wval]")
{
    wide_quadratic problem;
    Eigen::VectorXd x0{{-3.0}};
    solver_options opts;

    alternative::gcmma::rho_wval_policy<> policy;
    alternative::gcmma::rho_wval_policy<>::options_type popts;
    popts.rho_init = 1e-5;
    popts.rho_min = 1e-5;
    popts.max_inner_iterations = 1;
    popts.rho_decay = 1.0;
    popts.raai = 0.0;

    auto s = policy.init(problem, x0, opts, popts);
    const double f0 = s.f;
    const double rho0 = s.rho_obj;
    const auto r = policy.step(s);

    CHECK(r.is_null_step);
    CHECK_FALSE(r.improved);
    CHECK(s.x[0] == Approx(-3.0).margin(1e-15));  // x unchanged
    CHECK(s.f == Approx(f0).margin(1e-15));
    // rho was grown on the non-conservative trial and retained (the
    // null-step path returns before the inter-outer decay).
    CHECK(s.rho_obj > rho0);
}

// Positive control: a non-conservative but merit-improving trial is still
// committed (the conservativity loop is permissive toward descent). From
// x0 = -5 the same fresh approximation overshoots to x ~ 4 where the true
// objective (16) is below f(x0) = 25, so the step commits.
TEST_CASE("gcmma rho-wval commits a non-conservative improving trial",
          "[gcmma_rho_wval]")
{
    wide_quadratic problem;
    Eigen::VectorXd x0{{-5.0}};
    solver_options opts;

    alternative::gcmma::rho_wval_policy<> policy;
    alternative::gcmma::rho_wval_policy<>::options_type popts;
    popts.rho_init = 1e-5;
    popts.rho_min = 1e-5;
    popts.max_inner_iterations = 1;
    popts.rho_decay = 1.0;
    popts.raai = 0.0;

    auto s = policy.init(problem, x0, opts, popts);
    const double f0 = s.f;
    const auto r = policy.step(s);

    CHECK_FALSE(r.is_null_step);
    CHECK(r.improved);
    CHECK(s.f < f0);
    CHECK(s.x[0] != Approx(-5.0));  // x moved
}

// Per-iter-cost: the inner box-constrained dual solver is constructed once
// per outer iteration and cold-restarted (reset_clear) across the inner
// conservativity loop, not re-allocated on every inner iteration. HS024's
// cubic objective is non-conservative under the reciprocal approximation, so
// the inner loop runs several iterations per outer step -- a per-inner
// re-construction would call init up to max_inner times per outer.
TEST_CASE("gcmma rho-wval constructs the dual solver once per outer",
          "[gcmma_rho_wval]")
{
    hs024 problem;
    Eigen::Vector<double, 2> x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 500;

    alternative::gcmma::rho_wval_policy<
        hs024<>::problem_dimension, counting_lbfgsb> policy;
    auto s = policy.init(problem, x0, opts);

    g_dual_init_count = 0;
    g_dual_reset_clear_count = 0;

    constexpr int kOuter = 10;
    for(int i = 0; i < kOuter; ++i)
        (void)policy.step(s);

    // Exactly one construction per outer step (not one per inner iteration).
    CHECK(g_dual_init_count == kOuter);
    // The persistent state was cold-restarted across inner iterations, so at
    // least one outer step ran a multi-iteration conservativity loop that
    // reused the state rather than re-allocating it.
    CHECK(g_dual_reset_clear_count > 0);
}
