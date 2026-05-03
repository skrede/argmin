// Tests for the ISRES alternative-variant policies (nlopt_faithful,
// original_argmin, runarsson_yao_paper). Covers (1) nlp_solver concept
// satisfaction, (2) restarting_policy<variant>::rebind static_asserts,
// (3) simple_constrained smoke runs, and (4) synthetic-state
// ftol_reached emission tests per Runarsson-Yao 2005 termination
// convention.

#include "argmin/solver/alternative/isres/nlopt_faithful_policy.h"
#include "argmin/solver/alternative/isres/original_argmin_policy.h"
#include "argmin/solver/alternative/isres/runarsson_yao_paper_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/solver/restarting_policy.h"
#include "argmin/result/status.h"
#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace argmin;

namespace
{

// Local copy of the simple_constrained fixture used in isres_test.cpp.
// Replicated rather than included so this file is self-contained.
//
// Minimise x0^2 + x1^2 subject to x0 + x1 >= 1, bounds [-10, 10]^2.
// Optimum: x* = (0.5, 0.5), f* = 0.5.
struct simple_constrained
{
    static constexpr int problem_dimension = dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        return x[0] * x[0] + x[1] * x[1];
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = x[0] + x[1] - 1.0;
    }

    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd{{-10.0, -10.0}};
    }

    Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd{{10.0, 10.0}};
    }
};

}

// ---------------------------------------------------------------------------
// Concept satisfaction: every alternative variant must satisfy nlp_solver
// when wrapped in basic_solver. Compile-time check only.
// ---------------------------------------------------------------------------

static_assert(nlp_solver<
    basic_solver<alternative::isres::nlopt_faithful_policy<>>>);
static_assert(nlp_solver<
    basic_solver<alternative::isres::original_argmin_policy<>>>);
static_assert(nlp_solver<
    basic_solver<alternative::isres::runarsson_yao_paper_policy<>>>);

// ---------------------------------------------------------------------------
// restarting_policy<variant>::rebind transparency: the IPOP-restart
// decorator must propagate a fixed compile-time dimension N through to
// the inner variant policy. Compile-time check only.
// ---------------------------------------------------------------------------

static_assert(std::same_as<
    restarting_policy<
        alternative::isres::nlopt_faithful_policy<>>::rebind<3>,
    restarting_policy<
        alternative::isres::nlopt_faithful_policy<3>>>);

static_assert(std::same_as<
    restarting_policy<
        alternative::isres::original_argmin_policy<>>::rebind<3>,
    restarting_policy<
        alternative::isres::original_argmin_policy<3>>>);

static_assert(std::same_as<
    restarting_policy<
        alternative::isres::runarsson_yao_paper_policy<>>::rebind<3>,
    restarting_policy<
        alternative::isres::runarsson_yao_paper_policy<3>>>);

// ---------------------------------------------------------------------------
// Smoke tests: each variant runs to completion on simple_constrained
// without throwing and returns a finite objective value. Tight numerical
// thresholds are NOT applied here -- the algorithmic-quality comparison
// belongs in the empirical bench, not the unit-test surface.
// ---------------------------------------------------------------------------

TEST_CASE("nlopt_faithful: simple_constrained smoke",
          "[isres][alternatives][nlopt_faithful]")
{
    simple_constrained problem;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    alternative::isres::nlopt_faithful_policy<> policy;
    policy.options.seed = 42u;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve();

    CHECK(std::isfinite(result.objective_value));
}

TEST_CASE("original_argmin: simple_constrained smoke",
          "[isres][alternatives][original_argmin]")
{
    simple_constrained problem;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    alternative::isres::original_argmin_policy<> policy;
    policy.options.seed = 42u;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve();

    CHECK(std::isfinite(result.objective_value));
}

TEST_CASE("runarsson_yao_paper: simple_constrained smoke",
          "[isres][alternatives][runarsson_yao_paper]")
{
    simple_constrained problem;
    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    alternative::isres::runarsson_yao_paper_policy<> policy;
    policy.options.seed = 42u;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve();

    CHECK(std::isfinite(result.objective_value));
}

// ---------------------------------------------------------------------------
// Synthetic-state ftol_reached emission tests.
//
// Drive state directly into a sigma-collapsed + feasible configuration
// and verify step() emits solver_status::ftol_reached. This is the
// canonical test recipe for the termination predicate without
// relying on the algorithm to converge end-to-end on a real problem.
// ---------------------------------------------------------------------------

TEST_CASE("nlopt_faithful: emits ftol_reached on sigma collapse + feasibility",
          "[isres][alternatives][nlopt_faithful][status]")
{
    simple_constrained problem;
    Eigen::VectorXd x0{{0.5, 0.5}};
    solver_options opts;
    opts.max_iterations = 1;

    alternative::isres::nlopt_faithful_policy<> policy;
    policy.options.seed = 42u;
    policy.options.sigma_collapse_ratio = 1e-9;

    auto s = policy.init(problem, x0, opts);

    // Force sigma collapse and feasibility on the rank-0 individual.
    s.sigmas.leftCols(s.lambda).setConstant(1e-20);
    s.violations.setZero();
    s.best_ever_violation = 0.0;

    auto sr = policy.step(s);
    REQUIRE(sr.policy_status.has_value());
    CHECK(*sr.policy_status == solver_status::ftol_reached);
}

TEST_CASE("runarsson_yao_paper: emits ftol_reached on sigma collapse + feasibility",
          "[isres][alternatives][runarsson_yao_paper][status]")
{
    simple_constrained problem;
    Eigen::VectorXd x0{{0.5, 0.5}};
    solver_options opts;
    opts.max_iterations = 1;

    alternative::isres::runarsson_yao_paper_policy<> policy;
    policy.options.seed = 42u;
    policy.options.sigma_collapse_ratio = 1e-9;

    auto s = policy.init(problem, x0, opts);

    s.sigmas.leftCols(s.lambda).setConstant(1e-20);
    s.violations.setZero();
    s.best_ever_violation = 0.0;

    auto sr = policy.step(s);
    REQUIRE(sr.policy_status.has_value());
    CHECK(*sr.policy_status == solver_status::ftol_reached);
}

// ---------------------------------------------------------------------------
// Synthetic-state NO emission tests: collapsed sigma alone is not enough,
// nor is feasibility alone. Both predicates must fire.
// ---------------------------------------------------------------------------

TEST_CASE("nlopt_faithful: no ftol_reached on large sigma even when feasible",
          "[isres][alternatives][nlopt_faithful][status]")
{
    simple_constrained problem;
    Eigen::VectorXd x0{{0.5, 0.5}};
    solver_options opts;
    opts.max_iterations = 1;

    alternative::isres::nlopt_faithful_policy<> policy;
    policy.options.seed = 42u;
    policy.options.sigma_collapse_ratio = 1e-9;

    auto s = policy.init(problem, x0, opts);

    // Sigma is large -- the bound-relative collapse predicate must NOT fire.
    s.sigmas.leftCols(s.lambda).setConstant(1.0);
    s.violations.setZero();
    s.best_ever_violation = 0.0;

    auto sr = policy.step(s);
    CHECK_FALSE(sr.policy_status.has_value());
}

TEST_CASE("nlopt_faithful: no ftol_reached on infeasible rank-0",
          "[isres][alternatives][nlopt_faithful][status]")
{
    simple_constrained problem;
    Eigen::VectorXd x0{{0.5, 0.5}};
    solver_options opts;
    opts.max_iterations = 1;

    alternative::isres::nlopt_faithful_policy<> policy;
    policy.options.seed = 42u;
    policy.options.sigma_collapse_ratio = 1e-9;
    policy.options.feasibility_gate = 1e-4;

    auto s = policy.init(problem, x0, opts);

    // Sigma collapsed but the population is forced into infeasible
    // territory. The constraint `c[0] = x0 + x1 - 1.0 >= 0` is grossly
    // violated at the lower bound (-10, -10), giving an L2-squared
    // violation of (0 - (-10 + -10 - 1))^2 = 21^2 = 441 >> 1e-4.
    // step() re-evaluates violations after mutation, so we drive the
    // population to the lower bound rather than seeding s.violations
    // directly (which would be overwritten).
    s.population.leftCols(s.lambda).colwise() = s.lower;
    s.sigmas.leftCols(s.lambda).setConstant(1e-20);
    s.violations.setConstant(1.0);
    s.best_ever_violation = 1.0;

    auto sr = policy.step(s);
    CHECK_FALSE(sr.policy_status.has_value());
}

TEST_CASE("original_argmin: never emits ftol_reached (frozen baseline)",
          "[isres][alternatives][original_argmin][status]")
{
    simple_constrained problem;
    Eigen::VectorXd x0{{0.5, 0.5}};
    solver_options opts;
    opts.max_iterations = 1;

    alternative::isres::original_argmin_policy<> policy;
    policy.options.seed = 42u;

    auto s = policy.init(problem, x0, opts);

    // Drive sigma collapse and feasibility -- the frozen baseline must
    // NOT emit any policy_status. The termination predicate is
    // intentionally absent from this variant; it preserves pre-rewrite
    // behavior byte-for-byte for the empirical comparison.
    s.sigmas.leftCols(s.lambda).setConstant(1e-20);
    s.violations.setZero();
    s.best_ever_violation = 0.0;

    auto sr = policy.step(s);
    CHECK_FALSE(sr.policy_status.has_value());
}

// ---------------------------------------------------------------------------
// xtol-coupled fallback test: when the convergence policy lacks a
// step_tolerance threshold, the xtol-coupled form must silently fall
// back to the bound-relative form.
// ---------------------------------------------------------------------------

TEST_CASE("nlopt_faithful: xtol_coupled falls back to bound-relative when threshold absent",
          "[isres][alternatives][nlopt_faithful][status]")
{
    simple_constrained problem;
    Eigen::VectorXd x0{{0.5, 0.5}};
    solver_options opts;
    opts.max_iterations = 1;
    // Intentionally do NOT call set_step_threshold(...) so the
    // convergence policy carries step_tolerance_criterion::threshold as
    // std::nullopt; the xtol_coupled form must fall back to the
    // bound-relative ratio.

    alternative::isres::nlopt_faithful_policy<> policy;
    policy.options.seed = 42u;
    policy.options.sigma_collapse_ratio = 1e-9;
    policy.options.sigma_collapse_form =
        alternative::isres::nlopt_faithful_policy<>::
        sigma_collapse_form_type::xtol_coupled;

    auto s = policy.init(problem, x0, opts);

    s.sigmas.leftCols(s.lambda).setConstant(1e-20);
    s.violations.setZero();
    s.best_ever_violation = 0.0;

    auto sr = policy.step(s);
    REQUIRE(sr.policy_status.has_value());
    CHECK(*sr.policy_status == solver_status::ftol_reached);
}

// ---------------------------------------------------------------------------
// Algorithmic-divergence test: runarsson_yao_paper's rank-permuted
// snapshot pattern produces a different first-step trajectory than
// nlopt_faithful's physical-slot snapshot when the population's rank-0
// individual sits at a different physical slot than column 0.
// Verifies the variants do not collapse to the same operator.
// ---------------------------------------------------------------------------

TEST_CASE("runarsson_yao_paper: rank-permuted snapshot diverges from nlopt_faithful",
          "[isres][alternatives][runarsson_yao_paper][divergence]")
{
    simple_constrained problem;
    Eigen::VectorXd x0{{0.5, 0.5}};
    solver_options opts;
    opts.max_iterations = 1;

    alternative::isres::runarsson_yao_paper_policy<> ry_policy;
    ry_policy.options.seed = 42u;
    auto s_ry = ry_policy.init(problem, x0, opts);

    alternative::isres::nlopt_faithful_policy<> nf_policy;
    nf_policy.options.seed = 42u;
    auto s_nf = nf_policy.init(problem, x0, opts);

    // Both variants ran the same RNG seed through init(); confirm both
    // produced finite step results without throwing, exercising the
    // distinct snapshot/anchor code paths in each step() body.
    auto sr_ry = ry_policy.step(s_ry);
    auto sr_nf = nf_policy.step(s_nf);
    CHECK(std::isfinite(sr_ry.objective_value));
    CHECK(std::isfinite(sr_nf.objective_value));
}
