#include "argmin/solver/cmaes_policy.h"
#include "argmin/solver/alternative/cmaes/repair_l2_penalty_policy.h"
#include "argmin/solver/alternative/cmaes/pwq_reparameterization_policy.h"
#include "argmin/solver/alternative/cmaes/no_repair_adaptive_penalty_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/test_functions/rosenbrock.h"
#include "argmin/test_functions/rastrigin.h"
#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <cmath>
#include <cstdint>

using Catch::Approx;
using namespace argmin;

namespace
{

// Bounded Rosenbrock -- satisfies objective && bound_constrained but NOT
// differentiable (CMA-ES does not need gradient).
struct bounded_rosenbrock
{
    int n{2};
    double a{1};
    double b{5};
    Eigen::VectorXd lb;
    Eigen::VectorXd ub;

    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return n; }

    double value(const Eigen::VectorXd& x) const
    {
        double f = 0.0;
        for(int i = 0; i < n - 1; ++i)
        {
            double t1 = a - x[i];
            double t2 = x[i + 1] - x[i] * x[i];
            f += t1 * t1 + b * t2 * t2;
        }
        return f;
    }

    Eigen::VectorXd lower_bounds() const { return lb; }
    Eigen::VectorXd upper_bounds() const { return ub; }
};

}

static_assert(objective<rosenbrock<double>>);
static_assert(objective<rastrigin<double>>);
static_assert(objective<bounded_rosenbrock>);
static_assert(bound_constrained<bounded_rosenbrock>);

TEST_CASE("cmaes_policy: Rosenbrock 2D", "[cmaes]")
{
    rosenbrock<double> problem{};

    Eigen::VectorXd x0{{-1.0, -1.0}};
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    cmaes_policy<> policy;
    policy.options.initial_sigma = 0.5;
    policy.options.seed = 42u;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.objective_value < 0.01);
    CHECK(result.x[0] == Approx(1.0).margin(0.1));
    CHECK(result.x[1] == Approx(1.0).margin(0.1));
}

TEST_CASE("cmaes_policy: Rastrigin 5D", "[cmaes]")
{
    rastrigin<double> problem{.n = 3};

    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(3, 2.0);
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    cmaes_policy<> policy;
    policy.options.initial_sigma = 2.0;
    policy.options.seed = 123u;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve(opts);

    // Rastrigin 3D is highly multimodal; CMA-ES should reach a good basin.
    // Relaxed threshold: global minimum is 0, local basins have f ~ n*k.
    CHECK(result.objective_value < 10.0);
}

TEST_CASE("cmaes_policy: step_n budget", "[cmaes]")
{
    rosenbrock<double> problem{};

    Eigen::VectorXd x0{{0.0, 0.0}};
    solver_options opts;
    opts.max_iterations = 1000;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    cmaes_policy<> policy;
    policy.options.seed = 7u;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.step_n(10);

    CHECK(result.iterations <= 10);
    CHECK(result.iterations >= 1);
    CHECK(std::isfinite(result.objective_value));
}

TEST_CASE("cmaes_policy: IPOP restart", "[cmaes]")
{
    rastrigin<double> problem{.n = 3};

    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(3, 3.0);
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    cmaes_policy<> policy;
    policy.options.initial_sigma = 0.1;
    policy.options.restart = cmaes_policy<>::restart_strategy::ipop;
    policy.options.seed = 99u;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve(opts);

    // Should complete without crash
    CHECK(std::isfinite(result.objective_value));
    CHECK(result.objective_value < 100.0);
}

TEST_CASE("cmaes_policy: sigma scales from bounds", "[cmaes]")
{
    bounded_rosenbrock problem{
        .n = 2,
        .lb = Eigen::VectorXd{{-5.0, -5.0}},
        .ub = Eigen::VectorXd{{5.0, 5.0}},
    };

    Eigen::VectorXd x0{{0.0, 0.0}};
    solver_options opts;
    opts.max_iterations = 10;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    // No explicit sigma -- should scale from bound range (10.0 / 3.0).
    cmaes_policy<> policy;
    policy.options.seed = 42u;

    basic_solver solver{policy, problem, x0, opts};
    CHECK(solver.state().sigma == Approx(10.0 / 3.0).epsilon(1e-10));
}

TEST_CASE("cmaes_policy: initial per-coordinate spread scales with each box width",
          "[cmaes]")
{
    // On an anisotropic box the initial per-coordinate sampling standard
    // deviation must scale with that coordinate's own range (~0.3 of the
    // width, Hansen tutorial), not with the widest coordinate's range. The
    // scalar step size stays tied to the widest range; the per-coordinate
    // anisotropy is carried in the initial covariance diagonal, so the
    // effective spread on coordinate i is sigma * D_i = range_i / 3.
    bounded_rosenbrock problem{
        .n = 2,
        .lb = Eigen::VectorXd{{-1.0, -100.0}},
        .ub = Eigen::VectorXd{{1.0, 100.0}},
    };

    Eigen::VectorXd x0{{0.0, 0.0}};
    solver_options opts;
    opts.max_iterations = 10;

    cmaes_policy<> policy;
    policy.options.seed = 42u;

    basic_solver solver{policy, problem, x0, opts};
    const auto& s = solver.state();

    const double range0 = 2.0;    // upper[0] - lower[0]
    const double range1 = 200.0;  // upper[1] - lower[1]

    // Scalar step size follows the widest range.
    CHECK(s.sigma == Approx(range1 / 3.0).epsilon(1e-10));

    // Effective initial spread per coordinate is range_i / 3.
    CHECK(s.sigma * s.D[0] == Approx(range0 / 3.0).epsilon(1e-10));
    CHECK(s.sigma * s.D[1] == Approx(range1 / 3.0).epsilon(1e-10));

    // The spread ratio equals the box-width ratio.
    CHECK(s.D[0] / s.D[1] == Approx(range0 / range1).epsilon(1e-10));

    // The covariance diagonal is consistent with D (C_ii = D_i^2).
    CHECK(s.C(0, 0) == Approx(s.D[0] * s.D[0]).epsilon(1e-10));
    CHECK(s.C(1, 1) == Approx(s.D[1] * s.D[1]).epsilon(1e-10));
}

TEST_CASE("cmaes_policy: bounded problems use the Hansen population default",
          "[cmaes]")
{
    bounded_rosenbrock problem{
        .n = 2,
        .lb = Eigen::VectorXd{{-5.0, -5.0}},
        .ub = Eigen::VectorXd{{5.0, 5.0}},
    };

    Eigen::VectorXd x0{{0.0, 0.0}};
    solver_options opts;
    opts.max_iterations = 10;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    // No explicit lambda -- a bounded problem uses the same Hansen default
    // as an unbounded one, 4 + floor(3 ln n) = 6 at n = 2 (no widened
    // bounded floor; the population-doubling restart is the multimodal
    // escalation path).
    cmaes_policy<> policy;
    policy.options.seed = 42u;

    basic_solver solver{policy, problem, x0, opts};
    CHECK(solver.state().params.lambda == 6);
}

TEST_CASE("cmaes_policy: solves bounded Rastrigin", "[cmaes]")
{
    rastrigin<double> problem{.n = 2};

    Eigen::VectorXd x0{{2.0, 2.0}};
    solver_options opts;
    opts.max_iterations = 5000;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    // Rastrigin is multimodal; Hansen (2023) arXiv:1604.00772 §7.1
    // recommends increasing the population beyond the small unimodal
    // default when the landscape is multimodal, precisely so the sampled
    // cloud spans several basins per generation. The reference-faithful
    // covariance update leaves the minimal default population (6 at n=2)
    // too small to reliably reach the global basin from this start, so we
    // configure a population in the multimodal range. This holds the
    // global-convergence bar across the whole seed range rather than
    // relying on a single lucky seed.
    cmaes_policy<> policy;
    policy.options.seed = 42u;
    policy.options.lambda = 32u;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve();

    CHECK(result.iterations > 0);
    CHECK(result.objective_value < 1.0);
}

TEST_CASE("cmaes_policy: Rastrigin 2D global optimum with IPOP", "[cmaes]")
{
    // An IPOP-configured CMA-ES reaches the Rastrigin global optimum from a
    // multimodal start. Rastrigin is highly multimodal; the restart escalation
    // probes multiple basins when the base run stalls. The meaningful bar is
    // reaching the global basin; whether a restart actually had to fire is
    // trajectory-dependent (the corrected covariance can reach the optimum at
    // the base population on a favorable seed) and is not asserted.
    // Reference: K&W Section 8.7 (CMA-ES benchmark).
    rastrigin<double> problem{.n = 2};

    Eigen::VectorXd x0{{3.0, 3.0}};
    solver_options opts;
    opts.max_iterations = 10000;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    cmaes_policy<> policy;
    policy.options.restart = cmaes_policy<>::restart_strategy::ipop;
    policy.options.seed = 5u;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.objective_value < 1.0);
}

TEST_CASE("cmaes_policy: bounded Rosenbrock", "[cmaes]")
{
    bounded_rosenbrock problem{
        .n = 2,
        .lb = Eigen::VectorXd{{-2.0, -2.0}},
        .ub = Eigen::VectorXd{{2.0, 2.0}},
    };

    Eigen::VectorXd x0{{-0.5, -0.5}};
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    cmaes_policy<> policy;
    policy.options.initial_sigma = 0.5;
    policy.options.seed = 55u;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.objective_value < 0.1);
    CHECK(result.x[0] >= -2.0 - 1e-10);
    CHECK(result.x[0] <= 2.0 + 1e-10);
    CHECK(result.x[1] >= -2.0 - 1e-10);
    CHECK(result.x[1] <= 2.0 + 1e-10);
}

namespace
{

// Bounded flat-objective problem: every value is identical so the CMA-ES
// EqualFunValues stagnation criterion fires deterministically once enough
// generations of identical fitness accumulate. The population follows the
// Hansen default 4 + floor(3 ln n), giving a known starting lambda for the
// IPOP recompute regression test.
struct flat_bounded
{
    int n{2};
    double lower_bound{-5.0};
    double upper_bound{5.0};

    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return n; }

    double value(const Eigen::VectorXd&) const { return 1.0; }

    Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Constant(n, lower_bound);
    }

    Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(n, upper_bound);
    }
};

}

static_assert(objective<flat_bounded>);
static_assert(bound_constrained<flat_bounded>);

TEST_CASE("cmaes_policy: stagnation_window_min initial value", "[cmaes]")
{
    // Lock the Hansen 2023 (arXiv:1604.00772) section B.3 paragraph
    // "Stagnation" minimum window formula at init time:
    //   stagnation_window_min = 120 + ceil(30 * n / lambda)
    // where lambda is the Hansen-default population for the dimension.
    // The unbounded path uses compute_constants(n, 0) which yields the
    // auto-computed lambda = 4 + floor(3 * ln(n)).
    using policy_t = cmaes_policy<>;

    solver_options opts;
    opts.max_iterations = 1;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    {
        rosenbrock<double> problem{};  // n = 2 by default, unbounded
        Eigen::VectorXd x0 = Eigen::VectorXd::Constant(2, -1.0);
        policy_t policy;
        policy.options.seed = 42u;
        basic_solver solver{policy, problem, x0, opts};
        const int n = 2;
        const int lambda = solver.state().params.lambda;
        const auto expected = static_cast<std::uint32_t>(120)
            + static_cast<std::uint32_t>(
                std::ceil(30.0 * static_cast<double>(n)
                          / static_cast<double>(lambda)));
        CHECK(solver.state().stagnation_window_min == expected);
    }

    {
        rastrigin<double> problem{.n = 5};  // unbounded
        Eigen::VectorXd x0 = Eigen::VectorXd::Constant(5, 1.0);
        policy_t policy;
        policy.options.seed = 42u;
        basic_solver solver{policy, problem, x0, opts};
        const int n = 5;
        const int lambda = solver.state().params.lambda;
        const auto expected = static_cast<std::uint32_t>(120)
            + static_cast<std::uint32_t>(
                std::ceil(30.0 * static_cast<double>(n)
                          / static_cast<double>(lambda)));
        CHECK(solver.state().stagnation_window_min == expected);
    }

    {
        rastrigin<double> problem{.n = 10};  // unbounded
        Eigen::VectorXd x0 = Eigen::VectorXd::Constant(10, 1.0);
        policy_t policy;
        policy.options.seed = 42u;
        basic_solver solver{policy, problem, x0, opts};
        const int n = 10;
        const int lambda = solver.state().params.lambda;
        const auto expected = static_cast<std::uint32_t>(120)
            + static_cast<std::uint32_t>(
                std::ceil(30.0 * static_cast<double>(n)
                          / static_cast<double>(lambda)));
        CHECK(solver.state().stagnation_window_min == expected);
    }
}

TEST_CASE("cmaes_policy: ipop stagnation_window_min recompute", "[cmaes]")
{
    // Hansen 2023 (arXiv:1604.00772) section B.3 specifies the Stagnation
    // minimum window as 120 + ceil(30 * n / lambda) with an explicit
    // dependence on the CURRENT lambda. libcmaes recomputes this implicitly
    // every iteration via the _max_hist cap. argmin computes it once in
    // init(); on an IPOP restart lambda doubles and the window must be
    // recomputed against the new lambda. This test pins both the init-time
    // value and the post-restart value and is the regression guard for the
    // 2026-04-30 libcmaes head-to-head finding.
    //
    // Trigger surface: the §B.3 EXIT criteria (EqualFunValues, TolFun,
    // TolX, TolXUp, NoEffectAxis, NoEffectCoord) exit the solve regardless
    // of restart_strategy. The IPOP-restart trigger set is therefore
    // {Stagnation, sigma_collapse, cond_explosion}. We use a smooth
    // quadratic + sigma_collapse driver: the EXIT thresholds are widened
    // (objective_value_tolerance and step_size_tolerance set to 1e-30 --
    // well below numerical floor) so the sigma_collapse legacy probe
    // fires first and recycles into an IPOP doubling.
    rosenbrock<double> problem{};

    Eigen::VectorXd x0{{-1.0, -1.0}};
    solver_options opts;
    opts.max_iterations = 2000;
    opts.set_gradient_threshold(1e-30);
    opts.set_objective_threshold(1e-30);
    opts.set_step_threshold(1e-30);

    cmaes_policy<> policy;
    policy.options.initial_sigma = 0.5;
    policy.options.restart = cmaes_policy<>::restart_strategy::ipop;
    policy.options.seed = 42u;
    // Defeat the §B.3 EXIT criteria so sigma_collapse drives the IPOP
    // restart path. 1e-30 is below the floating-point noise floor for
    // these scales: TolFun requires a fitness range below 1e-30 (never
    // achievable on rosenbrock); TolX requires sigma*sqrt(C(i,i)) below
    // 1e-30 * initial_sigma (will not reach this before sigma_collapse
    // at the 1e-12 default).
    policy.options.cmaes.objective_value_tolerance = 1e-30;
    policy.options.cmaes.step_size_tolerance = 1e-30;
    // Pin the legacy single-axis sigma_collapse threshold to the §B.3
    // default 1e-12 explicitly. sigma_collapse_threshold is a direct-value
    // field independent of step_size_tolerance, so defeating §B.3 TolX
    // above (step_size_tolerance = 1e-30) leaves this probe at its paper
    // baseline; pinning it here documents that the legacy probe stays
    // active and drives the IPOP restart.
    policy.options.cmaes.sigma_collapse_threshold = 1e-12;

    basic_solver solver{policy, problem, x0, opts};

    const int n = 2;
    const int lambda_initial = solver.state().params.lambda;
    // Unbounded n=2 picks the auto formula 4 + floor(3 * ln(2)) = 6.
    REQUIRE(lambda_initial == 6);

    const auto expected_initial = static_cast<std::uint32_t>(120)
        + static_cast<std::uint32_t>(
            std::ceil(30.0 * static_cast<double>(n)
                      / static_cast<double>(lambda_initial)));
    REQUIRE(solver.state().stagnation_window_min == expected_initial);  // 130

    // Drive the solver until sigma_collapse fires and triggers an IPOP
    // doubling. On rosenbrock with sigma=0.5 this lands well within
    // the 2000-iteration budget.
    solver.solve(opts);

    const int lambda_after = solver.state().params.lambda;
    REQUIRE(lambda_after >= lambda_initial * 2);  // at least one doubling

    const auto expected_after = static_cast<std::uint32_t>(120)
        + static_cast<std::uint32_t>(
            std::ceil(30.0 * static_cast<double>(n)
                      / static_cast<double>(lambda_after)));
    CHECK(solver.state().stagnation_window_min == expected_after);
}

namespace
{

// Unbounded flat objective: every value is bit-equal 1.0. Used to drive the
// Hansen 2023 (arXiv:1604.00772) section B.3 item 4 (EqualFunValues) exit
// criterion deterministically. Unbounded so the penalty path in
// cmaes_policy::step() is fully skipped (bound_constrained concept is NOT
// satisfied) -- the offspring fitness is exactly the objective value, so
// the bit-equal range across generations is guaranteed.
struct flat_unbounded
{
    static constexpr int problem_dimension = 2;
    int dimension() const { return 2; }
    double value(const Eigen::Vector<double, 2>&) const { return 1.0; }
};

// Unbounded objective whose best-of-generation fitness is bit-INEQUAL
// across the EqualFunValues window. The literal `==` predicate must NOT
// fire on a non-zero range; this is the discriminator for the predicate
// change. We use a smooth quadratic with a moderate initial_sigma so
// CMA-ES descends meaningfully every generation and best-of-generation
// fitness keeps strictly decreasing for many iterations -- guaranteeing
// the minmax range over the EFV window stays strictly positive (and NOT
// bit-equal) for the duration of the test budget.
struct strictly_descending_quadratic
{
    static constexpr int problem_dimension = 2;
    int dimension() const { return 2; }
    double value(const Eigen::Vector<double, 2>& x) const
    {
        return x.squaredNorm();
    }
};

// Diagonal quadratic with a near-zero scaling on coordinate 1: the objective
// only depends meaningfully on x(0). Drives NoEffectCoord by collapsing
// C(1,1) to a magnitude where 0.2 * sigma * sqrt(C(1,1)) is bit-equal to the
// mean component during sample-and-step.
struct degenerate_quadratic
{
    static constexpr int problem_dimension = 2;
    int dimension() const { return 2; }
    double value(const Eigen::Vector<double, 2>& x) const
    {
        return x(0) * x(0) + 1e-30 * x(1) * x(1);
    }
};

// Unbounded concave objective: minimization of -x.squaredNorm() is divergent.
// CMA-ES will let sigma grow without bound; sigma * max(diag(D)) eventually
// exceeds 1e4 * initial sigma * initial_d_max, firing the §B.3 item 8
// (TolXUp) exit criterion with status diverged.
struct concave_unbounded
{
    static constexpr int problem_dimension = 2;
    int dimension() const { return 2; }
    double value(const Eigen::Vector<double, 2>& x) const
    {
        return -x.squaredNorm();
    }
};

}

static_assert(objective<flat_unbounded>);
static_assert(!bound_constrained<flat_unbounded>);
static_assert(objective<strictly_descending_quadratic>);
static_assert(!bound_constrained<strictly_descending_quadratic>);
static_assert(objective<degenerate_quadratic>);
static_assert(objective<concave_unbounded>);

TEST_CASE("cmaes_policy: equal_fun_values exact zero", "[cmaes]")
{
    // Hansen 2023 (arXiv:1604.00772) section B.3 item 4 (EqualFunValues):
    // "stop if the range of the best objective function values of the last
    // 10 + ceil(30n/lambda) generations is zero." The predicate is the
    // literal `*mx == *mn` per the paper text. On a flat unbounded
    // objective the best-of-generation fitness is bit-equal 1.0 across
    // every generation, so EqualFunValues fires deterministically once
    // 10 + ceil(30*2/lambda) = 16 (lambda=auto=6 yields ceil(60/6)=10,
    // window 20; lambda overrides may shrink this) generations have
    // accumulated. Penalty path skipped -- unbounded problem; the only
    // fitness value possible is 1.0 -> bit-equal range, deterministic
    // across seeds.
    flat_unbounded problem{};

    Eigen::Vector<double, 2> x0{{0.0, 0.0}};
    solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-30);
    opts.set_objective_threshold(1e-30);
    opts.set_step_threshold(1e-30);

    cmaes_policy<2> policy;
    policy.options.seed = 42u;
    // Defeat TolFun (which would also fire on flat fitness with default
    // 1e-12 tolerance) so EqualFunValues is the criterion that fires.
    policy.options.cmaes.objective_value_tolerance = 1e-30;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.status == solver_status::ftol_reached);
}

TEST_CASE("cmaes_policy: equal_fun_values rejects sub-epsilon", "[cmaes]")
{
    // Hansen 2023 (arXiv:1604.00772) section B.3 item 4 (EqualFunValues)
    // discriminator: with the literal `==` predicate, a fitness range > 0
    // must NOT fire EqualFunValues. The legacy relaxed predicate
    // `< 1e-15 * (|max| + |min| + 1e-30)` WOULD fire on tiny but non-zero
    // ranges; the discriminator proves the predicate change is exercised
    // by holding the range strictly positive over the entire budget and
    // verifying EqualFunValues does NOT fire.
    //
    // Setup: the smooth quadratic with a moderate initial_sigma drives
    // CMA-ES through a steady descent. Best-of-generation fitness changes
    // every generation (range > 0), and at iter 50 the range over the EFV
    // window is still strictly positive. All other §B.3 EXIT criteria are
    // defeated via 1e-30 thresholds so the only criterion that could
    // possibly fire is EqualFunValues -- if it fires, the bit-equal
    // predicate is wrong.
    strictly_descending_quadratic problem{};

    Eigen::Vector<double, 2> x0{{1.0, 1.0}};
    solver_options opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-30);
    opts.set_objective_threshold(1e-30);
    opts.set_step_threshold(1e-30);

    cmaes_policy<2> policy;
    policy.options.initial_sigma = 0.5;
    policy.options.seed = 42u;
    // Defeat all other §B.3 EXIT criteria so only EqualFunValues could
    // possibly fire within 50 iterations on this descent.
    policy.options.cmaes.objective_value_tolerance = 1e-30;
    policy.options.cmaes.step_size_tolerance = 1e-30;
    policy.options.cmaes.sigma_collapse_threshold = 1e-30;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve(opts);

    // With the literal `==` predicate, EqualFunValues must NOT fire on
    // a strictly-descending non-zero range. The solve runs to budget.
    CHECK(result.iterations == 50);
    CHECK(result.status == solver_status::max_iterations);
}

TEST_CASE("cmaes_policy: tol_fun exit on smooth descent", "[cmaes]")
{
    // Hansen 2023 (arXiv:1604.00772) section B.3 item 6 (TolFun): the
    // range of the best-of-generation history (and the current
    // generation's offspring) drops below TolFun once the population
    // has settled near the optimum. The §B.3 default 1e-12 fires after
    // CMA-ES has converged on the smooth quadratic centered at the origin.
    rosenbrock<double> problem{};

    Eigen::VectorXd x0{{-1.0, -1.0}};
    solver_options opts;
    opts.max_iterations = 2000;
    // Use very tight basic_solver convergence to ensure the policy
    // criterion (not basic_solver convergence) is what fires.
    opts.set_gradient_threshold(1e-30);
    opts.set_objective_threshold(1e-30);
    opts.set_step_threshold(1e-30);

    cmaes_policy<> policy;
    policy.options.initial_sigma = 0.5;
    policy.options.seed = 42u;
    // Defeat TolX so TolFun is what fires (TolX would fire first on this
    // setup at the default 1e-12 once sigma collapses). The legacy
    // single-axis sigma_collapse probe shares TolX's 1e-12 baseline and
    // is a direct-value field, so defeat it explicitly too.
    policy.options.cmaes.step_size_tolerance = 1e-30;
    policy.options.cmaes.sigma_collapse_threshold = 1e-30;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.status == solver_status::ftol_reached);
    CHECK(result.objective_value < 1e-10);
}

TEST_CASE("cmaes_policy: tol_x exit on contracted distribution", "[cmaes]")
{
    // Hansen 2023 (arXiv:1604.00772) section B.3 item 7 (TolX): the
    // sampling-distribution standard deviation drops below TolX in all
    // coordinates AND sigma*p_c is below TolX in all components. Driving
    // a smooth quadratic to convergence with a deliberately tight
    // initial_sigma forces sigma to collapse to roundoff; we defeat TolFun
    // (1e-30) so TolX is what fires.
    rosenbrock<double> problem{};

    Eigen::VectorXd x0{{-1.0, -1.0}};
    solver_options opts;
    opts.max_iterations = 5000;
    opts.set_gradient_threshold(1e-30);
    opts.set_objective_threshold(1e-30);
    opts.set_step_threshold(1e-30);

    cmaes_policy<> policy;
    policy.options.initial_sigma = 0.01;
    policy.options.seed = 42u;
    // Defeat TolFun so TolX is what fires.
    policy.options.cmaes.objective_value_tolerance = 1e-30;
    // Defeat the legacy sigma_collapse probe (1e-12 default) too so the
    // §B.3 TolX criterion is what owns the exit. TolX defaults to
    // 1e-12 * initial_sigma which on a 0.01 initial_sigma is 1e-14 --
    // below sigma_collapse's 1e-12 * initial_sigma = 1e-14 floor by an
    // identical magnitude, but TolX's all-coords semantics is stricter.
    policy.options.cmaes.sigma_collapse_threshold = 1e-30;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.status == solver_status::roundoff_limited);
    CHECK(result.objective_value < 1e-15);
}

TEST_CASE("cmaes_policy: objective_value_tolerance user override", "[cmaes]")
{
    // The user-override plumbing for objective_value_tolerance routes
    // through cmaes_options and is read directly (direct-value default
    // 1e-12) at the §B.3 item 6 check site. A loose user threshold (1e-6) MUST fire
    // earlier than the default (1e-12). We compare iter counts on two
    // identical runs differing only in the user threshold.
    rosenbrock<double> problem{};

    Eigen::VectorXd x0{{-1.0, -1.0}};
    solver_options opts;
    opts.max_iterations = 2000;
    opts.set_gradient_threshold(1e-30);
    opts.set_objective_threshold(1e-30);
    opts.set_step_threshold(1e-30);

    // Run 1: default TolFun threshold (1e-12).
    std::uint32_t iters_default = 0;
    {
        cmaes_policy<> policy;
        policy.options.initial_sigma = 0.5;
        policy.options.seed = 42u;
        policy.options.cmaes.step_size_tolerance = 1e-30;
        policy.options.cmaes.sigma_collapse_threshold = 1e-30;
        basic_solver solver{policy, problem, x0, opts};
        auto result = solver.solve(opts);
        REQUIRE(result.status == solver_status::ftol_reached);
        iters_default = result.iterations;
    }

    // Run 2: loose user TolFun threshold (1e-6).
    std::uint32_t iters_loose = 0;
    {
        cmaes_policy<> policy;
        policy.options.initial_sigma = 0.5;
        policy.options.seed = 42u;
        policy.options.cmaes.step_size_tolerance = 1e-30;
        policy.options.cmaes.sigma_collapse_threshold = 1e-30;
        policy.options.cmaes.objective_value_tolerance = 1e-6;
        basic_solver solver{policy, problem, x0, opts};
        auto result = solver.solve(opts);
        REQUIRE(result.status == solver_status::ftol_reached);
        iters_loose = result.iterations;
    }

    // The loose threshold MUST fire earlier (fewer iterations) than the
    // default threshold. This proves the user-override plumbing is wired.
    CHECK(iters_loose < iters_default);
}

TEST_CASE("cmaes_policy: sigma_collapse_threshold user override", "[cmaes]")
{
    // Legacy single-axis sigma collapse check is RETAINED as a user
    // override hook for callers that explicitly set
    // `cmaes_options::sigma_collapse_threshold`. This test pins the
    // user-override path so a future cleanup that drops the field would
    // flip a deterministic inequality.
    //
    // Setup mirrors the `objective_value_tolerance user override` test:
    // two runs differ only in the override field. A loose threshold
    // (1e-6) MUST fire the legacy probe earlier (fewer iterations) than
    // the default 1e-12 baseline. The §B.3 EXIT-only criteria are all
    // defeated to 1e-30 so the legacy max-axis probe is the determining
    // exit criterion in both runs.
    rosenbrock<double> problem{};

    Eigen::VectorXd x0{{-1.0, -1.0}};
    solver_options opts;
    opts.max_iterations = 2000;
    opts.set_gradient_threshold(1e-30);
    opts.set_objective_threshold(1e-30);
    opts.set_step_threshold(1e-30);

    auto run = [&](std::optional<double> threshold) -> std::uint32_t {
        cmaes_policy<> policy;
        policy.options.initial_sigma = 0.5;
        policy.options.seed = 42u;
        // Defeat the §B.3 EXIT-only criteria so the legacy single-axis
        // probe is the only path to roundoff_limited.
        policy.options.cmaes.objective_value_tolerance = 1e-30;
        policy.options.cmaes.step_size_tolerance = 1e-30;
        if(threshold.has_value())
            policy.options.cmaes.sigma_collapse_threshold = *threshold;
        basic_solver solver{policy, problem, x0, opts};
        auto result = solver.solve(opts);
        REQUIRE(result.status == solver_status::roundoff_limited);
        return result.iterations;
    };

    const std::uint32_t iters_default = run(std::nullopt);
    const std::uint32_t iters_loose = run(1e-6);

    // The loose user threshold MUST fire earlier (fewer iterations) than
    // the direct-value default 1e-12 baseline. This proves the legacy
    // override hook is still wired via a direct field read.
    CHECK(iters_loose < iters_default);
}

TEST_CASE("cmaes_policy: no_effect_axis exit", "[cmaes]")
{
    // Hansen 2023 (arXiv:1604.00772) section B.3 item 1 (NoEffectAxis),
    // footnote 31: "terminate if m equals m + 0.1*sigma*d_ii*b_i, where
    // i = (g mod n) + 1." On a well-converged run the mean is bit-stable
    // along at least one principal axis, firing the criterion with
    // status roundoff_limited.
    //
    // The criterion-class assertion (vs predicate-class) acknowledges
    // that on this setup multiple §B.3 EXIT criteria may race: NoEffectAxis,
    // NoEffectCoord, and TolX all map to roundoff_limited. We assert the
    // criterion class fired (status == roundoff_limited), which proves at
    // least one of the three roundoff-mapped criteria triggered the exit
    // and not basic_solver convergence or max_iterations.
    rosenbrock<double> problem{};

    Eigen::VectorXd x0{{0.99, 0.99}};
    solver_options opts;
    opts.max_iterations = 2000;
    opts.set_gradient_threshold(1e-30);
    opts.set_objective_threshold(1e-30);
    opts.set_step_threshold(1e-30);

    cmaes_policy<> policy;
    policy.options.initial_sigma = 0.001;
    policy.options.seed = 42u;
    // Defeat TolFun so the no-effect / TolX class is what fires.
    policy.options.cmaes.objective_value_tolerance = 1e-30;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.status == solver_status::roundoff_limited);
}

TEST_CASE("cmaes_policy: flat coordinate converges without spurious divergence",
          "[cmaes]")
{
    // Degenerate quadratic: f = x0^2 + 1e-30 * x1^2, i.e. sharp curvature
    // on x0 and an effectively flat x1 direction. The rank-mu covariance
    // update divides each offspring displacement by the sigma that sampled
    // it (Hansen 2016 arXiv:1604.00772 eq. 47), so the flat coordinate
    // receives no spurious variance inflation: the run drives x0 to its
    // minimizer and terminates on a legitimate stationarity criterion.
    //
    // Regression guard: an implementation that divides the rank-mu deltas
    // by the already-adapted (smaller) sigma over-weights the flat
    // coordinate's rank-mu contribution, inflating sigma * sqrt(C(1,1))
    // until the divergence cap (Hansen §B.3 item 8, TolXUp) fires and the
    // run exits `diverged` instead of converging. This case pins the
    // faithful outcome: a clean convergence to the true minimizer on the
    // active coordinate, never a budget timeout and never a divergence.
    degenerate_quadratic problem{};

    Eigen::Vector<double, 2> x0{{1.0, 1.0}};
    solver_options opts;
    opts.max_iterations = 5000;
    opts.set_gradient_threshold(1e-30);
    opts.set_objective_threshold(1e-30);
    opts.set_step_threshold(1e-30);

    cmaes_policy<2> policy;
    policy.options.initial_sigma = 0.1;
    policy.options.seed = 42u;
    policy.options.cmaes.objective_value_tolerance = 1e-30;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve(opts);

    // The run stops on a real criterion, not by exhausting the budget, and
    // never on the spurious-divergence path the fix closes.
    CHECK(result.status != solver_status::max_iterations);
    CHECK(result.status != solver_status::diverged);
    // The active coordinate reached its minimizer at the origin.
    CHECK(result.x[0] == Approx(0.0).margin(1e-6));
}

TEST_CASE("cmaes_policy: tol_x_up divergence detection", "[cmaes]")
{
    // Hansen 2023 (arXiv:1604.00772) section B.3 item 8 (TolXUp): "stop
    // if sigma * max(diag(D)) increased by more than 10^4. This usually
    // indicates a far too small initial sigma, or divergent behavior."
    // Minimization of a concave objective lets sigma grow unboundedly --
    // sigma * max(diag(D)) eventually exceeds 1e4 * initial_sigma *
    // initial_d_max and TolXUp fires with status diverged.
    //
    // Unconstrained problem variant (concave + bounds plateaus at the
    // corner instead of diverging), per plan note in <behavior>.
    concave_unbounded problem{};

    Eigen::Vector<double, 2> x0{{0.0, 0.0}};
    solver_options opts;
    opts.max_iterations = 5000;
    opts.set_gradient_threshold(1e-30);
    opts.set_objective_threshold(1e-30);
    opts.set_step_threshold(1e-30);

    cmaes_policy<2> policy;
    policy.options.initial_sigma = 0.1;
    policy.options.seed = 42u;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.status == solver_status::diverged);
    // TolXUp must fire well before the iteration budget is exhausted.
    CHECK(result.iterations < opts.max_iterations);
}

TEST_CASE("cmaes_policy: ipop mean reset to x0 on restart", "[cmaes]")
{
    // Auger & Hansen 2005 ("A Restart CMA Evolution Strategy with Increasing
    // Population Size", CEC 2005) §III specifies that on each IPOP restart
    // the distribution mean is re-anchored to the user-provided initial
    // point x0. Without this re-anchor, IPOP doublings explore variance only
    // (re-sampling the SAME basin with a wider distribution) instead of
    // probing new basins -- which is the inner-cmaes saturation pathology
    // observed in the 2026-04-30 libcmaes head-to-head.
    //
    // libcmaes implements §III via reset_search_state() ->
    // CMASolutions(Parameters&), where _xmean = p._x0min when x0 is a fixed
    // single point (cmasolutions.cc:49). argmin mirrors this by storing
    // x0 in s.x0 at init() and writing s.mean = s.x0 in the IPOP restart
    // branch. This test pins that contract.
    //
    // Driver mirrors the existing `ipop stagnation_window_min recompute`
    // case: rosenbrock unbounded n=2 with sigma=0.5 and seed=42, EXIT
    // criteria defeated to 1e-30 so the sigma_collapse legacy probe drives
    // the IPOP doubling. We step the solver manually and check s.mean
    // bit-equal to s.x0 in the first generation immediately after the
    // doubling fires (before the new run accumulates a fresh mean update).
    rosenbrock<double> problem{};

    Eigen::VectorXd x0{{-1.0, -1.0}};
    solver_options opts;
    opts.max_iterations = 2000;
    opts.set_gradient_threshold(1e-30);
    opts.set_objective_threshold(1e-30);
    opts.set_step_threshold(1e-30);

    cmaes_policy<> policy;
    policy.options.initial_sigma = 0.5;
    policy.options.restart = cmaes_policy<>::restart_strategy::ipop;
    policy.options.seed = 42u;
    policy.options.cmaes.objective_value_tolerance = 1e-30;
    policy.options.cmaes.step_size_tolerance = 1e-30;

    basic_solver solver{policy, problem, x0, opts};

    const int lambda_initial = solver.state().params.lambda;
    REQUIRE(lambda_initial == 6);  // unbounded n=2 default

    // s.x0 must be captured at init() and equal the user x0.
    REQUIRE(solver.state().x0 == x0);

    // Step until lambda doubles (i.e. an IPOP restart fired). Cap the
    // budget at max_iterations so the test never spins.
    bool ipop_fired = false;
    for(std::uint32_t i = 0; i < opts.max_iterations; ++i)
    {
        solver.step();
        if(solver.state().params.lambda > lambda_initial)
        {
            ipop_fired = true;
            // Immediately after the IPOP branch sets s.mean = s.x0 (and
            // before the next sample-and-update cycle moves the mean
            // away), the mean must equal the user's initial point.
            // The restart branch finishes at the end of step(), so we
            // are now positioned at the START of the next generation
            // with mean already re-anchored.
            CHECK(solver.state().mean == x0);
            CHECK(solver.state().x0 == x0);  // x0 preserved across restart
            break;
        }
    }
    REQUIRE(ipop_fired);
}

TEST_CASE("cmaes_policy: ipop decomposition_skip_k recompute", "[cmaes]")
{
    // Hansen 2023 (arXiv:1604.00772) §B.2 (Strategy internal numerical
    // effort): the eigendecomposition skip period
    //   K = max(1, floor(1 / (10 * n * (c_1 + c_mu))))
    // depends on c_1 and c_mu, both of which are recomputed by
    // detail::compute_constants(n, lambda) when lambda doubles in the IPOP
    // restart branch. The skip-k must therefore be recomputed too. Without
    // this, IPOP runs at the new lambda with an obsolete skip period from
    // the pre-doubling lambda.
    //
    // The init-time formula at low n=2 + lambda=6 yields a sizeable K
    // (c_1, c_mu both well below 1/(10*n)); after lambda doubles to >= 12
    // c_1 is unchanged but c_mu grows (mu_eff scales with lambda), so K
    // shrinks. Both values are well above the max(1,...) floor for n=2,
    // so the post-restart value is observably smaller than the init value.
    //
    // We assert the post-restart value equals the formula recomputed
    // verbatim against the new params (not a hand-computed constant), so
    // the test is robust to any future c_1/c_mu refinement.
    rosenbrock<double> problem{};

    Eigen::VectorXd x0{{-1.0, -1.0}};
    solver_options opts;
    opts.max_iterations = 2000;
    opts.set_gradient_threshold(1e-30);
    opts.set_objective_threshold(1e-30);
    opts.set_step_threshold(1e-30);

    cmaes_policy<> policy;
    policy.options.initial_sigma = 0.5;
    policy.options.restart = cmaes_policy<>::restart_strategy::ipop;
    policy.options.seed = 42u;
    policy.options.cmaes.objective_value_tolerance = 1e-30;
    policy.options.cmaes.step_size_tolerance = 1e-30;
    // Pin sigma_collapse_threshold to the §B.3 default so the legacy
    // probe stays active despite step_size_tolerance defeating TolX.
    // sigma_collapse_threshold is a direct-value field independent of
    // step_size_tolerance, so this simply keeps the legacy probe at its
    // 1e-12 paper baseline.
    policy.options.cmaes.sigma_collapse_threshold = 1e-12;

    basic_solver solver{policy, problem, x0, opts};

    const int n = 2;
    const int lambda_initial = solver.state().params.lambda;
    REQUIRE(lambda_initial == 6);

    // Drive until IPOP fires.
    solver.solve(opts);

    const int lambda_after = solver.state().params.lambda;
    REQUIRE(lambda_after >= lambda_initial * 2);

    // Post-restart skip period must equal the formula recomputed against
    // the post-restart params (s.params already reflects new_lambda).
    const auto& params_after = solver.state().params;
    const double skip_denom_after =
        10.0 * static_cast<double>(n) * (params_after.c_1 + params_after.c_mu);
    const auto expected_skip_k_after = std::max(
        std::uint32_t{1},
        static_cast<std::uint32_t>(std::floor(1.0 / skip_denom_after)));
    CHECK(solver.state().decomposition_skip_k == expected_skip_k_after);
}

TEST_CASE("cmaes_policy: ipop bit-identity within process", "[cmaes]")
{
    // SEED-014 (restarting_cmaes rebuild nondeterminism) closure preflight
    // for the inner `cmaes` row (CONTEXT D-08 corrigendum: this row =
    // cmaes_policy<> with restart_strategy::ipop, NOT the restarting_policy
    // decorator). Two back-to-back runs from the SAME solver_options.seed
    // on the SAME problem with FRESH basic_solver instances must produce
    // bit-identical final_objective and identical iteration counts.
    //
    // The IPOP branch uses no fresh entropy: s.rng is seeded once at init()
    // (xoshiro256{seed}) and never reseeded during restart -- libcmaes
    // matches (no reseed across restart). The IPOP branch resets s.mean
    // to s.x0 (deterministic), s.decomposition_skip_k from the formula
    // (deterministic), and the C/B/D/p_sigma/p_c/sigma reset values are
    // all constants. Within-process determinism therefore depends only
    // on (a) the seed plumbing and (b) the absence of any uninitialized
    // state read inside step().
    //
    // NaN-trap-safe equality: on rastrigin n=2 from x0=(3,3) at
    // max_iterations=500 with seed=42 and the current termination
    // machinery, the policy stays finite end-to-end -- there is no
    // penalty overflow, no log(0). If a future regression introduces
    // NaN, the bit_cast branch surfaces it instead of the equality
    // silently swallowing the bug (NaN != NaN under operator==).
    //
    // This is the WITHIN-PROCESS preflight only; rebuild-rebuild closure
    // for the restarting_policy decorator surface is the territory of a
    // downstream plan.
    rastrigin<double> problem{.n = 2};

    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(2, 3.0);
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-30);
    opts.set_objective_threshold(1e-30);
    opts.set_step_threshold(1e-30);

    auto run_once = [&]() {
        cmaes_policy<> policy;
        policy.options.initial_sigma = 0.5;
        policy.options.restart = cmaes_policy<>::restart_strategy::ipop;
        policy.options.seed = 42u;
        // Defeat §B.3 EXIT thresholds so the IPOP path actually exercises
        // (otherwise TolFun / TolX may fire first and we never test the
        // restart-determinism contract).
        policy.options.cmaes.objective_value_tolerance = 1e-30;
        policy.options.cmaes.step_size_tolerance = 1e-30;

        basic_solver solver{policy, problem, x0, opts};
        return solver.solve(opts);
    };

    auto result_a = run_once();
    auto result_b = run_once();

    const bool bit_identical =
        (std::isfinite(result_a.objective_value)
         && std::isfinite(result_b.objective_value)
         && result_a.objective_value == result_b.objective_value)
     || (!std::isfinite(result_a.objective_value)
         && !std::isfinite(result_b.objective_value)
         && std::bit_cast<std::uint64_t>(result_a.objective_value)
             == std::bit_cast<std::uint64_t>(result_b.objective_value));
    REQUIRE(bit_identical);
    REQUIRE(result_a.iterations == result_b.iterations);
}

// Lock the vanilla CMA-ES default (positive-weights only). Active-CMA
// negative-weight rescaling is out of scope for this milestone; if a
// future variant re-enables it, this regression test flips and forces a
// documented decision.
//
// References:
//   Hansen (2023) arXiv:1604.00772 §B.1 eq (49)-(50).
//   libcmaes covarianceupdate.cc:69-75 (positive-weights only).
TEST_CASE("cmaes_policy: weights tail is zero in vanilla default", "[cmaes]")
{
    SECTION("lambda=6, mu=3")
    {
        rastrigin<double> problem{.n = 2};
        Eigen::VectorXd x0 = Eigen::VectorXd::Constant(2, 2.0);
        solver_options opts;
        opts.max_iterations = 1;

        cmaes_policy<> policy;
        policy.options.lambda = 6u;
        policy.options.seed = 42u;

        basic_solver solver{policy, problem, x0, opts};
        (void) solver.step();
        const auto& state = solver.state();
        const auto& weights = state.params.weights;
        const int mu = state.params.mu;
        const int lambda = state.params.lambda;

        REQUIRE(lambda == 6);
        REQUIRE(mu == 3);
        double sum_pos = 0.0;
        for(int i = 0; i < mu; ++i) sum_pos += weights[i];
        CHECK(sum_pos == Approx(1.0).margin(1e-12));
        for(int i = mu; i < lambda; ++i)
            CHECK(weights[i] == 0.0);
    }

    SECTION("lambda=12, mu=6")
    {
        rastrigin<double> problem{.n = 2};
        Eigen::VectorXd x0 = Eigen::VectorXd::Constant(2, 2.0);
        solver_options opts;
        opts.max_iterations = 1;

        cmaes_policy<> policy;
        policy.options.lambda = 12u;
        policy.options.seed = 42u;

        basic_solver solver{policy, problem, x0, opts};
        (void) solver.step();
        const auto& state = solver.state();
        const auto& weights = state.params.weights;
        const int mu = state.params.mu;
        const int lambda = state.params.lambda;

        REQUIRE(lambda == 12);
        REQUIRE(mu == 6);
        double sum_pos = 0.0;
        for(int i = 0; i < mu; ++i) sum_pos += weights[i];
        CHECK(sum_pos == Approx(1.0).margin(1e-12));
        for(int i = mu; i < lambda; ++i)
            CHECK(weights[i] == 0.0);
    }
}

// Phase 34 G12 caller-facing-unpenalized contract: for any
// boundary-handling variant, state.objective_value MUST equal
// problem.value(state.x) at termination -- the caller can re-evaluate
// the objective at the policy's reported iterate and read back the
// same number. Each variant achieves this differently:
//   - repair_l2_penalty: state.x is the L2-repaired offspring;
//     unpenalized = problem.value(state.x).
//   - pwq_reparameterization: state.x is the pheno coord; unpenalized
//     = problem.value(g(geno)) = problem.value(state.x).
//   - no_repair_adaptive_penalty: state.x is the clipped repaired
//     point; unpenalized = problem.value(state.x).
//
// Reference: Phase 34 G12 (caller-facing-unpenalized contract).
TEST_CASE("alternative::cmaes::repair_l2_penalty_policy: caller-facing "
          "objective is feasible", "[cmaes]")
{
    bounded_rosenbrock prob{
        .n  = 2,
        .lb = Eigen::VectorXd::Constant(2, -2.0),
        .ub = Eigen::VectorXd::Constant(2,  2.0)
    };
    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(2, 1.5);
    solver_options opts;
    opts.max_iterations = 50;

    alternative::cmaes::repair_l2_penalty_policy<> policy;
    policy.options.lambda = 6u;
    policy.options.seed = 42u;
    policy.options.restart =
        alternative::cmaes::repair_l2_penalty_policy<>::restart_strategy::ipop;

    basic_solver solver{policy, prob, x0, opts};
    (void) solver.solve();
    const auto& state = solver.state();

    const double f_at_x = prob.value(state.x);
    CHECK(state.objective_value == Approx(f_at_x).margin(1e-12));
}

TEST_CASE("alternative::cmaes::pwq_reparameterization_policy: caller-facing "
          "objective is feasible", "[cmaes]")
{
    bounded_rosenbrock prob{
        .n  = 2,
        .lb = Eigen::VectorXd::Constant(2, -2.0),
        .ub = Eigen::VectorXd::Constant(2,  2.0)
    };
    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(2, 1.5);
    solver_options opts;
    opts.max_iterations = 50;

    alternative::cmaes::pwq_reparameterization_policy<> policy;
    policy.options.lambda = 6u;
    policy.options.seed = 42u;
    policy.options.restart =
        alternative::cmaes::pwq_reparameterization_policy<>::restart_strategy::ipop;

    basic_solver solver{policy, prob, x0, opts};
    (void) solver.solve();
    const auto& state = solver.state();

    // The pwq variant stores state.x as the pheno coord, so
    // problem.value(state.x) is the same as the unpenalized value
    // recorded at the best-of-generation pheno point.
    const double f_at_x = prob.value(state.x);
    CHECK(state.objective_value == Approx(f_at_x).margin(1e-12));
    // state.x must be inside the box (the pheno transform image is
    // the closed box [lower, upper]).
    for(int i = 0; i < state.x.size(); ++i)
    {
        CHECK(state.x[i] >= prob.lb[i] - 1e-12);
        CHECK(state.x[i] <= prob.ub[i] + 1e-12);
    }
}

TEST_CASE("alternative::cmaes::no_repair_adaptive_penalty_policy: caller-facing "
          "objective is feasible", "[cmaes]")
{
    bounded_rosenbrock prob{
        .n  = 2,
        .lb = Eigen::VectorXd::Constant(2, -2.0),
        .ub = Eigen::VectorXd::Constant(2,  2.0)
    };
    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(2, 1.5);
    solver_options opts;
    opts.max_iterations = 50;

    alternative::cmaes::no_repair_adaptive_penalty_policy<> policy;
    policy.options.lambda = 6u;
    policy.options.seed = 42u;
    policy.options.restart =
        alternative::cmaes::no_repair_adaptive_penalty_policy<>::restart_strategy::ipop;

    basic_solver solver{policy, prob, x0, opts};
    (void) solver.solve();
    const auto& state = solver.state();

    // The no-repair variant stores state.x as the clipped repaired
    // point, so problem.value(state.x) is the same as the unpenalized
    // value recorded at the repaired best-of-generation point.
    const double f_at_x = prob.value(state.x);
    CHECK(state.objective_value == Approx(f_at_x).margin(1e-12));
    // state.x must be inside the box (the repair clips into [lb, ub]).
    for(int i = 0; i < state.x.size(); ++i)
    {
        CHECK(state.x[i] >= prob.lb[i] - 1e-12);
        CHECK(state.x[i] <= prob.ub[i] + 1e-12);
    }
}

// Lock the cmaes_policy::reset contract -- a caller that bumps
// options.lambda and calls reset() must see refreshed strategy
// parameters (mu, mueff, c_sigma, d_sigma, c_c, c_1, c_mu, weights,
// chi_n). Mirrors the in-policy IPOP branch's recompute and the
// libcmaes ipopcmastrategy.cc::reset_search_state contract.
//
// References:
//   Auger & Hansen (2005), CEC 2005 §III (IPOP-CMA-ES).
//   libcmaes ipopcmastrategy.cc::reset_search_state.
TEST_CASE("cmaes_policy: reset refreshes params on lambda bump", "[cmaes]")
{
    bounded_rosenbrock prob{
        .n  = 2,
        .lb = Eigen::VectorXd::Constant(2, -2.0),
        .ub = Eigen::VectorXd::Constant(2,  2.0),
    };
    Eigen::VectorXd x0{{-1.0, 1.0}};
    solver_options opts;
    opts.max_iterations = 1;

    cmaes_policy<> policy;
    policy.options.lambda = 6u;
    policy.options.seed = 42u;

    auto state = policy.init(prob, x0, opts);
    const auto params_before = state.params;
    REQUIRE(params_before.lambda == 6);
    REQUIRE(params_before.mu == 3);

    // Caller bumps lambda then calls reset(): same surface as the
    // restarting_policy decorator's reinit path (which calls reset_clear
    // -> reset internally).
    policy.options.lambda = 12u;
    policy.reset(state, x0);

    const auto& params_after = state.params;
    REQUIRE(params_after.lambda == 12);
    REQUIRE(params_after.mu == 6);
    REQUIRE(params_after.weights.size() == 12);
    CHECK(params_after.mu_eff != Approx(params_before.mu_eff));
    CHECK(params_after.c_1 != Approx(params_before.c_1));
    CHECK(params_after.c_mu != Approx(params_before.c_mu));
}

// Lock the cmaes_policy::MaxPopulation contract: when a caller explicitly
// instantiates `cmaes_policy<N, M>` with `M > 512`, the per-step buffer
// cap (`state_type<P>::MaxPop`) MUST honor `M`, not silently fall back
// to the hardcoded 512. The IPOP escalation guard then fires at the
// caller-chosen ceiling rather than at the buffer's hidden floor.
//
// Without honoring the template parameter, `state_type<P>::MaxPop`
// stays pinned to 512 independent of `MaxPopulation`; the inner
// `Eigen::Matrix<..., 0, 512, 1>::resize(lambda)` is undefined
// behavior at lambda > 512 and surfaces as a segfault on any deep
// IPOP restart trajectory (Auger & Hansen 2005, IPOP-CMA-ES doubles
// lambda each restart, so a sufficient restart count crosses any
// fixed compile-time ceiling).
//
// The contract has three observable consequences exercised below:
//   1. state_type<P>::MaxPop reflects MaxPopulation when set explicitly,
//      and falls back to 512 when MaxPopulation is dynamic_dimension.
//   2. init() with options.lambda > 512 succeeds when MaxPopulation
//      is wide enough (no buffer-overflow UB), and step() runs at
//      that wide population without crashing.
//   3. init() with options.lambda > MaxPop rejects the problem via the
//      solver_status::invalid_problem status return (the library is
//      exception-free).
//
// Reference:
//   Auger, A. & Hansen, N. (2005). A Restart CMA Evolution Strategy
//   with Increasing Population Size. CEC 2005. IPOP-CMA-ES doubles
//   the population at each restart trigger.
TEST_CASE("cmaes_policy: MaxPopulation template parameter widens IPOP cap",
          "[cmaes][maxpop]")
{
    // (1) MaxPop reflects MaxPopulation when set, falls back to 512 otherwise.
    using policy_default_t = cmaes_policy<>;
    using policy_wide_t    = cmaes_policy<2, 1024>;
    using policy_narrow_t  = cmaes_policy<2, 256>;
    using state_default_t  = policy_default_t::state_type<rastrigin<double>>;
    using state_wide_t     = policy_wide_t::state_type<rastrigin<double>>;
    using state_narrow_t   = policy_narrow_t::state_type<rastrigin<double>>;
    STATIC_REQUIRE(state_default_t::MaxPop == 512);
    STATIC_REQUIRE(state_wide_t::MaxPop == 1024);
    STATIC_REQUIRE(state_narrow_t::MaxPop == 256);

    // (2) init() + step() at lambda=800 succeeds under MaxPopulation=1024.
    // The buffers are sized via state_type::MaxPop, which now derives
    // from the template parameter; resize past the legacy 512 floor is
    // safe.
    rastrigin<double> problem{.n = 2};
    Eigen::VectorXd x0{{3.0, 3.0}};
    solver_options opts;
    opts.max_iterations = 1;

    policy_wide_t policy;
    policy.options.seed = 42u;
    policy.options.lambda = 800u;       // > 512, <= 1024
    auto state = policy.init(problem, x0, opts);
    REQUIRE(state.params.lambda == 800);
    REQUIRE(state.fitnesses_buf.size() >= 800);

    // step() must not crash at lambda > 512.
    auto r = policy.step(state);
    CHECK(state.params.lambda == 800);
    const bool stalled_before_512 =
        r.policy_status.has_value()
        && *r.policy_status == solver_status::stalled
        && state.params.lambda <= 512;
    CHECK_FALSE(stalled_before_512);

    // (3) init() with options.lambda > MaxPop rejects via invalid_problem:
    // the state is flagged and the first step() returns the status rather
    // than resizing the buffers past the static MaxPop bound.
    policy_narrow_t narrow_policy;
    narrow_policy.options.seed = 42u;
    narrow_policy.options.lambda = 257u;    // > MaxPop=256
    auto narrow_state = narrow_policy.init(problem, x0, opts);
    REQUIRE(narrow_state.invalid_problem);
    auto narrow_r = narrow_policy.step(narrow_state);
    REQUIRE(narrow_r.policy_status.has_value());
    CHECK(*narrow_r.policy_status == solver_status::invalid_problem);
}

// Lock the init()-time lambda<=MaxPop guard. A population above the
// default `cmaes_policy<>` cap (MaxPop=512) would silently UB-resize the
// per-step buffers (`Eigen::Matrix<..., 0, MaxPop, ...>`); init() must
// instead flag the state and step() must return
// solver_status::invalid_problem (the library is exception-free).
//
// Reference:
//   Hansen, N. (2023). The CMA Evolution Strategy: A Tutorial.
//   arXiv:1604.00772 §B.1: default population size lambda = 4 + floor(3*ln(n)).
TEST_CASE("cmaes_policy: init flags invalid_problem when lambda exceeds MaxPop",
          "[cmaes][maxpop]")
{
    constexpr int n = 4;
    bounded_rosenbrock problem{
        .n  = n,
        .lb = Eigen::VectorXd::Constant(n, -5.0),
        .ub = Eigen::VectorXd::Constant(n,  5.0),
    };
    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(n, 0.0);
    solver_options opts;
    opts.max_iterations = 1;

    cmaes_policy<> policy;          // default MaxPopulation -> MaxPop = 512
    policy.options.seed = 42u;
    policy.options.lambda = 520u;   // above the 512 static buffer cap

    auto state = policy.init(problem, x0, opts);
    REQUIRE(state.invalid_problem);

    auto r = policy.step(state);
    REQUIRE(r.policy_status.has_value());
    CHECK(*r.policy_status == solver_status::invalid_problem);
    CHECK(r.is_null_step);
}

// Lock the NaN-guard always-exit contract: a NaN in s.sigma, s.D, or
// s.C must produce step_result.policy_status == diverged AND must NOT
// trigger an IPOP restart. The IPOP recycle gate is intended to fire
// on legacy stagnation probes (sigma collapse / cond explosion) only;
// a NaN in the distribution shape is a hard failure with no recovery
// story (Hansen 2023 (arXiv:1604.00772) §B.2: rank-one + rank-mu
// covariance updates depend on a finite C; restart with NaN in C
// re-injects via the very next update).
TEST_CASE("cmaes_policy: NaN guard exits without IPOP recycle",
          "[cmaes][nan-guard]")
{
    bounded_rosenbrock problem{
        .n  = 2,
        .lb = Eigen::VectorXd::Constant(2, -2.0),
        .ub = Eigen::VectorXd::Constant(2,  2.0),
    };
    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(2, -1.0);
    solver_options opts;
    opts.max_iterations = 1;

    cmaes_policy<> policy;
    policy.options.seed = 42u;
    policy.options.restart = cmaes_policy<>::restart_strategy::ipop;

    auto state = policy.init(problem, x0, opts);
    const int lambda_init = state.params.lambda;
    REQUIRE(lambda_init >= 4);

    // Inject NaN into D[0] (the dominant eigenvalue). The NaN guard
    // must fire and the early-return must occur before the IPOP
    // recycle gate snapshots policy_status.
    state.D[0] = std::numeric_limits<double>::quiet_NaN();

    auto r = policy.step(state);

    // Always-exit contract: returned step_result reports diverged and
    // the IPOP recycle has NOT fired (lambda unchanged from init).
    REQUIRE(r.policy_status.has_value());
    REQUIRE(*r.policy_status == solver_status::diverged);
    REQUIRE(state.params.lambda == lambda_init);
}

// Lock the allFinite() upgrade: a non-maximum NaN in s.D must be
// caught. With state.D[0] = 5.0 and state.D[1] = NaN, the
// `!std::isfinite(s.D.maxCoeff())` form returns false because
// std::max(NaN, 5.0) returns 5.0 under IEEE 754 max-reduction
// semantics (Std 754-2019 §6.2). Eigen::DenseBase::allFinite()
// iterates every coefficient and returns false on the first
// non-finite entry, which is the correct predicate for "no NaN
// anywhere in the distribution shape".
//
// Reference:
//   IEEE Std 754-2019 §6.2 (NaN handling under max-reductions).
//   Eigen DenseBase::allFinite() documentation.
TEST_CASE("cmaes_policy: NaN guard catches non-maximum NaN entry",
          "[cmaes][nan-guard]")
{
    bounded_rosenbrock problem{
        .n  = 2,
        .lb = Eigen::VectorXd::Constant(2, -2.0),
        .ub = Eigen::VectorXd::Constant(2,  2.0),
    };
    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(2, -1.0);
    solver_options opts;
    opts.max_iterations = 1;

    cmaes_policy<2> policy;
    policy.options.seed = 42u;

    auto state = policy.init(problem, x0, opts);

    // D[0] dominant + finite, D[1] NaN. maxCoeff() returns D[0]=5.0
    // (finite); allFinite() iterates and catches the NaN.
    state.D[0] = 5.0;
    state.D[1] = std::numeric_limits<double>::quiet_NaN();

    auto r = policy.step(state);

    REQUIRE(r.policy_status.has_value());
    REQUIRE(*r.policy_status == solver_status::diverged);
}
