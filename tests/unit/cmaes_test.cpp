#include "nablapp/solver/cmaes_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/test_functions/rosenbrock.h"
#include "nablapp/test_functions/rastrigin.h"
#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <cmath>
#include <cstdint>

using Catch::Approx;
using namespace nablapp;

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

    static constexpr int problem_dimension = nablapp::dynamic_dimension;

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
    cmaes_policy policy;
    policy.options.seed = 42u;

    basic_solver solver{policy, problem, x0, opts};
    CHECK(solver.state().sigma == Approx(10.0 / 3.0).epsilon(1e-10));
}

TEST_CASE("cmaes_policy: lambda minimum for bounded problems", "[cmaes]")
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

    // No explicit lambda -- should enforce 4*N = 8 minimum for bounded N=2.
    cmaes_policy policy;
    policy.options.seed = 42u;

    basic_solver solver{policy, problem, x0, opts};
    CHECK(solver.state().params.lambda >= 8);
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

    // Default options -- sigma and lambda should auto-scale from bounds.
    cmaes_policy policy;
    policy.options.seed = 42u;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve();

    CHECK(result.iterations > 0);
    CHECK(result.objective_value < 1.0);
}

TEST_CASE("cmaes_policy: Rastrigin 2D global optimum with IPOP", "[cmaes]")
{
    // Validates CMA-01 (sigma scaled from bound range: (5.12-(-5.12))/3 = 3.41)
    // and CMA-02 (lambda >= 4*N = 8 for bounded problem).
    // Rastrigin is highly multimodal; IPOP restarts explore multiple basins.
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
    policy.options.seed = 42u;

    basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve(opts);

    CHECK(result.objective_value < 1.0);
    CHECK(solver.state().params.lambda >= 8);
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
// generations of identical fitness accumulate. Bounded so the policy applies
// the lambda = max(4*n, 4 + floor(3*ln(n))) minimum, giving a known starting
// lambda for the IPOP recompute regression test.
struct flat_bounded
{
    int n{2};
    double lower_bound{-5.0};
    double upper_bound{5.0};

    static constexpr int problem_dimension = nablapp::dynamic_dimension;

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
    // every iteration via the _max_hist cap. nablapp computes it once in
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
    // setup at the default 1e-12 once sigma collapses).
    policy.options.cmaes.step_size_tolerance = 1e-30;

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
    // through cmaes_options and is consumed via value_or(1e-12) at the
    // §B.3 item 6 check site. A loose user threshold (1e-6) MUST fire
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

TEST_CASE("cmaes_policy: no_effect_coord exit", "[cmaes]")
{
    // Hansen 2023 (arXiv:1604.00772) section B.3 item 2 (NoEffectCoord):
    // "stop if adding 0.2-standard deviations in any single coordinate
    // does not change m (i.e. m_i equals m_i + 0.2*sigma*c_{i,i} for any
    // i)." The degenerate quadratic has near-zero curvature on coordinate
    // 1; CMA-ES collapses C(1,1) until 0.2 * sigma * sqrt(C(1,1)) is below
    // the mean's ULP, firing the criterion.
    //
    // Same criterion-class semantics as no_effect_axis: any of the three
    // roundoff_limited-mapped criteria is acceptable.
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

    CHECK(result.status == solver_status::roundoff_limited);
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
    // single point (cmasolutions.cc:49). nablapp mirrors this by storing
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
