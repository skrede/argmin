#include "nablapp/solver/cmaes_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/test_functions/rosenbrock.h"
#include "nablapp/test_functions/rastrigin.h"
#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

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
    flat_bounded problem{.n = 2};

    Eigen::VectorXd x0{{0.0, 0.0}};
    solver_options opts;
    // Enough budget to accumulate the EqualFunValues window
    // (10 + ceil(30*n/lambda)) generations on a flat objective and trigger
    // one IPOP doubling. The flat fitness guarantees stagnation fires.
    opts.max_iterations = 300;
    opts.set_gradient_threshold(1e-30);
    opts.set_objective_threshold(1e-30);
    opts.set_step_threshold(1e-30);

    cmaes_policy<> policy;
    policy.options.initial_sigma = 1.0;
    policy.options.restart = cmaes_policy<>::restart_strategy::ipop;
    policy.options.seed = 42u;

    basic_solver solver{policy, problem, x0, opts};

    const int n = 2;
    const int lambda_initial = solver.state().params.lambda;
    REQUIRE(lambda_initial == 8);  // bounded n=2: max(4*2, 4 + floor(3*ln(2))) = 8

    const auto expected_initial = static_cast<std::uint32_t>(120)
        + static_cast<std::uint32_t>(
            std::ceil(30.0 * static_cast<double>(n)
                      / static_cast<double>(lambda_initial)));
    REQUIRE(solver.state().stagnation_window_min == expected_initial);  // 128

    // Drive the solver until at least one IPOP restart fires. With a flat
    // objective and lambda=8 the EqualFunValues window is 10 + ceil(60/8) = 18
    // generations, so a doubling lands well before the 300-iteration budget.
    solver.solve(opts);

    const int lambda_after = solver.state().params.lambda;
    REQUIRE(lambda_after >= lambda_initial * 2);  // at least one doubling

    const auto expected_after = static_cast<std::uint32_t>(120)
        + static_cast<std::uint32_t>(
            std::ceil(30.0 * static_cast<double>(n)
                      / static_cast<double>(lambda_after)));
    CHECK(solver.state().stagnation_window_min == expected_after);
}
