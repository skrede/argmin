// restarting_policy<cmaes_policy<>> decorator unit tests.
//
// Locks the within-process bit-identity contract that closes the
// rebuild-rebuild nondeterminism on the `restarting_cmaes` summary
// row in publish_bench. Two back-to-back invocations of a fresh
// `basic_solver` from default-constructed decorator options at the
// SAME inner seed must produce bit-identical `final_objective` and
// identical `iterations`. A second TEST_CASE pins the seed-plumbing
// path so a future regression that silently bypasses the inner seed
// (e.g. by re-routing through std::random_device) flips a deterministic
// inequality.
//
// Reference: Auger & Hansen (2005), "A Restart CMA Evolution Strategy
//            with Increasing Population Size", CEC 2005 (decorator
//            algorithm); Hansen (2023), "The CMA Evolution Strategy:
//            A Tutorial", arXiv:1604.00772 §B.5 (reproducibility).

#include "nablapp/solver/restarting_policy.h"
#include "nablapp/solver/cmaes_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/detail/cmaes_constants.h"
#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <cmath>
#include <cstdint>

using namespace nablapp;

namespace
{

// Bounded Ackley n=2 -- satisfies objective && bound_constrained,
// matches the rebuild-nondeterminism closure-criterion problem
// (Ackley n=2 from seed=42 is the canonical reproducer for the
// publish_bench `restarting_cmaes` row regression captured in the
// project's seed/backlog tracker).
struct bounded_ackley
{
    static constexpr int problem_dimension = nablapp::dynamic_dimension;

    int n{2};
    Eigen::VectorXd lb;
    Eigen::VectorXd ub;

    int dimension() const { return n; }

    double value(const Eigen::VectorXd& x) const
    {
        const double a = 20.0;
        const double b = 0.2;
        const double c = 2.0 * M_PI;
        double sum_sq = 0.0;
        double sum_cos = 0.0;
        for(int i = 0; i < n; ++i)
        {
            sum_sq += x[i] * x[i];
            sum_cos += std::cos(c * x[i]);
        }
        const double dn = static_cast<double>(n);
        return -a * std::exp(-b * std::sqrt(sum_sq / dn))
             - std::exp(sum_cos / dn) + a + std::exp(1.0);
    }

    Eigen::VectorXd lower_bounds() const { return lb; }
    Eigen::VectorXd upper_bounds() const { return ub; }
};

}

static_assert(objective<bounded_ackley>);
static_assert(bound_constrained<bounded_ackley>);

TEST_CASE("restarting_policy bit-identity within process", "[restarting_policy][cmaes]")
{
    // Decorator-surface within-process determinism preflight. Two fresh
    // `basic_solver` instances constructed back-to-back inside the same
    // process from the SAME seed on the SAME problem must produce
    // bit-identical `final_objective` and identical iteration counts.
    //
    // The decorator forwards seed into the inner cmaes policy via
    // `inner_policy_.options = options.inner` at init() time. When
    // `options.inner.seed` is unset, the decorator must NOT permit the
    // inner policy to fall through to `std::random_device` (such a
    // fallback caused publish_bench rebuild-rebuild nondeterminism on
    // the `restarting_cmaes` summary row at fixed seed: 25 cells
    // shifted final_objective between two consecutive rebuilds at the
    // same SHA, while the directly-driven `cmaes` row was rebuild-stable
    // because its bench wiring sets cmaes_options.seed explicitly).
    //
    // The fresh-solver-per-run pattern defeats first-run-cache effects
    // (state-type buffers carrying over within a single solver instance);
    // the bit_cast equality branch defeats NaN-trap silent passes (NaN
    // != NaN under operator==).
    bounded_ackley problem{
        .n = 2,
        .lb = Eigen::VectorXd::Constant(2, -5.0),
        .ub = Eigen::VectorXd::Constant(2, 5.0),
    };

    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(2, 3.0);
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-30);
    opts.set_objective_threshold(1e-30);
    opts.set_step_threshold(1e-30);

    auto run_once = [&]() {
        // Default-constructed decorator -- mirrors bench_nablapp.h
        // wiring of the `restarting_cmaes` row. The fix at
        // restarting_policy::init() must guarantee a deterministic
        // inner seed in this exact configuration; without it, the
        // inner cmaes_policy falls through to std::random_device and
        // this assertion fails.
        restarting_policy<cmaes_policy<>> policy;

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

TEST_CASE("restarting_policy seed plumbing through restarts", "[restarting_policy][cmaes]")
{
    // Sanity check on seed discrimination. Two decorators with
    // DIFFERENT explicit inner seeds run from the same x0 must
    // produce different sample paths, observable here as different
    // final objective values within the iteration budget. If a
    // future regression makes the decorator silently ignore the
    // inner seed (e.g. by clobbering it with a hard-coded constant
    // or routing through std::random_device), this assertion would
    // either pass by random chance OR fail deterministically -- in
    // either case the regression is detectable.
    //
    // On the chosen problem / x0 / iteration budget, the two seeds
    // exercise different IPOP-restart-decorator branches and converge
    // to numerically distinct final_objective values; if a future
    // change makes them coincide, that is itself signal worth
    // investigating.
    bounded_ackley problem{
        .n = 2,
        .lb = Eigen::VectorXd::Constant(2, -5.0),
        .ub = Eigen::VectorXd::Constant(2, 5.0),
    };

    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(2, 3.0);
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-30);
    opts.set_objective_threshold(1e-30);
    opts.set_step_threshold(1e-30);

    auto run_with_seed = [&](std::uint64_t inner_seed) {
        restarting_policy<cmaes_policy<>> policy;
        policy.options.inner.seed = inner_seed;

        basic_solver solver{policy, problem, x0, opts};
        return solver.solve(opts);
    };

    auto result_42 = run_with_seed(42u);
    auto result_43 = run_with_seed(43u);

    // The two seeds must drive distinguishable trajectories. We do
    // NOT require any particular ordering of the two objective
    // values -- just that they are not bit-identical. A deterministic
    // tie here would indicate the seed is being discarded.
    const bool different =
        std::bit_cast<std::uint64_t>(result_42.objective_value)
        != std::bit_cast<std::uint64_t>(result_43.objective_value);
    REQUIRE(different);
}

// When restarting_policy::reinit bumps the inner's options.lambda and
// calls reset_clear, cmaes_policy::reset MUST recompute s.inner.params
// via detail::compute_constants(n, new_lambda). Without the recompute,
// s.inner.params holds init-time values (mu, mueff, c_sigma, d_sigma,
// c_c, c_1, c_mu, weights, chi_n) that no longer match the bumped
// lambda; the inner policy then adapts incorrectly on the new
// generation.
//
// inner_policy_ is private (no public accessor), so we drive the
// lambda bump through the decorator's natural restart trigger: a tiny
// stagnation_limit forces reinit() to fire after the budget runs out;
// reinit() bumps inner.options.lambda to initial_lambda * 2 and calls
// reset_clear, which (post-fix) recomputes s.inner.params via
// detail::compute_constants. We then compare s.inner.params against a
// freshly-computed reference cmaes_params at the post-restart lambda.
//
// References:
//   Auger & Hansen (2005), CEC 2005 §III (IPOP-CMA-ES).
//   libcmaes ipopcmastrategy.cc::reset_search_state.
TEST_CASE("restarting_policy: lambda bump propagates to inner params after reset",
          "[restarting_policy][cmaes]")
{
    bounded_ackley problem{
        .n  = 2,
        .lb = Eigen::VectorXd::Constant(2, -5.0),
        .ub = Eigen::VectorXd::Constant(2,  5.0),
    };
    Eigen::VectorXd x0 = Eigen::VectorXd::Constant(2, 0.0);
    solver_options opts;
    opts.max_iterations = 1;

    // Drive the lambda bump through the decorator's inner_policy()
    // accessor, mirroring the surface that reinit() uses internally
    // (inner_policy_.options.lambda = new_pop; inner_policy_.reset_clear(...)).
    // The accessor is the test-side equivalent of the private
    // mutation reinit() performs -- both paths converge on
    // cmaes_policy::reset, where the F-R-01 fix lives.
    restarting_policy<cmaes_policy<>> rp;
    rp.options.inner.seed = 42u;
    rp.options.inner.lambda = 6u;
    auto state = rp.init(problem, x0, opts);
    const auto params_before = state.inner.params;
    REQUIRE(params_before.lambda == 6);
    REQUIRE(static_cast<std::uint32_t>(params_before.lambda)
            == state.initial_lambda);

    // Simulate one IPOP-decorator restart: bump inner.options.lambda
    // and call reset_clear. The F-R-01 fix in cmaes_policy::reset
    // recomputes s.inner.params via detail::compute_constants on the
    // bumped lambda.
    const std::uint32_t new_lambda = state.initial_lambda * 2u;
    rp.inner_policy().options.lambda = new_lambda;
    rp.reset_clear(state, x0);

    const auto& params_after = state.inner.params;
    REQUIRE(params_after.lambda == static_cast<int>(new_lambda));
    REQUIRE(params_after.mu == params_after.lambda / 2);
    REQUIRE(params_after.weights.size() == params_after.lambda);

    // Cross-check against a freshly-computed reference: the
    // post-restart inner.params must match what compute_constants
    // would produce at the post-restart lambda. This is the strongest
    // form of the F-R-01 contract -- not just "lambda updated" but
    // "every dependent strategy parameter (mu, mueff, c_sigma,
    // d_sigma, c_c, c_1, c_mu, weights, chi_n) refreshed".
    const auto reference = nablapp::detail::compute_constants<double>(
        static_cast<int>(state.dimension),
        static_cast<int>(new_lambda));
    REQUIRE(params_after.lambda == reference.lambda);
    REQUIRE(params_after.mu == reference.mu);
    REQUIRE(params_after.mu_eff == reference.mu_eff);
    REQUIRE(params_after.c_sigma == reference.c_sigma);
    REQUIRE(params_after.d_sigma == reference.d_sigma);
    REQUIRE(params_after.c_c == reference.c_c);
    REQUIRE(params_after.c_1 == reference.c_1);
    REQUIRE(params_after.c_mu == reference.c_mu);
    REQUIRE(params_after.chi_n == reference.chi_n);
    REQUIRE(params_after.weights.size() == reference.weights.size());
    for(int i = 0; i < params_after.weights.size(); ++i)
        REQUIRE(params_after.weights[i] == reference.weights[i]);

    // Pre/post sanity: at least one strategy parameter must differ
    // between the init-time and post-bump snapshots; otherwise we
    // have not exercised the contract.
    REQUIRE(params_before.mu_eff != params_after.mu_eff);
    REQUIRE(params_before.c_1 != params_after.c_1);
}
