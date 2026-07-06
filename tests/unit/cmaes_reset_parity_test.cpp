// CMA-ES restart / reset parity instruments.
//
// Three behaviors of the production CMA-ES policy are pinned here:
//
//   * Restart-mean randomization (Auger & Hansen 2005 §III): a restart on
//     a bounded problem draws the distribution mean uniformly within the
//     search box so successive restarts probe distinct basins instead of
//     re-sampling the previous one around a fixed initial point.
//
//   * reset()/init() derived-quantity parity: reset() must re-derive the
//     same configuration-dependent quantities init() does (the bounded
//     population floor, the stagnation history window, the
//     eigendecomposition skip period, and the boundary-axis cycle index),
//     so a reset run behaves identically to a fresh run.
//
//   * Eigendecomposition cadence: the covariance dirty flag means "C has
//     changed since the last decomposition"; when a generation neither
//     dirties C nor lands on the skip period, the cached B/D are reused
//     unchanged.

#include "argmin/solver/alternative/cmaes/pwq_reparameterization_policy.h"
#include "argmin/solver/cmaes_policy.h"
#include "argmin/solver/options.h"
#include "argmin/types.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace argmin;

namespace
{

// Bounded convex quadratic: objective && bound_constrained, no gradient.
struct bounded_quadratic
{
    Eigen::VectorXd lb;
    Eigen::VectorXd ub;

    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return static_cast<int>(lb.size()); }

    double value(const Eigen::VectorXd& x) const { return x.squaredNorm(); }

    Eigen::VectorXd lower_bounds() const { return lb; }
    Eigen::VectorXd upper_bounds() const { return ub; }
};

// Unbounded convex quadratic: objective only, no search box.
struct unbounded_quadratic
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;
    int dimension() const { return 2; }
    double value(const Eigen::VectorXd& x) const { return x.squaredNorm(); }
};

}

TEST_CASE("cmaes_policy: bounded restart draws the mean uniformly in the box",
          "[cmaes][restart]")
{
    // A restart must NOT pin the mean to the fixed initial point on a
    // bounded problem: it draws each coordinate uniformly in [lower, upper]
    // so the widened population explores a new basin. Two consecutive
    // restarts therefore land on distinct means, both strictly inside the
    // box, and neither equal to x0. reset_clear() is the restart entry the
    // external restart decorator calls; it exercises the same randomization
    // as the in-policy IPOP branch.
    bounded_quadratic problem{
        .lb = Eigen::VectorXd::Constant(2, -5.0),
        .ub = Eigen::VectorXd::Constant(2, 5.0)};

    Eigen::VectorXd x0{{0.0, 0.0}};
    solver_options opts;
    cmaes_policy<> policy;
    policy.options.seed = 42u;
    policy.options.initial_sigma = 1.0;

    auto s = policy.init(problem, x0, opts);

    policy.reset_clear(s, x0);
    const Eigen::VectorXd mean1 = s.mean;

    policy.reset_clear(s, x0);
    const Eigen::VectorXd mean2 = s.mean;

    // Both draws are strictly inside the box.
    for(int i = 0; i < 2; ++i)
    {
        CHECK(mean1[i] > problem.lb[i]);
        CHECK(mean1[i] < problem.ub[i]);
        CHECK(mean2[i] > problem.lb[i]);
        CHECK(mean2[i] < problem.ub[i]);
    }

    // Neither restart is anchored at x0, and consecutive restarts differ.
    CHECK(mean1 != x0);
    CHECK(mean2 != x0);
    CHECK(mean1 != mean2);
}

TEST_CASE("cmaes_policy: unbounded restart keeps the initial-point anchor",
          "[cmaes][restart]")
{
    // With no finite search domain there is nothing to sample: the restart
    // mean stays at the user's initial point, so the randomization is a
    // pure no-op on unbounded problems (and reset_clear stays deterministic
    // there).
    unbounded_quadratic problem;

    Eigen::VectorXd x0{{0.3, -0.7}};
    solver_options opts;
    cmaes_policy<> policy;
    policy.options.seed = 42u;
    policy.options.initial_sigma = 1.0;

    auto s = policy.init(problem, x0, opts);
    policy.reset_clear(s, x0);

    CHECK(s.mean == x0);
}

TEST_CASE("cmaes_policy: reset() re-derives the same quantities as init()",
          "[cmaes][reset]")
{
    // reset() must reproduce every configuration-derived quantity init()
    // computes, so a reset run behaves identically to a fresh run. On a
    // bounded problem with no explicit population this exercises the
    // bounded population floor (which reset() previously skipped), plus the
    // stagnation window, the eigendecomposition skip period, and the
    // boundary-axis cycle index (which reset() previously left stale).
    bounded_quadratic problem{
        .lb = Eigen::VectorXd::Constant(2, -5.0),
        .ub = Eigen::VectorXd::Constant(2, 5.0)};

    Eigen::VectorXd x0{{1.0, -2.0}};
    solver_options opts;
    opts.max_iterations = 50;
    cmaes_policy<> policy;
    policy.options.seed = 7u;
    policy.options.initial_sigma = 1.0;

    // Fresh init() derivations.
    auto s = policy.init(problem, x0, opts);
    const auto init_lambda = s.params.lambda;
    const auto init_stagnation = s.stagnation_window_min;
    const auto init_skip_k = s.decomposition_skip_k;
    const auto init_axis = s.axis_cycle_index;

    // The bounded floor must have engaged (max(4n, 4 + floor(3 ln n)) = 8
    // at n = 2, above the unimodal default of 6).
    REQUIRE(init_lambda == 8);

    // Advance several generations so the covariance state, generation
    // counter, and step-size move away from their init values, then force
    // the boundary-axis cycle index to a non-init position so reset()'s
    // restoration of it is observable independent of the mod-n wrap.
    for(int i = 0; i < 12; ++i)
        (void)policy.step(s);
    s.axis_cycle_index = init_axis + 1;
    REQUIRE(s.axis_cycle_index != init_axis);
    REQUIRE(s.generation != 0);

    // reset() must restore every derived quantity to its init value.
    policy.reset(s, x0);
    CHECK(s.params.lambda == init_lambda);
    CHECK(s.stagnation_window_min == init_stagnation);
    CHECK(s.decomposition_skip_k == init_skip_k);
    CHECK(s.axis_cycle_index == init_axis);

    // And the covariance state is a clean identity basis, matching init.
    CHECK(s.covariance_dirty == false);
    CHECK(s.C == Eigen::MatrixXd::Identity(2, 2));
    CHECK(s.D == Eigen::VectorXd::Ones(2));
    CHECK(s.initial_d_max == 1.0);
}
