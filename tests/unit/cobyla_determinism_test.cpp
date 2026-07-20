// Reset-reproducibility pin for the COBYLA policy.
//
// Scope, deliberately narrow: these pins assert that one binary reproduces its
// own trajectory exactly within a single process. COBYLA has no injectable
// seed -- the degenerate-geometry jitter LCG is seeded from the problem
// dimensions (seed = n + m) and reset() re-runs that initialization -- so
// no cross-platform, cross-toolchain, or cross-instantiation bit-identity is
// claimed here, and these pins must not be extended into such a claim.
//
// Because the seed is derived rather than injected, the discrimination control
// cannot vary a seed; it perturbs the start point instead, proving the pin
// catches a genuine divergence rather than passing on a trajectory that never
// moves.

#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <vector>
#include <cstdint>

using namespace argmin;

namespace
{

// Minimise x0^2 + x1^2 s.t. x0 + x1 >= 1 (argmin: c >= 0) over [-10, 10]^2;
// optimum (0.5, 0.5). Restated here so this translation unit stays independent.
struct simple_constrained
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        return x[0] * x[0] + x[1] * x[1];
    }

    void constraints(const Eigen::Vector<double, 2>& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = x[0] + x[1] - 1.0;
    }

    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    Eigen::Vector<double, 2> lower_bounds() const
    {
        Eigen::Vector<double, 2> lb;
        lb << -10.0, -10.0;
        return lb;
    }

    Eigen::Vector<double, 2> upper_bounds() const
    {
        Eigen::Vector<double, 2> ub;
        ub << 10.0, 10.0;
        return ub;
    }
};

// Long enough that a divergence has room to surface, short enough to stay
// clear of trust-radius termination.
constexpr int trajectory_steps = 40;

solver_options<> trajectory_options()
{
    solver_options<> opts;
    opts.max_iterations = trajectory_steps;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);
    return opts;
}

template <typename Solver>
std::vector<std::array<double, 5>> drain_trajectory(Solver& solver)
{
    std::vector<std::array<double, 5>> trace;
    for(int i = 0; i < trajectory_steps; ++i)
    {
        auto r = solver.step();
        const auto& s = solver.state();
        trace.push_back({r.objective_value, r.step_size, r.gradient_norm,
                         s.x(0), s.x(1)});
        if(r.policy_status)
            break;
    }
    return trace;
}

std::vector<std::array<double, 5>> record_trajectory(const Eigen::VectorXd& start)
{
    simple_constrained problem;
    step_budget_solver solver{cobyla_policy{}, problem, start, trajectory_options()};
    return drain_trajectory(solver);
}

std::vector<std::array<double, 5>> record_trajectory_after_reset(
    const Eigen::VectorXd& start)
{
    simple_constrained problem;
    step_budget_solver solver{cobyla_policy{}, problem, start, trajectory_options()};

    (void)drain_trajectory(solver);
    solver.reset(start);
    return drain_trajectory(solver);
}

void require_bit_identical(const std::vector<std::array<double, 5>>& lhs,
                           const std::vector<std::array<double, 5>>& rhs)
{
    REQUIRE(lhs.size() == rhs.size());
    REQUIRE(lhs.size() > 5);
    for(std::size_t i = 0; i < lhs.size(); ++i)
        for(std::size_t k = 0; k < 5; ++k)
            CHECK(lhs[i][k] == rhs[i][k]); // exact, bit-for-bit
}

bool differs_somewhere(const std::vector<std::array<double, 5>>& lhs,
                       const std::vector<std::array<double, 5>>& rhs)
{
    if(lhs.size() != rhs.size())
        return true;
    for(std::size_t i = 0; i < lhs.size(); ++i)
        for(std::size_t k = 0; k < 5; ++k)
            if(lhs[i][k] != rhs[i][k])
                return true;
    return false;
}

Eigen::VectorXd start_point()
{
    return Eigen::VectorXd{{2.0, 2.0}};
}

}

TEST_CASE("cobyla is bit-for-bit deterministic within one process",
          "[cobyla][determinism]")
{
    const auto run1 = record_trajectory(start_point());
    const auto run2 = record_trajectory(start_point());

    require_bit_identical(run1, run2);
}

TEST_CASE("cobyla reset re-seeds the generator so the post-reset run is identical",
          "[cobyla][determinism]")
{
    // reset() re-runs the evaluator initialization, which re-derives the LCG
    // seed from the problem dimensions and rebuilds the simplex from the start
    // point, so a post-reset run retraces a fresh run exactly.
    const auto reset_run = record_trajectory_after_reset(start_point());
    const auto fresh_run = record_trajectory(start_point());

    require_bit_identical(reset_run, fresh_run);
}

TEST_CASE("cobyla trajectory discrimination control",
          "[cobyla][determinism]")
{
    // The negative control: without an injectable seed, a start-point
    // perturbation is the divergence source. If this did not fire, every pin
    // above could pass on a trajectory that never moves.
    const auto base = record_trajectory(start_point());
    const auto perturbed = record_trajectory(Eigen::VectorXd{{5.0, 5.0}});

    CHECK(differs_somewhere(base, perturbed));
}
