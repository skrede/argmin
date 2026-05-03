#include "argmin/detail/halton.h"
#include "argmin/solver/multistart_policy.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace argmin;

namespace
{

// Bound-constrained Booth function for multi-start testing.
struct multistart_booth
{
    static constexpr int problem_dimension = dynamic_dimension;
    Eigen::VectorXd lb;
    Eigen::VectorXd ub;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        double t1 = x[0] + 2.0 * x[1] - 7.0;
        double t2 = 2.0 * x[0] + x[1] - 5.0;
        return t1 * t1 + t2 * t2;
    }

    Eigen::VectorXd lower_bounds() const { return lb; }
    Eigen::VectorXd upper_bounds() const { return ub; }
};

// Multimodal test function: shifted Rosenbrock with an exponential trap.
// The exponential term at the origin creates a local minimum that can
// trap single-start solvers starting from negative coordinates.
struct multimodal_2d
{
    static constexpr int problem_dimension = dynamic_dimension;
    Eigen::VectorXd lb;
    Eigen::VectorXd ub;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        double t1 = 1.0 - x[0];
        double t2 = x[1] - x[0] * x[0];
        return t1 * t1 + 100.0 * t2 * t2 + 5.0 * std::exp(-x.squaredNorm());
    }

    Eigen::VectorXd lower_bounds() const { return lb; }
    Eigen::VectorXd upper_bounds() const { return ub; }
};

}

TEST_CASE("halton van der Corput sequence", "[multistart]")
{
    using detail::van_der_corput;

    CHECK(van_der_corput(1, 2) == Approx(0.5).margin(1e-15));
    CHECK(van_der_corput(2, 2) == Approx(0.25).margin(1e-15));
    CHECK(van_der_corput(3, 2) == Approx(0.75).margin(1e-15));
    CHECK(van_der_corput(1, 3) == Approx(1.0 / 3.0).margin(1e-15));
    CHECK(van_der_corput(2, 3) == Approx(2.0 / 3.0).margin(1e-15));
}

TEST_CASE("halton point in unit cube", "[multistart]")
{
    auto p0 = detail::halton_point(0, 2);
    CHECK(p0[0] == Approx(0.5).margin(1e-15));
    CHECK(p0[1] == Approx(1.0 / 3.0).margin(1e-15));

    auto p1 = detail::halton_point(1, 2);
    CHECK(p1[0] == Approx(0.25).margin(1e-15));
    CHECK(p1[1] == Approx(2.0 / 3.0).margin(1e-15));

    // All values must be in (0, 1).
    for(int i = 0; i < 10; ++i)
    {
        auto p = detail::halton_point(static_cast<std::uint32_t>(i), 3);
        for(int d = 0; d < 3; ++d)
        {
            CHECK(p[d] > 0.0);
            CHECK(p[d] < 1.0);
        }
    }
}

TEST_CASE("halton to bounds mapping", "[multistart]")
{
    Eigen::VectorXd lower{{-5.0, -5.0}};
    Eigen::VectorXd upper{{5.0, 5.0}};

    auto p = detail::halton_to_bounds(0, 2, lower, upper);

    // Should map [0,1]^2 into [-5, 5]^2.
    CHECK(p[0] >= -5.0);
    CHECK(p[0] <= 5.0);
    CHECK(p[1] >= -5.0);
    CHECK(p[1] <= 5.0);

    // First Halton point in base 2 is 0.5 -> maps to 0.0 in [-5, 5].
    CHECK(p[0] == Approx(0.0).margin(1e-12));
}

TEST_CASE("multistart_policy concept satisfaction", "[multistart]")
{
    multistart_booth problem{
        .lb = Eigen::VectorXd{{-10.0, -10.0}},
        .ub = Eigen::VectorXd{{10.0, 10.0}},
    };

    Eigen::VectorXd x0{{0.0, 0.0}};
    solver_options opts;
    opts.max_iterations = 300;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-12);

    using ms_policy = multistart_policy<bobyqa_policy<>>;
    basic_solver solver{ms_policy{}, problem, x0, opts};
    auto result = solver.solve();

    // Booth minimum at (1, 3).
    CHECK(result.x[0] == Approx(1.0).margin(0.1));
    CHECK(result.x[1] == Approx(3.0).margin(0.1));
    CHECK(result.objective_value < 1.0);
}

TEST_CASE("multistart finds better solution than single start", "[multistart]")
{
    multimodal_2d problem{
        .lb = Eigen::VectorXd{{-5.0, -5.0}},
        .ub = Eigen::VectorXd{{5.0, 5.0}},
    };

    Eigen::VectorXd x0{{-3.0, -3.0}};
    solver_options opts;
    opts.max_iterations = 1000;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-12);

    // Single-start BOBYQA.
    basic_solver single{bobyqa_policy{}, problem, x0, opts};
    auto single_result = single.solve();

    // Multi-start BOBYQA.
    using ms_policy = multistart_policy<bobyqa_policy<>>;
    solver_options ms_opts = opts;
    ms_opts.max_iterations = 3000;
    basic_solver multi{ms_policy{}, problem, x0, ms_opts};
    auto multi_result = multi.solve();

    // Multi-start should find a solution at least as good.
    CHECK(multi_result.objective_value <= single_result.objective_value + 0.1);
}

TEST_CASE("multistart respects max_restarts", "[multistart]")
{
    multimodal_2d problem{
        .lb = Eigen::VectorXd{{-5.0, -5.0}},
        .ub = Eigen::VectorXd{{5.0, 5.0}},
    };

    Eigen::VectorXd x0{{-3.0, -3.0}};
    solver_options opts;
    opts.max_iterations = 5000;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-14);
    opts.set_step_threshold(1e-14);

    using ms_policy = multistart_policy<bobyqa_policy<>>;
    ms_policy::options_type policy_opts;
    policy_opts.max_restarts = 2;

    basic_solver solver{ms_policy{}, problem, x0, opts, policy_opts};
    auto result = solver.solve();

    // Should terminate (not run forever) -- either converged or hit max_iterations.
    CHECK(std::isfinite(result.objective_value));
    CHECK(result.iterations <= 5000);
}
