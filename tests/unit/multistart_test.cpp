#include "argmin/detail/halton.h"
#include "argmin/solver/multistart_policy.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/cobyla_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <type_traits>

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

// Bound- and inequality-constrained problem for exercising constraint
// mirroring through the multistart decorator. Feasible when x0 + x1 >= 10
// (argmin convention: constraints() returns c with c >= 0 meaning
// feasible; see tests/unit/cobyla_test.cpp's simple_constrained). The
// feasible half-plane is pushed far enough from the origin that no
// initial-simplex vertex near x0=(0,0) can reach it, so cobyla's
// feasibility-seeking best-point selection cannot mask the violation at
// x0 the way it would for a barely-infeasible starting point.
struct infeasible_constrained_2d
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
        c[0] = x[0] + x[1] - 10.0;
    }

    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd::Constant(2, -10.0); }
    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd::Constant(2, 10.0); }
};

// Fixed-dimension (compile-time N=2) bound-constrained problem, used to pin
// that multistart_policy propagates the inner policy's fixed-size vector
// type through its decorator state instead of collapsing it to a dynamic
// Eigen::VectorXd.
struct fixed_dim_quadratic
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        return (x[0] - 1.0) * (x[0] - 1.0) + (x[1] + 2.0) * (x[1] + 2.0);
    }

    Eigen::Vector<double, 2> lower_bounds() const
    {
        return Eigen::Vector<double, 2>::Constant(-10.0);
    }

    Eigen::Vector<double, 2> upper_bounds() const
    {
        return Eigen::Vector<double, 2>::Constant(10.0);
    }
};

}

static_assert(constrained_values<infeasible_constrained_2d>);
static_assert(bound_constrained<infeasible_constrained_2d>);
static_assert(bound_constrained<fixed_dim_quadratic>);

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

TEST_CASE("multistart reports an honest status on restart exhaustion, not converged",
          "[multistart]")
{
    // Drive the exhaustion branch directly through the decorator (mirrors
    // restarting_policy_test.cpp's direct state/step manipulation) so the
    // assertion is deterministic and does not depend on whether a real
    // optimization run happens to stagnate enough times to exhaust
    // restarts within budget.
    multistart_booth problem{
        .lb = Eigen::VectorXd{{-10.0, -10.0}},
        .ub = Eigen::VectorXd{{10.0, 10.0}},
    };
    Eigen::VectorXd x0{{0.0, 0.0}};
    solver_options opts;

    using ms_policy = multistart_policy<bobyqa_policy<>>;
    ms_policy::options_type policy_opts;
    policy_opts.max_restarts = 0;  // exhausted as soon as a restart is attempted

    ms_policy policy;
    auto state = policy.init(problem, x0, opts, policy_opts);

    // Force the exhaustion branch: restart_count (0) >= max_restarts (0).
    state.restart_pending = true;
    auto result = policy.step(state);

    REQUIRE(result.policy_status.has_value());
    CHECK(*result.policy_status == solver_status::budget_exhausted);
    CHECK(*result.policy_status != solver_status::converged);
}

TEST_CASE("multistart mirrors inner constraint violation through the decorator",
          "[multistart]")
{
    infeasible_constrained_2d problem;
    Eigen::VectorXd x0{{0.0, 0.0}};
    solver_options opts;

    using ms_policy = multistart_policy<cobyla_policy>;
    basic_solver solver{ms_policy{}, problem, x0, opts};

    // The initial iterate is deeply infeasible (c = 0 + 0 - 10 = -10) and
    // stays infeasible however cobyla's initial-simplex best-point
    // selection resolves it. If multistart_policy failed to mirror the
    // inner cobyla policy's c_eq / c_ineq into its own decorator state,
    // constraint_violation() would silently read 0 instead of the real
    // inner violation.
    CHECK(solver.constraint_violation() > 0.0);
}

TEST_CASE("multistart propagates the inner policy's fixed-N vector type",
          "[multistart]")
{
    // Compile-time pin: a fixed-N inner policy keeps fully-fixed state
    // inside the decorator instead of collapsing to Eigen::VectorXd.
    static_assert(std::is_same_v<
        decltype(std::declval<multistart_policy<bobyqa_policy<2>>
            ::state_type<fixed_dim_quadratic>>().x),
        Eigen::Vector<double, 2>>);

    // Regression pin: a dynamic-N inner policy keeps a dynamic vector.
    static_assert(std::is_same_v<
        decltype(std::declval<multistart_policy<bobyqa_policy<>>
            ::state_type<multistart_booth>>().x),
        Eigen::VectorXd>);

    // Runtime exercise of the fixed-N path: basic_solver's CTAD rebinds
    // the un-rebound multistart_policy<bobyqa_policy<>> to N=2 from the
    // problem's compile-time dimension, so the decorator's state actually
    // instantiates with Eigen::Vector<double, 2> end to end.
    fixed_dim_quadratic problem;
    Eigen::Vector<double, 2> x0;
    x0 << 0.0, 0.0;

    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-12);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    using ms_policy = multistart_policy<bobyqa_policy<>>;
    basic_solver solver{ms_policy{}, problem, x0, opts};
    auto result = solver.solve();

    CHECK(result.x[0] == Approx(1.0).margin(0.05));
    CHECK(result.x[1] == Approx(-2.0).margin(0.05));
}
