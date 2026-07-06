// Strict-Svanberg-1987 MMA (reciprocal moving asymptotes) policy tests.
//
// Plain MMA: no conservativity loop, unconditional acceptance of the
// dual-subproblem trial each outer iter. The conservativity-protected
// variant is GCMMA (gcmma_policy); the quadratic-penalty CCSA variant
// is ccsa_quadratic_policy.
//
// Plain MMA's convergence guarantee is restricted to *separable*
// objectives and constraints (the reciprocal approximation is exact
// when each function depends on each x_j independently). On
// non-separable problems with cross-terms (e.g., HS076's x0*x2 term)
// plain MMA can drift infeasible -- by design, that's GCMMA's job to
// fix. These tests therefore exercise:
//   1. Algorithmic plumbing (init, step, dual solve, reset) does not
//      crash and produces finite output.
//   2. A structural-design-style problem (beam_design) where the
//      reciprocal approximation is well-fitted.
//
// Reference: Svanberg 1987, "The method of moving asymptotes",
//            Int. J. Numer. Methods Engng 24:359-373, Section 5
//            (separability assumption for convergence).

#include "argmin/detail/mma_reciprocal_dual_problem.h"
#include "argmin/detail/mma_subproblem.h"
#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/options.h"
#include "argmin/solver/mma_policy.h"
#include "argmin/solver/ccsa_quadratic_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace argmin;

namespace {

// Beam-design surrogate: separable objective, separable constraint, the
// classic structural-optimization shape MMA was invented for. Minimize
// f(x) = x0^2 + x1^2 subject to 1/x0 + 1/x1 <= 2 (i.e. argmin
// convention c0 = 2 - 1/x0 - 1/x1 >= 0), bounds 0.1 <= x_i <= 10.
// The optimum is at x0 = x1 = 1 with f* = 2.
struct separable_beam
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }
    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        return x[0] * x[0] + x[1] * x[1];
    }

    void gradient(const Eigen::Vector<double, 2>& x,
                  Eigen::Vector<double, 2>& g) const
    {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
    }

    void constraints(const Eigen::Vector<double, 2>& x,
                     Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = 2.0 - 1.0 / x[0] - 1.0 / x[1];
    }

    void constraint_jacobian(const Eigen::Vector<double, 2>& x,
                             Eigen::MatrixXd& J) const
    {
        J.resize(1, 2);
        J(0, 0) = 1.0 / (x[0] * x[0]);
        J(0, 1) = 1.0 / (x[1] * x[1]);
    }

    Eigen::Vector<double, 2> lower_bounds() const
    {
        return Eigen::Vector<double, 2>::Constant(0.1);
    }

    Eigen::Vector<double, 2> upper_bounds() const
    {
        return Eigen::Vector<double, 2>::Constant(10.0);
    }
};

}

// Dual-solve termination must key on the projected KKT residual, not the
// raw dual gradient norm. The MMA dual is box-constrained (y >= 0); at a
// solution with an inactive constraint (y_i* = 0 with g_tilde_i(x*) < 0)
// the raw gradient component g_i = -g_tilde_i stays bounded away from
// zero, so a raw-gradient-gated inner loop cannot recognize optimality and
// falls through to its iteration cap. The projected KKT residual (the
// bound-constrained stationarity measure lbfgsb reports on step_result)
// correctly certifies the solution.
//
// Worked dual: n = 1, m = 1. L = 0, U = 2, alpha = 0.05, beta = 1.95;
// objective p0 = q0 = 1, r_obj = 0 (unconstrained primal minimizer x = 1,
// interior); constraint pc = qc = 0.1, rc = -5 so g_tilde_1(1) = 0.1 +
// 0.1 - 5 = -4.8 < 0 (slack). The dual objective -W(y) = -(gval - 4.8 y)
// increases in y, so y* = 0 and the raw gradient there is |g| = 4.8.
TEST_CASE("mma dual solve terminates on the projected KKT residual",
          "[mma_reciprocal]")
{
    Eigen::VectorXd L{{0.0}}, U{{2.0}};
    Eigen::VectorXd alpha{{0.05}}, beta{{1.95}};
    Eigen::VectorXd p0{{1.0}}, q0{{1.0}};
    Eigen::MatrixXd pc(1, 1), qc(1, 1);
    pc(0, 0) = 0.1;
    qc(0, 0) = 0.1;
    Eigen::VectorXd rc{{-5.0}};

    detail::mma_reciprocal_dual_problem<double, Eigen::Dynamic,
                                        Eigen::Dynamic> dual;
    dual.L_out = &L;
    dual.U_out = &U;
    dual.alpha_out = &alpha;
    dual.beta_out = &beta;
    dual.p_obj_out = &p0;
    dual.q_obj_out = &q0;
    dual.p_con_out = &pc;
    dual.q_con_out = &qc;
    dual.r_obj = 0.0;
    dual.r_con_out = &rc;
    dual.n_primal = 1;
    dual.m_dual = 1;

    // Solve the dual from a warm start y0 = 1 with the production inner
    // loop, gated on the projected KKT residual.
    Eigen::VectorXd y0{{1.0}};
    lbfgsb_policy<Eigen::Dynamic> dp;
    solver_options<default_convergence> dopts;
    dopts.max_iterations = 100;
    dopts.set_gradient_threshold(1e-9);
    dopts.set_step_threshold(1e-15);
    auto ds = dp.init(dual, y0, dopts);

    step_result<double> last{};
    int iters = 0;
    for(int k = 0; k < 100; ++k)
    {
        last = dp.step(ds);
        ++iters;
        if(last.kkt_residual.value_or(last.gradient_norm) < 1e-9
           || last.step_size < 1e-15)
            break;
    }

    // The inner solver surfaces the projected KKT residual (the measure
    // the AL composition reads for its inner tolerance).
    REQUIRE(last.kkt_residual.has_value());

    // Optimality: the projected residual certifies the solution, and it
    // was reached well inside the 100-iteration cap.
    CHECK(*last.kkt_residual < 1e-7);
    CHECK(iters < 100);

    // The inactive constraint drives y* to its lower bound, where the raw
    // dual gradient norm stays bounded away from zero -- a raw-gradient
    // gate could never have certified this solution.
    CHECK(ds.x[0] == Approx(0.0).margin(1e-6));
    CHECK(last.gradient_norm > 1.0);
}

// Plumbing test: init, step, dual solve, reset all execute without crash
// and produce finite output, even on a non-separable problem (HS076)
// where plain MMA is not expected to converge to the optimum.
TEST_CASE("strict mma plumbing executes on HS076", "[mma_reciprocal]")
{
    hs076 problem;
    Eigen::VectorXd x0 = problem.initial_point();

    solver_options opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-5);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);

    basic_solver solver{
        mma_policy<hs076<>::problem_dimension>{}, problem, x0, opts};
    const auto result = solver.solve(opts);

    CHECK(std::isfinite(result.objective_value));
    CHECK(static_cast<int>(result.iterations) >= 1);
}

// Convergence on a separable structural-design problem: this is where
// MMA's reciprocal approximation is theoretically exact. Plain MMA
// should reach the optimum to good accuracy without conservativity.
TEST_CASE("strict mma converges on separable_beam", "[mma_reciprocal]")
{
    separable_beam problem;
    Eigen::Vector<double, 2> x0{2.0, 2.0};

    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-6);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);

    basic_solver solver{
        mma_policy<separable_beam::problem_dimension>{}, problem, x0, opts};
    const auto result = solver.solve(opts);

    // f* = 2.0 at x0 = x1 = 1.0 (verified via KKT: lambda such that
    // 2*x_i = lambda / x_i^2 -> 2 x_i^3 = lambda for both i, so x0=x1
    // by symmetry; the constraint binds at 2/x = 2 -> x = 1, f = 2).
    CHECK(result.objective_value == Approx(2.0).margin(1e-2));
    // Constraint roughly satisfied (allow 1e-3 violation since plain
    // MMA without conservativity can drift slightly infeasible).
    Eigen::VectorXd c(1);
    problem.constraints(Eigen::Vector<double, 2>{
        Eigen::Vector<double, 2>::Map(result.x.data(), 2)}, c);
    CHECK(c[0] >= -1e-3);
}

// ---------------------------------------------------------------------------
// Bounded-dual elastics (Svanberg 2002 relaxed subproblem, a_i = 0 instance)
// ---------------------------------------------------------------------------

namespace {

// min x0^2 + x1^2 s.t. x0 >= 3 (argmin convention c0 = x0 - 3 >= 0),
// box [-10, 10]^2. The start (0, 0) is inequality-infeasible (c0 = -3).
// The optimum is x* = (3, 0), f* = 9, with the single constraint active.
struct halfspace_cell
{
    static constexpr int problem_dimension = 2;
    int dimension() const { return 2; }
    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        return x[0] * x[0] + x[1] * x[1];
    }
    void gradient(const Eigen::Vector<double, 2>& x,
                  Eigen::Vector<double, 2>& g) const
    {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
    }
    void constraints(const Eigen::Vector<double, 2>& x, auto& c) const
    {
        c[0] = x[0] - 3.0;
    }
    void constraint_jacobian(const Eigen::Vector<double, 2>&, auto& J) const
    {
        J(0, 0) = 1.0; J(0, 1) = 0.0;
    }
    Eigen::Vector<double, 2> lower_bounds() const
    {
        return Eigen::Vector<double, 2>::Constant(2, -10.0);
    }
    Eigen::Vector<double, 2> upper_bounds() const
    {
        return Eigen::Vector<double, 2>::Constant(2, 10.0);
    }
    Eigen::Vector<double, 2> initial_point() const
    {
        return Eigen::Vector<double, 2>{{0.0, 0.0}};
    }
};

}

// Half b of the elastics: the primal y-recovery. Hand-computed KKT cases.
TEST_CASE("bounded-dual elastics recover the primal slacks",
          "[mma_reciprocal][elastics]")
{
    using argmin::detail::recover_elastic_slacks;
    Eigen::Vector<double, 1> lambda, c_dual, gcval, y, relaxed;

    // Saturated at c_i with a positive raw approximation value: the
    // elastic absorbs the whole violation, y_i = g_tilde_i, relaxed = 0.
    lambda[0] = 5.0; c_dual[0] = 5.0; gcval[0] = 3.0;
    recover_elastic_slacks(lambda, c_dual, gcval, y, relaxed);
    CHECK(y[0] == Approx(3.0));
    CHECK(relaxed[0] == Approx(0.0));

    // Interior multiplier (lambda < c_i): the elastic is inert, y_i = 0,
    // the relaxed value is the raw g_tilde_i.
    lambda[0] = 2.0; c_dual[0] = 5.0; gcval[0] = 3.0;
    recover_elastic_slacks(lambda, c_dual, gcval, y, relaxed);
    CHECK(y[0] == Approx(0.0));
    CHECK(relaxed[0] == Approx(3.0));

    // Saturated but with a non-positive raw value: the max(., 0) floor
    // keeps y_i = 0 (no negative slack), relaxed = raw.
    lambda[0] = 5.0; c_dual[0] = 5.0; gcval[0] = -1.0;
    recover_elastic_slacks(lambda, c_dual, gcval, y, relaxed);
    CHECK(y[0] == Approx(0.0));
    CHECK(relaxed[0] == Approx(-1.0));

    // A +infinity bound never saturates (the un-elastic classic dual).
    lambda[0] = 1e12;
    c_dual[0] = std::numeric_limits<double>::infinity();
    gcval[0] = 4.0;
    recover_elastic_slacks(lambda, c_dual, gcval, y, relaxed);
    CHECK(y[0] == Approx(0.0));
    CHECK(relaxed[0] == Approx(4.0));
}

// Half a of the elastics: the dual box bounds the multiplier on an
// inequality-infeasible start where the un-elastic dual is unbounded (it
// would otherwise run the inner solver to its iteration cap on an ever-
// growing multiplier). HS024 from the infeasible interior point
// x0 = (0.5, 4.0) (c0 = 0.5/sqrt(3) - 4 < 0, c2 = 6 - 0.5 - sqrt(3)*4 < 0).
TEST_CASE("bounded dual keeps the multiplier bounded on an infeasible start",
          "[mma_reciprocal][elastics]")
{
    hs024 problem;
    Eigen::Vector<double, 2> x0{0.5, 4.0};
    solver_options opts;
    opts.max_iterations = 200;

    auto max_multiplier = [&](double scale) {
        ccsa_quadratic_policy<hs024<>::problem_dimension> policy;
        typename ccsa_quadratic_policy<
            hs024<>::problem_dimension>::options_type popts;
        popts.dual_bound_scale = scale;
        auto s = policy.init(problem, x0, opts, popts);
        double worst = 0.0;
        double worst_over_bound = 0.0;
        for(int k = 0; k < 200; ++k)
        {
            (void)policy.step(s);
            worst = std::max(worst, s.y_dual.cwiseAbs().maxCoeff());
            for(int i = 0; i < s.y_dual.size(); ++i)
                worst_over_bound = std::max(worst_over_bound,
                                            s.y_dual[i] / s.c_dual[i]);
        }
        return std::pair{worst, worst_over_bound};
    };

    // Bounded (swept default scale): the multiplier never exceeds its box
    // c_i = 1000 * max(|g_i(x0)|, 1), so the saturation ratio stays <= 1.
    auto [bounded_max, bounded_ratio] = max_multiplier(1000.0);
    CHECK(bounded_ratio <= 1.0 + 1e-9);
    CHECK(bounded_max < 1e5);   // c_i ~ 3.7e3 here; comfortably bounded.

    // Unbounded control (+infinity): the same infeasible subproblem drives
    // the dual multiplier to astronomically large values (the DoS the box
    // mitigates).
    auto [unbounded_max, unbounded_ratio] =
        max_multiplier(std::numeric_limits<double>::infinity());
    (void)unbounded_ratio;
    CHECK(unbounded_max > 1e8);
    CHECK(unbounded_max > 1e3 * bounded_max);
}

// End-to-end: with the bounded-dual box on the constraint multipliers,
// CCSA reaches the true optimum from an inequality-infeasible start where
// the classic (unbounded) dual would otherwise stall. min x0^2 + x1^2
// s.t. x0 >= 3 from (0, 0); x* = (3, 0), f* = 9. The conservativity test
// compares the raw approximation values (Svanberg concheck); the bounded
// dual is what keeps the infeasible-start subproblem well posed.
TEST_CASE("ccsa with bounded-dual elastics converges from an infeasible start",
          "[mma_reciprocal][elastics]")
{
    halfspace_cell problem;
    Eigen::Vector<double, 2> x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 300;
    opts.set_gradient_threshold(1e-8);
    opts.set_step_threshold(1e-15);

    basic_solver solver{
        ccsa_quadratic_policy<halfspace_cell::problem_dimension>{},
        problem, x0, opts};

    double best_feasible = 1e10;
    for(int i = 0; i < 300; ++i)
    {
        auto sr = solver.step();
        if(solver.constraint_violation() < 1e-6
           && sr.objective_value < best_feasible)
            best_feasible = sr.objective_value;
    }

    // Reached the exact constrained optimum f* = 9 at x* = (3, 0).
    CHECK(best_feasible == Approx(9.0).margin(1e-4));
}
