// Independent single-step correctness witnesses for the line-search SLSQP
// globalization gates. Each TEST_CASE is a hand-derived assertion of the
// CORRECT post-fix behavior, tagged [!shouldfail] because the defect it
// targets is still present: the case registers as an expected failure against
// the current code and the corresponding fix removes the tag. These witnesses
// are the verification anchors; the Hock-Schittkowski acceptance matrix is a
// re-baseline recording step, never a verification step.
//
// Reference: Wachter & Biegler 2006 "On the implementation of an interior-point
//            filter line-search algorithm", Algorithm A and Section 2.4;
//            Nocedal & Wright 2e Section 18.3 (Maratos effect, second-order
//            correction); IPOPT Section 2.4 (second-order-correction trigger).

#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace argmin;

namespace
{

// f-type filter-bypass problem.
//
//   min  f(x) = 1e-3 * x0
//   s.t. c(x) = x1 - 1e11 * x0^2 = 0   (a high-curvature equality)
//
// At x_k = (0, 0): f_k = 0, c = 0 so h_k = 0 (feasible). The gradient is
// g = (1e-3, 0) and the Jacobian is J = (-2e11*x0, 1) = (0, 1). With the
// initial BFGS Hessian B = I the equality QP forces p1 = -c = 0 and then
// minimizes 0.5*p0^2 + 1e-3*p0, giving p0 = -1e-3. The unit step is therefore
//   p = (-1e-3, 0),  x_k + p = (-1e-3, 0).
// Hand-derived trial quantities at the unit step:
//   f(x_k + p)  = 1e-3 * (-1e-3)         = -1e-6   -> df  = -1e-6 (marginal)
//   c(x_k + p)  = 0 - 1e11 * (1e-3)^2    = -1e5    -> h   =  1e5  (explodes)
// The iteration is f-type (h_k = 0, grad_f.p = -1e-6 < 0) and the trial
// satisfies the Armijo f-descent test. The current code accepts on Armijo-on-f
// alone, so the iterate jumps to constraint violation 1e5. The correct rule
// (W-B Algorithm A) additionally requires filter acceptability and h <= h_max;
// with the caller-set ceiling h_max = 1e4 * max(1, h_0) = 1e4 the trial has
// h = 1e5 > h_max and MUST be rejected. The larger-than-1e4 violation (vs. a
// merely-illustrative 1e3) is deliberate: it exceeds the default ceiling so
// the h <= h_max conjunct actually bites.
struct filter_bypass_problem
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const { return 1.0e-3 * x[0]; }

    void gradient(const Eigen::VectorXd& /*x*/, Eigen::VectorXd& g) const
    {
        g.resize(2);
        g[0] = 1.0e-3;
        g[1] = 0.0;
    }

    int num_equality() const { return 1; }
    int num_inequality() const { return 0; }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = x[1] - 1.0e11 * x[0] * x[0];
    }

    void constraint_jacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const
    {
        J.resize(1, 2);
        J(0, 0) = -2.0e11 * x[0];
        J(0, 1) = 1.0;
    }
};

// Maratos-effect equality problem (Powell's circle example).
//
//   min  f(x) = 2*(x0^2 + x1^2 - 1) - x0
//   s.t. c(x) = x0^2 + x1^2 - 1 = 0     (the unit circle)
//
// On the circle f reduces to -x0, so the optimum is x* = (1, 0) with
// multiplier lambda* = 3/2. The Lagrangian Hessian is
//   grad^2 f - lambda * grad^2 c = 4I - (3/2)*2I = I,
// so with the initial BFGS Hessian B = I the QP reproduces the exact SQP step.
struct maratos_problem
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        return 2.0 * (x[0] * x[0] + x[1] * x[1] - 1.0) - x[0];
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g.resize(2);
        g[0] = 4.0 * x[0] - 1.0;
        g[1] = 4.0 * x[1];
    }

    int num_equality() const { return 1; }
    int num_inequality() const { return 0; }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = x[0] * x[0] + x[1] * x[1] - 1.0;
    }

    void constraint_jacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const
    {
        J.resize(1, 2);
        J(0, 0) = 2.0 * x[0];
        J(0, 1) = 2.0 * x[1];
    }
};

// L1 constraint violation of a single-equality problem at x.
double violation_at(const maratos_problem& p, const Eigen::VectorXd& x)
{
    Eigen::VectorXd c;
    p.constraints(x, c);
    return std::abs(c[0]);
}

}  // namespace

// WITNESS (SLSQP-01 / finding: filter bypass on f-type iterations).
//
// A marginally-objective-improving, catastrophically-infeasible unit step is
// currently ACCEPTED because the f-type branch consults only the Armijo
// f-descent test and never the filter / h_max ceiling. The correct behavior
// rejects it (filter acceptability AND h <= h_max must hold in addition to
// Armijo), so the accepted iterate keeps its constraint violation at or below
// the ceiling h_max = 1e4. Pinned [!shouldfail]: the pre-fix code accepts the
// bypass and the post-step violation is 1e5 >> 1e4.
TEST_CASE("filter_slsqp rejects a marginal-f, exploding-h unit step",
          "[filter_slsqp][witness]")
{
    filter_bypass_problem problem;
    Eigen::VectorXd x0{{0.0, 0.0}};
    solver_options<> opts;
    opts.max_iterations = 500;

    step_budget_solver solver{filter_slsqp_policy<>{}, problem, x0, opts};
    auto sr = solver.step();

    // Caller-set filter ceiling h_max = 1e4 * max(1, h_0) with h_0 = 0.
    // The correct gate rejects any trial whose violation exceeds this ceiling,
    // so an accepted step can never carry violation 1e5. Pre-fix the f-type
    // Armijo-only branch accepts the bypass trial and this fails.
    CHECK(sr.constraint_violation < 1.0e4);
}

// WITNESS (SLSQP-02 / finding: SOC gate inverted vs the Maratos effect).
//
// At a near-feasible iterate (h_k = 0) on the curved equality the full unit
// step increases the constraint violation (h(x_k + p) > h_k): the Maratos
// regime. The correct trigger fires a second-order correction on this
// rejection whenever theta(x_k + p) >= theta(x_k), independent of the current
// iterate's own violation, AND composes the corrected trial as x + p_soc: the
// SOC RHS -c(x + p) + J*p re-anchors the linearized constraints at the full
// unit step, so the re-solved QP direction p_soc is already the full corrected
// step from the current iterate. The earlier x + p + p_soc composition
// double-counts p and lands at x + 2p + O(||p||^2), farther from the optimum.
//
// Hand-derived step at x_k = (cos 0.1, sin 0.1), exact SQP with B = I:
//   tangent direction t_hat = (-sin th, cos th); QP step p = sin(th) * (-t_hat)
//     p          = (sin^2 th, -sin th cos th)   = ( 0.00996673, -0.09933467)
//     x_k + p    = (cos th + sin^2 th, sin th (1 - cos th))
//                = ( 1.00497093,  0.00049875)
//     h(x_k)     = 0
//     h(x_k + p) = |(x_k + p)|^2 - 1 = 0.00996884 > 0   -> Maratos regime.
//
// The soc_retry_count >= 1 signature alone does NOT witness the composition:
// the trigger fires for both the correct (x + p_soc) and the double-counting
// (x + p + p_soc) trials. The landing assertions below discriminate. Measured
// single-step landing at x_k (identical for both policies, f* = -1):
//   double-count composition : f = -0.998744802, h = 1.55211e-06
//   corrected composition    : f = -0.999962749, h = 2.48338e-05
// The corrected step lands an order of magnitude closer to the optimum (f <
// -0.9999 fails on the double-count's -0.9987) while keeping the violation at
// O(||p||^2) (the hand-derived residual c = -0.00997 + 0.00999 ~ 2e-5; a
// first-order landing would carry h ~ ||p|| ~ 0.1).
TEST_CASE("slsqp fires a second-order correction at a near-feasible Maratos step",
          "[slsqp][soc][witness]")
{
    const double th = 0.1;
    Eigen::VectorXd xk{{std::cos(th), std::sin(th)}};

    // Precondition (asserted directly, not gated on any violation threshold):
    // the iterate is feasible and the full unit step increases the violation.
    maratos_problem geom;
    const double sin_th = std::sin(th);
    const double cos_th = std::cos(th);
    Eigen::VectorXd p{{sin_th * sin_th, -sin_th * cos_th}};
    const double h_k = violation_at(geom, xk);
    const double h_full = violation_at(geom, (xk + p).eval());
    CHECK(h_k == Approx(0.0).margin(1e-12));
    CHECK(h_full > h_k);

    // O(||p||^2) scale for the corrected landing's residual bound.
    const double p_sq = p.squaredNorm();

    solver_options<> opts;
    opts.max_iterations = 500;

    SECTION("filter_slsqp")
    {
        maratos_problem problem;
        step_budget_solver solver{filter_slsqp_policy<>{}, problem, xk, opts};
        auto sr = solver.step();
        // SOC fires on the Maratos rejection.
        CHECK(sr.diagnostics.soc_retry_count >= std::size_t{1});
        // Corrected full step (x + p_soc) lands near the optimum; the
        // double-count composition (x + p + p_soc) parks at f ~ -0.9987 and
        // fails this bound.
        CHECK(sr.objective_value < -0.9999);
        // Landing violation is second-order in ||p||, not first-order.
        CHECK(sr.constraint_violation < 0.01 * p_sq);
    }

    SECTION("kraft_slsqp")
    {
        maratos_problem problem;
        step_budget_solver solver{kraft_slsqp_policy<>{}, problem, xk, opts};
        auto sr = solver.step();
        CHECK(sr.diagnostics.soc_retry_count >= std::size_t{1});
        CHECK(sr.objective_value < -0.9999);
        CHECK(sr.constraint_violation < 0.01 * p_sq);
    }
}
