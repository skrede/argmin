// Independent correctness witnesses for the nw_sqp-family (dense-BFGS) SQP
// second-order-correction gate. Two cases, each a hand-derived or
// re-baseline-independent assertion of the correct behavior. The first is a
// single-step Maratos SOC-trigger witness paralleling the SLSQP one; the
// second is the HS043 strictly-feasible-descent over-rejection witness.
//
// Reference: Nocedal & Wright 2e Section 18.3 (Maratos effect, second-order
//            correction); IPOPT Section 2.4 (SOC trigger); Wachter & Biegler
//            2006 Section 2.3 (filter envelope); Hock & Schittkowski 1981
//            Problem 43.

#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace argmin;

namespace
{

// Maratos-effect equality problem (Powell's circle example); see the SLSQP
// witness file for the full derivation. On the circle f reduces to -x0 with
// optimum x* = (1, 0); the Lagrangian Hessian is the identity, so with the
// initial BFGS Hessian B = I the QP reproduces the exact SQP step.
//
//   min  f(x) = 2*(x0^2 + x1^2 - 1) - x0
//   s.t. c(x) = x0^2 + x1^2 - 1 = 0
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

double violation_at(const maratos_problem& p, const Eigen::VectorXd& x)
{
    Eigen::VectorXd c;
    p.constraints(x, c);
    return std::abs(c[0]);
}

}  // namespace

// WITNESS (nw_sqp-family SOC trigger vs the Maratos effect).
//
// Parallels the SLSQP single-step SOC witness: at a near-feasible iterate
// (h_k = 0) the full unit step increases the violation (h(x_k + p) > h_k). The
// correct trigger fires a second-order correction on this full-step rejection
// whenever theta(x_k + p) >= theta(x_k); a gate keyed to the violation at the
// current iterate (h_k above a fixed threshold) can never fire at h_k = 0,
// which is exactly the Maratos regime. The single-step signature of "SOC
// fired" is diagnostics.soc_retry_count >= 1. Hand-derived quantities at
// x_k = (cos 0.1, sin 0.1):
//   p          = (sin^2 th, -sin th cos th) = ( 0.00996673, -0.09933467)
//   x_k + p    = ( 1.00497093, 0.00049875)
//   h(x_k)     = 0,   h(x_k + p) = 0.00996884 > 0  -> Maratos regime.
TEST_CASE("nw_sqp family fires a second-order correction at a near-feasible "
          "Maratos step",
          "[nw_sqp][soc][witness]")
{
    const double th = 0.1;
    Eigen::VectorXd xk{{std::cos(th), std::sin(th)}};

    maratos_problem geom;
    const double sin_th = std::sin(th);
    const double cos_th = std::cos(th);
    Eigen::VectorXd p{{sin_th * sin_th, -sin_th * cos_th}};
    const double h_k = violation_at(geom, xk);
    const double h_full = violation_at(geom, (xk + p).eval());
    CHECK(h_k == Approx(0.0).margin(1e-12));
    CHECK(h_full > h_k);

    solver_options<> opts;
    opts.max_iterations = 500;

    SECTION("nw_sqp")
    {
        maratos_problem problem;
        basic_solver solver{nw_sqp_policy<>{}, problem, xk, opts};
        auto sr = solver.step();
        CHECK(sr.diagnostics.soc_retry_count >= std::size_t{1});
    }

    SECTION("filter_nw_sqp")
    {
        maratos_problem problem;
        basic_solver solver{filter_nw_sqp_policy<>{}, problem, xk, opts};
        auto sr = solver.step();
        CHECK(sr.diagnostics.soc_retry_count >= std::size_t{1});
    }
}

// WITNESS (filter_nw_sqp HS043 strictly-feasible-descent over-rejection).
//
// HS043 (4 variables, 3 nonlinear inequalities, optimum f* = -44 at
// (0, 1, 2, -1)) is the canonical filter-lineage over-rejection case: the
// filter line search rejects strictly-feasible quadratic descent trials near
// the curved active manifold and the run parks at the best strictly-feasible
// iterate f ~ -40.4 instead of descending to -44. The correct behavior accepts
// the quadratic / second-order-corrected step and reaches the optimum. Pinned
// [!shouldfail]: the pre-fix run returns f ~ -40.375.
//
// This is the diagnostic for the nw_sqp-family SOC defect. If restoring the
// Maratos SOC trigger (theta(x + p) >= theta(x_k), replacing the h_k > 1e-3
// heuristic) alone does not resolve the over-rejection, the next suspect is the
// filter_nw_sqp SOC residual assembly, which drives the correction RHS from the
// stored trial-constraint buffer WITHOUT re-evaluating the full-step constraint
// c(x + p) and WITHOUT the +J*p linearization term that the SLSQP-side residual
// carries. Both candidate root causes are stated here without asserting which;
// the fix plan resolves it and records an honest resolved / partial verdict.
TEST_CASE("filter_nw_sqp reaches the HS043 optimum without over-rejection",
          "[filter_nw_sqp][hs043][witness][!shouldfail]")
{
    hs043<> problem;
    auto x0 = problem.initial_point();
    solver_options<> opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-4);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-15);

    basic_solver solver{filter_nw_sqp_policy<hs043<>::problem_dimension>{},
                        problem, x0, opts};
    auto result = solver.solve(opts);

    // The optimum is f* = -44. The pre-fix run over-rejects and parks at the
    // best strictly-feasible iterate f ~ -40.375, outside this band.
    CHECK(result.objective_value == Approx(-44.0).margin(1.0));
    CHECK(result.constraint_violation <= opts.feasibility_tolerance);
}
