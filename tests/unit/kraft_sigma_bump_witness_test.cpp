// Independent single-step correctness witness for the kraft_slsqp Armijo
// merit baseline across the penalty cold-bump. The TEST_CASE is a
// hand-derived assertion of the CORRECT post-fix behavior, tagged
// [!shouldfail] because the defect it targets is still present: the case
// registers as an expected failure against the current code and the
// corresponding fix removes the tag.
//
// Defect mechanics: kraft_slsqp computes the Armijo baseline merit_0 with
// the pre-bump penalty sigma, then bump_sigma_for_descent raises sigma and
// only the directional derivative is recomputed. The trial merits inside
// the line search read the post-bump sigma, so the Armijo test carries a
// constant spurious offset (sigma_new - sigma_old) * v0 > 0 whenever the
// bump fires at a violated iterate: sufficient decrease is unsatisfiable
// even as alpha -> 0, the 40-eval backtracking budget is exhausted, and a
// spurious BFGS reset follows. The algebraic re-baseline
//   merit_0 += (sigma_new - sigma_old) * v0
// (exact for the L1 merit) removes the offset.
//
// Reachability: at any iterate INSIDE its box bounds the bump cannot fire.
// With the QP KKT identity g.p = -h4 * lambda^T c + (bound terms) - p^T B p
// and the multiplier-driven penalty update enforcing
// sigma >= |lambda|_inf + 1/2, the merit slope satisfies
// dphi <= -h4 * v0 / 2 - p^T B p < 0, because for in-box iterates the box
// gives p_lo <= 0 <= p_hi and every active-bound multiplier term is
// non-positive. The one reachable trigger is iteration 0 with a
// user-supplied x0 OUTSIDE its box: the policy does not project the
// initial iterate (accepted iterates are projected, so all later iterates
// are in-box), the violated bound forces p_lo > 0 so the QP drives p
// uphill in f, and the box multiplier that balances g.p in the KKT
// identity is discarded from the QP's returned lambda -- the penalty
// update never sees it, sigma stays at the (small) constraint-multiplier
// scale, and the bump fires.
//
// Reference: Nocedal & Wright 2e Section 18.5 eq. 18.36 (penalty
//            sufficient for L1-merit descent); Section 15.4 (L1 exact
//            penalty); Kraft 1988 DFVLR-FB 88-28 Section 2.2.6 (penalty
//            update).

#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/basic_solver.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <limits>

using Catch::Approx;
using namespace argmin;

namespace
{

// Penalty cold-bump problem.
//
//   min  f(x) = 10 * x0
//   s.t. c(x) = x1 - 0.1 = 0        (equality, violated at x0_init)
//        1 <= x0 <= 2               (box; x0_init sits below the box)
//
// Start at x_init = (0, 0): outside the box on coordinate 0 and with
// constraint violation v0 = |0 - 0.1| = 0.1 > 0.
//
// Hand-derived iteration-0 trace (B = I initially, exact LSEI QP):
//   p_lo = (1, -inf), p_hi = (2, +inf)   -> the QP forces p0 >= 1.
//   QP solution: p = (1, 0.1), equality multiplier lambda = p1 = 0.1
//   (the active-bound multiplier z0 = p0 + g0 = 11 is NOT part of the
//   returned lambda).
//   Cold-start calibration: sigma = max(1.0, |lambda| + 1) = 1.1.
//   Multiplier update:      |lambda| + 0.5 = 0.6 < 1.1  -> no change.
//   Merit baseline (pre-bump):  merit_0 = f(0,0) + 1.1 * 0.1 = 0.11.
//   Merit slope: dphi = g.p - sigma * v0 = 10 - 1.1 * 0.1 = 9.89 >= 0
//     -> cold-bump fires: sigma_new = |g.p| / v0 + 1 = 101,
//        dphi_new = 10 - 101 * 0.1 = -0.1 < 0.
//   Stale-baseline offset: (101 - 1.1) * 0.1 = 9.99.
//   Every line-search trial projects x0 back to the box, so
//     phi(alpha) = 10 + 101 * (1 - alpha) * 0.1 >= 10   for alpha in (0,1],
//   while the stale Armijo threshold is 0.11 + 1e-4 * alpha * (-0.1) < 0.11:
//   all 40 backtracking evaluations fail, the second-order correction
//   retry fails identically, and BFGS is spuriously reset. The retry pass
//   (sigma already 101) recomputes merit_0 = 10.1 and accepts alpha = 1
//   immediately -- proving the first-pass rejection was purely the stale
//   baseline. With the re-baseline (merit_0 = 10.1 in the first pass) the
//   unit step is accepted at once: phi(1) = 10 <= 10.1 - 1e-5.
struct sigma_bump_problem
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const { return 10.0 * x[0]; }

    void gradient(const Eigen::VectorXd& /*x*/, Eigen::VectorXd& g) const
    {
        g.resize(2);
        g[0] = 10.0;
        g[1] = 0.0;
    }

    Eigen::VectorXd lower_bounds() const
    {
        Eigen::VectorXd lo(2);
        lo << 1.0, -std::numeric_limits<double>::infinity();
        return lo;
    }

    Eigen::VectorXd upper_bounds() const
    {
        Eigen::VectorXd hi(2);
        hi << 2.0, std::numeric_limits<double>::infinity();
        return hi;
    }

    int num_equality() const { return 1; }
    int num_inequality() const { return 0; }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = x[1] - 0.1;
    }

    void constraint_jacobian(const Eigen::VectorXd& /*x*/, Eigen::MatrixXd& J) const
    {
        J.resize(1, 2);
        J(0, 0) = 0.0;
        J(0, 1) = 1.0;
    }
};

}  // namespace

// WITNESS (finding: stale Armijo merit baseline across the penalty bump).
//
// The penalty cold-bump fires at a violated iterate (v0 = 0.1) and the
// stale merit_0 = 0.11 (pre-bump sigma) sits 9.99 below every reachable
// trial merit (post-bump sigma), exhausting the line search and the SOC
// retry and forcing a spurious BFGS reset -- after which the retry pass,
// with a consistent baseline, accepts the very same unit step. The correct
// behavior re-baselines merit_0 across the bump and accepts alpha = 1 on
// the first pass with no reset. Pinned [!shouldfail]: pre-fix
// bfgs_reset_count is 1.
TEST_CASE("kraft_slsqp accepts the unit step across a penalty cold-bump without a spurious BFGS reset",
          "[kraft_slsqp][sigma_bump][witness]")
{
    sigma_bump_problem problem;
    Eigen::VectorXd x0{{0.0, 0.0}};

    // Precondition: the starting iterate is constraint-violated (the bump
    // offset (sigma_new - sigma_old) * v0 is material) and below the box.
    Eigen::VectorXd c0;
    problem.constraints(x0, c0);
    CHECK(std::abs(c0[0]) == Approx(0.1));
    CHECK(x0[0] < problem.lower_bounds()[0]);

    solver_options<> opts;
    opts.max_iterations = 100;

    basic_solver solver{kraft_slsqp_policy<>{}, problem, x0, opts};
    auto sr = solver.step();

    // The accepted step is identical pre- and post-fix (x1 = (1, 0.1),
    // f = 10, feasible); what differs is HOW it is reached. Post-fix the
    // first-pass Armijo accepts alpha = 1 and no BFGS reset occurs.
    // Pre-fix the stale baseline exhausts 40 + 40 merit evaluations and
    // resets BFGS once: bfgs_reset_count == 1.
    CHECK(sr.diagnostics.bfgs_reset_count == std::size_t{0});

    // Sanity on the accepted iterate (holds pre- and post-fix; documents
    // that the witness pins the wasted-work signature, not the endpoint).
    CHECK(sr.objective_value == Approx(10.0));
    CHECK(sr.constraint_violation == Approx(0.0).margin(1e-12));
}
