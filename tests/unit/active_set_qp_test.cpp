#include "argmin/detail/active_set_qp.h"
#include "argmin/detail/givens_qr_update.h"

#include <Eigen/Core>
#include <Eigen/QR>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

using Catch::Approx;
using namespace argmin;
using namespace argmin::detail;

TEST_CASE("solve_qp unconstrained", "[qp]")
{
    // min 0.5 x^T (2I) x + [-2, -4]^T x
    // = x1^2 + x2^2 - 2*x1 - 4*x2
    // Gradient = 0 => x* = [1, 2]
    Eigen::MatrixXd G = 2.0 * Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd d{{-2.0, -4.0}};
    Eigen::MatrixXd A_eq(0, 2);
    Eigen::VectorXd b_eq(0);
    Eigen::MatrixXd A_ineq(0, 2);
    Eigen::VectorXd b_ineq(0);
    Eigen::VectorXd x0{{0.0, 0.0}};

    auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);

    CHECK(result.status == qp_status::optimal);
    CHECK(result.x[0] == Approx(1.0).margin(1e-10));
    CHECK(result.x[1] == Approx(2.0).margin(1e-10));
}

TEST_CASE("solve_qp equality only", "[qp]")
{
    // min 0.5 x^T (2I) x   s.t.  x1 + x2 = 1
    // Solution: x* = [0.5, 0.5]
    Eigen::MatrixXd G = 2.0 * Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd d = Eigen::VectorXd::Zero(2);
    Eigen::MatrixXd A_eq(1, 2);
    A_eq << 1.0, 1.0;
    Eigen::VectorXd b_eq{{1.0}};
    Eigen::MatrixXd A_ineq(0, 2);
    Eigen::VectorXd b_ineq(0);
    Eigen::VectorXd x0{{0.5, 0.5}};

    auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);

    CHECK(result.status == qp_status::optimal);
    CHECK(result.x[0] == Approx(0.5).margin(1e-10));
    CHECK(result.x[1] == Approx(0.5).margin(1e-10));
}

TEST_CASE("solve_qp inequality constraints", "[qp]")
{
    // N&W Example 16.3 style:
    // min (x1 - 1)^2 + (x2 - 2.5)^2
    // = 0.5 x^T (2I) x + [-2, -5]^T x + const
    //
    // s.t.  x1 - 2*x2 + 2 >= 0    (constraint 1)
    //      -x1 - 2*x2 + 6 >= 0    (constraint 2)
    //      -x1 + 2*x2 + 2 >= 0    (constraint 3)
    //       x1 >= 0                (constraint 4)
    //       x2 >= 0                (constraint 5)
    //
    // Solution: x* = (1.4, 1.7)
    Eigen::MatrixXd G = 2.0 * Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd d{{-2.0, -5.0}};
    Eigen::MatrixXd A_eq(0, 2);
    Eigen::VectorXd b_eq(0);

    Eigen::MatrixXd A_ineq(5, 2);
    Eigen::VectorXd b_ineq(5);
    A_ineq << 1.0, -2.0,
             -1.0, -2.0,
             -1.0,  2.0,
              1.0,  0.0,
              0.0,  1.0;
    b_ineq << -2.0, -6.0, -2.0, 0.0, 0.0;

    Eigen::VectorXd x0{{2.0, 0.0}};

    auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);

    CHECK(result.status == qp_status::optimal);
    CHECK(result.x[0] == Approx(1.4).margin(1e-8));
    CHECK(result.x[1] == Approx(1.7).margin(1e-8));
}

TEST_CASE("solve_qp box constraints", "[qp]")
{
    // min 0.5 x^T I x + [-2, -3]^T x
    // Unconstrained min: [2, 3]. With bounds [0,1]^2: x* = [1, 1].
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd d{{-2.0, -3.0}};
    Eigen::MatrixXd A_eq(0, 2);
    Eigen::VectorXd b_eq(0);
    Eigen::MatrixXd A_ineq(0, 2);
    Eigen::VectorXd b_ineq(0);
    Eigen::VectorXd lower{{0.0, 0.0}};
    Eigen::VectorXd upper{{1.0, 1.0}};
    Eigen::VectorXd x0{{0.5, 0.5}};

    auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, lower, upper, x0);

    CHECK(result.status == qp_status::optimal);
    CHECK(result.x[0] == Approx(1.0).margin(1e-8));
    CHECK(result.x[1] == Approx(1.0).margin(1e-8));
}

TEST_CASE("solve_qp mixed equality inequality box", "[qp]")
{
    // 3D problem: min 0.5 x^T I x
    // s.t. x1 + x2 + x3 = 1         (equality)
    //      x1 - x2 >= 0              (inequality)
    //      0 <= xi <= 2              (box)
    //
    // With equality x3 = 1 - x1 - x2, objective becomes function of x1,x2.
    // With x1 >= x2 and bounds, solution at x1 = x2 = 1/3, x3 = 1/3.
    // Actually unconstrained min on the equality plane is x1=x2=x3=1/3.
    // Check x1>=x2: 1/3 >= 1/3 satisfied (weakly).
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(3, 3);
    Eigen::VectorXd d = Eigen::VectorXd::Zero(3);
    Eigen::MatrixXd A_eq(1, 3);
    A_eq << 1.0, 1.0, 1.0;
    Eigen::VectorXd b_eq{{1.0}};
    Eigen::MatrixXd A_ineq(1, 3);
    A_ineq << 1.0, -1.0, 0.0;
    Eigen::VectorXd b_ineq{{0.0}};
    Eigen::VectorXd lower{{0.0, 0.0, 0.0}};
    Eigen::VectorXd upper{{2.0, 2.0, 2.0}};

    // Feasible starting point on equality plane with x1 >= x2 and in bounds
    Eigen::VectorXd x0{{0.5, 0.3, 0.2}};

    auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, lower, upper, x0);

    CHECK(result.status == qp_status::optimal);
    // x1 + x2 + x3 = 1
    CHECK(result.x.sum() == Approx(1.0).margin(1e-8));
    // x1 >= x2
    CHECK(result.x[0] >= result.x[1] - 1e-8);
    // All components in bounds
    for(int i = 0; i < 3; ++i)
    {
        CHECK(result.x[i] >= -1e-8);
        CHECK(result.x[i] <= 2.0 + 1e-8);
    }
    // Optimal: x* ~ [1/3, 1/3, 1/3]
    CHECK(result.x[0] == Approx(1.0 / 3.0).margin(1e-6));
    CHECK(result.x[1] == Approx(1.0 / 3.0).margin(1e-6));
    CHECK(result.x[2] == Approx(1.0 / 3.0).margin(1e-6));
}

TEST_CASE("solve_qp does not return optimal for contradictory inequalities", "[qp]")
{
    // x >= 1 and -x >= 0 have no common point. From x0 = 0.5, which violates both rows by 0.5,
    // the unconstrained step -0.5 blocks on x >= 1 at alpha = 0, that row then fills the single
    // dimension with multiplier g = 0.5 >= 0, and the loop reaches its terminal branch at x = 0.5,
    // which the terminal verification rejects. An infeasible status needs a certificate, and the
    // only one this solver builds is for inconsistent equality rows; inconsistent inequalities
    // carry none, so the honest outcome is the bounded-iteration status.
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(1, 1);
    Eigen::VectorXd d{{0.0}};
    Eigen::MatrixXd A_eq(0, 1);
    Eigen::VectorXd b_eq(0);
    Eigen::MatrixXd A_ineq(2, 1);
    A_ineq << 1.0, -1.0;
    Eigen::VectorXd b_ineq(2);
    b_ineq << 1.0, 0.0;  // x >= 1 and -x >= 0 => x <= 0
    Eigen::VectorXd x0{{0.5}};
    qp_options opts{.max_iterations = 50};

    auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0, opts);

    CHECK(result.status == qp_status::max_iterations);
}

TEST_CASE("solve_qp degenerate weakly active", "[qp]")
{
    // min x1^2 + (x2+1)^2   s.t. x1 >= 0, x2 >= 0
    // = 0.5 x^T (2I) x + [0, 2]^T x + const
    // Unconstrained min: (0, -1). With x2 >= 0: x* = (0, 0).
    // Constraint x1 >= 0 is weakly active (lambda_1 = 0 at solution).
    Eigen::MatrixXd G = 2.0 * Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd d{{0.0, 2.0}};
    Eigen::MatrixXd A_eq(0, 2);
    Eigen::VectorXd b_eq(0);
    Eigen::MatrixXd A_ineq(2, 2);
    A_ineq << 1.0, 0.0,
              0.0, 1.0;
    Eigen::VectorXd b_ineq{{0.0, 0.0}};
    Eigen::VectorXd x0{{1.0, 1.0}};

    auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);

    CHECK(result.status == qp_status::optimal);
    CHECK(result.x[0] == Approx(0.0).margin(1e-8));
    CHECK(result.x[1] == Approx(0.0).margin(1e-8));
}

TEST_CASE("solve_qp N&W Example 16.1 equality", "[qp]")
{
    // N&W Example 16.1, p. 443:
    // min 3*x1^2 + 2*x1*x2 + x1*x3 + 2.5*x2^2 + 2*x2*x3 + 2*x3^2
    //     - 8*x1 - 3*x2 - 3*x3
    // s.t. x1 + x3 = 3, x2 + x3 = 0
    //
    // G = [[6,2,1],[2,5,2],[1,2,4]], d = [-8,-3,-3]
    // Solution: x* = [2, -1, 1], lambda* = [3, -2]
    Eigen::MatrixXd G(3, 3);
    G << 6.0, 2.0, 1.0,
         2.0, 5.0, 2.0,
         1.0, 2.0, 4.0;
    Eigen::VectorXd d{{-8.0, -3.0, -3.0}};
    Eigen::MatrixXd A_eq(2, 3);
    A_eq << 1.0, 0.0, 1.0,
            0.0, 1.0, 1.0;
    Eigen::VectorXd b_eq{{3.0, 0.0}};
    Eigen::MatrixXd A_ineq(0, 3);
    Eigen::VectorXd b_ineq(0);
    // x0 must satisfy constraints: x1+x3=3, x2+x3=0 => e.g. x=(1,-2,2)
    Eigen::VectorXd x0{{1.0, -2.0, 2.0}};

    auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);

    CHECK(result.status == qp_status::optimal);
    CHECK(result.x[0] == Approx(2.0).margin(1e-8));
    CHECK(result.x[1] == Approx(-1.0).margin(1e-8));
    CHECK(result.x[2] == Approx(1.0).margin(1e-8));
}

// ---------------------------------------------------------------------------
// Dual-multiplier assertions on hand-solvable QPs.
//
// The active-set solver returns lambda satisfying the stationarity relation
// A_W^T lambda = G x + d at the solution (N&W eq. 16.30/16.37), where A_W
// collects the rows of the active constraints. Each case below derives the
// multipliers in closed form from the KKT conditions and pins both their
// values and their signs (dual feasibility for inequalities).
//
// Reference: Nocedal & Wright, "Numerical Optimization" 2e, Section 16.5
//            (KKT conditions and multiplier recovery), Examples 16.1/16.3.
// ---------------------------------------------------------------------------

TEST_CASE("solve_qp equality multipliers match hand-derived KKT duals", "[qp][dual]")
{
    // N&W Example 16.1: min 0.5 x^T G x + d^T x  s.t. A_eq x = b_eq with
    //   G = [[6,2,1],[2,5,2],[1,2,4]],  d = [-8,-3,-3],
    //   A_eq = [[1,0,1],[0,1,1]],  b_eq = [3, 0].
    // At x* = [2, -1, 1] the gradient is
    //   G x* + d = [11, 1, 4] + [-8, -3, -3] = [3, -2, 1].
    // Stationarity A_eq^T lambda = G x* + d gives
    //   [lambda1, lambda2, lambda1 + lambda2] = [3, -2, 1]
    //   => lambda1 = 3,  lambda2 = -2   (and 3 + (-2) = 1 checks out).
    // Equality multipliers are free in sign: lambda1 = +3, lambda2 = -2.
    Eigen::MatrixXd G(3, 3);
    G << 6.0, 2.0, 1.0,
         2.0, 5.0, 2.0,
         1.0, 2.0, 4.0;
    Eigen::VectorXd d{{-8.0, -3.0, -3.0}};
    Eigen::MatrixXd A_eq(2, 3);
    A_eq << 1.0, 0.0, 1.0,
            0.0, 1.0, 1.0;
    Eigen::VectorXd b_eq{{3.0, 0.0}};
    Eigen::MatrixXd A_ineq(0, 3);
    Eigen::VectorXd b_ineq(0);
    Eigen::VectorXd x0{{1.0, -2.0, 2.0}};

    auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);

    CHECK(result.status == qp_status::optimal);
    REQUIRE(result.lambda.size() == 2);
    CHECK(result.lambda[0] == Approx(3.0).margin(1e-8));
    CHECK(result.lambda[1] == Approx(-2.0).margin(1e-8));
}

TEST_CASE("solve_qp single-active-inequality multiplier is a nonnegative KKT dual",
          "[qp][dual]")
{
    // min 0.5 (x1^2 + x2^2)  s.t.  x1 + x2 >= 1   (feasible form a^T x >= b).
    // KKT with G = I, d = 0, a = [1, 1], mu >= 0:
    //   stationarity  G x + d = a^T mu  =>  x = mu * [1, 1]
    //   active        x1 + x2 = 1       =>  2 mu = 1  =>  mu = 1/2
    //   dual feas.    mu = 0.5 >= 0
    //   primal        x* = (0.5, 0.5)
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd d = Eigen::VectorXd::Zero(2);
    Eigen::MatrixXd A_eq(0, 2);
    Eigen::VectorXd b_eq(0);
    Eigen::MatrixXd A_ineq(1, 2);
    A_ineq << 1.0, 1.0;
    Eigen::VectorXd b_ineq{{1.0}};
    Eigen::VectorXd x0{{1.0, 0.0}};

    auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);

    CHECK(result.status == qp_status::optimal);
    CHECK(result.x[0] == Approx(0.5).margin(1e-10));
    CHECK(result.x[1] == Approx(0.5).margin(1e-10));
    REQUIRE(result.lambda.size() == 1);
    CHECK(result.lambda[0] == Approx(0.5).margin(1e-8));
    CHECK(result.lambda[0] >= 0.0);  // dual feasibility
}

TEST_CASE("solve_qp N&W 16.3 active-inequality multiplier matches KKT dual",
          "[qp][dual]")
{
    // min (x1 - 1)^2 + (x2 - 2.5)^2  =>  G = 2I, d = [-2, -5].
    // Constraints a_i^T x >= b_i, rows of A_ineq below. At x* = (1.4, 1.7)
    // only constraint 0 (x1 - 2 x2 + 2 >= 0) is active:
    //   grad = G x* + d = (0.8, -1.6),  a_0 = [1, -2].
    //   stationarity grad = mu_0 a_0:  0.8 = mu_0,  -1.6 = -2 mu_0
    //     => mu_0 = 0.8   (consistent),  dual feas. mu_0 = 0.8 >= 0.
    //   x1 = 1 + mu_0/2 = 1.4,  x2 = 2.5 - mu_0 = 1.7, active row satisfied.
    // All other inequality multipliers are zero (inactive/interior).
    Eigen::MatrixXd G = 2.0 * Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd d{{-2.0, -5.0}};
    Eigen::MatrixXd A_eq(0, 2);
    Eigen::VectorXd b_eq(0);
    Eigen::MatrixXd A_ineq(5, 2);
    A_ineq << 1.0, -2.0,
             -1.0, -2.0,
             -1.0,  2.0,
              1.0,  0.0,
              0.0,  1.0;
    Eigen::VectorXd b_ineq(5);
    b_ineq << -2.0, -6.0, -2.0, 0.0, 0.0;
    Eigen::VectorXd x0{{2.0, 0.0}};

    auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);

    CHECK(result.status == qp_status::optimal);
    REQUIRE(result.lambda.size() == 5);
    CHECK(result.lambda[0] == Approx(0.8).margin(1e-7));
    CHECK(result.lambda[0] >= 0.0);  // dual feasibility on the active row
    for(int i = 1; i < 5; ++i)
        CHECK(result.lambda[i] == Approx(0.0).margin(1e-7));
}

// Max-iteration exit now preserves the last multipliers computed for the
// working set they were solved against, instead of scattering a zero vector.
// On the truncated N&W 16.3 solve below the initial working set has two
// active inequalities with hand-derivable duals (-2, -1), so the returned
// lambda must carry a nonzero computed multiplier. (Previously tagged
// [!shouldfail] against the zero-scatter substrate; the tag is removed now
// that the exit path returns the computed duals.)
TEST_CASE("solve_qp returns computed multipliers on max-iteration exit",
          "[qp][dual]")
{
    // A truncated solve on the N&W 16.3 geometry keeps at least
    // one inequality active in the working set, so the returned lambda must
    // carry a nonzero computed multiplier -- not be identically zero.
    Eigen::MatrixXd G = 2.0 * Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd d{{-2.0, -5.0}};
    Eigen::MatrixXd A_eq(0, 2);
    Eigen::VectorXd b_eq(0);
    Eigen::MatrixXd A_ineq(5, 2);
    A_ineq << 1.0, -2.0,
             -1.0, -2.0,
             -1.0,  2.0,
              1.0,  0.0,
              0.0,  1.0;
    Eigen::VectorXd b_ineq(5);
    b_ineq << -2.0, -6.0, -2.0, 0.0, 0.0;
    Eigen::VectorXd x0{{2.0, 0.0}};
    qp_options opts{.max_iterations = 1};

    auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0, opts);

    REQUIRE(result.status == qp_status::max_iterations);
    CHECK(result.lambda.cwiseAbs().maxCoeff() > 0.0);
}

// ---------------------------------------------------------------------------
// Working-set independence and anti-cycling pins.
//
// The null-space method (N&W Algorithm 16.3) requires the working-set matrix
// A_W to have full row rank. Admitting linearly dependent gradient rows
// (parallel active constraints at a degenerate vertex) violates that
// precondition. The rank-revealing admission keeps the lowest-index
// independent basis; Bland's rule (Bland 1977; N&W Section 13.5) guarantees
// termination once the iteration budget is exceeded.
// ---------------------------------------------------------------------------

TEST_CASE("initial_working_set admits only linearly independent rows",
          "[qp][working-set]")
{
    // x1 >= 0 and 2*x1 >= 0 have parallel gradients and are both near-active
    // at x0 = (0, 0.5). Their candidate set {0, 1} is rank 1. Full-row-rank
    // admission must retain only the first (lowest-index) row.
    //
    // Pre-fix: W = {0, 1} with rank(A_W) = 1 < |W| (dependent row admitted).
    // Post-fix: W = {0} with rank(A_W) = 1 = |W| (full row rank).
    Eigen::MatrixXd A(2, 2);
    A << 1.0, 0.0,
         2.0, 0.0;
    Eigen::VectorXd b{{0.0, 0.0}};
    Eigen::VectorXd x0{{0.0, 0.5}};

    auto W = initial_working_set<double>(x0, A, b, /*n_eq=*/0, /*tol=*/1e-9);

    REQUIRE_FALSE(W.empty());
    Eigen::MatrixXd A_W(static_cast<int>(W.size()), 2);
    for(int k = 0; k < static_cast<int>(W.size()); ++k)
        A_W.row(k) = A.row(W[static_cast<std::size_t>(k)]);
    const int rank = static_cast<int>(A_W.colPivHouseholderQr().rank());
    CHECK(rank == static_cast<int>(W.size()));  // no linearly dependent row admitted
}

TEST_CASE("solve_qp leaves a degenerate active vertex for the interior optimum",
          "[qp][working-set][dual]")
{
    // finding-2 geometry: min 0.5||x||^2 - (x1 + x2)  (G = I, d = (-1,-1))
    // s.t. x1 >= 0, 2*x1 >= 0 (parallel), x2 >= 0, started at the origin where
    // all three are active. The unconstrained minimizer (1, 1) is feasible, so
    // the solver must leave the degenerate vertex and return it with every
    // multiplier zero (no constraint active at the interior optimum). The
    // active dual is hand-derivable: grad = G x* + d = 0 at (1, 1), so
    // A_W^T lambda = 0 forces lambda = 0 for any working set.
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd d{{-1.0, -1.0}};
    Eigen::MatrixXd A_eq(0, 2);
    Eigen::VectorXd b_eq(0);
    Eigen::MatrixXd A_ineq(3, 2);
    A_ineq << 1.0, 0.0,
              2.0, 0.0,
              0.0, 1.0;
    Eigen::VectorXd b_ineq{{0.0, 0.0, 0.0}};
    Eigen::VectorXd x0{{0.0, 0.0}};

    auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);

    CHECK(result.status == qp_status::optimal);
    CHECK(result.x[0] == Approx(1.0).margin(1e-8));
    CHECK(result.x[1] == Approx(1.0).margin(1e-8));
    REQUIRE(result.lambda.size() == 3);
    for(int i = 0; i < 3; ++i)
        CHECK(result.lambda[i] == Approx(0.0).margin(1e-8));  // interior optimum: zero duals
}

TEST_CASE("solve_qp terminates on a redundant-constraint degenerate vertex",
          "[qp][working-set]")
{
    // The optimum (0, 0) sits at a vertex with redundant parallel active
    // constraints (x1 >= 0 and 3*x1 >= 0; x2 >= 0 and 5*x2 >= 0), a classic
    // degeneracy that can cycle under an arbitrary-tie leaving rule. Rank-
    // revealing admission plus the Bland fallback guarantee termination; the
    // solver must return the correct optimum with status optimal rather than
    // exhausting its iteration budget.
    Eigen::MatrixXd G = 2.0 * Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd d{{2.0, 2.0}};  // unconstrained min (-1,-1); with x >= 0 -> (0,0)
    Eigen::MatrixXd A_eq(0, 2);
    Eigen::VectorXd b_eq(0);
    Eigen::MatrixXd A_ineq(4, 2);
    A_ineq << 1.0, 0.0,
              3.0, 0.0,
              0.0, 1.0,
              0.0, 5.0;
    Eigen::VectorXd b_ineq{{0.0, 0.0, 0.0, 0.0}};
    Eigen::VectorXd x0{{0.0, 0.0}};
    qp_options opts{.max_iterations = 50};

    auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0, opts);

    CHECK(result.status == qp_status::optimal);
    CHECK(result.x[0] == Approx(0.0).margin(1e-8));
    CHECK(result.x[1] == Approx(0.0).margin(1e-8));
}

TEST_CASE("solve_qp projects an equality-infeasible start onto the equality manifold",
          "[qp][qp_rank]")
{
    // Shared objective: min 0.5 ||x||^2 + d^T x with d = (-1, 0, 1), started at x0 = 0. Under
    // x1 + x2 + x3 = 3, stationarity x + d = t (1, 1, 1) gives x = (1 + t, t, t - 1) and the row
    // gives 3t = 3, so x* = (2, 1, 0) with multiplier 1. Without a projection every step stays in
    // the row's null space through x0 and the loop ends at -d = (1, 0, -1), where A_eq x = 0.
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(3, 3);
    Eigen::VectorXd d{{-1.0, 0.0, 1.0}};
    Eigen::MatrixXd A_ineq(0, 3);
    Eigen::VectorXd b_ineq(0);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(3);

    SECTION("a single equality row")
    {
        Eigen::MatrixXd A_eq(1, 3);
        A_eq << 1.0, 1.0, 1.0;
        Eigen::VectorXd b_eq{{3.0}};
        auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);
        REQUIRE(result.status == qp_status::optimal);
        CHECK((A_eq * result.x - b_eq).cwiseAbs().maxCoeff() <= 1e-12 * (1.0 + 3.0));
        CHECK(result.x[0] == Approx(2.0).margin(1e-10));
        CHECK(result.x[1] == Approx(1.0).margin(1e-10));
        CHECK(result.x[2] == Approx(0.0).margin(1e-10));
        CHECK(result.lambda[0] == Approx(1.0).margin(1e-10));
    }

    SECTION("duplicated equality rows")
    {
        // The row stated twice has rank 1 but is consistent, so the rank-truncated correction
        // drops the duplicate and the optimum is the single-row one.
        Eigen::MatrixXd A_eq(2, 3);
        A_eq << 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0;
        Eigen::VectorXd b_eq{{3.0, 3.0}};
        auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);
        REQUIRE(result.status == qp_status::optimal);
        CHECK((A_eq * result.x - b_eq).cwiseAbs().maxCoeff() <= 1e-12 * (1.0 + 3.0));
        CHECK(result.x[0] == Approx(2.0).margin(1e-10));
        CHECK(result.x[1] == Approx(1.0).margin(1e-10));
        CHECK(result.x[2] == Approx(0.0).margin(1e-10));
    }

    SECTION("a rank-zero equality row with a zero right-hand side")
    {
        // 0^T x = 0 holds everywhere, so nothing is projected and the solution is -d.
        Eigen::MatrixXd A_eq = Eigen::MatrixXd::Zero(1, 3);
        Eigen::VectorXd b_eq{{0.0}};
        auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);
        REQUIRE(result.status == qp_status::optimal);
        CHECK(result.x[0] == Approx(1.0).margin(1e-10));
        CHECK(result.x[1] == Approx(0.0).margin(1e-10));
        CHECK(result.x[2] == Approx(-1.0).margin(1e-10));
    }

    SECTION("an inequality active only at the projected point")
    {
        // min 0.5 ||x||^2 + x2 s.t. x1 = 1 and x1 + x2 >= 1, from x0 = (0, 0), where the
        // inequality is violated by 1. The projection lands on (1, 0), where it is active. With
        // x1 = 1 the row reads x2 >= 0 against a free minimizer x2 = -1, so x* = (1, 0), and
        // stationarity (1, 1) = lambda_eq (1, 0) + lambda_in (1, 1) gives lambda = (0, 1).
        // Classified at the projected point both rows start in the working set and the first
        // iteration terminates; classified at x0 the inequality enters only after a zero-length
        // blocked step, one iteration later.
        Eigen::MatrixXd G2 = Eigen::MatrixXd::Identity(2, 2);
        Eigen::VectorXd d2{{0.0, 1.0}};
        Eigen::MatrixXd A_eq(1, 2);
        A_eq << 1.0, 0.0;
        Eigen::VectorXd b_eq{{1.0}};
        Eigen::MatrixXd A_in(1, 2);
        A_in << 1.0, 1.0;
        Eigen::VectorXd b_in{{1.0}};
        auto result = solve_qp(G2, d2, A_eq, b_eq, A_in, b_in, Eigen::VectorXd::Zero(2).eval());
        REQUIRE(result.status == qp_status::optimal);
        CHECK(result.iterations == 0);
        CHECK(result.x[0] == Approx(1.0).margin(1e-12));
        CHECK(result.x[1] == Approx(0.0).margin(1e-12));
        CHECK(result.lambda[0] == Approx(0.0).margin(1e-12));
        CHECK(result.lambda[1] == Approx(1.0).margin(1e-12));
    }
}

TEST_CASE("solve_qp reports infeasible on contradictory equality rows", "[qp][qp_rank]")
{
    // x1 + x2 + x3 = 3 and x1 + x2 + x3 = 4 have identical rows and different right-hand sides:
    // (3, 4) is not in the range of A_eq = (1, 1)^T (1, 1, 1), so a rank-1 correction satisfies at
    // most one row and leaves a residual of 1 against a bound of tol (1 + 4) = 5e-12. A zero row
    // with a nonzero right-hand side is the rank-zero instance: its range is empty.
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(3, 3);
    Eigen::VectorXd d{{-1.0, 0.0, 1.0}};
    Eigen::MatrixXd A_ineq(0, 3);
    Eigen::VectorXd b_ineq(0);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(3);

    SECTION("duplicated rows with different right-hand sides")
    {
        Eigen::MatrixXd A_eq(2, 3);
        A_eq << 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0;
        Eigen::VectorXd b_eq{{3.0, 4.0}};
        auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);
        CHECK(result.status == qp_status::infeasible);
        CHECK(result.iterations == 0);
        REQUIRE(result.lambda.size() == 2);
        CHECK(result.lambda.isZero());
    }

    SECTION("a rank-zero row with a nonzero right-hand side")
    {
        Eigen::MatrixXd A_eq = Eigen::MatrixXd::Zero(1, 3);
        Eigen::VectorXd b_eq{{1.0}};
        auto result = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);
        CHECK(result.status == qp_status::infeasible);
        CHECK(result.iterations == 0);
    }
}

TEST_CASE("solve_qp does not return optimal for an infeasible terminal iterate", "[qp][qp_kkt]")
{
    // Hand-built terminal states in the free function's row layout (equality x1 = 1 first, then
    // inequality x2 >= 1), verified at the pinned floor sqrt(epsilon). The reference state
    // x = (1, 1), g = (-1, 1), lambda = (-1, 1) is feasible and stationary, and its negative
    // multiplier sits on the equality row, whose sign is free, so it satisfies all three conditions.
    const double tol = std::sqrt(std::numeric_limits<double>::epsilon());
    const Eigen::MatrixXd A = Eigen::MatrixXd::Identity(2, 2);
    const Eigen::VectorXd b{{1.0, 1.0}};
    const Eigen::VectorXd x{{1.0, 1.0}};
    Eigen::VectorXd g{{-1.0, 1.0}};
    Eigen::VectorXd lambda{{-1.0, 1.0}};
    Eigen::VectorXd scratch(2);
    const auto holds = [&] { return terminal_kkt_holds(A, b, x, g, lambda, 1, 2, tol, scratch); };
    REQUIRE(holds());

    SECTION("primal feasibility")
    {
        // x >= 2 alone, from x0 = 0: the gradient there is zero, the working set is empty, so the
        // first iteration takes no step and reaches the terminal branch at x = 0, which violates
        // the row by 2. The iterate is reported as a bounded-iteration outcome, not optimal.
        Eigen::MatrixXd G = Eigen::MatrixXd::Identity(1, 1);
        Eigen::VectorXd d{{0.0}};
        Eigen::MatrixXd A_eq(0, 1);
        Eigen::VectorXd b_eq(0);
        Eigen::MatrixXd A_in{{1.0}};
        Eigen::VectorXd b_in{{2.0}};
        auto result = solve_qp(G, d, A_eq, b_eq, A_in, b_in, Eigen::VectorXd::Zero(1).eval());
        CHECK(result.status == qp_status::max_iterations);
        CHECK(result.iterations == 0);
        CHECK(result.x[0] == 0.0);
    }

    SECTION("dual sign")
    {
        // A multiplier of -1 on the inequality row against a floor of -tol (1 + 1). The leaving
        // rule already rejects any multiplier below -opts.tolerance and the floor is never tighter,
        // so this state is reachable only by hand.
        g << -1.0, -1.0;
        lambda << -1.0, -1.0;
        CHECK_FALSE(holds());
    }

    SECTION("stationarity")
    {
        // g - A^T lambda = (0, 1e-3), far above tol (1 + 1.001); 1e-10 stays inside the bound.
        g << -1.0, 1.001;
        CHECK_FALSE(holds());
        g << -1.0, 1.0 + 1e-10;
        CHECK(holds());
    }

    SECTION("a correct terminal on an ill-conditioned working set")
    {
        // The equalities x1 = 1 and x1 + delta x2 = 1 + delta with delta = 2e-6 fix x1 = x2 = 1
        // and x3 = 0 minimizes 0.5 ||x||^2 - x1 over the free coordinate, so g = (0, 1, 0) and
        // A^T lambda = g gives lambda = (-1 / delta, 1 / delta). The stationarity residual carries
        // rounding of order epsilon / delta, above the 1e-12 inner tolerance but far below the
        // sqrt(epsilon) floor. Started at the solution, so no projection runs.
        const double delta = 2e-6;
        Eigen::MatrixXd G = Eigen::MatrixXd::Identity(3, 3);
        Eigen::VectorXd d{{-1.0, 0.0, 0.0}};
        Eigen::MatrixXd A_eq(2, 3);
        A_eq << 1.0, 0.0, 0.0,
                1.0, delta, 0.0;
        Eigen::VectorXd b_eq{{1.0, 1.0 + delta}};
        Eigen::MatrixXd A_in(0, 3);
        Eigen::VectorXd b_in(0);
        auto result = solve_qp(G, d, A_eq, b_eq, A_in, b_in, Eigen::VectorXd{{1.0, 1.0, 0.0}});
        REQUIRE(result.status == qp_status::optimal);
        CHECK(result.x[0] == Approx(1.0).margin(1e-10));
        CHECK(result.x[1] == Approx(1.0).margin(1e-10));
        CHECK(result.x[2] == Approx(0.0).margin(1e-10));
        CHECK(result.lambda[1] == Approx(1.0 / delta).epsilon(1e-6));
        CHECK(result.lambda[0] == Approx(-1.0 / delta).epsilon(1e-6));
    }
}

// Givens QR update tests.
// Reference: Golub & Van Loan, "Matrix Computations" 4th ed., Algorithm 5.1.3.
//            N&W Algorithm 16.3 (QR update in active-set context).

TEST_CASE("givens_rotation zeroes target element", "[qp][givens]")
{
    auto g = givens_rotation<double>::compute(3.0, 4.0);
    double r = g.c * 3.0 + g.s * 4.0;
    double z = -g.s * 3.0 + g.c * 4.0;
    CHECK(z == Approx(0.0).margin(1e-14));
    CHECK(r == Approx(5.0).margin(1e-14));
    CHECK(g.c * g.c + g.s * g.s == Approx(1.0).margin(1e-14));
}

TEST_CASE("givens_rotation edge cases", "[qp][givens]")
{
    SECTION("b = 0")
    {
        auto g = givens_rotation<double>::compute(5.0, 0.0);
        CHECK(g.c == 1.0);
        CHECK(g.s == 0.0);
    }
    SECTION("|b| > |a|")
    {
        auto g = givens_rotation<double>::compute(1.0, 10.0);
        double z = -g.s * 1.0 + g.c * 10.0;
        CHECK(z == Approx(0.0).margin(1e-14));
    }
    SECTION("negative values")
    {
        auto g = givens_rotation<double>::compute(-3.0, -4.0);
        double z = -g.s * (-3.0) + g.c * (-4.0);
        CHECK(z == Approx(0.0).margin(1e-14));
    }
}

TEST_CASE("add_constraint_qr preserves Q*R = A_W^T", "[qp][givens]")
{
    // Start with 2 constraints on a 4D problem, add a 3rd.
    Eigen::MatrixXd A(2, 4);
    A << 1.0, 0.0, 1.0, 0.0,
         0.0, 1.0, 0.0, 1.0;

    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A.transpose());
    Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(4, 4);
    auto Rqr = qr.matrixQR();
    for(int j = 0; j < 2; ++j)
        for(int i = 0; i <= j; ++i)
            R(i, j) = Rqr(i, j);

    // Verify initial factorization
    CHECK((Q * R.leftCols(2) - A.transpose()).norm() < 1e-12);

    // Add a rank-increasing constraint
    Eigen::VectorXd a_new(4);
    a_new << 1.0, 1.0, 0.0, 0.0;

    add_constraint_qr<double>(Q, R, a_new, 3);

    // Verify Q*R = [A^T | a_new]
    Eigen::MatrixXd AT_aug(4, 3);
    AT_aug.leftCols(2) = A.transpose();
    AT_aug.col(2) = a_new;
    CHECK((Q * R.leftCols(3) - AT_aug).norm() < 1e-12);

    // Verify Q orthogonality
    CHECK((Q.transpose() * Q - Eigen::MatrixXd::Identity(4, 4)).norm() < 1e-12);

    // Verify R is upper triangular
    for(int j = 0; j < 3; ++j)
        for(int i = j + 1; i < 4; ++i)
            CHECK(std::abs(R(i, j)) < 1e-12);

    // Verify null-space: A_aug * Z = 0
    auto Z = Q.rightCols(1);
    Eigen::MatrixXd A_aug(3, 4);
    A_aug.topRows(2) = A;
    A_aug.row(2) = a_new.transpose();
    CHECK((A_aug * Z).norm() < 1e-12);
}

TEST_CASE("remove_constraint_qr preserves Q*R = A_W^T", "[qp][givens]")
{
    // Start with 3 constraints on a 4D problem, remove the middle one.
    Eigen::MatrixXd A(3, 4);
    A << 1.0, 0.0, 1.0, 0.0,
         0.0, 1.0, 0.0, 1.0,
         1.0, 1.0, 0.0, 0.0;

    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A.transpose());
    Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(4, 4);
    auto Rqr = qr.matrixQR();
    for(int j = 0; j < 3; ++j)
        for(int i = 0; i <= j; ++i)
            R(i, j) = Rqr(i, j);

    CHECK((Q * R.leftCols(3) - A.transpose()).norm() < 1e-12);

    // Remove constraint 1 (the middle one)
    remove_constraint_qr<double>(Q, R, 1, 3);

    // After removal, A_W has rows 0 and 2 of original A
    Eigen::MatrixXd A_reduced(2, 4);
    A_reduced.row(0) = A.row(0);
    A_reduced.row(1) = A.row(2);
    CHECK((Q * R.leftCols(2) - A_reduced.transpose()).norm() < 1e-10);

    // Verify Q orthogonality
    CHECK((Q.transpose() * Q - Eigen::MatrixXd::Identity(4, 4)).norm() < 1e-10);

    // Verify null-space: A_reduced * Z = 0
    auto Z = Q.rightCols(2);
    CHECK((A_reduced * Z).norm() < 1e-10);
}

TEST_CASE("stateful solver with Givens matches free-function baseline", "[qp][givens]")
{
    // N&W Example 16.3 style problem solved by both paths.
    Eigen::MatrixXd G = 2.0 * Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd d{{-2.0, -5.0}};
    Eigen::MatrixXd A_eq(0, 2);
    Eigen::VectorXd b_eq(0);

    Eigen::MatrixXd A_ineq(5, 2);
    Eigen::VectorXd b_ineq(5);
    A_ineq << 1.0, -2.0,
             -1.0, -2.0,
             -1.0,  2.0,
              1.0,  0.0,
              0.0,  1.0;
    b_ineq << -2.0, -6.0, -2.0, 0.0, 0.0;

    Eigen::VectorXd x0{{2.0, 0.0}};

    // Free function (no Givens, uses the standalone solve_equality_qp)
    auto result_free = solve_qp(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);

    // Stateful solver (uses Givens QR updates)
    active_set_qp_solver<double> solver(2, 5);
    auto result_stateful = solver.solve(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);

    CHECK(result_free.status == qp_status::optimal);
    CHECK(result_stateful.status == qp_status::optimal);
    CHECK(result_stateful.x[0] == Approx(result_free.x[0]).margin(1e-10));
    CHECK(result_stateful.x[1] == Approx(result_free.x[1]).margin(1e-10));
}

TEST_CASE("active_set_qp_solver projects infeasible iter-0 onto equality manifold",
          "[qp][regression]")
{
    // Synthetic minimal reproducer: n=2, n_eq=2 forces the m>=n branch on
    // entry; x0=(0,0) violates b_eq=(1,1) so the iterate is infeasible
    // w.r.t. the working set. Pre-fix: solve_equality_subproblem returns
    // p=0 unconditionally on the m>=n branch, the active-set loop
    // terminates with x still at x0, and the post-condition
    // (A_eq * x == b_eq) fails. Post-fix: the solver projects x onto
    // the equality manifold before entering the active-set loop,
    // satisfying the feasibility invariant of N&W Algorithm 16.1
    // (N&W 2e Section 16.1, pp. 460-463).
    //
    // This fixture instantiates the stateful active_set_qp_solver
    // class directly and calls its public solve() member -- the same
    // code path consumed by the line-search SQP policies. The free
    // function solve_qp(...) is a separate implementation that carries
    // its own projection, covered by the solve_qp projection fixtures.
    //
    // Closed form: A_eq = I_2, b_eq = (1,1), G = I_2, d = 0, n_eq = n = 2.
    // The equality system fully determines x:
    //   x = x0 + A_eq^T (A_eq A_eq^T)^{-1} (b_eq - A_eq x0)
    //     = (0,0) + I (I)^{-1} ((1,1) - (0,0))
    //     = (1, 1).
    // With G = I_2, d = 0, the unconstrained QP minimum is at 0, but
    // the equality pins x to (1, 1) -- no remaining degree of freedom.
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd d = Eigen::VectorXd::Zero(2);
    Eigen::MatrixXd A_eq = Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd b_eq{{1.0, 1.0}};
    Eigen::MatrixXd A_ineq(0, 2);
    Eigen::VectorXd b_ineq(0);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(2);

    // n=2 unknowns, max_constraints=2 (room for the two equalities; no inequalities).
    active_set_qp_solver<double> solver(2, 2);
    auto result = solver.solve(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);

    CHECK(result.status == qp_status::optimal);
    CHECK((A_eq * result.x - b_eq).cwiseAbs().maxCoeff() < 1e-8);
    CHECK(result.x[0] == Approx(1.0).margin(1e-10));
    CHECK(result.x[1] == Approx(1.0).margin(1e-10));
}

TEST_CASE("active_set_qp_solver escalates to a pivoted projection on duplicated equality rows",
          "[qp][qp_rank]")
{
    // min 0.5 ||x||^2 + d^T x with d = (-1, 0, 1), subject to x1 + x2 + x3 = 3
    // stated twice. The equality block has rank 1 but is consistent, so the
    // duplicate row changes nothing: stationarity x + d = t (1, 1, 1) gives
    // x = (1 + t, t, t - 1), and the constraint gives 3t = 3, so t = 1 and
    // x* = (2, 1, 0). From x0 = 0 the residual (3, 3) forces the phase-1
    // projection, whose unpivoted factor has an (exactly or nearly) zero
    // second diagonal entry.
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(3, 3);
    Eigen::VectorXd d{{-1.0, 0.0, 1.0}};
    Eigen::MatrixXd A_eq(2, 3);
    A_eq << 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0;
    Eigen::VectorXd b_eq{{3.0, 3.0}};
    Eigen::MatrixXd A_ineq(0, 3);
    Eigen::VectorXd b_ineq(0);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(3);

    active_set_qp_solver<double> solver(3, 2);
    auto result = solver.solve(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);

    REQUIRE(result.status == qp_status::optimal);
    CHECK((A_eq * result.x - b_eq).cwiseAbs().maxCoeff() < 1e-12);
    CHECK(result.x[0] == Approx(2.0).margin(1e-10));
    CHECK(result.x[1] == Approx(1.0).margin(1e-10));
    CHECK(result.x[2] == Approx(0.0).margin(1e-10));
}

TEST_CASE("active_set_qp_solver reports infeasible on contradictory equality rows",
          "[qp][qp_rank]")
{
    // x1 + x2 + x3 = 3 and x1 + x2 + x3 = 4 have identical rows and different
    // right-hand sides: (3, 4) is not in the range of A_eq = (1, 1)^T (1, 1, 1),
    // so no x satisfies both. Any minimum-norm correction satisfies at most one
    // row, leaving a residual of 1 against a bound of tol (1 + 4) = 5e-12. The
    // solver must report infeasible from the projection, before any active-set
    // iteration, instead of optimizing over one of the two rows.
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(3, 3);
    Eigen::VectorXd d{{-1.0, 0.0, 1.0}};
    Eigen::MatrixXd A_eq(2, 3);
    A_eq << 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0;
    Eigen::VectorXd b_eq{{3.0, 4.0}};
    Eigen::MatrixXd A_ineq(0, 3);
    Eigen::VectorXd b_ineq(0);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(3);

    active_set_qp_solver<double> solver(3, 2);
    auto result = solver.solve(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);

    CHECK(result.status == qp_status::infeasible);
    CHECK(result.iterations == 0);
    REQUIRE(result.lambda.size() == 2);
    CHECK(result.lambda.isZero());
}

TEST_CASE("active_set_qp_solver handles an empty and a rank-zero equality block",
          "[qp][qp_rank]")
{
    // Shared objective: min 0.5 ||x||^2 + d^T x with d = (-1, 0, 1), whose
    // unconstrained minimizer is -d = (1, 0, -1).
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(3, 3);
    Eigen::VectorXd d{{-1.0, 0.0, 1.0}};
    Eigen::MatrixXd A_ineq(0, 3);
    Eigen::VectorXd b_ineq(0);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(3);

    SECTION("no equality rows")
    {
        // Only x1 >= 2, started feasible at (3, 0, 0). It cuts off the
        // unconstrained minimizer, so x* = (2, 0, -1) with multiplier
        // g1 = x1* - 1 = 1 on the active row.
        Eigen::MatrixXd A_eq(0, 3);
        Eigen::VectorXd b_eq(0);
        Eigen::MatrixXd A_in(1, 3);
        A_in << 1.0, 0.0, 0.0;
        Eigen::VectorXd b_in{{2.0}};
        active_set_qp_solver<double> solver(3, 1);
        auto result = solver.solve(G, d, A_eq, b_eq, A_in, b_in, Eigen::VectorXd{{3.0, 0.0, 0.0}});
        REQUIRE(result.status == qp_status::optimal);
        CHECK(result.x[0] == Approx(2.0).margin(1e-10));
        CHECK(result.x[1] == Approx(0.0).margin(1e-10));
        CHECK(result.x[2] == Approx(-1.0).margin(1e-10));
        CHECK(result.lambda[0] == Approx(1.0).margin(1e-10));
    }

    SECTION("a rank-zero equality row with a zero right-hand side")
    {
        // 0^T x = 0 holds for every x, so the row constrains nothing and the
        // solution is the unconstrained minimizer (1, 0, -1).
        Eigen::MatrixXd A_eq = Eigen::MatrixXd::Zero(1, 3);
        Eigen::VectorXd b_eq{{0.0}};
        active_set_qp_solver<double> solver(3, 1);
        auto result = solver.solve(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);
        REQUIRE(result.status == qp_status::optimal);
        CHECK(result.x[0] == Approx(1.0).margin(1e-10));
        CHECK(result.x[1] == Approx(0.0).margin(1e-10));
        CHECK(result.x[2] == Approx(-1.0).margin(1e-10));
    }

    SECTION("a rank-zero equality row with a nonzero right-hand side")
    {
        // 0^T x = 1 holds for no x: the residual 1 lies outside the (empty)
        // range of A_eq, which is a demonstration of infeasibility.
        Eigen::MatrixXd A_eq = Eigen::MatrixXd::Zero(1, 3);
        Eigen::VectorXd b_eq{{1.0}};
        active_set_qp_solver<double> solver(3, 1);
        auto result = solver.solve(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);
        CHECK(result.status == qp_status::infeasible);
        CHECK(result.iterations == 0);
    }

    SECTION("a single equality row")
    {
        // x1 + x2 + x3 = 3 alone: the same stationarity argument as the
        // duplicated-row case gives x* = (2, 1, 0) with multiplier 1.
        Eigen::MatrixXd A_eq(1, 3);
        A_eq << 1.0, 1.0, 1.0;
        Eigen::VectorXd b_eq{{3.0}};
        active_set_qp_solver<double> solver(3, 1);
        auto result = solver.solve(G, d, A_eq, b_eq, A_ineq, b_ineq, x0);
        REQUIRE(result.status == qp_status::optimal);
        CHECK(result.x[0] == Approx(2.0).margin(1e-10));
        CHECK(result.x[1] == Approx(1.0).margin(1e-10));
        CHECK(result.x[2] == Approx(0.0).margin(1e-10));
        CHECK(result.lambda[0] == Approx(1.0).margin(1e-10));
    }

    SECTION("a zero-length decision vector")
    {
        // With no variables and no rows the empty point is trivially optimal.
        Eigen::MatrixXd G0(0, 0);
        Eigen::VectorXd e(0);
        Eigen::MatrixXd A0(0, 0);
        active_set_qp_solver<double> solver(0, 0);
        auto result = solver.solve(G0, e, A0, e, A0, e, e);
        CHECK(result.status == qp_status::optimal);
        CHECK(result.x.size() == 0);
        CHECK(result.lambda.size() == 0);
    }
}

TEST_CASE("active_set_qp_solver does not return optimal for an infeasible terminal iterate",
          "[qp][qp_kkt]")
{
    // Hand-built terminal states for min 0.5 ||x||^2 + d^T x s.t. x1 + x2 >= 1, verified at the
    // pinned floor sqrt(epsilon). The reference state x = (0.5, 0.5), lambda = 0.5, g = (0.5, 0.5)
    // satisfies all three conditions exactly; each section breaks one of them.
    const double tol = std::sqrt(std::numeric_limits<double>::epsilon());
    Eigen::MatrixXd A(1, 2);
    A << 1.0, 1.0;
    const Eigen::VectorXd b{{1.0}};
    Eigen::VectorXd x{{0.5, 0.5}};
    Eigen::VectorXd g{{0.5, 0.5}};
    Eigen::VectorXd lambda{{0.5}};
    Eigen::VectorXd scratch(2);
    const auto holds = [&] { return terminal_kkt_holds(A, b, x, g, lambda, 0, 1, tol, scratch); };
    REQUIRE(holds());

    SECTION("primal feasibility")
    {
        // x = (0.4, 0.4) violates the row by 0.2 against a bound of tol (1 + 0.8 + 1).
        x << 0.4, 0.4;
        g << 0.4, 0.4;
        lambda << 0.0;
        CHECK_FALSE(holds());

        // Through the solver: x >= 1 and -x >= 0 from x0 = 0.5, where both rows are violated by
        // 0.5. The first step blocks on x >= 1 at alpha = 0, the working set {x >= 1} then
        // fills the single dimension, p = 0, and its multiplier g / 1 = 0.5 is nonnegative, so
        // the loop reaches its terminal branch at x = 0.5 with both rows still violated. That
        // iterate has no infeasibility certificate, so it is reported as a bounded-iteration
        // outcome rather than optimal or infeasible.
        Eigen::MatrixXd G = Eigen::MatrixXd::Identity(1, 1);
        Eigen::VectorXd d{{0.0}};
        Eigen::MatrixXd A_eq(0, 1);
        Eigen::VectorXd b_eq(0);
        Eigen::MatrixXd A_in(2, 1);
        A_in << 1.0, -1.0;
        Eigen::VectorXd b_in{{1.0, 0.0}};
        active_set_qp_solver<double> solver(1, 2);
        auto result = solver.solve(G, d, A_eq, b_eq, A_in, b_in, Eigen::VectorXd{{0.5}});
        CHECK(result.status == qp_status::max_iterations);
        CHECK(result.x[0] == Approx(0.5).margin(1e-12));
    }

    SECTION("dual sign")
    {
        // Stationary and feasible, but the inequality multiplier is -0.5 against a floor of
        // -tol (1 + 0.5): the point is a constrained maximizer of the linearization, not a
        // minimizer. The solver's own leaving rule already rejects any multiplier below
        // -opts.tolerance, and the floor is never tighter than that, so this state is only
        // reachable by hand.
        g << -0.5, -0.5;
        lambda << -0.5;
        CHECK_FALSE(holds());
    }

    SECTION("stationarity")
    {
        // g - A^T lambda = (0, 1e-3), far above tol (1 + 0.5001); a perturbation of 1e-10
        // stays inside that relative bound.
        g << 0.5, 0.501;
        CHECK_FALSE(holds());
        g << 0.5, 0.5 + 1e-10;
        CHECK(holds());
    }

    SECTION("a correct terminal on an ill-conditioned working set")
    {
        // Equalities x1 = 1 and x1 + delta x2 = 1 + delta with delta = 2e-6 (condition number
        // about 1e6), objective 0.5 ||x||^2 - x1. The equalities fix x1 = x2 = 1, x3 = 0
        // minimizes the free coordinate, and g = x + d = (0, 1, 0). Stationarity
        // A^T lambda = g gives delta lambda2 = 1 and lambda1 = -lambda2, so lambda = (-5e5, 5e5):
        // multipliers of order 1 / delta cancel to an order-one gradient, and the computed
        // stationarity residual carries rounding of order epsilon / delta, above the 1e-12
        // inner tolerance but far below the sqrt(epsilon) floor. Started at the solution, so no
        // projection runs and the terminal is reached on the first iteration.
        const double delta = 2e-6;
        Eigen::MatrixXd G = Eigen::MatrixXd::Identity(3, 3);
        Eigen::VectorXd d{{-1.0, 0.0, 0.0}};
        Eigen::MatrixXd A_eq(2, 3);
        A_eq << 1.0, 0.0, 0.0,
                1.0, delta, 0.0;
        Eigen::VectorXd b_eq{{1.0, 1.0 + delta}};
        Eigen::MatrixXd A_in(0, 3);
        Eigen::VectorXd b_in(0);
        active_set_qp_solver<double> solver(3, 2);
        auto result = solver.solve(G, d, A_eq, b_eq, A_in, b_in, Eigen::VectorXd{{1.0, 1.0, 0.0}});
        REQUIRE(result.status == qp_status::optimal);
        CHECK(result.x[0] == Approx(1.0).margin(1e-10));
        CHECK(result.x[1] == Approx(1.0).margin(1e-10));
        CHECK(result.x[2] == Approx(0.0).margin(1e-10));
        CHECK(result.lambda[1] == Approx(1.0 / delta).epsilon(1e-6));
        CHECK(result.lambda[0] == Approx(-1.0 / delta).epsilon(1e-6));
    }
}

TEST_CASE("active-set projection judges a large start by its backward error", "[qp][qp_rank]")
{
    // One consistent row through the origin, so the minimizer of 0.5 ||x||^2 is x = 0. From a
    // start of magnitude s the projected residual is rounding of order epsilon * s, which an
    // absolute tolerance would read as an inconsistent system.
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(3, 3);
    Eigen::VectorXd d = Eigen::VectorXd::Zero(3);
    Eigen::MatrixXd A_eq(1, 3);
    A_eq << 0.3, 0.7, 1.1;
    const Eigen::VectorXd b_eq{{0.0}};
    Eigen::MatrixXd A_in(0, 3);
    Eigen::VectorXd b_in(0);
    const Eigen::VectorXd direction{{1.37, -0.91, 0.53}};
    const auto solve_both = [&](double scale) {
        active_set_qp_solver<double> solver(3, 1);
        const Eigen::VectorXd x0 = scale * direction;
        return std::pair{solver.solve(G, d, A_eq, b_eq, A_in, b_in, x0),
                         solve_qp(G, d, A_eq, b_eq, A_in, b_in, x0)};
    };

    for(const double scale : {1e5, 1e6})
    {
        const auto [stateful, free] = solve_both(scale);
        for(const auto& result : {stateful, free})
        {
            CHECK(result.status == qp_status::optimal);
            CHECK(result.x.lpNorm<Eigen::Infinity>() < 1e-12 * scale);
        }
    }

    // At 1e8 the terminal check still bounds the residual by the landed x ~ 0 rather than the
    // cancelled start, so the correct point is reported unverified, never infeasible.
    const auto [stateful, free] = solve_both(1e8);
    CHECK(stateful.status == qp_status::max_iterations);
    CHECK(free.status == qp_status::max_iterations);
}

TEST_CASE("active-set terminal verification is not excused by a far bound on another row",
          "[qp][qp_kkt]")
{
    // The contradictory pair x1 >= 1, -x1 >= 0 from x1 = 0.5 ends unverified; adding the slack row
    // x2 >= -1e30 must not lend its magnitude to the violated rows.
    Eigen::MatrixXd G = Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd d = Eigen::VectorXd::Zero(2);
    Eigen::MatrixXd A_eq(0, 2);
    Eigen::VectorXd b_eq(0);
    Eigen::MatrixXd A_in(3, 2);
    A_in << 1.0, 0.0,
            -1.0, 0.0,
            0.0, 1.0;
    const Eigen::VectorXd b_in{{1.0, 0.0, -1e30}};
    const Eigen::VectorXd x0{{0.5, 0.0}};
    active_set_qp_solver<double> solver(2, 3);
    CHECK(solver.solve(G, d, A_eq, b_eq, A_in, b_in, x0).status == qp_status::max_iterations);
    CHECK(solve_qp(G, d, A_eq, b_eq, A_in, b_in, x0).status == qp_status::max_iterations);
}

TEST_CASE("active-set step test scales with the iterate", "[qp][qp_kkt]")
{
    // The unconstrained minimizer c is cut off by x1 >= 0.5 s, so the solution sits on that row and
    // the equality. At magnitude s a null step is rounding of order epsilon * s, which an absolute
    // step test never accepts, so the solver cycled to its iteration cap.
    Eigen::MatrixXd G(3, 3);
    G << 2.0, 0.6, 0.1,
         0.6, 1.0, 0.2,
         0.1, 0.2, 1.5;
    Eigen::MatrixXd A_eq(1, 3);
    A_eq << 0.3, 1.1, -0.5;
    const Eigen::VectorXd b_eq{{0.0}};
    Eigen::MatrixXd A_in(2, 3);
    A_in << -1.0, 0.0, 0.0,
            0.0, 0.0, 1.0;
    const Eigen::Vector3d c{1.3, -0.7, 0.4};
    const Eigen::Vector3d starts[] = {{0.0, 0.0, 0.0}, {0.1, 0.2, 0.1}, {-0.37, 0.11, 0.02}};

    for(const double s : {1e5, 1e6, 1e7})
    {
        const Eigen::VectorXd d = -G * (s * c);
        const Eigen::VectorXd b_in{{-0.5 * s, -10.0 * s}};
        for(const Eigen::Vector3d& start : starts)
        {
            const Eigen::VectorXd x0 = s * start;
            active_set_qp_solver<double> solver(3, 3);
            const auto stateful = solver.solve(G, d, A_eq, b_eq, A_in, b_in, x0);
            const auto free = solve_qp(G, d, A_eq, b_eq, A_in, b_in, x0);
            for(const auto& result : {stateful, free})
            {
                CHECK(result.status == qp_status::optimal);
                CHECK(result.iterations < 5);
                CHECK(result.x[0] == Approx(0.5 * s).epsilon(1e-12));
            }
        }
    }
}
