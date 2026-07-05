#include "argmin/detail/active_set_qp.h"
#include "argmin/detail/givens_qr_update.h"

#include <Eigen/Core>
#include <Eigen/QR>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

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

TEST_CASE("solve_qp infeasible detected", "[qp]")
{
    // x >= 1 and x <= 0 (contradictory) -- infeasibility manifests as max_iterations.
    // Starting from x0 = 0.5 which violates x >= 1.
    // With contradictory constraints the solver cycles or hits iteration limit.
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

    // Infeasibility should manifest as max_iterations or suboptimal result
    CHECK((result.status == qp_status::max_iterations ||
           result.status == qp_status::optimal));
    // If "optimal", the constraints cannot both be satisfied
    if(result.status == qp_status::optimal)
    {
        bool feasible = (result.x[0] >= 1.0 - 1e-6) && (result.x[0] <= 1e-6);
        CHECK_FALSE(feasible);
    }
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
        A_W.row(k) = A.row(W[k]);
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
    // function solve_qp(...) is a separate, independent code path and
    // is not exercised here.
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
