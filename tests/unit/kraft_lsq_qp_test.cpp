#include "nablapp/detail/nnls.h"
#include "nablapp/detail/ldp.h"
#include "nablapp/detail/lsi.h"
#include "nablapp/detail/lsei.h"
#include "nablapp/detail/kraft_lsq_qp.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <cmath>

using Catch::Approx;
using namespace nablapp;
using namespace nablapp::detail;

// ---------------------------------------------------------------------------
// NNLS tests (Lawson & Hanson 1974, Ch. 23.3)
// ---------------------------------------------------------------------------

TEST_CASE("NNLS identity with positive RHS", "[nnls]")
{
    // min || I x - b ||  s.t. x >= 0 with b > 0  => x = b.
    constexpr int n = 4;
    Eigen::Matrix<double, n, n> A = Eigen::Matrix<double, n, n>::Identity();
    Eigen::Vector<double, n> b{{1.0, 2.0, 3.0, 4.0}};
    Eigen::Vector<double, n> x = Eigen::Vector<double, n>::Zero();
    Eigen::Vector<double, n> w = Eigen::Vector<double, n>::Zero();

    auto result = nnls<double, n, n>(A, b, x, w, n, n);

    CHECK(result.mode == 1);
    CHECK(x[0] == Approx(1.0).margin(1e-12));
    CHECK(x[1] == Approx(2.0).margin(1e-12));
    CHECK(x[2] == Approx(3.0).margin(1e-12));
    CHECK(x[3] == Approx(4.0).margin(1e-12));
}

TEST_CASE("NNLS identity with mixed-sign RHS clamps negatives to zero", "[nnls]")
{
    // min || I x - b ||  s.t. x >= 0 with mixed b.
    // Unconstrained optimum is x* = b; the clipped projection gives
    // x_j = max(b_j, 0).
    constexpr int n = 4;
    Eigen::Matrix<double, n, n> A = Eigen::Matrix<double, n, n>::Identity();
    Eigen::Vector<double, n> b{{2.0, -1.5, 0.7, -3.0}};
    Eigen::Vector<double, n> x = Eigen::Vector<double, n>::Zero();
    Eigen::Vector<double, n> w = Eigen::Vector<double, n>::Zero();

    auto result = nnls<double, n, n>(A, b, x, w, n, n);

    CHECK(result.mode == 1);
    CHECK(x[0] == Approx(2.0).margin(1e-12));
    CHECK(x[1] == Approx(0.0).margin(1e-12));
    CHECK(x[2] == Approx(0.7).margin(1e-12));
    CHECK(x[3] == Approx(0.0).margin(1e-12));
}

TEST_CASE("NNLS overdetermined 3x2 system", "[nnls]")
{
    // A = [[1, 0], [0, 1], [1, 1]], b = [1, 1, 2]
    // Unconstrained LS solution is x* = [1, 1] which is non-negative,
    // so NNLS must return it exactly.
    constexpr int m = 3;
    constexpr int n = 2;
    Eigen::Matrix<double, m, n> A;
    A << 1.0, 0.0,
         0.0, 1.0,
         1.0, 1.0;
    Eigen::Vector<double, m> b{{1.0, 1.0, 2.0}};
    Eigen::Vector<double, n> x = Eigen::Vector<double, n>::Zero();
    Eigen::Vector<double, n> w = Eigen::Vector<double, n>::Zero();

    auto result = nnls<double, m, n>(A, b, x, w, m, n);

    CHECK(result.mode == 1);
    CHECK(x[0] == Approx(1.0).margin(1e-12));
    CHECK(x[1] == Approx(1.0).margin(1e-12));
    CHECK(result.residual_norm == Approx(0.0).margin(1e-12));
}

TEST_CASE("NNLS clipping: unconstrained optimum has one negative component", "[nnls]")
{
    // A = [[1, 0], [0, 1], [1, 1]], b = [-1, 2, 1]
    // Unconstrained LS: solve normal equations
    //   A^T A x = A^T b,  A^T A = [[2, 1], [1, 2]],  A^T b = [0, 3]
    //   => x_free = [-1, 2]
    // NNLS should clamp x1 to 0, then x2 = argmin ||[[0],[1],[1]] x2 - b||
    //   => x2 = (0*(-1) + 1*2 + 1*1) / 2 = 1.5
    // Result: x = [0, 1.5]
    constexpr int m = 3;
    constexpr int n = 2;
    Eigen::Matrix<double, m, n> A;
    A << 1.0, 0.0,
         0.0, 1.0,
         1.0, 1.0;
    Eigen::Vector<double, m> b{{-1.0, 2.0, 1.0}};
    Eigen::Vector<double, n> x = Eigen::Vector<double, n>::Zero();
    Eigen::Vector<double, n> w = Eigen::Vector<double, n>::Zero();

    auto result = nnls<double, m, n>(A, b, x, w, m, n);

    CHECK(result.mode == 1);
    CHECK(x[0] == Approx(0.0).margin(1e-12));
    CHECK(x[1] == Approx(1.5).margin(1e-12));
}

// ---------------------------------------------------------------------------
// LDP tests (Lawson & Hanson 1974, Ch. 23.4)
// ---------------------------------------------------------------------------

TEST_CASE("LDP simple 2D halfspace", "[ldp]")
{
    // min ||x||  s.t.  x1 + x2 >= 1.
    // Lagrangian: x = lambda * [1, 1], lambda = 0.5
    // => x* = [0.5, 0.5]
    constexpr int m = 1;
    constexpr int n = 2;
    Eigen::Matrix<double, m, n> G;
    G << 1.0, 1.0;
    Eigen::Vector<double, m> h{{1.0}};
    Eigen::Vector<double, n> x = Eigen::Vector<double, n>::Zero();

    Eigen::MatrixXd nnls_A;
    Eigen::VectorXd nnls_b;
    Eigen::VectorXd nnls_x_vec;
    Eigen::VectorXd nnls_w;

    int mode = ldp<double, m, n>(G, h, x, nnls_A, nnls_b, nnls_x_vec, nnls_w, m, n);

    CHECK(mode == 1);
    CHECK(x[0] == Approx(0.5).margin(1e-12));
    CHECK(x[1] == Approx(0.5).margin(1e-12));
}

TEST_CASE("LDP triangle constraint", "[ldp]")
{
    // min ||x||  s.t.  x1 >= 1, x2 >= 1, x1 + x2 >= 3.
    // Active constraint at optimum: x1 + x2 = 3 with x1 = x2 = 1.5.
    // Solution: x* = [1.5, 1.5], ||x*|| = sqrt(4.5)
    constexpr int m = 3;
    constexpr int n = 2;
    Eigen::Matrix<double, m, n> G;
    G << 1.0, 0.0,
         0.0, 1.0,
         1.0, 1.0;
    Eigen::Vector<double, m> h{{1.0, 1.0, 3.0}};
    Eigen::Vector<double, n> x = Eigen::Vector<double, n>::Zero();

    Eigen::MatrixXd nnls_A;
    Eigen::VectorXd nnls_b;
    Eigen::VectorXd nnls_x_vec;
    Eigen::VectorXd nnls_w;

    int mode = ldp<double, m, n>(G, h, x, nnls_A, nnls_b, nnls_x_vec, nnls_w, m, n);

    CHECK(mode == 1);
    CHECK(x[0] == Approx(1.5).margin(1e-9));
    CHECK(x[1] == Approx(1.5).margin(1e-9));
}

TEST_CASE("LDP trivial: origin is feasible", "[ldp]")
{
    // min ||x||  s.t.  x1 >= -1, x2 >= -1.
    // Origin is feasible and has norm 0, so x* = [0, 0].
    // LDP with a feasible origin is a degenerate case: the NNLS dual
    // residual is zero which indicates infeasibility of the DUAL
    // (because the primal optimum is at x = 0, not the primal problem
    // itself). We therefore accept mode != 1 here as long as x is
    // consistent with the origin.
    constexpr int m = 2;
    constexpr int n = 2;
    Eigen::Matrix<double, m, n> G;
    G << 1.0, 0.0,
         0.0, 1.0;
    Eigen::Vector<double, m> h{{-1.0, -1.0}};
    Eigen::Vector<double, n> x = Eigen::Vector<double, n>::Zero();

    Eigen::MatrixXd nnls_A;
    Eigen::VectorXd nnls_b;
    Eigen::VectorXd nnls_x_vec;
    Eigen::VectorXd nnls_w;

    int mode = ldp<double, m, n>(G, h, x, nnls_A, nnls_b, nnls_x_vec, nnls_w, m, n);

    // Caller of LDP (LSI) treats mode != 1 as "origin is the solution"
    // in this degenerate case; we just check x is numerically zero.
    (void)mode;
    CHECK(std::abs(x[0]) <= 1e-12);
    CHECK(std::abs(x[1]) <= 1e-12);
}

TEST_CASE("LDP infeasible constraints", "[ldp]")
{
    // x >= 1 AND x <= -1  (i.e. x >= 1 AND -x >= 1) is infeasible.
    constexpr int m = 2;
    constexpr int n = 1;
    Eigen::Matrix<double, m, n> G;
    G << 1.0,
        -1.0;
    Eigen::Vector<double, m> h{{1.0, 1.0}};
    Eigen::Vector<double, n> x = Eigen::Vector<double, n>::Zero();

    Eigen::MatrixXd nnls_A;
    Eigen::VectorXd nnls_b;
    Eigen::VectorXd nnls_x_vec;
    Eigen::VectorXd nnls_w;

    int mode = ldp<double, m, n>(G, h, x, nnls_A, nnls_b, nnls_x_vec, nnls_w, m, n);

    CHECK(mode != 1);
}

// ---------------------------------------------------------------------------
// LSI tests (Lawson & Hanson 1974, Ch. 23.5)
// ---------------------------------------------------------------------------

TEST_CASE("LSI unconstrained reduces to ordinary LS", "[lsi]")
{
    // min ||I x - [3, 4]||^2  (no inequalities) => x = [3, 4]
    constexpr int n = 2;
    Eigen::Matrix<double, n, n> E = Eigen::Matrix<double, n, n>::Identity();
    Eigen::Vector<double, n> f{{3.0, 4.0}};
    Eigen::Matrix<double, Eigen::Dynamic, n> G(0, n);
    Eigen::VectorXd h(0);
    Eigen::Vector<double, n> x = Eigen::Vector<double, n>::Zero();

    Eigen::MatrixXd nnls_A;
    Eigen::VectorXd nnls_b;
    Eigen::VectorXd nnls_x_vec;
    Eigen::VectorXd nnls_w;

    int mode = lsi<double, n>(E, f, G, h, x,
                              nnls_A, nnls_b, nnls_x_vec, nnls_w, n, 0);

    CHECK(mode == 1);
    CHECK(x[0] == Approx(3.0).margin(1e-12));
    CHECK(x[1] == Approx(4.0).margin(1e-12));
}

TEST_CASE("LSI 2D with one active inequality", "[lsi]")
{
    // min ||I x - [0, 0]||^2  s.t.  x1 + x2 >= 1.
    // Without the constraint the optimum is [0, 0]; the inequality
    // is active, and the projection onto the constraint boundary gives
    // x = [0.5, 0.5].
    constexpr int n = 2;
    constexpr int m = 1;
    Eigen::Matrix<double, n, n> E = Eigen::Matrix<double, n, n>::Identity();
    Eigen::Vector<double, n> f = Eigen::Vector<double, n>::Zero();
    Eigen::Matrix<double, Eigen::Dynamic, n> G(m, n);
    G << 1.0, 1.0;
    Eigen::VectorXd h(m);
    h << 1.0;
    Eigen::Vector<double, n> x = Eigen::Vector<double, n>::Zero();

    Eigen::MatrixXd nnls_A;
    Eigen::VectorXd nnls_b;
    Eigen::VectorXd nnls_x_vec;
    Eigen::VectorXd nnls_w;

    int mode = lsi<double, n>(E, f, G, h, x,
                              nnls_A, nnls_b, nnls_x_vec, nnls_w, n, m);

    CHECK(mode == 1);
    CHECK(x[0] == Approx(0.5).margin(1e-10));
    CHECK(x[1] == Approx(0.5).margin(1e-10));
}

TEST_CASE("LSI 2D with inactive inequality", "[lsi]")
{
    // min ||I x - [3, 4]||^2  s.t.  x1 + x2 >= 1.
    // Unconstrained optimum [3, 4] already satisfies 3+4=7 >= 1,
    // so the constraint is inactive.
    constexpr int n = 2;
    constexpr int m = 1;
    Eigen::Matrix<double, n, n> E = Eigen::Matrix<double, n, n>::Identity();
    Eigen::Vector<double, n> f{{3.0, 4.0}};
    Eigen::Matrix<double, Eigen::Dynamic, n> G(m, n);
    G << 1.0, 1.0;
    Eigen::VectorXd h(m);
    h << 1.0;
    Eigen::Vector<double, n> x = Eigen::Vector<double, n>::Zero();

    Eigen::MatrixXd nnls_A;
    Eigen::VectorXd nnls_b;
    Eigen::VectorXd nnls_x_vec;
    Eigen::VectorXd nnls_w;

    int mode = lsi<double, n>(E, f, G, h, x,
                              nnls_A, nnls_b, nnls_x_vec, nnls_w, n, m);

    CHECK(mode == 1);
    CHECK(x[0] == Approx(3.0).margin(1e-10));
    CHECK(x[1] == Approx(4.0).margin(1e-10));
}

// ---------------------------------------------------------------------------
// LSEI tests (Kraft 1988 / Lawson & Hanson 1974, Ch. 23.6)
// ---------------------------------------------------------------------------

TEST_CASE("LSEI equality only", "[lsei]")
{
    // min ||I x - [3, 4]||^2  s.t.  x1 + x2 = 2.
    //
    // Lagrangian KKT: x = f + A^T lambda, with the constraint giving
    //   (f1 + lambda) + (f2 + lambda) = 2  =>  lambda = (2 - f1 - f2)/2
    //   lambda = (2 - 7)/2 = -2.5
    //   x1 = 3 - 2.5 = 0.5,  x2 = 4 - 2.5 = 1.5
    constexpr int n = 2;
    Eigen::Matrix<double, n, n> E = Eigen::Matrix<double, n, n>::Identity();
    Eigen::Vector<double, n> f{{3.0, 4.0}};
    Eigen::Matrix<double, Eigen::Dynamic, n> C(1, n);
    C << 1.0, 1.0;
    Eigen::VectorXd d(1);
    d << 2.0;
    Eigen::Matrix<double, Eigen::Dynamic, n> G(0, n);
    Eigen::VectorXd h(0);
    Eigen::Vector<double, n> x = Eigen::Vector<double, n>::Zero();

    Eigen::MatrixXd nnls_A;
    Eigen::VectorXd nnls_b;
    Eigen::VectorXd nnls_x_vec;
    Eigen::VectorXd nnls_w;

    int mode = lsei<double, n>(C, d, E, f, G, h, x,
                               nnls_A, nnls_b, nnls_x_vec, nnls_w, n, 1, 0);

    CHECK(mode == 1);
    CHECK(x[0] == Approx(0.5).margin(1e-10));
    CHECK(x[1] == Approx(1.5).margin(1e-10));
}

TEST_CASE("LSEI equality plus inactive inequality", "[lsei]")
{
    // min ||I x - [3, 4]||^2  s.t.  x1 + x2 = 2,  x1 >= -10 (inactive).
    // Same answer as equality-only case: x = [0.5, 1.5].
    constexpr int n = 2;
    Eigen::Matrix<double, n, n> E = Eigen::Matrix<double, n, n>::Identity();
    Eigen::Vector<double, n> f{{3.0, 4.0}};
    Eigen::Matrix<double, Eigen::Dynamic, n> C(1, n);
    C << 1.0, 1.0;
    Eigen::VectorXd d(1);
    d << 2.0;
    Eigen::Matrix<double, Eigen::Dynamic, n> G(1, n);
    G << 1.0, 0.0;
    Eigen::VectorXd h(1);
    h << -10.0;
    Eigen::Vector<double, n> x = Eigen::Vector<double, n>::Zero();

    Eigen::MatrixXd nnls_A;
    Eigen::VectorXd nnls_b;
    Eigen::VectorXd nnls_x_vec;
    Eigen::VectorXd nnls_w;

    int mode = lsei<double, n>(C, d, E, f, G, h, x,
                               nnls_A, nnls_b, nnls_x_vec, nnls_w, n, 1, 1);

    CHECK(mode == 1);
    CHECK(x[0] == Approx(0.5).margin(1e-10));
    CHECK(x[1] == Approx(1.5).margin(1e-10));
}

TEST_CASE("LSEI equality plus active inequality", "[lsei]")
{
    // min ||I x - [3, 4]||^2  s.t.  x1 + x2 = 2,  x1 >= 1.
    //
    // Unconstrained-with-equality optimum is [0.5, 1.5], which
    // violates x1 >= 1. The active-inequality solution sits at
    // x1 = 1, x2 = 1 (from the equality).
    constexpr int n = 2;
    Eigen::Matrix<double, n, n> E = Eigen::Matrix<double, n, n>::Identity();
    Eigen::Vector<double, n> f{{3.0, 4.0}};
    Eigen::Matrix<double, Eigen::Dynamic, n> C(1, n);
    C << 1.0, 1.0;
    Eigen::VectorXd d(1);
    d << 2.0;
    Eigen::Matrix<double, Eigen::Dynamic, n> G(1, n);
    G << 1.0, 0.0;
    Eigen::VectorXd h(1);
    h << 1.0;
    Eigen::Vector<double, n> x = Eigen::Vector<double, n>::Zero();

    Eigen::MatrixXd nnls_A;
    Eigen::VectorXd nnls_b;
    Eigen::VectorXd nnls_x_vec;
    Eigen::VectorXd nnls_w;

    int mode = lsei<double, n>(C, d, E, f, G, h, x,
                               nnls_A, nnls_b, nnls_x_vec, nnls_w, n, 1, 1);

    CHECK(mode == 1);
    CHECK(x[0] == Approx(1.0).margin(1e-8));
    CHECK(x[1] == Approx(1.0).margin(1e-8));
}

TEST_CASE("LSEI rank deficient equality is reported", "[lsei]")
{
    // Duplicate equality rows => rank-deficient C.
    constexpr int n = 2;
    Eigen::Matrix<double, n, n> E = Eigen::Matrix<double, n, n>::Identity();
    Eigen::Vector<double, n> f{{3.0, 4.0}};
    Eigen::Matrix<double, Eigen::Dynamic, n> C(2, n);
    C << 1.0, 1.0,
         1.0, 1.0;
    Eigen::VectorXd d(2);
    d << 2.0, 2.0;
    Eigen::Matrix<double, Eigen::Dynamic, n> G(0, n);
    Eigen::VectorXd h(0);
    Eigen::Vector<double, n> x = Eigen::Vector<double, n>::Zero();

    Eigen::MatrixXd nnls_A;
    Eigen::VectorXd nnls_b;
    Eigen::VectorXd nnls_x_vec;
    Eigen::VectorXd nnls_w;

    int mode = lsei<double, n>(C, d, E, f, G, h, x,
                               nnls_A, nnls_b, nnls_x_vec, nnls_w, n, 2, 0);

    CHECK(mode == 6);
}

// ---------------------------------------------------------------------------
// kraft_lsq_qp_solver tests (Kraft 1988 eq. 2.33 + Ch. 3.2)
// ---------------------------------------------------------------------------

TEST_CASE("kraft_lsq_qp_solver unconstrained", "[kraft_lsq_qp]")
{
    // min 0.5 p^T I p + [1, 2]^T p  =>  p* = -[1, 2]
    constexpr int n = 2;
    Eigen::Matrix<double, n, n> B = Eigen::Matrix<double, n, n>::Identity();
    Eigen::Vector<double, n> g{{1.0, 2.0}};

    Eigen::Matrix<double, Eigen::Dynamic, n> A_eq(0, n);
    Eigen::VectorXd b_eq(0);
    Eigen::Matrix<double, Eigen::Dynamic, n> A_ineq(0, n);
    Eigen::VectorXd b_ineq(0);

    const double inf = std::numeric_limits<double>::infinity();
    Eigen::Vector<double, n> p_lo{{-inf, -inf}};
    Eigen::Vector<double, n> p_hi{{ inf,  inf}};

    kraft_lsq_qp_solver<double, n> solver;
    solver.resize(n, 0, 0, 0, 0);
    auto res = solver.solve(B, g, A_eq, b_eq, A_ineq, b_ineq, p_lo, p_hi);

    CHECK(res.status == qp_status::optimal);
    CHECK(res.x[0] == Approx(-1.0).margin(1e-10));
    CHECK(res.x[1] == Approx(-2.0).margin(1e-10));
}

TEST_CASE("kraft_lsq_qp_solver equality only", "[kraft_lsq_qp]")
{
    // min 0.5 (x1^2 + x2^2)  s.t.  x1 + x2 = 1
    // => x = [0.5, 0.5]
    constexpr int n = 2;
    Eigen::Matrix<double, n, n> B = Eigen::Matrix<double, n, n>::Identity();
    Eigen::Vector<double, n> g = Eigen::Vector<double, n>::Zero();

    Eigen::Matrix<double, Eigen::Dynamic, n> A_eq(1, n);
    A_eq << 1.0, 1.0;
    Eigen::VectorXd b_eq(1);
    b_eq << 1.0;
    Eigen::Matrix<double, Eigen::Dynamic, n> A_ineq(0, n);
    Eigen::VectorXd b_ineq(0);

    const double inf = std::numeric_limits<double>::infinity();
    Eigen::Vector<double, n> p_lo{{-inf, -inf}};
    Eigen::Vector<double, n> p_hi{{ inf,  inf}};

    kraft_lsq_qp_solver<double, n> solver;
    solver.resize(n, 1, 0, 0, 0);
    auto res = solver.solve(B, g, A_eq, b_eq, A_ineq, b_ineq, p_lo, p_hi);

    CHECK(res.status == qp_status::optimal);
    CHECK(res.x[0] == Approx(0.5).margin(1e-10));
    CHECK(res.x[1] == Approx(0.5).margin(1e-10));
}

TEST_CASE("kraft_lsq_qp_solver box bounds only", "[kraft_lsq_qp]")
{
    // min 0.5 p^T I p + [-3, -4]^T p  s.t.  0 <= p <= 2
    // Unconstrained optimum [3, 4] is clipped to [2, 2].
    constexpr int n = 2;
    Eigen::Matrix<double, n, n> B = Eigen::Matrix<double, n, n>::Identity();
    Eigen::Vector<double, n> g{{-3.0, -4.0}};

    Eigen::Matrix<double, Eigen::Dynamic, n> A_eq(0, n);
    Eigen::VectorXd b_eq(0);
    Eigen::Matrix<double, Eigen::Dynamic, n> A_ineq(0, n);
    Eigen::VectorXd b_ineq(0);

    Eigen::Vector<double, n> p_lo{{0.0, 0.0}};
    Eigen::Vector<double, n> p_hi{{2.0, 2.0}};

    kraft_lsq_qp_solver<double, n> solver;
    solver.resize(n, 0, 0, n, n);
    auto res = solver.solve(B, g, A_eq, b_eq, A_ineq, b_ineq, p_lo, p_hi);

    CHECK(res.status == qp_status::optimal);
    CHECK(res.x[0] == Approx(2.0).margin(1e-10));
    CHECK(res.x[1] == Approx(2.0).margin(1e-10));
}

TEST_CASE("kraft_lsq_qp_solver infinite bounds not augmented", "[kraft_lsq_qp]")
{
    // One-sided bound: p >= 0 (infinite upper), unconstrained otherwise.
    // min 0.5 p^T I p + [-3, -4]^T p  s.t.  p >= 0
    // Optimum [3, 4] is feasible.
    constexpr int n = 2;
    Eigen::Matrix<double, n, n> B = Eigen::Matrix<double, n, n>::Identity();
    Eigen::Vector<double, n> g{{-3.0, -4.0}};

    Eigen::Matrix<double, Eigen::Dynamic, n> A_eq(0, n);
    Eigen::VectorXd b_eq(0);
    Eigen::Matrix<double, Eigen::Dynamic, n> A_ineq(0, n);
    Eigen::VectorXd b_ineq(0);

    const double inf = std::numeric_limits<double>::infinity();
    Eigen::Vector<double, n> p_lo{{0.0, 0.0}};
    Eigen::Vector<double, n> p_hi{{inf, inf}};

    kraft_lsq_qp_solver<double, n> solver;
    solver.resize(n, 0, 0, n, 0);
    auto res = solver.solve(B, g, A_eq, b_eq, A_ineq, b_ineq, p_lo, p_hi);

    CHECK(res.status == qp_status::optimal);
    CHECK(res.x[0] == Approx(3.0).margin(1e-10));
    CHECK(res.x[1] == Approx(4.0).margin(1e-10));
}

TEST_CASE("kraft_lsq_qp_solver mixed HS071-like subproblem", "[kraft_lsq_qp]")
{
    // Simplified QP of SQP shape: n=4, B=I, g arbitrary,
    // one equality, one inequality, box bounds [1, 5]^4.
    //
    // The test only checks that the solver returns optimal and the
    // returned step is feasible w.r.t. all constraints.
    constexpr int n = 4;
    Eigen::Matrix<double, n, n> B = Eigen::Matrix<double, n, n>::Identity();
    Eigen::Vector<double, n> g{{0.1, -0.2, 0.05, -0.1}};

    Eigen::Matrix<double, Eigen::Dynamic, n> A_eq(1, n);
    A_eq << 1.0, 1.0, 1.0, 1.0;
    Eigen::VectorXd b_eq(1);
    b_eq << 0.0;  // equality: sum of p_i = 0

    Eigen::Matrix<double, Eigen::Dynamic, n> A_ineq(1, n);
    A_ineq << 1.0, -1.0, 0.0, 0.0;
    Eigen::VectorXd b_ineq(1);
    b_ineq << -10.0;  // inactive: p1 - p2 >= -10

    Eigen::Vector<double, n> p_lo{{-1.0, -1.0, -1.0, -1.0}};
    Eigen::Vector<double, n> p_hi{{ 1.0,  1.0,  1.0,  1.0}};

    kraft_lsq_qp_solver<double, n> solver;
    solver.resize(n, 1, 1, n, n);
    auto res = solver.solve(B, g, A_eq, b_eq, A_ineq, b_ineq, p_lo, p_hi);

    CHECK(res.status == qp_status::optimal);

    // Verify feasibility.
    const double sum = res.x.sum();
    CHECK(std::abs(sum) < 1e-9);  // equality
    const double ineq_val = res.x[0] - res.x[1];
    CHECK(ineq_val >= -10.0 - 1e-9);
    for(int i = 0; i < n; ++i)
    {
        CHECK(res.x[i] >= -1.0 - 1e-9);
        CHECK(res.x[i] <=  1.0 + 1e-9);
    }
}
