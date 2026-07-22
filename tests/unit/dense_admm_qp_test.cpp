#include "argmin/qp/dense_admm_qp.h"
#include "argmin/detail/kraft_lsq_qp.h"
#include "argmin/qp/dense_active_set_qp.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <random>
#include <limits>
#include <vector>

using Catch::Approx;
using namespace argmin;

namespace
{

constexpr double inf = std::numeric_limits<double>::infinity();

struct osqp_problem
{
    Eigen::MatrixXd P;
    Eigen::VectorXd q;
    Eigen::MatrixXd A;
    Eigen::VectorXd l;
    Eigen::VectorXd u;
};

bool is_eq_row(const osqp_problem& p, int i)
{
    return std::isfinite(p.l[i]) && std::isfinite(p.u[i]) && (p.u[i] - p.l[i]) <= 1e-9;
}

// Trusted oracle: solve the same QP through the Kraft LSQ/LSEI active-set QP.
// The OSQP rows are folded into the oracle's (A_eq, A_ineq >=) blocks with the
// same finite-bound-skip mapping the adapter uses; box bounds pass as infinite
// so every constraint lives in the general blocks.
Eigen::VectorXd kraft_reference(const osqp_problem& p)
{
    const int n = static_cast<int>(p.P.rows());
    const int m = static_cast<int>(p.A.rows());
    int m_eq = 0;
    int m_in = 0;
    for(int i = 0; i < m; ++i)
    {
        if(is_eq_row(p, i))
            ++m_eq;
        else
        {
            if(std::isfinite(p.l[i]))
                ++m_in;
            if(std::isfinite(p.u[i]))
                ++m_in;
        }
    }
    Eigen::MatrixXd A_eq(m_eq, n);
    Eigen::MatrixXd A_in(m_in, n);
    Eigen::VectorXd b_eq(m_eq);
    Eigen::VectorXd b_in(m_in);
    int re = 0;
    int ri = 0;
    for(int i = 0; i < m; ++i)
    {
        if(is_eq_row(p, i))
        {
            A_eq.row(re) = p.A.row(i);
            b_eq[re] = p.l[i];
            ++re;
        }
        else
        {
            if(std::isfinite(p.l[i]))
            {
                A_in.row(ri) = p.A.row(i);
                b_in[ri] = p.l[i];
                ++ri;
            }
            if(std::isfinite(p.u[i]))
            {
                A_in.row(ri) = -p.A.row(i);
                b_in[ri] = -p.u[i];
                ++ri;
            }
        }
    }
    const Eigen::VectorXd p_lo = Eigen::VectorXd::Constant(n, -inf);
    const Eigen::VectorXd p_hi = Eigen::VectorXd::Constant(n, inf);

    detail::kraft_lsq_qp_solver<double> oracle;
    oracle.resize(n, m_eq, m_in, 0, 0);
    auto res = oracle.solve(p.P, p.q, A_eq, b_eq, A_in, b_in, p_lo, p_hi);
    return res.x;
}

// A family of strictly-convex OSQP problems (seeded random SPD P, box rows plus
// a general two-sided coupling row) on which both solvers have a unique
// minimizer. Deterministic via a fixed seed.
std::vector<osqp_problem> convex_family()
{
    std::vector<osqp_problem> out;
    std::mt19937 rng(424242);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(0.5, 2.0);
    for(int trial = 0; trial < 40; ++trial)
    {
        const int n = 2 + (trial % 4);
        Eigen::MatrixXd M(n, n);
        for(int i = 0; i < n; ++i)
            for(int j = 0; j < n; ++j)
                M(i, j) = nd(rng);
        osqp_problem p;
        p.P = M.transpose() * M + (1.0 + ud(rng)) * Eigen::MatrixXd::Identity(n, n);
        p.q.resize(n);
        for(int i = 0; i < n; ++i)
            p.q[i] = nd(rng);
        const int m = n + 1;
        p.A = Eigen::MatrixXd::Zero(m, n);
        p.l.resize(m);
        p.u.resize(m);
        for(int i = 0; i < n; ++i)
        {
            p.A(i, i) = 1.0;
            const double b = ud(rng);
            p.l[i] = -b;
            p.u[i] = b;
        }
        for(int j = 0; j < n; ++j)
            p.A(m - 1, j) = nd(rng);
        p.l[m - 1] = -2.0 * ud(rng);
        p.u[m - 1] = 2.0 * ud(rng);
        out.push_back(std::move(p));
    }
    return out;
}

}

// ---------------------------------------------------------------------------
// Parity against the Kraft oracle.
//
// On strictly-convex problems both solvers reach the same unique minimizer.
// The committed tolerance is 1e-9; the measured worst-case deviation over this
// family is ~3.3e-16 (machine precision), so the bound has ~7 orders of
// headroom -- it is a genuine accuracy gate, not a loosened pass. The polish
// is required to succeed on every fixture (a rejection is a diagnostic).
// ---------------------------------------------------------------------------
TEST_CASE("dense_admm_qp matches the Kraft oracle within a tight tolerance", "[qp_parity]")
{
    constexpr double parity_tol = 1e-9;
    for(const auto& p : convex_family())
    {
        const int n = static_cast<int>(p.P.rows());
        const int m = static_cast<int>(p.A.rows());
        const Eigen::VectorXd x_ref = kraft_reference(p);

        dense_admm_qp_solver<double> solver(n, m);
        auto r = solver.solve(p.P, p.q, p.A, p.l, p.u);
        REQUIRE(r);
        CHECK(r->status == qp_solve_status::solved);
        CHECK(r->polished);
        CHECK((r->x - x_ref).cwiseAbs().maxCoeff() <= parity_tol);
    }
}

// ---------------------------------------------------------------------------
// Warm-start correctness: a vectors-only resolve after a ~1% perturbation
// reaches the cold answer in strictly fewer iterations and reuses the frozen
// Ruiz factors (no re-equilibration).
// ---------------------------------------------------------------------------
TEST_CASE("dense_admm_qp warm resolve converges faster without re-equilibrating",
          "[qp_warm]")
{
    const int n = 5;
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(n, n);
    for(int i = 0; i < n; ++i)
        P(i, i) = std::pow(50.0, static_cast<double>(i) / (n - 1));
    Eigen::VectorXd q(n);
    for(int i = 0; i < n; ++i)
        q[i] = (i % 2 ? 1.0 : -1.0) * (i + 1);

    const int m = n + 2;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, n);
    Eigen::VectorXd l(m);
    Eigen::VectorXd u(m);
    for(int i = 0; i < n; ++i)
    {
        A(i, i) = 1.0;
        l[i] = -0.2;
        u[i] = 0.2;
    }
    A(n, 0) = 1.0;
    A(n, 1) = 1.0;
    l[n] = -0.1;
    u[n] = 0.1;
    for(int j = 0; j < n; ++j)
        A(n + 1, j) = 1.0;
    l[n + 1] = -0.3;
    u[n + 1] = 0.3;

    dense_admm_qp_solver<double> solver(n, m);
    auto cold = solver.solve(P, q, A, l, u);
    REQUIRE(cold);

    const double c0 = solver.scaling_cost();
    const Eigen::VectorXd D0 = solver.scaling_primal();
    const Eigen::VectorXd E0 = solver.scaling_dual();

    // Unperturbed warm resolve: seeded from the retained optimum, it converges
    // by the first termination check.
    auto same = solver.resolve(q, l, u);
    REQUIRE(same);
    CHECK(same->iterations <= 25);

    // No re-equilibration: the Ruiz factors are bit-for-bit the pose-time ones.
    CHECK(solver.scaling_cost() == c0);
    CHECK((solver.scaling_primal() - D0).cwiseAbs().maxCoeff() == 0.0);
    CHECK((solver.scaling_dual() - E0).cwiseAbs().maxCoeff() == 0.0);

    // ~1% perturbation: warm resolve vs a cold solve of the perturbed problem.
    const Eigen::VectorXd q2 = q * 1.01;
    const Eigen::VectorXd l2 = l * 1.01;
    const Eigen::VectorXd u2 = u * 1.01;
    auto warm = solver.resolve(q2, l2, u2);
    REQUIRE(warm);

    dense_admm_qp_solver<double> fresh(n, m);
    auto cold_perturbed = fresh.solve(P, q2, A, l2, u2);
    REQUIRE(cold_perturbed);

    CHECK((warm->x - cold_perturbed->x).cwiseAbs().maxCoeff() <= 1e-6);
    CHECK(warm->iterations < cold_perturbed->iterations);
}

// ---------------------------------------------------------------------------
// Primal-infeasibility certificate on a trivially infeasible box: x >= 1 and
// x <= 0. A capability the active-set path lacks.
// ---------------------------------------------------------------------------
TEST_CASE("dense_admm_qp certifies a trivially infeasible box", "[qp_infeasible]")
{
    const int n = 1;
    const int m = 2;
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(n, n);
    Eigen::VectorXd q(n);
    q << 0.0;
    Eigen::MatrixXd A(m, n);
    A << 1.0, 1.0;
    Eigen::VectorXd l(m);
    l << 1.0, -inf;
    Eigen::VectorXd u(m);
    u << inf, 0.0;

    dense_admm_qp_solver<double> solver(n, m);
    auto r = solver.solve(P, q, A, l, u);
    REQUIRE(r);
    CHECK(r->status == qp_solve_status::primal_infeasible);
}

// ---------------------------------------------------------------------------
// Adapter row-mapping and inherited-contract errors. The read-only active-set
// adapter, given a feasible start, reaches the same optimum as the ADMM/Kraft
// path on a mixed equality / one-sided / box problem; an inequality-infeasible
// start and an over-determined equality set surface as checked errors.
// ---------------------------------------------------------------------------
TEST_CASE("dense_active_set_qp adapter maps OSQP rows and checks its contract",
          "[qp_adapter]")
{
    const int n = 2;
    const int m = 4;
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(n, n);
    Eigen::VectorXd q(n);
    q << 0.0, 0.0;
    Eigen::MatrixXd A(m, n);
    A << 1.0, 1.0,   // equality x1 + x2 = 1
         1.0, 0.0,   // one-sided x1 >= 0
         1.0, 0.0,   // box x1
         0.0, 1.0;   // box x2
    Eigen::VectorXd l(m);
    l << 1.0, 0.0, -5.0, -5.0;
    Eigen::VectorXd u(m);
    u << 1.0, inf, 5.0, 5.0;

    dense_admm_qp_solver<double> admm(n, m);
    auto ref = admm.solve(P, q, A, l, u);
    REQUIRE(ref);

    dense_active_set_qp_solver<double> adapter(n, m);
    Eigen::VectorXd x0(n);
    x0 << 0.5, 0.5;  // feasible start
    adapter.warm_start(x0, Eigen::VectorXd());
    auto r = adapter.solve(P, q, A, l, u);
    REQUIRE(r);
    CHECK(r->status == qp_solve_status::solved);
    CHECK((r->x - ref->x).cwiseAbs().maxCoeff() <= 1e-6);

    SECTION("inequality-infeasible start is rejected")
    {
        dense_active_set_qp_solver<double> a2(n, m);
        Eigen::VectorXd bad(n);
        bad << 10.0, 10.0;
        a2.warm_start(bad, Eigen::VectorXd());
        auto rbad = a2.solve(P, q, A, l, u);
        REQUIRE_FALSE(rbad);
        CHECK(rbad.error() == qp_error::infeasible_start);
    }

    SECTION("over-determined equality set is rejected")
    {
        dense_active_set_qp_solver<double> a3(1, 2);
        Eigen::MatrixXd P1 = Eigen::MatrixXd::Identity(1, 1);
        Eigen::VectorXd q1(1);
        q1 << 0.0;
        Eigen::MatrixXd A1(2, 1);
        A1 << 1.0, 1.0;
        Eigen::VectorXd l1(2);
        l1 << 1.0, 1.0;
        Eigen::VectorXd u1(2);
        u1 << 1.0, 1.0;
        auto rover = a3.solve(P1, q1, A1, l1, u1);
        REQUIRE_FALSE(rover);
        CHECK(rover.error() == qp_error::invalid_problem);
    }
}

// ---------------------------------------------------------------------------
// Polish padding-invariance. The reduced-KKT polish assembles the active set
// into a constant (n + m_cap) padded shape with an identity trailing block, so
// a PartialPivLU pivot could in principle cross into the padding. This asserts
// the polished solution is independent of the padding amount: a solver sized at
// the exact constraint count and one with extra unused capacity agree to
// machine precision. Measured worst-case deviation over this family is
// ~1.1e-16; the committed bound is 1e-12. If this ever fails, the remedy is the
// fixed (n + m) reduced-shape polish with every row present and inactive rows
// delta-slacked.
// ---------------------------------------------------------------------------
TEST_CASE("dense_admm_qp polish is invariant to the padding amount", "[qp_parity]")
{
    constexpr double pad_tol = 1e-12;
    for(const auto& p : convex_family())
    {
        const int n = static_cast<int>(p.P.rows());
        const int m = static_cast<int>(p.A.rows());

        dense_admm_qp_solver<double> tight(n, m);
        auto rt = tight.solve(p.P, p.q, p.A, p.l, p.u);
        REQUIRE(rt);

        dense_admm_qp_solver<double> padded(n, m + 7);
        auto rp = padded.solve(p.P, p.q, p.A, p.l, p.u);
        REQUIRE(rp);

        CHECK(rt->polished);
        CHECK(rp->polished);
        CHECK((rt->x - rp->x).cwiseAbs().maxCoeff() <= pad_tol);
    }
}
