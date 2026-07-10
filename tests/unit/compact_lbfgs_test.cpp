#include "argmin/detail/compact_lbfgs.h"
#include "argmin/detail/dense_ldl_bfgs.h"

#include <Eigen/Core>
#include <Eigen/LU>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <vector>
#include <cmath>
#include <limits>
#include <cstddef>
#include <algorithm>

using Catch::Approx;
using namespace argmin::detail;
using Eigen::VectorXd;

// ---------------------------------------------------------------------------
// compact_lbfgs middle-matrix factorization and curvature-clamp pins.
//
// The compact representation is B = theta*I - W M^{-1} W^T with the 2m x 2m
// middle matrix M = [[theta S^T S, L], [L^T, -D]] (N&W Section 9.2, eq. 9.15;
// Byrd-Nocedal-Schnabel 1994). M is structurally indefinite, so the earlier
// Eigen::LDLT (a (semi)definite-only factorization) was out of contract, and
// the curvature clamp was applied to rho_/theta_ but not to the raw s^T y
// stored on the -D diagonal. The reference below evaluates the *intended*
// representation -- clamped curvature everywhere -- in long double via a
// robust FullPivLU, so double-precision B*v from compact_lbfgs is compared
// against a high-precision ground truth of the same object.
// ---------------------------------------------------------------------------

namespace
{
// High-precision B*v for the compact representation with the class's clamp
// rules (curvature clamp floor eps*||s||^2 with double eps, theta from the
// latest accepted pair). Independent of compact_lbfgs's factorization.
VectorXd reference_Bv(const std::vector<VectorXd>& ss,
                      const std::vector<VectorXd>& ys, int n, const VectorXd& v)
{
    using ld = long double;
    using MatL = Eigen::Matrix<ld, Eigen::Dynamic, Eigen::Dynamic>;
    using VecL = Eigen::Matrix<ld, Eigen::Dynamic, 1>;
    const double epsd = std::numeric_limits<double>::epsilon();

    std::vector<int> acc;
    for(int i = 0; i < static_cast<int>(ss.size()); ++i)
    {
        const auto ui = static_cast<std::size_t>(i);
        if(ss[ui].squaredNorm() < epsd * epsd) continue;
        if(ss[ui].dot(ys[ui]) <= 0.0) continue;
        acc.push_back(i);
    }
    const int k = static_cast<int>(acc.size());
    if(k == 0) return v;  // theta = 1

    MatL S(n, k), Y(n, k);
    VecL Dc(k);
    for(int c = 0; c < k; ++c)
    {
        const int i = acc[static_cast<std::size_t>(c)];
        const auto ui = static_cast<std::size_t>(i);
        for(int r = 0; r < n; ++r)
        {
            S(r, c) = static_cast<ld>(ss[ui](r));
            Y(r, c) = static_cast<ld>(ys[ui](r));
        }
        const double floor_i = epsd * ss[ui].squaredNorm();
        Dc(c) = static_cast<ld>(std::max(ss[ui].dot(ys[ui]), floor_i));
    }
    const int last = acc.back();
    const auto ulast = static_cast<std::size_t>(last);
    const double floor_l = epsd * ss[ulast].squaredNorm();
    const double sTy_eff_l = std::max(ss[ulast].dot(ys[ulast]), floor_l);
    const ld theta = static_cast<ld>(
        std::clamp(ys[ulast].squaredNorm() / sTy_eff_l, 1e-10, 1e10));

    MatL STS = S.transpose() * S;
    MatL StY = S.transpose() * Y;
    MatL L = MatL::Zero(k, k);
    for(int i = 0; i < k; ++i)
        for(int j = 0; j < i; ++j) L(i, j) = StY(i, j);  // strictly lower

    MatL M(2 * k, 2 * k);
    M.topLeftCorner(k, k) = theta * STS;
    M.topRightCorner(k, k) = L;
    M.bottomLeftCorner(k, k) = L.transpose();
    M.bottomRightCorner(k, k) = -MatL(Dc.asDiagonal());

    MatL W(n, 2 * k);
    W.leftCols(k) = theta * S;
    W.rightCols(k) = Y;

    VecL vl(n);
    for(int r = 0; r < n; ++r) vl(r) = static_cast<ld>(v(r));
    VecL z = M.fullPivLu().solve(W.transpose() * vl);
    VecL Bv = theta * vl - W * z;

    VectorXd out(n);
    for(int r = 0; r < n; ++r) out(r) = static_cast<double>(Bv(r));
    return out;
}

// A near-degenerate 3-pair sequence: pair 2 carries a genuinely tiny curvature
// s^T y = 1e-25 (a unit axis for s, so no cancellation) that sits far below
// the eps*||s||^2 clamp floor.
void make_sequence(std::vector<VectorXd>& ss, std::vector<VectorXd>& ys)
{
    VectorXd s1(4); s1 << 1.0, 0.5, -0.3, 0.2;
    VectorXd y1(4); y1 << 2.0, 1.0, 0.4, 0.1;
    VectorXd s2(4); s2 << 0.0, 1.0, 0.0, 0.0;
    VectorXd y2(4); y2 << 0.8, 1e-25, -0.6, 0.5;  // s2.y2 = 1e-25
    VectorXd s3(4); s3 << -0.2, 0.9, 0.3, -0.5;
    VectorXd y3(4); y3 << 0.5, 1.2, 0.6, 0.3;
    ss = {s1, s2, s3};
    ys = {y1, y2, y3};
}

std::vector<VectorXd> basket(int n)
{
    std::vector<VectorXd> vs;
    for(int i = 0; i < n; ++i) vs.push_back(VectorXd::Unit(n, i));
    VectorXd vr(n); vr << 1.0, -2.0, 3.0, -4.0;
    vs.push_back(vr);
    return vs;
}
}  // namespace

TEST_CASE("compact_lbfgs B*v matches a high-precision reference on a "
          "near-degenerate sequence", "[compact_lbfgs][bv]")
{
    // The two representations are algebraically identical for the same clamped
    // pair set (BNS 1994 representation theorem), so the in-contract double
    // factorization must reproduce the long-double reference to a tight
    // relative tolerance. 1e-10 is justified by that algebraic identity, not
    // shopped: any larger residual signals element growth in the middle-matrix
    // solve (the pre-fix Eigen::LDLT-on-indefinite / raw-D failure mode).
    std::vector<VectorXd> ss, ys;
    make_sequence(ss, ys);
    const int n = 4;

    compact_lbfgs<double, Eigen::Dynamic, 7> lbfgs;  // default: PartialPivLU
    for(int i = 0; i < 3; ++i)
        lbfgs.push(ss[static_cast<std::size_t>(i)], ys[static_cast<std::size_t>(i)]);

    for(const auto& v : basket(n))
    {
        VectorXd got = lbfgs.multiply(v);
        VectorXd ref = reference_Bv(ss, ys, n, v);
        CHECK((got - ref).norm() <= 1e-10 * (ref.norm() + 1e-300));
    }
}

TEST_CASE("compact_lbfgs clamps the stored curvature consistently",
          "[compact_lbfgs][clamp]")
{
    // D-clamp invariant: every stored curvature s^T y on the middle-matrix -D
    // diagonal is at or above the clamp floor eps*||s||^2 that also governs
    // rho_ and theta_. Pushing a pair with s^T y = 1e-25 (well below the
    // floor) must leave no D entry at the raw tiny value.
    //
    // Pre-fix: D stored the raw 1e-25 (rho_/theta_ used the clamped value,
    // so the -D block, rho_ and theta_ disagreed). Post-fix: min stored
    // curvature is the clamp floor, strictly positive.
    std::vector<VectorXd> ss, ys;
    make_sequence(ss, ys);

    compact_lbfgs<double, Eigen::Dynamic, 7> lbfgs;
    for(int i = 0; i < 3; ++i)
        lbfgs.push(ss[static_cast<std::size_t>(i)], ys[static_cast<std::size_t>(i)]);

    const double eps = std::numeric_limits<double>::epsilon();
    // The tiny pair is s2 = e_2, so its floor is eps*1. No stored curvature
    // may fall below that floor, and all must be strictly positive.
    const double floor2 = eps * ss[1].squaredNorm();
    CHECK(lbfgs.min_stored_curvature() >= floor2);
    CHECK(lbfgs.min_stored_curvature() > 0.0);
    CHECK(lbfgs.min_stored_curvature() > 1e-25);  // not the raw curvature
}

TEST_CASE("compact_lbfgs single-pair B*v equals dense_ldl_bfgs",
          "[compact_lbfgs][bv]")
{
    // For a single curvature pair both representations reduce to the same BFGS
    // update from B0 = theta*I with theta = y^T y / s^T y (no Powell damping
    // is triggered), so their B*v agree to machine precision. This ties the
    // compact form to the verified-good dense_ldl_bfgs reference.
    const int n = 4;
    VectorXd s(n); s << 1.0, 0.5, -0.3, 0.2;
    VectorXd y(n); y << 2.0, 1.0, 0.4, 0.1;

    compact_lbfgs<double, Eigen::Dynamic, 7> lbfgs;
    dense_ldl_bfgs<double, Eigen::Dynamic> dense(n);
    lbfgs.push(s, y);
    dense.push(s, y);

    for(const auto& v : basket(n))
    {
        VectorXd a = lbfgs.multiply(v);
        VectorXd b = dense.multiply(v);
        CHECK((a - b).norm() <= 1e-12 * (b.norm() + 1e-300));
    }
}

TEST_CASE("compact_lbfgs PartialPivLU beats the BNS Schur route on "
          "near-degenerate curvature", "[compact_lbfgs][factorization-ab]")
{
    // Empirical A/B recorded in the plan summary. Both strategies are in
    // contract for the indefinite middle matrix, but once a D diagonal reaches
    // the clamp floor the BNS route forms D^{-1} explicitly (Schur complement
    // theta S^T S + L D^{-1} L^T) and amplifies the tiny pivot, while
    // PartialPivLU pivots the assembled M and never inverts D. PartialPivLU is
    // therefore the default; this pin locks in the ordering.
    std::vector<VectorXd> ss, ys;
    make_sequence(ss, ys);
    const int n = 4;

    compact_lbfgs<double, Eigen::Dynamic, 7, lbfgs_middle_solve::partial_piv_lu> plu;
    compact_lbfgs<double, Eigen::Dynamic, 7, lbfgs_middle_solve::bns> bns;
    for(int i = 0; i < 3; ++i)
    {
        const auto ui = static_cast<std::size_t>(i);
        plu.push(ss[ui], ys[ui]);
        bns.push(ss[ui], ys[ui]);
    }

    double plu_rel = 0.0, bns_rel = 0.0;
    for(const auto& v : basket(n))
    {
        VectorXd ref = reference_Bv(ss, ys, n, v);
        const double denom = ref.norm() + 1e-300;
        plu_rel = std::max(plu_rel, (plu.multiply(v) - ref).norm() / denom);
        bns_rel = std::max(bns_rel, (bns.multiply(v) - ref).norm() / denom);
    }
    CHECK(plu_rel <= 1e-10);        // the default stays tight
    CHECK(plu_rel <= bns_rel);      // and is no worse than BNS (here far better)
}
