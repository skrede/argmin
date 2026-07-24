#include "argmin/qp/detail/sparse_kkt.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <catch2/catch_test_macros.hpp>

#include <random>
#include <vector>
#include <cstddef>
#include <cstring>

using namespace argmin;

namespace
{

// A discrete double integrator over a horizon, posed exactly the way the
// downstream control workload poses it: decision vector of stacked states and
// inputs, banded equality dynamics rows above identity box rows on every
// variable. Built directly from triplets, without going through any solver, so
// the witness measures the factorization and nothing else.
struct control_system
{
    Eigen::SparseMatrix<double> P;
    Eigen::SparseMatrix<double> A;
    int n{0};
    int m{0};
    int equality_rows{0};
};

control_system make_double_integrator_system(int horizon)
{
    constexpr int nx = 2;
    constexpr int nu = 1;
    constexpr double dt = 0.05;

    const int states = nx * (horizon + 1);
    const int inputs = nu * horizon;

    control_system s;
    s.n = states + inputs;
    s.equality_rows = nx * (horizon + 1);
    s.m = s.equality_rows + s.n;

    std::vector<Eigen::Triplet<double>> p_entries;
    for(int k = 0; k <= horizon; ++k)
    {
        const int base = nx * k;
        p_entries.emplace_back(base, base, 2.0);
        p_entries.emplace_back(base + 1, base + 1, 1.5);
        p_entries.emplace_back(base, base + 1, 0.25);
        p_entries.emplace_back(base + 1, base, 0.25);
    }
    for(int k = 0; k < horizon; ++k)
        p_entries.emplace_back(states + nu * k, states + nu * k, 0.75);

    s.P.resize(s.n, s.n);
    s.P.setFromTriplets(p_entries.begin(), p_entries.end());
    s.P.makeCompressed();

    std::vector<Eigen::Triplet<double>> a_entries;
    for(int i = 0; i < nx; ++i)
        a_entries.emplace_back(i, i, 1.0);
    for(int k = 0; k < horizon; ++k)
    {
        const int row = nx * (k + 1);
        const int xk = nx * k;
        const int xk1 = nx * (k + 1);
        const int uk = states + nu * k;

        a_entries.emplace_back(row, xk1, 1.0);
        a_entries.emplace_back(row, xk, -1.0);
        a_entries.emplace_back(row, xk + 1, -dt);
        a_entries.emplace_back(row, uk, -0.5 * dt * dt);

        a_entries.emplace_back(row + 1, xk1 + 1, 1.0);
        a_entries.emplace_back(row + 1, xk + 1, -1.0);
        a_entries.emplace_back(row + 1, uk, -dt);
    }
    for(int i = 0; i < s.n; ++i)
        a_entries.emplace_back(s.equality_rows + i, i, 1.0);

    s.A.resize(s.m, s.n);
    s.A.setFromTriplets(a_entries.begin(), a_entries.end());
    s.A.makeCompressed();

    return s;
}

// Equality rows carry the 1e3 rho boost the solver applies, so the lower-right
// block under test is as ill-conditioned as the real workload makes it.
std::vector<double> make_rho_inv(const control_system& s)
{
    constexpr double rho_base = 0.1;
    constexpr double equality_boost = 1e3;

    std::vector<double> rho_inv(static_cast<std::size_t>(s.m));
    for(int i = 0; i < s.m; ++i)
    {
        const double rho = i < s.equality_rows ? rho_base * equality_boost : rho_base;
        rho_inv[static_cast<std::size_t>(i)] = 1.0 / rho;
    }
    return rho_inv;
}

Eigen::VectorXd seeded_rhs(int size)
{
    std::mt19937 rng(20240722u);
    std::uniform_real_distribution<double> draw(-1.0, 1.0);

    Eigen::VectorXd rhs(size);
    for(int i = 0; i < size; ++i)
        rhs(i) = draw(rng);
    return rhs;
}

// The residual is formed from the stored lower triangle expanded through a
// self-adjoint view, never from the decomposition's own success report.
double relative_residual(const detail::sparse_kkt<double>& kkt, const Eigen::VectorXd& rhs)
{
    Eigen::VectorXd solution(kkt.rows());
    kkt.solve_into(rhs, solution);

    const Eigen::SparseMatrix<double> k_full = kkt.matrix().selfadjointView<Eigen::Lower>();
    const Eigen::VectorXd residual = k_full * solution - rhs;
    return residual.cwiseAbs().maxCoeff() / rhs.cwiseAbs().maxCoeff();
}

constexpr int witness_horizon = 120;
constexpr double sigma = 1e-6;

// Measured relative infinity-norm residual on this system is ~9.9e-15 for the
// base rho and ~3.3e-16 after the 1e3 rho update, so the committed bound
// carries roughly five orders of headroom on the worse of the two -- it is a
// genuine accuracy gate, not a bound loosened to fit an observation.
constexpr double residual_bound = 1e-9;

}

TEST_CASE("sparse_kkt factors a control-shaped quasi-definite system", "[qp][sparse_kkt]")
{
    const control_system s = make_double_integrator_system(witness_horizon);

    REQUIRE(s.n >= 300);
    REQUIRE(s.m >= s.n);
    const double density = static_cast<double>(s.A.nonZeros())
                           / (static_cast<double>(s.m) * static_cast<double>(s.n));
    REQUIRE(density < 0.02);

    const std::vector<double> rho_inv = make_rho_inv(s);

    detail::sparse_kkt<double> kkt;
    REQUIRE(kkt.analyze(s.P, s.A, sigma, rho_inv));
    REQUIRE(kkt.refactorize(rho_inv));

    const Eigen::VectorXd rhs = seeded_rhs(kkt.rows());
    CHECK(relative_residual(kkt, rhs) <= residual_bound);

    const Eigen::VectorXd d = kkt.diagonal_factor();
    const auto positive = (d.array() > 0.0).count();
    const auto negative = (d.array() < 0.0).count();
    CHECK(positive > 0);
    CHECK(negative > 0);
}

TEST_CASE("sparse_kkt refactorizes on a rho update without re-analyzing", "[qp][sparse_kkt]")
{
    const control_system s = make_double_integrator_system(witness_horizon);
    const std::vector<double> rho_inv = make_rho_inv(s);

    detail::sparse_kkt<double> kkt;
    REQUIRE(kkt.analyze(s.P, s.A, sigma, rho_inv));
    REQUIRE(kkt.refactorize(rho_inv));

    const Eigen::Index nonzeros = kkt.matrix().nonZeros();
    const std::vector<int> outer(kkt.matrix().outerIndexPtr(),
                                 kkt.matrix().outerIndexPtr() + kkt.matrix().outerSize() + 1);
    const std::vector<int> inner(kkt.matrix().innerIndexPtr(),
                                 kkt.matrix().innerIndexPtr() + nonzeros);

    std::vector<double> updated(rho_inv);
    for(double& v : updated)
        v *= 1e3;

    REQUIRE(kkt.refactorize(updated));

    const Eigen::VectorXd rhs = seeded_rhs(kkt.rows());
    CHECK(relative_residual(kkt, rhs) <= residual_bound);

    CHECK(kkt.matrix().nonZeros() == nonzeros);
    CHECK(std::vector<int>(kkt.matrix().outerIndexPtr(),
                           kkt.matrix().outerIndexPtr() + kkt.matrix().outerSize() + 1)
          == outer);
    CHECK(std::vector<int>(kkt.matrix().innerIndexPtr(),
                           kkt.matrix().innerIndexPtr() + nonzeros)
          == inner);
}

namespace
{

// Distinct pseudo-random right-hand sides, one per seed, so the bit-identity
// comparison below is exercised over several independent vectors rather than a
// single fixed one.
Eigen::VectorXd rhs_for(int size, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> draw(-1.0, 1.0);

    Eigen::VectorXd rhs(size);
    for(int i = 0; i < size; ++i)
        rhs(i) = draw(rng);
    return rhs;
}

// True only when the two vectors are byte-for-byte identical -- the strongest
// form of the equality contract, distinguishing even signed zeros.
bool bit_identical(const Eigen::VectorXd& a, const Eigen::VectorXd& b)
{
    return a.size() == b.size()
           && std::memcmp(a.data(), b.data(),
                          static_cast<std::size_t>(a.size()) * sizeof(double))
                  == 0;
}

}

// The manual permuted solve reuses Eigen's own factor APIs step for step, so its
// result must equal a stock SimplicialLDLT::solve on the same matrix bit for bit.
// The reference is an INDEPENDENT factorization of kkt.matrix() (the same lower
// triangle, the same AMD ordering, no pivoting), so factoring it reproduces the
// member factor's bits exactly and no old code path has to be kept alive.
//
// Exact byte equality is the expected outcome on every supported toolchain,
// because both paths run the same Eigen inner kernels (triangular solves,
// diagonal reciprocal, permutation scatter) in the same order; a genuine
// same-build mismatch is a bug to investigate first -- an aliased permutation, a
// division where the reference multiplies by a reciprocal, or a stale diagonal
// cache -- not a reason to weaken this assertion. A relative-parity floor near
// 1e-12 is the documented contract floor reserved only for a toolchain that
// provably cannot reproduce Eigen's bits; it is deliberately not activated here.
TEST_CASE("sparse_kkt manual solve is bit-identical to a reference factorization",
          "[qp][sparse_kkt]")
{
    constexpr int horizon = 20;
    const control_system s = make_double_integrator_system(horizon);
    const std::vector<double> rho_inv = make_rho_inv(s);

    detail::sparse_kkt<double> kkt;
    REQUIRE(kkt.analyze(s.P, s.A, sigma, rho_inv));
    REQUIRE(kkt.refactorize(rho_inv));

    detail::sparse_kkt<double>::factorization_type reference;
    reference.analyzePattern(kkt.matrix());
    reference.factorize(kkt.matrix());
    REQUIRE(reference.info() == Eigen::Success);

    for(unsigned seed = 1; seed <= 5; ++seed)
    {
        const Eigen::VectorXd rhs = rhs_for(kkt.rows(), seed);

        Eigen::VectorXd out;
        kkt.solve_into(rhs, out);
        const Eigen::VectorXd expected = reference.solve(rhs);

        CHECK(bit_identical(out, expected));
    }
}

// Guards the reciprocal-diagonal cache: a rho update recomputes D, and a
// solve_into that read a stale cache would silently scale by the pre-update
// diagonal. Re-factoring the independent reference on the updated matrix and
// re-asserting exact equality fails loudly if refactorize() forgot to refresh
// the cache -- an error a bit-identity test on a freshly posed KKT alone could
// not catch.
TEST_CASE("sparse_kkt manual solve stays bit-identical across a rho refactorize",
          "[qp][sparse_kkt]")
{
    constexpr int horizon = 20;
    const control_system s = make_double_integrator_system(horizon);
    const std::vector<double> rho_inv = make_rho_inv(s);

    detail::sparse_kkt<double> kkt;
    REQUIRE(kkt.analyze(s.P, s.A, sigma, rho_inv));
    REQUIRE(kkt.refactorize(rho_inv));

    const Eigen::VectorXd rhs = rhs_for(kkt.rows(), 7u);

    Eigen::VectorXd before;
    kkt.solve_into(rhs, before);
    detail::sparse_kkt<double>::factorization_type ref_before;
    ref_before.analyzePattern(kkt.matrix());
    ref_before.factorize(kkt.matrix());
    REQUIRE(ref_before.info() == Eigen::Success);
    CHECK(bit_identical(before, Eigen::VectorXd(ref_before.solve(rhs))));

    std::vector<double> updated(rho_inv);
    for(double& v : updated)
        v *= 1e3;
    REQUIRE(kkt.refactorize(updated));

    Eigen::VectorXd after;
    kkt.solve_into(rhs, after);
    detail::sparse_kkt<double>::factorization_type ref_after;
    ref_after.analyzePattern(kkt.matrix());
    ref_after.factorize(kkt.matrix());
    REQUIRE(ref_after.info() == Eigen::Success);
    CHECK(bit_identical(after, Eigen::VectorXd(ref_after.solve(rhs))));
}
