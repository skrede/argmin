#include "argmin/qp/qp_types.h"
#include "argmin/qp/dense_admm_qp.h"
#include "argmin/qp/sparse_admm_qp.h"
#include "argmin/detail/kraft_lsq_qp.h"

#include "portable_random.h"

#include "mm_data/mm_problems.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <optional>

using namespace argmin;

namespace
{

constexpr double inf = std::numeric_limits<double>::infinity();

using sparse = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using row_major = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Zero drop tolerance, so the structural pattern is exactly the dense data's
// nonzero set and the sparse solver sees the same problem rather than a pruned
// approximation of it.
sparse to_sparse(const Eigen::MatrixXd& m)
{
    return m.sparseView(0.0, 0.0);
}

// The solver's stated input convention is a FULL symmetric P; handing it a
// triangle would silently halve the off-diagonal quadratic terms.
bool is_full_symmetric(const sparse& P)
{
    const Eigen::MatrixXd d = Eigen::MatrixXd(P);
    return (d - d.transpose()).cwiseAbs().maxCoeff() == 0.0;
}

double relative_gap(double a, double b)
{
    return std::abs(a - b) / std::max(1.0, std::abs(b));
}

struct osqp_problem
{
    Eigen::MatrixXd P;
    Eigen::VectorXd q;
    Eigen::MatrixXd A;
    Eigen::VectorXd l;
    Eigen::VectorXd u;
};

// The same seeded strictly-convex family the dense suite uses -- same seed,
// same trial count, same construction -- so both suites exercise one family and
// the primal minimizer is unique on every member.
std::vector<osqp_problem> convex_family()
{
    std::vector<osqp_problem> out;
    std::mt19937 rng(424242);
    const auto nd = [&rng] { return portable_random::normal(rng); };
    const auto ud = [&rng] { return portable_random::uniform(rng, 0.5, 2.0); };
    for(int trial = 0; trial < 40; ++trial)
    {
        const int n = 2 + (trial % 4);
        Eigen::MatrixXd M(n, n);
        for(int i = 0; i < n; ++i)
            for(int j = 0; j < n; ++j)
                M(i, j) = nd();
        osqp_problem p;
        p.P = M.transpose() * M + (1.0 + ud()) * Eigen::MatrixXd::Identity(n, n);
        p.q.resize(n);
        for(int i = 0; i < n; ++i)
            p.q[i] = nd();
        const int m = n + 1;
        p.A = Eigen::MatrixXd::Zero(m, n);
        p.l.resize(m);
        p.u.resize(m);
        for(int i = 0; i < n; ++i)
        {
            p.A(i, i) = 1.0;
            const double b = ud();
            p.l[i] = -b;
            p.u[i] = b;
        }
        for(int j = 0; j < n; ++j)
            p.A(m - 1, j) = nd();
        p.l[m - 1] = -2.0 * ud();
        p.u[m - 1] = 2.0 * ud();
        out.push_back(std::move(p));
    }
    return out;
}

// A seeded family carrying a genuine equality row alongside boxes that are
// mostly inactive. It is the shape that exposes the polish's active-set frailty:
// the equality row's multiplier is large, every inactive box multiplier is pure
// round-off, and a round-off sign read as a bound assignment pins an interior
// variable to a bound.
std::vector<osqp_problem> equality_family(int trials)
{
    std::vector<osqp_problem> out;
    std::mt19937 rng(20260722);
    const auto nd = [&rng] { return portable_random::normal(rng); };
    const auto ud = [&rng] { return portable_random::uniform(rng, 0.3, 1.5); };
    for(int trial = 0; trial < trials; ++trial)
    {
        const int n = 3 + (trial % 4);
        Eigen::MatrixXd M(n, n);
        for(int i = 0; i < n; ++i)
            for(int j = 0; j < n; ++j)
                M(i, j) = nd();
        osqp_problem p;
        p.P = M.transpose() * M + 0.5 * Eigen::MatrixXd::Identity(n, n);
        p.q.resize(n);
        for(int i = 0; i < n; ++i)
            p.q[i] = 3.0 * nd();
        const int m = n + 1;
        p.A = Eigen::MatrixXd::Zero(m, n);
        p.l.resize(m);
        p.u.resize(m);
        for(int i = 0; i < n; ++i)
        {
            p.A(i, i) = 1.0;
            const double b = ud();
            p.l[i] = -b;
            p.u[i] = b;
        }
        for(int j = 0; j < n; ++j)
            p.A(m - 1, j) = 1.0;
        p.l[m - 1] = 0.5;
        p.u[m - 1] = 0.5;
        out.push_back(std::move(p));
    }
    return out;
}

// An ill-conditioned diagonal objective with box rows plus two coupling rows --
// the sparse twin of the dense warm-start fixture.
osqp_problem warm_fixture()
{
    const int n = 5;
    const int m = n + 2;
    osqp_problem p;
    p.P = Eigen::MatrixXd::Zero(n, n);
    for(int i = 0; i < n; ++i)
        p.P(i, i) = std::pow(50.0, static_cast<double>(i) / (n - 1));
    p.q.resize(n);
    for(int i = 0; i < n; ++i)
        p.q[i] = (i % 2 ? 1.0 : -1.0) * (i + 1);
    p.A = Eigen::MatrixXd::Zero(m, n);
    p.l.resize(m);
    p.u.resize(m);
    for(int i = 0; i < n; ++i)
    {
        p.A(i, i) = 1.0;
        p.l[i] = -0.2;
        p.u[i] = 0.2;
    }
    p.A(n, 0) = 1.0;
    p.A(n, 1) = 1.0;
    p.l[n] = -0.1;
    p.u[n] = 0.1;
    for(int j = 0; j < n; ++j)
        p.A(n + 1, j) = 1.0;
    p.l[n + 1] = -0.3;
    p.u[n + 1] = 0.3;
    return p;
}

// The leading m rows of a fixed pool over one strictly convex objective, the
// same construction the dense seed cases use. The unconstrained minimizer sits
// outside the boxes, so the multipliers are nonzero and a consumed dual seed
// changes the trajectory.
osqp_problem seed_problem(int m)
{
    osqp_problem p;
    p.P = Eigen::MatrixXd{{2.0, 0.5, 0.0}, {0.5, 1.0, 0.0}, {0.0, 0.0, 1.5}};
    p.q = Eigen::VectorXd{{-3.0, 2.0, -1.0}};
    const Eigen::MatrixXd pool{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 1.0}, {1.0, -1.0, 0.0}};
    const Eigen::VectorXd lower{{-0.3, -0.4, -0.5, -1.0}};
    const Eigen::VectorXd upper{{0.3, 0.4, 0.5, 1.0}};
    p.A = pool.topRows(m);
    p.l = lower.head(m);
    p.u = upper.head(m);
    return p;
}

std::optional<qp_error> refusal(const expected<qp_result<double>, qp_error>& result)
{
    if(result)
        return std::nullopt;
    return result.error();
}

bool same_bits(const Eigen::VectorXd& a, const Eigen::VectorXd& b)
{
    if(a.size() != b.size())
        return false;
    for(Eigen::Index i = 0; i < a.size(); ++i)
        if(std::bit_cast<std::uint64_t>(a[i]) != std::bit_cast<std::uint64_t>(b[i]))
            return false;
    return true;
}

bool same_solve(const qp_result<double>& a, const qp_result<double>& b)
{
    return same_bits(a.x, b.x) && same_bits(a.y, b.y) && a.iterations == b.iterations
           && a.warm_started == b.warm_started;
}

const Eigen::VectorXd seed_x{{0.1, -0.2, 0.05}};
const Eigen::VectorXd seed_y{{0.4, -0.3, 0.2, 0.1}};

// Polish snaps nearby iterates onto one active-set solution and termination is
// checked only every few iterations; both can hide which seed was consumed. A
// short unpolished run exposes the raw trajectory instead.
sparse_qp_options trajectory_options()
{
    sparse_qp_options opts;
    opts.polish = false;
    opts.check_termination = 1;
    opts.max_iterations = 8;
    return opts;
}

// Poses the four-row seed problem, then resolves it from the given seed, so the
// pose-time reshaping never sees the seed.
expected<qp_result<double>, qp_error> resolve_from(const Eigen::VectorXd& x, const Eigen::VectorXd& y,
                                                   const sparse_qp_options& resolve_opts)
{
    const osqp_problem p = seed_problem(4);
    sparse_admm_qp_solver<double> solver;
    REQUIRE(solver.solve(to_sparse(p.P), p.q, to_sparse(p.A), p.l, p.u, trajectory_options()));
    solver.warm_start(x, y);
    return solver.resolve(1.05 * p.q, p.l, p.u, resolve_opts);
}

expected<qp_result<double>, qp_error> boxed_solve(const Eigen::MatrixXd& P, bool validate)
{
    const Eigen::VectorXd q{{1.0, -1.0}};
    const Eigen::MatrixXd A = Eigen::MatrixXd::Identity(2, 2);
    const Eigen::VectorXd l{{-1.0, -1.0}};
    const Eigen::VectorXd u{{1.0, 1.0}};
    sparse_qp_options opts;
    opts.validate_convexity = validate;
    sparse_admm_qp_solver<double> solver;
    return solver.solve(P.sparseView(0.0, 0.0), q, to_sparse(A), l, u, opts);
}

}

// ---------------------------------------------------------------------------
// Parity against the dense operator-splitting solver over the committed
// problem set.
//
// The dense solver is a legitimate oracle here because its own accuracy is
// enforced against an independent active-set reference, so agreement with it is
// a transitive accuracy statement rather than two implementations agreeing on
// the same mistake. The leg does not rest on that transitivity alone: every
// committed problem carries an independently verified optimum, and the SPARSE
// objective is checked against that absolute anchor directly.
//
// Committed bounds: 1e-8 on the primal iterate in infinity norm, 1e-9 relative
// on the objective. Measured worst cases over the set are 1.11e-16 primal and
// 6.19e-16 for the objective both against the dense solver and against the
// verified optimum -- machine precision, so the bounds retain six to eight
// orders of headroom and are accuracy gates rather than loosened passes.
// ---------------------------------------------------------------------------
TEST_CASE("sparse_admm_qp matches the dense solver on the committed problem set",
          "[qp][sparse_parity]")
{
    constexpr double primal_tol = 1e-8;
    constexpr double objective_tol = 1e-9;

    struct fixture
    {
        const char* label;
        int n;
        int m;
        const double* P;
        const double* q;
        const double* A;
        const double* l;
        const double* u;
        bool has_optimum;
        double optimum;
    };

    std::vector<fixture> fixtures;
#define MM_PROBLEM(nm)                                                                    \
    fixtures.push_back(fixture{argmin::mm_data::nm::label, argmin::mm_data::nm::n,        \
                               argmin::mm_data::nm::m, argmin::mm_data::nm::P.data(),     \
                               argmin::mm_data::nm::q.data(), argmin::mm_data::nm::A.data(),\
                               argmin::mm_data::nm::l.data(), argmin::mm_data::nm::u.data(),\
                               argmin::mm_data::nm::has_optimum, argmin::mm_data::nm::optimum});
#include "mm_data/mm_problems.inc"
#undef MM_PROBLEM

    REQUIRE_FALSE(fixtures.empty());

    double worst_primal = 0.0;
    double worst_objective = 0.0;
    double worst_optimum = 0.0;
    for(const auto& f : fixtures)
    {
        INFO("problem " << f.label);
        const Eigen::MatrixXd Pd = Eigen::Map<const row_major>(f.P, f.n, f.n);
        const Eigen::VectorXd q = Eigen::Map<const Eigen::VectorXd>(f.q, f.n);
        const Eigen::MatrixXd Ad = Eigen::Map<const row_major>(f.A, f.m, f.n);
        const Eigen::VectorXd l = Eigen::Map<const Eigen::VectorXd>(f.l, f.m);
        const Eigen::VectorXd u = Eigen::Map<const Eigen::VectorXd>(f.u, f.m);

        const sparse Ps = to_sparse(Pd);
        const sparse As = to_sparse(Ad);
        CHECK(is_full_symmetric(Ps));

        dense_admm_qp_solver<double> dense(f.n, f.m);
        auto rd = dense.solve(Pd, q, Ad, l, u);
        REQUIRE(rd);

        sparse_admm_qp_solver<double> solver;
        auto rs = solver.solve(Ps, q, As, l, u);
        REQUIRE(rs);

        CHECK(rs->status == rd->status);

        const double dp = (rs->x - rd->x).cwiseAbs().maxCoeff();
        const double dobj = relative_gap(rs->objective_value, rd->objective_value);
        worst_primal = std::max(worst_primal, dp);
        worst_objective = std::max(worst_objective, dobj);
        CHECK(dp <= primal_tol);
        CHECK(dobj <= objective_tol);

        if(f.has_optimum)
        {
            const double dopt = relative_gap(rs->objective_value, f.optimum);
            worst_optimum = std::max(worst_optimum, dopt);
            CHECK(dopt <= objective_tol);
        }
    }

    std::printf("[sparse parity, committed set] worst primal %.3e, worst objective %.3e, "
                "worst objective-vs-optimum %.3e\n",
                worst_primal, worst_objective, worst_optimum);
}

// ---------------------------------------------------------------------------
// Parity over the seeded strictly-convex family. Strict convexity makes the
// primal minimizer unique, so agreement is a real statement about the answer
// rather than a coincidence of two runs of the same algorithm.
//
// The dual bound is deliberately looser than the primal one: a degenerate
// active set admits a non-unique dual, so two correct solvers may legitimately
// return different multipliers for the same primal point. Committed bounds are
// 1e-8 primal and 1e-6 dual; measured worst cases over the family are 1.67e-16
// primal and 1.11e-16 dual.
// ---------------------------------------------------------------------------
TEST_CASE("sparse_admm_qp matches the dense solver on the seeded convex family",
          "[qp][sparse_parity]")
{
    constexpr double primal_tol = 1e-8;
    constexpr double dual_tol = 1e-6;

    double worst_primal = 0.0;
    double worst_dual = 0.0;
    for(const auto& p : convex_family())
    {
        const int n = static_cast<int>(p.P.rows());
        const int m = static_cast<int>(p.A.rows());
        const sparse Ps = to_sparse(p.P);
        const sparse As = to_sparse(p.A);
        CHECK(is_full_symmetric(Ps));

        dense_admm_qp_solver<double> dense(n, m);
        auto rd = dense.solve(p.P, p.q, p.A, p.l, p.u);
        REQUIRE(rd);

        sparse_admm_qp_solver<double> solver;
        auto rs = solver.solve(Ps, p.q, As, p.l, p.u);
        REQUIRE(rs);

        CHECK(rd->status == qp_solve_status::solved);
        CHECK(rs->status == qp_solve_status::solved);
        CHECK(rd->polished);
        CHECK(rs->polished);

        const double dp = (rs->x - rd->x).cwiseAbs().maxCoeff();
        const double dy = (rs->y - rd->y).cwiseAbs().maxCoeff();
        worst_primal = std::max(worst_primal, dp);
        worst_dual = std::max(worst_dual, dy);
        CHECK(dp <= primal_tol);
        CHECK(dy <= dual_tol);
    }

    std::printf("[sparse parity, convex family] worst primal %.3e, worst dual %.3e\n",
                worst_primal, worst_dual);
}

// ---------------------------------------------------------------------------
// Warm-start correctness: a vectors-only resolve after a ~1% perturbation
// reaches the cold answer in strictly fewer iterations and reuses the frozen
// Ruiz factors. The scaling assertions are exact equality, not tolerance
// comparisons, because the claim is that no re-equilibration happened at all.
// ---------------------------------------------------------------------------
TEST_CASE("sparse_admm_qp warm resolve converges faster without re-equilibrating",
          "[qp][sparse_warm]")
{
    const osqp_problem p = warm_fixture();
    const sparse Ps = to_sparse(p.P);
    const sparse As = to_sparse(p.A);

    sparse_admm_qp_solver<double> solver;
    auto cold = solver.solve(Ps, p.q, As, p.l, p.u);
    REQUIRE(cold);

    const double c0 = solver.scaling_cost();
    const Eigen::VectorXd D0 = solver.scaling_primal();
    const Eigen::VectorXd E0 = solver.scaling_dual();

    auto same = solver.resolve(p.q, p.l, p.u);
    REQUIRE(same);
    CHECK(same->iterations <= 25);

    CHECK(solver.scaling_cost() == c0);
    CHECK((solver.scaling_primal() - D0).cwiseAbs().maxCoeff() == 0.0);
    CHECK((solver.scaling_dual() - E0).cwiseAbs().maxCoeff() == 0.0);

    const Eigen::VectorXd q2 = p.q * 1.01;
    const Eigen::VectorXd l2 = p.l * 1.01;
    const Eigen::VectorXd u2 = p.u * 1.01;
    auto warm = solver.resolve(q2, l2, u2);
    REQUIRE(warm);

    sparse_admm_qp_solver<double> fresh;
    auto cold_perturbed = fresh.solve(Ps, q2, As, l2, u2);
    REQUIRE(cold_perturbed);

    CHECK((warm->x - cold_perturbed->x).cwiseAbs().maxCoeff() <= 1e-6);
    CHECK(warm->iterations < cold_perturbed->iterations);
}

// ---------------------------------------------------------------------------
// The factor-once-at-pose contract, asserted behaviorally rather than by source
// inspection. The solver counts the symbolic analyses and numeric
// factorizations it actually performs, so "resolve does no pattern work and no
// factorization" becomes an observation instead of a claim: with the adaptive
// rho schedule disabled a resolve moves neither counter at all, and with it
// enabled the only counter that moves is the numeric one.
//
// The polish is counted on its own instrument. It reuses its reduced-KKT
// analysis and factor whenever the active-set pattern, the delta and the pose
// are unchanged, so on this stable-pattern resolve the polish counters stay
// frozen -- the reuse is what the frozen counter observes -- while folding them
// into the iteration counts would hide exactly the distinction being pinned.
// ---------------------------------------------------------------------------
TEST_CASE("sparse_admm_qp resolve performs no analysis and no unscheduled refactorization",
          "[qp][sparse_resolve_work]")
{
    const osqp_problem p = warm_fixture();
    const sparse Ps = to_sparse(p.P);
    const sparse As = to_sparse(p.A);

    sparse_qp_options fixed_rho;
    fixed_rho.adaptive_rho = false;

    sparse_admm_qp_solver<double> solver;
    auto cold = solver.solve(Ps, p.q, As, p.l, p.u, fixed_rho);
    REQUIRE(cold);

    const std::size_t analyses_after_pose = solver.iteration_analyses();
    const std::size_t factorizations_after_pose = solver.iteration_factorizations();
    CHECK(analyses_after_pose == 1u);
    CHECK(factorizations_after_pose == 1u);

    const Eigen::VectorXd q2 = p.q * 1.01;
    const Eigen::VectorXd l2 = p.l * 1.01;
    const Eigen::VectorXd u2 = p.u * 1.01;

    auto fixed = solver.resolve(q2, l2, u2, fixed_rho);
    REQUIRE(fixed);
    CHECK(solver.iteration_analyses() == analyses_after_pose);
    CHECK(solver.iteration_factorizations() == factorizations_after_pose);

    // Cold-started so the run is long enough for the schedule to fire; the
    // pattern is untouched, so any factorization it triggers is values-only.
    sparse_qp_options adaptive;
    adaptive.warm_start = false;
    REQUIRE(adaptive.adaptive_rho);

    // The pattern, the delta and the pose are all unchanged across this resolve,
    // so the polish reuses its analysis and numeric factor: both polish counters
    // stay frozen -- that frozen counter IS the observable reuse -- while the
    // iteration factorization still moves once for the scheduled rho update.
    const std::size_t polish_analyses_before = solver.polish_analyses();
    const std::size_t polish_factorizations_before = solver.polish_factorizations();
    auto scheduled = solver.resolve(p.q, p.l, p.u, adaptive);
    REQUIRE(scheduled);
    CHECK(solver.iteration_analyses() == analyses_after_pose);
    CHECK(solver.iteration_factorizations() > factorizations_after_pose);
    CHECK(scheduled->iterations >= static_cast<int>(adaptive.adaptive_rho_interval));

    CHECK(solver.polish_analyses() == polish_analyses_before);
    CHECK(solver.polish_factorizations() == polish_factorizations_before);
    CHECK(solver.polish_analyses() == solver.polish_factorizations());
}

// ---------------------------------------------------------------------------
// The polish may never return an answer worse than the iterate it was handed,
// and on this family the polished answer is gated directly against the
// independent active-set oracle rather than against another operator-splitting
// solver.
//
// This is a regression gate for a real and severe failure, not a theoretical
// nicety. Selecting the active set from the strict sign of the dual -- the
// reference implementation's rule -- reads the arbitrary sign of a round-off
// multiplier as a bound assignment and pins an interior variable to a bound.
// The resulting point is primal feasible and solves its own reduced system
// exactly, so both residuals improve and a residual-only accept test takes it:
// a grossly suboptimal answer, labeled solved. Over a 400-member superset of
// this family that produced a relative objective error against the active-set
// oracle of up to 11.3 on 75 problems. With the relative activity threshold on
// the multipliers and the tolerance-scaled objective guard in place, the
// measured worst relative objective error over that superset is 2.96e-6 with
// nothing above 1e-4, and the polish is still accepted on 399 of 400 members --
// the guards reject broken polishes, not all of them. Over the 60 members
// exercised here the polish is accepted on every one, the worst objective gap
// against the oracle is 5.16e-14 and the worst regression against the solver's
// own unpolished iterate is 8.78e-7, so the committed 1e-4 bound keeps ten
// orders of headroom on the first and two on the second.
// ---------------------------------------------------------------------------
TEST_CASE("sparse_admm_qp polish agrees with the active-set oracle on equality-constrained rows",
          "[qp][sparse_polish_guard]")
{
    constexpr double oracle_tol = 1e-4;

    sparse_qp_options with_polish;
    sparse_qp_options without_polish;
    without_polish.polish = false;

    int accepted = 0;
    int members = 0;
    double worst_oracle_gap = 0.0;
    double worst_regression = 0.0;
    for(const auto& p : equality_family(60))
    {
        const int n = static_cast<int>(p.P.rows());
        const int m = static_cast<int>(p.A.rows());
        const sparse Ps = to_sparse(p.P);
        const sparse As = to_sparse(p.A);

        Eigen::MatrixXd A_eq(1, n);
        A_eq.row(0) = p.A.row(m - 1);
        Eigen::VectorXd b_eq(1);
        b_eq[0] = p.l[m - 1];
        const Eigen::MatrixXd A_in(0, n);
        const Eigen::VectorXd b_in(0);
        detail::kraft_lsq_qp_solver<double> oracle;
        oracle.resize(n, 1, 0, n, n);
        auto ref = oracle.solve(p.P, p.q, A_eq, b_eq, A_in, b_in, p.l.head(n), p.u.head(n));
        const double reference_objective =
            0.5 * ref.x.dot(p.P * ref.x) + p.q.dot(ref.x);

        sparse_admm_qp_solver<double> polished_solver;
        auto rp = polished_solver.solve(Ps, p.q, As, p.l, p.u, with_polish);
        REQUIRE(rp);
        sparse_admm_qp_solver<double> raw_solver;
        auto rr = raw_solver.solve(Ps, p.q, As, p.l, p.u, without_polish);
        REQUIRE(rr);
        if(rr->status != qp_solve_status::solved)
            continue;

        ++members;
        if(rp->polished)
            ++accepted;

        const double oracle_gap = relative_gap(rp->objective_value, reference_objective);
        worst_oracle_gap = std::max(worst_oracle_gap, oracle_gap);
        CHECK(oracle_gap <= oracle_tol);

        const double regression =
            (rp->objective_value - rr->objective_value) / std::max(1.0, std::abs(reference_objective));
        worst_regression = std::max(worst_regression, regression);
        CHECK(regression <= oracle_tol);

        // Feasibility at the solver's own convergence tolerance, which is what
        // an accepted iterate is entitled to; the polished point is normally
        // exact on its active rows.
        const Eigen::VectorXd ax = p.A * rp->x;
        CHECK((ax - p.l).minCoeff() >= -1e-5);
        CHECK((p.u - ax).minCoeff() >= -1e-5);
    }

    REQUIRE(members > 0);
    CHECK(accepted * 2 > members);
    std::printf("[sparse polish guard] %d/%d members polished, worst objective gap against the "
                "active-set oracle %.3e, worst regression %.3e\n",
                accepted, members, worst_oracle_gap, worst_regression);
}

// ---------------------------------------------------------------------------
// Primal-infeasibility certificate on a trivially infeasible box: x >= 1 and
// x <= 0, the same construction the dense suite uses.
// ---------------------------------------------------------------------------
TEST_CASE("sparse_admm_qp certifies a trivially infeasible box", "[qp][sparse_infeasible]")
{
    const int n = 1;
    const int m = 2;
    Eigen::MatrixXd Pd = Eigen::MatrixXd::Identity(n, n);
    Eigen::VectorXd q(n);
    q << 0.0;
    Eigen::MatrixXd Ad(m, n);
    Ad << 1.0, 1.0;
    Eigen::VectorXd l(m);
    l << 1.0, -inf;
    Eigen::VectorXd u(m);
    u << inf, 0.0;

    dense_admm_qp_solver<double> dense(n, m);
    auto rd = dense.solve(Pd, q, Ad, l, u);
    REQUIRE(rd);

    sparse_admm_qp_solver<double> solver;
    auto rs = solver.solve(to_sparse(Pd), q, to_sparse(Ad), l, u);
    REQUIRE(rs);
    CHECK(rs->status == qp_solve_status::primal_infeasible);
    CHECK(rs->status == rd->status);
}

// ---------------------------------------------------------------------------
// Argument and precondition violations. Every one of these is a returned error
// value: the solver's posture is exception-free, so a violation must never
// throw and must never come back as a plausible-looking result.
// ---------------------------------------------------------------------------
TEST_CASE("sparse_admm_qp reports argument and precondition violations on the error channel",
          "[qp][sparse_contract]")
{
    const int n = 2;
    const int m = 2;
    Eigen::MatrixXd Pd = Eigen::MatrixXd::Identity(n, n);
    Eigen::VectorXd q(n);
    q << -1.0, -1.0;
    Eigen::MatrixXd Ad = Eigen::MatrixXd::Identity(m, n);
    Eigen::VectorXd l = Eigen::VectorXd::Constant(m, -1.0);
    Eigen::VectorXd u = Eigen::VectorXd::Constant(m, 1.0);
    const sparse Ps = to_sparse(Pd);
    const sparse As = to_sparse(Ad);

    sparse_admm_qp_solver<double> solver;
    auto unposed = solver.resolve(q, l, u);
    REQUIRE_FALSE(unposed);
    CHECK(unposed.error() == qp_error::invalid_problem);

    auto posed = solver.solve(Ps, q, As, l, u);
    REQUIRE(posed);

    // A differently shaped problem cannot enter through the resolve path at
    // all, which is what makes the pattern genuinely frozen at pose.
    auto short_q = solver.resolve(Eigen::VectorXd::Zero(n + 1), l, u);
    REQUIRE_FALSE(short_q);
    CHECK(short_q.error() == qp_error::dimension_mismatch);

    auto short_l = solver.resolve(q, Eigen::VectorXd::Zero(m + 1), u);
    REQUIRE_FALSE(short_l);
    CHECK(short_l.error() == qp_error::dimension_mismatch);

    auto short_u = solver.resolve(q, l, Eigen::VectorXd::Zero(m + 1));
    REQUIRE_FALSE(short_u);
    CHECK(short_u.error() == qp_error::dimension_mismatch);

    Eigen::VectorXd q_nan = q;
    q_nan[0] = std::numeric_limits<double>::quiet_NaN();
    sparse_admm_qp_solver<double> nan_solver;
    auto nonfinite = nan_solver.solve(Ps, q_nan, As, l, u);
    REQUIRE_FALSE(nonfinite);
    CHECK(nonfinite.error() == qp_error::non_finite_input);

    Eigen::VectorXd l_crossed = l;
    l_crossed[0] = 2.0;
    sparse_admm_qp_solver<double> crossed_solver;
    auto crossed = crossed_solver.solve(Ps, q, As, l_crossed, u);
    REQUIRE_FALSE(crossed);
    CHECK(crossed.error() == qp_error::invalid_bounds);
}

// ---------------------------------------------------------------------------
// The re-pose lifecycle: the only legal route by which the sparsity pattern may
// change. This guards the facility's direct value-array writes, where a slot
// table surviving a shape change would corrupt silently rather than fail
// loudly. One instance solves pattern A, then a differently shaped pattern B,
// then A again -- each verified against the dense solver at the parity bound,
// with A's second answer required to be bit-for-bit its first.
// ---------------------------------------------------------------------------
TEST_CASE("sparse_admm_qp re-poses a used instance onto a different sparsity pattern",
          "[qp][sparse_repose]")
{
    constexpr double parity_tol = 1e-8;

    osqp_problem a;
    a.P.resize(2, 2);
    a.P << 2.0, 0.5, 0.5, 3.0;
    a.q.resize(2);
    a.q << -1.0, -2.0;
    a.A = Eigen::MatrixXd::Identity(2, 2);
    a.l = Eigen::VectorXd::Constant(2, -0.4);
    a.u = Eigen::VectorXd::Constant(2, 0.4);

    osqp_problem b;
    b.P = Eigen::MatrixXd::Zero(3, 3);
    b.P(0, 0) = 4.0;
    b.P(1, 1) = 1.0;
    b.P(2, 2) = 2.0;
    b.P(0, 2) = 0.25;
    b.P(2, 0) = 0.25;
    b.q.resize(3);
    b.q << 1.0, -3.0, 0.5;
    b.A = Eigen::MatrixXd::Zero(4, 3);
    for(int i = 0; i < 3; ++i)
        b.A(i, i) = 1.0;
    b.A(3, 0) = 1.0;
    b.A(3, 1) = 1.0;
    b.A(3, 2) = 1.0;
    b.l.resize(4);
    b.l << -1.0, -1.0, -1.0, 0.5;
    b.u.resize(4);
    b.u << 1.0, 1.0, 1.0, 2.0;

    const sparse Pa = to_sparse(a.P);
    const sparse Aa = to_sparse(a.A);
    const sparse Pb = to_sparse(b.P);
    const sparse Ab = to_sparse(b.A);
    REQUIRE(is_full_symmetric(Pa));
    REQUIRE(is_full_symmetric(Pb));

    dense_admm_qp_solver<double> dense_a(2, 2);
    auto ref_a = dense_a.solve(a.P, a.q, a.A, a.l, a.u);
    REQUIRE(ref_a);
    dense_admm_qp_solver<double> dense_b(3, 4);
    auto ref_b = dense_b.solve(b.P, b.q, b.A, b.l, b.u);
    REQUIRE(ref_b);

    sparse_admm_qp_solver<double> solver;
    auto first_a = solver.solve(Pa, a.q, Aa, a.l, a.u);
    REQUIRE(first_a);
    CHECK((first_a->x - ref_a->x).cwiseAbs().maxCoeff() <= parity_tol);
    const Eigen::VectorXd x_a = first_a->x;
    const double c_a = solver.scaling_cost();
    const Eigen::VectorXd D_a = solver.scaling_primal();
    const Eigen::VectorXd E_a = solver.scaling_dual();

    auto on_b = solver.solve(Pb, b.q, Ab, b.l, b.u);
    REQUIRE(on_b);
    CHECK((on_b->x - ref_b->x).cwiseAbs().maxCoeff() <= parity_tol);

    // The equilibration is recomputed for the new data, which is correct on a
    // re-pose and would be wrong on a resolve.
    CHECK(solver.scaling_primal().size() == 3);
    CHECK(solver.scaling_dual().size() == 4);
    CHECK(solver.scaling_cost() != c_a);

    auto wrong_shape = solver.resolve(a.q, a.l, a.u);
    REQUIRE_FALSE(wrong_shape);
    CHECK(wrong_shape.error() == qp_error::dimension_mismatch);

    auto second_a = solver.solve(Pa, a.q, Aa, a.l, a.u);
    REQUIRE(second_a);
    CHECK((second_a->x - ref_a->x).cwiseAbs().maxCoeff() <= parity_tol);
    CHECK((second_a->x - x_a).cwiseAbs().maxCoeff() == 0.0);
    CHECK(solver.scaling_cost() == c_a);
    CHECK((solver.scaling_primal() - D_a).cwiseAbs().maxCoeff() == 0.0);
    CHECK((solver.scaling_dual() - E_a).cwiseAbs().maxCoeff() == 0.0);
}

// ---------------------------------------------------------------------------
// Option values outside their domain are refused on the error channel at both
// entry points, with the value and the field set the dense splitting solver
// uses: the two are compared against each other, so a domain one refuses and
// the other accepts is a defect in whichever of them moved.
// ---------------------------------------------------------------------------
TEST_CASE("sparse_admm_qp rejects option values outside their domain", "[qp_options]")
{
    const osqp_problem p = seed_problem(4);
    const sparse Ps = to_sparse(p.P);
    const sparse As = to_sparse(p.A);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const auto rejected = [&](const auto& mutate)
    {
        sparse_qp_options opts;
        mutate(opts);
        sparse_admm_qp_solver<double> fresh;
        const auto solved = fresh.solve(Ps, p.q, As, p.l, p.u, opts);
        sparse_admm_qp_solver<double> posed;
        REQUIRE(posed.solve(Ps, p.q, As, p.l, p.u));
        const auto resolved = posed.resolve(p.q, p.l, p.u, opts);
        return refusal(solved) == qp_error::invalid_options
               && refusal(resolved) == qp_error::invalid_options;
    };
    const auto agrees_with_dense = [&](const auto& mutate)
    {
        sparse_qp_options sparse_opts;
        mutate(sparse_opts);
        dense_qp_options dense_opts;
        mutate(dense_opts);
        sparse_admm_qp_solver<double> sparse_solver;
        dense_admm_qp_solver<double> dense_solver(3, 4);
        return refusal(sparse_solver.solve(Ps, p.q, As, p.l, p.u, sparse_opts))
               == refusal(dense_solver.solve(p.P, p.q, p.A, p.l, p.u, dense_opts));
    };

    SECTION("penalty parameter")
    {
        for(const double v : {0.0, -0.1, nan, inf})
        {
            const auto set = [v](auto& o) { o.rho = v; };
            CHECK(rejected(set));
            CHECK(agrees_with_dense(set));
        }
    }
    SECTION("regularization")
    {
        for(const double v : {0.0, -1e-6, nan, inf})
        {
            const auto set = [v](auto& o) { o.sigma = v; };
            CHECK(rejected(set));
            CHECK(agrees_with_dense(set));
        }
    }
    SECTION("over-relaxation")
    {
        for(const double v : {0.0, 2.0, -0.5, 2.5, nan, inf})
        {
            const auto set = [v](auto& o) { o.alpha = v; };
            CHECK(rejected(set));
            CHECK(agrees_with_dense(set));
        }
    }
    SECTION("polish regularization")
    {
        for(const double v : {0.0, -1e-6, nan, inf})
        {
            const auto set = [v](auto& o) { o.delta = v; };
            CHECK(rejected(set));
            CHECK(agrees_with_dense(set));
        }
    }
    SECTION("adaptive ratio threshold")
    {
        for(const double v : {0.999, 0.0, nan, inf})
        {
            const auto set = [v](auto& o) { o.adaptive_rho_tolerance = v; };
            CHECK(rejected(set));
            CHECK(agrees_with_dense(set));
        }
    }
    SECTION("convergence and certificate tolerances")
    {
        const auto both_zero = [](auto& o)
        {
            o.eps_abs = 0.0;
            o.eps_rel = 0.0;
        };
        CHECK(rejected(both_zero));
        CHECK(agrees_with_dense(both_zero));
        for(const double v : {-1e-6, nan, inf})
        {
            const auto set_abs = [v](auto& o) { o.eps_abs = v; };
            const auto set_rel = [v](auto& o) { o.eps_rel = v; };
            CHECK(rejected(set_abs));
            CHECK(rejected(set_rel));
            CHECK(agrees_with_dense(set_abs));
            CHECK(agrees_with_dense(set_rel));
        }
        for(const double v : {0.0, -1e-4, nan, inf})
        {
            const auto set_prim = [v](auto& o) { o.eps_prim_inf = v; };
            const auto set_dual = [v](auto& o) { o.eps_dual_inf = v; };
            CHECK(rejected(set_prim));
            CHECK(rejected(set_dual));
            CHECK(agrees_with_dense(set_prim));
            CHECK(agrees_with_dense(set_dual));
        }
    }
    SECTION("iteration budget")
    {
        const auto set = [](auto& o) { o.max_iterations = 0; };
        CHECK(rejected(set));
        CHECK(agrees_with_dense(set));
    }
    SECTION("in-domain boundary values solve")
    {
        const auto unit_threshold = [](auto& o) { o.adaptive_rho_tolerance = 1.0; };
        const auto absolute_only = [](auto& o) { o.eps_rel = 0.0; };
        const auto coerced = [](auto& o)
        {
            o.check_termination = 0;
            o.adaptive_rho_interval = 0;
        };
        const auto solves = [&](const auto& mutate)
        {
            sparse_qp_options opts;
            mutate(opts);
            sparse_admm_qp_solver<double> solver;
            const auto result = solver.solve(Ps, p.q, As, p.l, p.u, opts);
            REQUIRE(result);
            CHECK(result->status == qp_solve_status::solved);
            CHECK(solver.resolve(p.q, p.l, p.u, opts));
            CHECK(agrees_with_dense(mutate));
        };
        solves([](auto&) {});
        solves(unit_threshold);
        solves(absolute_only);
        solves(coerced);
    }
}

// ---------------------------------------------------------------------------
// A system the no-pivot factorization cannot complete is a numerical failure,
// not a defect in the caller's problem, and is reported as one; the solver then
// refuses a resolve until it is posed again. With equilibration off, P = -sigma
// cancels the regularization exactly and leaves a zero pivot.
// ---------------------------------------------------------------------------
TEST_CASE("sparse_admm_qp reports a factorization failure as a factorization failure",
          "[qp_factorization]")
{
    sparse_qp_options opts;
    opts.scaling = 0;
    const osqp_problem valid = seed_problem(4);

    SECTION("at pose")
    {
        const sparse P = to_sparse(Eigen::MatrixXd{{-opts.sigma}});
        const Eigen::VectorXd q{{1.0}};
        const sparse A(0, 1);
        const Eigen::VectorXd empty(0);
        sparse_admm_qp_solver<double> solver;
        REQUIRE(solver.solve(to_sparse(valid.P), valid.q, to_sparse(valid.A), valid.l, valid.u));
        const auto result = solver.solve(P, q, A, empty, empty, opts);
        REQUIRE_FALSE(result);
        CHECK(result.error() == qp_error::factorization_failed);
        CHECK_FALSE(result.error() == qp_error::invalid_problem);
        const auto stale = solver.resolve(q, empty, empty, opts);
        REQUIRE_FALSE(stale);
        CHECK(stale.error() == qp_error::invalid_problem);
    }
    SECTION("after an accepted penalty update")
    {
        // P + sigma = -1e6 makes the system singular exactly at the penalty
        // ceiling. Without clipping z_tilde = A x_tilde, so the first primal
        // residual would be pure round-off; q puts x_tilde = q / (1e6 - rho) = 10
        // past u = 1, the clip leaves residual ratio sqrt(10), and the first
        // update from 9e5 lands on the clamp regardless of FP contraction.
        const double ceiling = 1e6;
        opts.rho = 9e5;
        opts.adaptive_rho_interval = 1;
        opts.adaptive_rho_tolerance = 1.0;
        opts.check_termination = 1000;
        opts.max_iterations = 50;
        const sparse P = to_sparse(Eigen::MatrixXd{{-ceiling - opts.sigma}});
        REQUIRE(-ceiling - opts.sigma + opts.sigma == -ceiling);
        const Eigen::VectorXd q{{1e6}};
        const sparse A = to_sparse(Eigen::MatrixXd{{1.0}});
        const Eigen::VectorXd l{{-1.0}};
        const Eigen::VectorXd u{{1.0}};
        sparse_qp_options fixed_penalty = opts;
        fixed_penalty.adaptive_rho = false;
        sparse_admm_qp_solver<double> control;
        REQUIRE(control.solve(P, q, A, l, u, fixed_penalty));

        sparse_admm_qp_solver<double> solver;
        const auto result = solver.solve(P, q, A, l, u, opts);
        REQUIRE_FALSE(result);
        CHECK(result.error() == qp_error::factorization_failed);
        CHECK(solver.iteration_factorizations() == 2u);
        const auto stale = solver.resolve(q, l, u, fixed_penalty);
        REQUIRE_FALSE(stale);
        CHECK(stale.error() == qp_error::invalid_problem);
    }
}

// ---------------------------------------------------------------------------
// The seed fallback is observable. A seed counts as used exactly when its
// primal part was consumed: a dual of the wrong length is replaced by zeros
// while the primal is kept, and the result matches an explicit zero-dual seed
// bit for bit; a primal of the wrong length, no seed at all, or a seed carried
// across a change of problem shape starts from zero and says so.
// ---------------------------------------------------------------------------
TEST_CASE("sparse_admm_qp reports whether the caller's seed was used", "[qp_warm]")
{
    const sparse_qp_options probe = trajectory_options();
    sparse_qp_options cold = probe;
    cold.warm_start = false;

    SECTION("a full seed")
    {
        const auto result = resolve_from(seed_x, seed_y, probe);
        REQUIRE(result);
        CHECK(result->warm_started);
        const auto zero_dual = resolve_from(seed_x, Eigen::VectorXd::Zero(4), probe);
        REQUIRE(zero_dual);
        CHECK_FALSE(same_bits(result->x, zero_dual->x));
    }
    SECTION("a primal seed with a dual of the wrong length")
    {
        const auto result = resolve_from(seed_x, seed_y.head(3), probe);
        REQUIRE(result);
        CHECK(result->warm_started);
        const auto control = resolve_from(seed_x, Eigen::VectorXd::Zero(4), probe);
        REQUIRE(control);
        CHECK(same_solve(*result, *control));
    }
    SECTION("a primal seed of the wrong length")
    {
        const auto control = resolve_from(seed_x, seed_y, cold);
        REQUIRE(control);
        CHECK_FALSE(control->warm_started);
        for(const Eigen::VectorXd& x : {Eigen::VectorXd(seed_x.head(2)), Eigen::VectorXd()})
        {
            const auto result = resolve_from(x, seed_y, probe);
            REQUIRE(result);
            CHECK_FALSE(result->warm_started);
            CHECK(same_solve(*result, *control));
        }
    }
    SECTION("no seed")
    {
        const osqp_problem p = seed_problem(4);
        sparse_admm_qp_solver<double> unseeded;
        const auto result = unseeded.solve(to_sparse(p.P), p.q, to_sparse(p.A), p.l, p.u, probe);
        REQUIRE(result);
        CHECK_FALSE(result->warm_started);
        sparse_admm_qp_solver<double> never_warm;
        const auto control = never_warm.solve(to_sparse(p.P), p.q, to_sparse(p.A), p.l, p.u, cold);
        REQUIRE(control);
        CHECK(same_solve(*result, *control));
    }
    SECTION("a seed carried across a change of problem shape")
    {
        const osqp_problem small = seed_problem(2);
        const osqp_problem large = seed_problem(4);
        sparse_admm_qp_solver<double> reshaped;
        REQUIRE(reshaped.solve(to_sparse(small.P), small.q, to_sparse(small.A), small.l, small.u, probe));
        reshaped.warm_start(seed_x, seed_y);
        const auto result =
            reshaped.solve(to_sparse(large.P), large.q, to_sparse(large.A), large.l, large.u, probe);
        REQUIRE(result);
        CHECK_FALSE(result->warm_started);
        sparse_admm_qp_solver<double> fresh;
        const auto control =
            fresh.solve(to_sparse(large.P), large.q, to_sparse(large.A), large.l, large.u, cold);
        REQUIRE(control);
        CHECK(same_solve(*result, *control));
    }
}

// ---------------------------------------------------------------------------
// The system factor reads only the lower triangle of P while the residuals and
// the objective read all of it, so an asymmetric P would be solved as one
// problem and reported as another. Asymmetry at round-off level is accepted;
// a rejected solve leaves the previously posed problem in place; the flag turns
// the check off.
// ---------------------------------------------------------------------------
TEST_CASE("sparse_admm_qp rejects an asymmetric objective matrix", "[qp_convexity]")
{
    const Eigen::MatrixXd unequal{{2.0, 0.5}, {0.7, 1.0}};
    const Eigen::MatrixXd one_sided{{2.0, 0.5}, {0.0, 1.0}};
    for(const Eigen::MatrixXd& asymmetric : {unequal, one_sided})
    {
        const auto rejected = boxed_solve(asymmetric, true);
        REQUIRE_FALSE(rejected);
        CHECK(rejected.error() == qp_error::invalid_problem);
    }

    Eigen::MatrixXd rounded{{2.0, 0.5}, {0.5, 1.0}};
    rounded(1, 0) = std::nextafter(0.5, 1.0);
    const auto accepted = boxed_solve(rounded, true);
    REQUIRE(accepted);
    CHECK(accepted->status == qp_solve_status::solved);

    SECTION("a rejected solve keeps the posed problem")
    {
        const osqp_problem p = seed_problem(4);
        sparse_admm_qp_solver<double> solver;
        REQUIRE(solver.solve(to_sparse(p.P), p.q, to_sparse(p.A), p.l, p.u));
        const Eigen::VectorXd q{{1.0, -1.0}};
        const sparse A = to_sparse(Eigen::MatrixXd::Identity(2, 2));
        const Eigen::VectorXd bound = Eigen::VectorXd::Ones(2);
        CHECK_FALSE(solver.solve(to_sparse(unequal), q, A, -bound, bound));
        CHECK(solver.resolve(p.q, p.l, p.u));
    }
    SECTION("the flag turns the check off")
    {
        CHECK(boxed_solve(unequal, false));
        CHECK(boxed_solve(one_sided, false));
    }
}
