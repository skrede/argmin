#include "float_gate.h"

#include "argmin/qp/dense_admm_qp.h"
#include "argmin/qp/sparse_admm_qp.h"
#include "argmin/qp/dense_active_set_qp.h"

#include "argmin/options/dense_qp_options.h"
#include "argmin/options/sparse_qp_options.h"
#include "argmin/options/dense_active_set_qp_options.h"

#include "argmin/expected.h"
#include "argmin/types.h"

#include "../portable_random.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <catch2/catch_test_macros.hpp>

#include <random>
#include <cstdint>

using namespace argmin;
using namespace argmin::float_gate;

namespace
{

struct qp_problem
{
    Eigen::MatrixXd P;
    Eigen::VectorXd q;
    Eigen::MatrixXd A;
    Eigen::VectorXd l;
    Eigen::VectorXd u;
};

// Seeded SPD P with box rows plus one coupling row, built from portable draws; the coupling row is
// either two-sided or an equality, the shape that stresses the equality-row rho boost.
qp_problem convex_problem(std::int32_t n, std::uint32_t seed, bool equality_row)
{
    std::mt19937 rng(seed);
    const auto nd = [&rng] { return portable_random::normal(rng); };
    const auto ud = [&rng] { return portable_random::uniform(rng, 0.5, 2.0); };
    Eigen::MatrixXd M(n, n);
    for(std::int32_t i = 0; i < n; ++i)
        for(std::int32_t j = 0; j < n; ++j)
            M(i, j) = nd();
    qp_problem p;
    p.P = M.transpose() * M + (1.0 + ud()) * Eigen::MatrixXd::Identity(n, n);
    p.q.resize(n);
    for(std::int32_t i = 0; i < n; ++i)
        p.q[i] = nd();
    const std::int32_t m = n + 1;
    p.A = Eigen::MatrixXd::Zero(m, n);
    p.l.resize(m);
    p.u.resize(m);
    for(std::int32_t i = 0; i < n; ++i)
    {
        p.A(i, i) = 1.0;
        const double b = ud();
        p.l[i] = -b;
        p.u[i] = b;
    }
    for(std::int32_t j = 0; j < n; ++j)
        p.A(m - 1, j) = equality_row ? 1.0 : nd();
    p.l[m - 1] = equality_row ? 0.5 : -2.0 * ud();
    p.u[m - 1] = equality_row ? 0.5 : 2.0 * ud();
    return p;
}

enum class qp_kind : std::uint8_t
{
    dense_splitting,
    sparse_splitting,
    active_set
};

template <scalar S>
expected<qp_result<S>, qp_error> solve(qp_kind kind, const qp_problem& p, bool equality_row)
{
    const auto n = static_cast<int>(p.q.size());
    const auto m = static_cast<int>(p.l.size());
    const matrix<S> P = p.P.cast<S>();
    const matrix<S> A = p.A.cast<S>();
    const vector<S> q = p.q.cast<S>();
    const vector<S> l = p.l.cast<S>();
    const vector<S> u = p.u.cast<S>();
    if(kind == qp_kind::dense_splitting)
        return dense_admm_qp_solver<S>(n, m).solve(P, q, A, l, u, dense_qp_options<S>{});
    if(kind == qp_kind::sparse_splitting)
        return sparse_admm_qp_solver<S>().solve(P.sparseView(S(0), S(0)), q, A.sparseView(S(0), S(0)), l, u, sparse_qp_options<S>{});
    dense_active_set_qp_solver<S> solver(n, m);
    const vector<double> start = vector<double>::Constant(n, equality_row ? 0.5 / n : 0.0);
    solver.warm_start(start.cast<S>(), vector<S>());
    return solver.solve(P, q, A, l, u, dense_active_set_qp_options<S>{});
}

void check_tracks(qp_kind kind)
{
    for(const std::int32_t n : {4, 12})
        for(std::uint32_t seed = 1; seed <= 10; ++seed)
            for(const bool equality_row : {false, true})
            {
                const qp_problem p = convex_problem(n, 7919u * seed + static_cast<std::uint32_t>(n), equality_row);
                const auto solved_d = solve<double>(kind, p, equality_row);
                const auto solved_f = solve<float>(kind, p, equality_row);
                REQUIRE(solved_d.has_value());
                REQUIRE(solved_f.has_value());
                const qp_result<double>& rd = *solved_d;
                const qp_result<float>& rf = *solved_f;
                INFO("n " << n << " seed " << seed << " equality row " << equality_row << " iterations " << rd.iterations << " / " << rf.iterations
                          << " gap " << relative_gap(rf.x, rd.x));
                REQUIRE(rd.status == qp_solve_status::solved);
                CHECK(rf.status == qp_solve_status::solved);
                CHECK(tracks(rf.x, rd.x, k_qp));
            }
}

}

TEST_CASE("dense splitting QP in float tracks the double solution of the seeded convex family at its own defaults", "[float]")
{
    check_tracks(qp_kind::dense_splitting);
}

TEST_CASE("sparse splitting QP in float tracks the double solution of the seeded convex family at its own defaults", "[float]")
{
    check_tracks(qp_kind::sparse_splitting);
}

TEST_CASE("dense active-set QP in float tracks the double solution of the seeded convex family at its own defaults", "[float]")
{
    check_tracks(qp_kind::active_set);
}
