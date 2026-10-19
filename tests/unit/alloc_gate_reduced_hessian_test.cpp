#include "alloc_gate_support.h"

#include "argmin/detail/active_set_qp.h"

#include "argmin/options/qp_options.h"

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>

namespace
{

using argmin::alloc_gate::run_armed;
using argmin::alloc_gate::window_counts;
using argmin::detail::qp_status;
using solver_type = argmin::detail::active_set_qp_solver<double>;

class lcg
{
public:
    explicit lcg(std::uint64_t seed)
        : state_(seed)
    {
    }

    double next()
    {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(state_ >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
    }

private:
    std::uint64_t state_;
};

template <typename Derived>
void fill(Eigen::MatrixBase<Derived>& m, lcg& rng)
{
    for(Eigen::Index j = 0; j < m.cols(); ++j)
        for(Eigen::Index i = 0; i < m.rows(); ++i)
            m(i, j) = rng.next();
}

struct equality_qp
{
    Eigen::MatrixXd G;
    Eigen::VectorXd d;
    Eigen::MatrixXd A_eq;
    Eigen::VectorXd b_eq;
    Eigen::MatrixXd A_ineq;
    Eigen::VectorXd b_ineq;
    Eigen::VectorXd x0;
};

equality_qp make_equality_qp(std::int32_t n, std::int32_t m_eq, std::uint64_t seed)
{
    lcg rng(seed);
    Eigen::MatrixXd R(n, n);
    fill(R, rng);
    equality_qp qp{R.transpose() * R, Eigen::VectorXd(n), Eigen::MatrixXd(m_eq, n),
                   Eigen::VectorXd(m_eq), Eigen::MatrixXd(0, n), Eigen::VectorXd(0), Eigen::VectorXd(n)};
    qp.G.diagonal().array() += 0.5;
    fill(qp.d, rng);
    fill(qp.A_eq, rng);
    fill(qp.x0, rng);
    qp.b_eq = qp.A_eq * qp.x0;
    return qp;
}

// Non-condensed horizon-8 MPC with three states and one input: z = [u_0, x_1, u_1, x_2, ...], dynamics
// equalities, and an input box tight enough that the active set changes across the warm solves.
struct mpc_qp
{
    equality_qp qp;
    Eigen::VectorXd lower;
    Eigen::VectorXd upper;
};

mpc_qp make_mpc_qp(std::uint64_t seed)
{
    constexpr std::int32_t horizon = 8, nx = 3, nu = 1, blk = nx + nu, n = horizon * blk;
    lcg rng(seed);
    Eigen::MatrixXd A(nx, nx);
    Eigen::MatrixXd B(nx, nu);
    Eigen::VectorXd x_init(nx);
    fill(A, rng);
    fill(B, rng);
    fill(x_init, rng);
    A = Eigen::MatrixXd::Identity(nx, nx) + 0.1 * A;
    B *= 0.5;
    x_init *= 4.0;
    mpc_qp mpc{{Eigen::MatrixXd::Zero(n, n), Eigen::VectorXd::Zero(n), Eigen::MatrixXd::Zero(horizon * nx, n),
                Eigen::VectorXd::Zero(horizon * nx), Eigen::MatrixXd(0, n), Eigen::VectorXd(0), Eigen::VectorXd::Zero(n)},
               Eigen::VectorXd::Constant(n, -1e3), Eigen::VectorXd::Constant(n, 1e3)};
    Eigen::VectorXd state = x_init;
    for(std::int32_t k = 0; k < horizon; ++k)
    {
        const std::int32_t u = k * blk, x = u + nu;
        mpc.qp.G(u, u) = 0.1;
        mpc.qp.G.block(x, x, nx, nx).diagonal().setConstant(1.0);
        mpc.lower[u] = -0.3;
        mpc.upper[u] = 0.3;
        mpc.qp.A_eq.block(k * nx, x, nx, nx).setIdentity();
        mpc.qp.A_eq.block(k * nx, u, nx, nu) = -B;
        if(k == 0)
            mpc.qp.b_eq.head(nx) = A * x_init;
        else
            mpc.qp.A_eq.block(k * nx, x - blk, nx, nx) = -A;
        state = A * state;
        mpc.qp.x0.segment(x, nx) = state;
    }
    return mpc;
}

}

TEST_CASE("qp_reduced_hessian_allocation_count", "[qp-reduced-hessian-alloc]")
{
    const equality_qp qp = make_equality_qp(7, 4, 1);
    const argmin::qp_options<> opts{};
    solver_type solver(7, 4);
    solver_type::result_type out;
    solver.solve_into(qp.G, qp.d, qp.A_eq, qp.b_eq, qp.A_ineq, qp.b_ineq, qp.x0, opts, out);
    REQUIRE(out.status == qp_status::optimal);

    const window_counts counts = run_armed([&] {
        solver.solve_into(qp.G, qp.d, qp.A_eq, qp.b_eq, qp.A_ineq, qp.b_ineq, qp.x0, opts, out);
    });

    CHECK(out.status == qp_status::optimal);
    CHECK(out.iterations == 1);
    // The equality subproblem allocates nothing; the two left are the iteration gradient's G x
    // temporary, one per active-set pass.
    CHECK(counts.eigen == 2);
}

TEST_CASE("qp_equality_subproblem_allocation_count", "[qp-reduced-hessian-alloc]")
{
    const mpc_qp mpc = make_mpc_qp(11);
    const equality_qp& qp = mpc.qp;
    const argmin::qp_options<> opts{};
    const auto rows = static_cast<std::int32_t>(qp.A_eq.rows() + 2 * qp.G.rows());
    solver_type solver(static_cast<std::int32_t>(qp.G.rows()), rows);
    solver_type::result_type out;
    Eigen::VectorXd d = qp.d;
    solver.solve_into(qp.G, d, qp.A_eq, qp.b_eq, qp.A_ineq, qp.b_ineq, mpc.lower, mpc.upper, qp.x0, opts, out);
    REQUIRE(out.status == qp_status::optimal);

    lcg rng(99);
    for(std::int32_t solve = 0; solve < 6; ++solve)
    {
        for(Eigen::Index i = 0; i < d.size(); ++i)
            d[i] += 0.2 * rng.next();
        const window_counts counts = run_armed([&] {
            solver.solve_into(qp.G, d, qp.A_eq, qp.b_eq, qp.A_ineq, qp.b_ineq, mpc.lower, mpc.upper, qp.x0, opts, out);
        });
        INFO("warm solve " << solve);
        REQUIRE(out.status == qp_status::optimal);
        CHECK(out.iterations > 4);
        // One allocation per active-set pass is the gradient's G x temporary outside the subproblem;
        // the subproblem itself allocated about n per pass before its solves moved into member storage.
        const auto passes = static_cast<std::size_t>(out.iterations) + 1;
        CHECK(counts.eigen == passes);
        CHECK(counts.c == passes);
    }
}
