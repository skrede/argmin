#include "alloc_gate_support.h"

#include "argmin/detail/active_set_qp.h"

#include "argmin/options/qp_options.h"

#include <Eigen/Core>

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
    // Two allocations per warm solve used to be the reduced-Hessian product's hidden Z^T G temporary,
    // one per equality-subproblem solve; the other 21 are the dynamic path's remaining temporaries,
    // so a count of 23 means the temporary came back.
    CHECK(counts.eigen == 21);
}
