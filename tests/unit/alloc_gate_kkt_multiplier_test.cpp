#include "alloc_gate_support.h"

#include "argmin/detail/lagrangian.h"

#include <Eigen/Core>

#include <cstdint>

namespace
{

using argmin::alloc_gate::run_armed;
using argmin::alloc_gate::window_counts;

double next(std::uint64_t& state)
{
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(state >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
}

Eigen::MatrixXd random_matrix(Eigen::Index rows, Eigen::Index cols, std::uint64_t& state)
{
    Eigen::MatrixXd m(rows, cols);
    for(Eigen::Index j = 0; j < cols; ++j)
        for(Eigen::Index i = 0; i < rows; ++i)
            m(i, j) = next(state);
    return m;
}

}

// Every fourth inequality sits on its bound, so 24 equalities and 4 active inequalities give the
// multiplier estimate a 40 x 28 least-squares solve on each re-estimation.
TEST_CASE("kkt_multiplier_estimate_allocation_count", "[alloc-gate]")
{
    constexpr Eigen::Index n = 40, n_eq = 24, n_ineq = 16;
    std::uint64_t state = 5;
    const Eigen::MatrixXd J_eq = random_matrix(n_eq, n, state);
    const Eigen::MatrixXd J_ineq = random_matrix(n_ineq, n, state);
    Eigen::VectorXd c_ineq = random_matrix(n_ineq, 1, state).array() - 2.0;
    for(Eigen::Index i = 0; i < n_ineq; i += 4)
        c_ineq[i] = 0.0;
    Eigen::VectorXd g = random_matrix(n, 1, state);
    argmin::detail::kkt_multiplier_workspace<double> ws;
    Eigen::VectorXd lam_eq(n_eq);
    Eigen::VectorXd mu_ineq(n_ineq);
    argmin::detail::compute_kkt_multipliers_active_set<double>(g, J_eq, J_ineq, c_ineq, ws, lam_eq, mu_ineq);

    for(std::int32_t step = 0; step < 4; ++step)
    {
        g += 0.1 * random_matrix(n, 1, state);
        const window_counts counts = run_armed([&] {
            argmin::detail::compute_kkt_multipliers_active_set<double>(g, J_eq, J_ineq, c_ineq, ws, lam_eq, mu_ineq);
        });
        INFO("warm estimate " << step);
        CHECK(lam_eq.allFinite());
        CHECK(counts.eigen == 0);
        CHECK(counts.c == 0);
    }
}
