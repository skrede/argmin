// Compile-isolation test for the SQP scaffolding header.
//
// Verifies that detail/sqp_common.h composes without unresolved
// dependencies and that sqp_state_buffers instantiates in both the
// fixed-N and dynamic-N flavors. The forward-declared helper
// functions are NOT called here because their bodies live in a
// follow-up plan; calling them would link-fail.
//
// Also exercises the no-op path of detail/bench/alloc_counter.h
// (without ARGMIN_BENCH_TRACE_ALLOC defined the API is a sized-zero
// counter).

#include "argmin/types.h"
#include "argmin/detail/lagrangian.h"
#include "argmin/detail/sqp_common.h"
#include "argmin/detail/merit_function.h"
#include "argmin/detail/bench/alloc_counter.h"

#include <Eigen/Core>

#include <cstddef>

int main()
{
    argmin::detail::sqp_state_buffers<double, 4> fixed_buf;
    fixed_buf.resize(4, 1, 2);

    argmin::detail::sqp_state_buffers<double, argmin::dynamic_dimension> dyn_buf;
    dyn_buf.resize(4, 1, 2);

    argmin::detail::bench::reset_alloc_count();
    argmin::detail::bench::arm_alloc_trace();
    argmin::detail::bench::disarm_alloc_trace();
    const std::size_t count = argmin::detail::bench::read_alloc_count();
    (void)count;

    // Compile + link smoke test for the lagrangian.h / merit_function.h
    // helpers. The bodies are header-only inline templates; calling each
    // here verifies the new helpers compose cleanly with the scaffold.
    {
        constexpr int N = 4;
        const int n_eq = 1;
        const int n_ineq = 2;

        Eigen::Vector<double, N> g = Eigen::Vector<double, N>::Zero();
        Eigen::Matrix<double, Eigen::Dynamic, N> J_eq(n_eq, N);
        Eigen::Matrix<double, Eigen::Dynamic, N> J_ineq(n_ineq, N);
        J_eq.setZero();
        J_ineq.setZero();
        Eigen::VectorXd c_ineq = Eigen::VectorXd::Zero(n_ineq);
        Eigen::Vector<double, Eigen::Dynamic> c_eq_v(n_eq);
        c_eq_v.setZero();
        Eigen::VectorXd lam_eq = Eigen::VectorXd::Zero(n_eq);
        Eigen::VectorXd mu_ineq = Eigen::VectorXd::Zero(n_ineq);

        argmin::detail::compute_kkt_multipliers_active_set<double, N>(
            g, J_eq, J_ineq, c_ineq, lam_eq, mu_ineq);

        const double dphi = argmin::detail::l1_merit_dphi_h4<double>(
            -1.0, c_eq_v, c_ineq, 1.0, 1.0);
        (void)dphi;

        const double sigma = argmin::detail::bump_sigma_for_descent<double>(
            1.0, -1.0, 0.5, 1.0, 1e10);
        (void)sigma;
    }

    return 0;
}
