// Compile-isolation test for the SQP scaffolding header.
//
// Verifies that detail/sqp_common.h composes without unresolved
// dependencies and that sqp_state_buffers instantiates in both the
// fixed-N and dynamic-N flavors. Each of the five sqp_common helpers
// (null_step_result, extract_qp_multipliers, compute_bfgs_pair_fused,
// equality_feasibility_warmstart, step_with_projection) is invoked
// here as a header-only inline-template smoke test that confirms the
// helper bodies link cleanly without policy adoption.
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
#include <Eigen/Cholesky>

#include <cstddef>
#include <optional>

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

    // Compile + link smoke test for the five sqp_common helpers.
    // Each helper is invoked with both fixed-N (N = 4) and dynamic-N
    // (N = argmin::dynamic_dimension) template arguments so that ODR
    // and template-instantiation cycles surface here, before any
    // policy adoption.
    {
        constexpr int N = 4;
        const int n_eq = 1;
        const int n_ineq = 2;
        const int m_total = n_eq + n_ineq;

        Eigen::Vector<double, N> x = Eigen::Vector<double, N>::Zero();
        Eigen::Vector<double, N> p = Eigen::Vector<double, N>::Zero();
        Eigen::Vector<double, N> g = Eigen::Vector<double, N>::Zero();
        Eigen::Vector<double, N> lower = Eigen::Vector<double, N>::Constant(-1.0);
        Eigen::Vector<double, N> upper = Eigen::Vector<double, N>::Constant(1.0);
        Eigen::Vector<double, N> x_trial = Eigen::Vector<double, N>::Zero();
        Eigen::Vector<double, N> grad_L_old = Eigen::Vector<double, N>::Zero();
        Eigen::Vector<double, N> grad_L_new = Eigen::Vector<double, N>::Zero();
        Eigen::Vector<double, N> sk = Eigen::Vector<double, N>::Zero();
        Eigen::Vector<double, N> yk = Eigen::Vector<double, N>::Zero();
        Eigen::Vector<double, N> p0 = Eigen::Vector<double, N>::Zero();

        Eigen::Matrix<double, Eigen::Dynamic, N> J_eq(n_eq, N);
        Eigen::Matrix<double, Eigen::Dynamic, N> J_ineq(n_ineq, N);
        Eigen::MatrixXd J_all(m_total, N);
        Eigen::MatrixXd J_all_old(m_total, N);
        J_eq.setZero();
        J_ineq.setZero();
        J_all.setZero();
        J_all_old.setZero();

        Eigen::Vector<double, Eigen::Dynamic> c_eq_v(n_eq);
        Eigen::Vector<double, Eigen::Dynamic> c_ineq_v(n_ineq);
        c_eq_v.setZero();
        c_ineq_v.setZero();
        Eigen::VectorXd qp_lambda = Eigen::VectorXd::Zero(m_total);
        Eigen::VectorXd lam_eq_v = Eigen::VectorXd::Zero(n_eq);
        Eigen::VectorXd mu_ineq_v = Eigen::VectorXd::Zero(n_ineq);
        Eigen::VectorXd lam_full = Eigen::VectorXd::Zero(m_total);
        Eigen::VectorXd b_eq_v = Eigen::VectorXd::Zero(n_eq);

        argmin::detail::extract_qp_multipliers<double>(
            qp_lambda, n_eq, n_ineq, lam_eq_v, mu_ineq_v);

        argmin::detail::step_with_projection<double, N>(
            x, 0.5, p, lower, upper, x_trial);

        Eigen::MatrixXd AAt = Eigen::MatrixXd::Zero(n_eq, n_eq);
        Eigen::LDLT<Eigen::MatrixXd> ldlt(n_eq);
        argmin::detail::equality_feasibility_warmstart<double, N>(
            J_eq, b_eq_v, AAt, ldlt, p0);

        argmin::detail::compute_bfgs_pair_fused<double, N>(
            g, g, J_all_old, J_all, lam_full, m_total,
            grad_L_old, grad_L_new, sk, yk, x, x);

        const auto r =
            argmin::detail::null_step_result<double, N, Eigen::Dynamic, Eigen::Dynamic>(
                0.0, g, J_eq, J_ineq, lam_eq_v, mu_ineq_v, c_eq_v, c_ineq_v,
                0.0, 0u, std::nullopt);
        (void)r;
    }

    // Dynamic-N instantiation (N = argmin::dynamic_dimension): mirrors the
    // fixed-N block to surface ODR or template-instantiation issues that
    // the compile-time-N path would not catch alone.
    {
        constexpr int N = argmin::dynamic_dimension;
        const int n = 4;
        const int n_eq = 1;
        const int n_ineq = 2;
        const int m_total = n_eq + n_ineq;

        Eigen::Vector<double, N> x = Eigen::Vector<double, N>::Zero(n);
        Eigen::Vector<double, N> p = Eigen::Vector<double, N>::Zero(n);
        Eigen::Vector<double, N> g = Eigen::Vector<double, N>::Zero(n);
        Eigen::Vector<double, N> lower = Eigen::Vector<double, N>::Constant(n, -1.0);
        Eigen::Vector<double, N> upper = Eigen::Vector<double, N>::Constant(n, 1.0);
        Eigen::Vector<double, N> x_trial = Eigen::Vector<double, N>::Zero(n);
        Eigen::Vector<double, N> grad_L_old = Eigen::Vector<double, N>::Zero(n);
        Eigen::Vector<double, N> grad_L_new = Eigen::Vector<double, N>::Zero(n);
        Eigen::Vector<double, N> sk = Eigen::Vector<double, N>::Zero(n);
        Eigen::Vector<double, N> yk = Eigen::Vector<double, N>::Zero(n);
        Eigen::Vector<double, N> p0 = Eigen::Vector<double, N>::Zero(n);

        Eigen::Matrix<double, Eigen::Dynamic, N> J_eq(n_eq, n);
        Eigen::Matrix<double, Eigen::Dynamic, N> J_ineq(n_ineq, n);
        Eigen::MatrixXd J_all(m_total, n);
        Eigen::MatrixXd J_all_old(m_total, n);
        J_eq.setZero();
        J_ineq.setZero();
        J_all.setZero();
        J_all_old.setZero();

        Eigen::Vector<double, Eigen::Dynamic> c_eq_v(n_eq);
        Eigen::Vector<double, Eigen::Dynamic> c_ineq_v(n_ineq);
        c_eq_v.setZero();
        c_ineq_v.setZero();
        Eigen::VectorXd qp_lambda = Eigen::VectorXd::Zero(m_total);
        Eigen::VectorXd lam_eq_v = Eigen::VectorXd::Zero(n_eq);
        Eigen::VectorXd mu_ineq_v = Eigen::VectorXd::Zero(n_ineq);
        Eigen::VectorXd lam_full = Eigen::VectorXd::Zero(m_total);
        Eigen::VectorXd b_eq_v = Eigen::VectorXd::Zero(n_eq);

        argmin::detail::extract_qp_multipliers<double>(
            qp_lambda, n_eq, n_ineq, lam_eq_v, mu_ineq_v);

        argmin::detail::step_with_projection<double, N>(
            x, 0.5, p, lower, upper, x_trial);

        Eigen::MatrixXd AAt = Eigen::MatrixXd::Zero(n_eq, n_eq);
        Eigen::LDLT<Eigen::MatrixXd> ldlt(n_eq);
        argmin::detail::equality_feasibility_warmstart<double, N>(
            J_eq, b_eq_v, AAt, ldlt, p0);

        argmin::detail::compute_bfgs_pair_fused<double, N>(
            g, g, J_all_old, J_all, lam_full, m_total,
            grad_L_old, grad_L_new, sk, yk, x, x);

        const auto r =
            argmin::detail::null_step_result<double, N, Eigen::Dynamic, Eigen::Dynamic>(
                0.0, g, J_eq, J_ineq, lam_eq_v, mu_ineq_v, c_eq_v, c_ineq_v,
                0.0, 0u, std::nullopt);
        (void)r;
    }

    return 0;
}
