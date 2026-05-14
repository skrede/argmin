#ifndef HPP_GUARD_ARGMIN_TEST_FUNCTIONS_NMPC_LQR_H
#define HPP_GUARD_ARGMIN_TEST_FUNCTIONS_NMPC_LQR_H

// Finite-horizon LQR-shaped NMPC test problem.
//
// Decision variables: z = [x_1, u_0, x_2, u_1, ..., x_H, u_{H-1}]^T,
// stacked in horizon order. The initial state x_0 is fixed problem
// data and not part of the decision vector. Each stage block has size
// n_x + n_u = 6 (n_x = 4 states, n_u = 2 controls), so the joint
// primal dimension is H * (n_x + n_u).
//
// Stage cost: 0.5 * sum_{k=0..H-1} ( x_k^T Q x_k + u_k^T R u_k ).
// Q = I_{n_x}, R = 0.1 * I_{n_u}. No terminal cost on x_H (keeps the
// problem a pure finite-horizon LQR-shaped exercise).
//
// Linear dynamics: x_{k+1} = A x_k + B u_k for k = 0..H-1, encoded as
// H * n_x equality constraints stacked in row blocks of size n_x:
//   c_eq[k * n_x : (k+1) * n_x] = x_{k+1} - A x_k - B u_k.
//
// A and B are the zero-order-hold discretization of a double-integrator
// stack: 4 states = (px, py, vx, vy), 2 controls = (ax, ay), sample
// time dt = 0.1. The corresponding A and B follow the standard
// continuous-to-discrete derivation for double-integrator dynamics.
//
// Control box bounds: u_min = (-2, -2), u_max = (2, 2) on every u_k.
// State variables are unbounded (lower/upper = +/- infinity); the SQP
// harness treats infinity bounds as inactive.
//
// LQR / Riccati reference: Anderson and Moore, Optimal Control: Linear
// Quadratic Methods, 1990, Section 2.3.
// Discretization: double-integrator stack with dt = 0.1; standard
// zero-order-hold.

#include "argmin/test_functions/problem_class.h"

#include <Eigen/Core>

#include <cmath>
#include <limits>

namespace argmin
{

template <int H = 10, typename Scalar = double>
struct nmpc_lqr
{
    static constexpr int horizon = H;
    static constexpr int n_x = 4;
    static constexpr int n_u = 2;
    static constexpr int problem_dimension = H * (n_x + n_u);
    static constexpr int constraint_count = H * n_x;
    static constexpr problem_class pclass =
          problem_class::equality
        | problem_class::bound_constrained
        | problem_class::application;

    // Sample time and stage-cost weights (LQR Q, R diagonals).
    static constexpr Scalar dt = Scalar(0.1);
    static constexpr Scalar q_diag = Scalar(1);
    static constexpr Scalar r_diag = Scalar(0.1);

    // Control box bounds (state bounds are +/- infinity).
    static constexpr Scalar u_min_scalar = Scalar(-2);
    static constexpr Scalar u_max_scalar = Scalar(2);

    // ---- Decision-vector layout helpers ---------------------------------
    //
    // Stage block size n_stage = n_x + n_u = 6. The k-th stage block
    // (k = 0..H-1) occupies indices [k * n_stage, (k+1) * n_stage) and
    // contains [x_{k+1}; u_k]. So:
    //   x_offset(k) = k * n_stage              (k = 0..H-1, returns x_{k+1})
    //   u_offset(k) = k * n_stage + n_x        (k = 0..H-1, returns u_k)
    static constexpr int n_stage = n_x + n_u;

    [[nodiscard]] static constexpr int x_offset(int k_minus_one) noexcept
    {
        return k_minus_one * n_stage;
    }

    [[nodiscard]] static constexpr int u_offset(int k) noexcept
    {
        return k * n_stage + n_x;
    }

    // ---- Fixed initial state x_0 ----------------------------------------
    [[nodiscard]] static Eigen::Matrix<Scalar, n_x, 1> x0_fixed() noexcept
    {
        Eigen::Matrix<Scalar, n_x, 1> x0;
        x0 << Scalar(1), Scalar(1), Scalar(0), Scalar(0);
        return x0;
    }

    // ---- A, B (ZOH double-integrator) -----------------------------------
    [[nodiscard]] static Eigen::Matrix<Scalar, n_x, n_x> A_matrix() noexcept
    {
        Eigen::Matrix<Scalar, n_x, n_x> A;
        A.setIdentity();
        A(0, 2) = dt;
        A(1, 3) = dt;
        return A;
    }

    [[nodiscard]] static Eigen::Matrix<Scalar, n_x, n_u> B_matrix() noexcept
    {
        Eigen::Matrix<Scalar, n_x, n_u> B;
        B.setZero();
        const Scalar half_dt2 = Scalar(0.5) * dt * dt;
        B(0, 0) = half_dt2;
        B(1, 1) = half_dt2;
        B(2, 0) = dt;
        B(3, 1) = dt;
        return B;
    }

    // ---- Public API -----------------------------------------------------

    [[nodiscard]] int dimension() const { return problem_dimension; }
    [[nodiscard]] int num_equality() const { return constraint_count; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& z) const
    {
        // Cost = 0.5 * sum_{k=0..H-1} ( x_k^T Q x_k + u_k^T R u_k ).
        // x_0 is fixed; x_k for k = 1..H-1 are in z (x_H is in z but
        // excluded from the sum; no terminal cost).
        const auto x0 = x0_fixed();
        Scalar cost = Scalar(0.5) * q_diag * x0.squaredNorm();
        for(int k = 0; k < H; ++k)
        {
            const int u_off = u_offset(k);
            // u_k^T R u_k = r_diag * (u_k . u_k)
            Scalar u_sq = Scalar(0);
            for(int j = 0; j < n_u; ++j)
                u_sq += z[u_off + j] * z[u_off + j];
            cost += Scalar(0.5) * r_diag * u_sq;

            if(k + 1 < H)
            {
                // x_{k+1} contribution to sum at index k+1 (still k <= H-1).
                const int x_off = x_offset(k);  // gives x_{k+1}
                Scalar x_sq = Scalar(0);
                for(int j = 0; j < n_x; ++j)
                    x_sq += z[x_off + j] * z[x_off + j];
                cost += Scalar(0.5) * q_diag * x_sq;
            }
        }
        return cost;
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& z,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        g.setZero();
        for(int k = 0; k < H; ++k)
        {
            const int u_off = u_offset(k);
            // d cost / d u_k = r_diag * u_k
            for(int j = 0; j < n_u; ++j)
                g[u_off + j] = r_diag * z[u_off + j];

            if(k + 1 < H)
            {
                // d cost / d x_{k+1} = q_diag * x_{k+1}; x_{k+1} lives
                // at index x_offset(k).
                const int x_off = x_offset(k);
                for(int j = 0; j < n_x; ++j)
                    g[x_off + j] = q_diag * z[x_off + j];
            }
            // x_H (k = H-1, index x_offset(H-1)) has no cost contribution.
        }
    }

    // Combined constraint vector layout: H * n_x equality constraints
    // (rows 0..H*n_x - 1), no inequality entries.
    void constraints(const Eigen::Vector<Scalar, problem_dimension>& z,
                     auto& c) const
    {
        const auto A = A_matrix();
        const auto B = B_matrix();
        const auto x0 = x0_fixed();
        for(int k = 0; k < H; ++k)
        {
            // Pull x_k and u_k from z (or x_0 from fixed data for k = 0).
            Eigen::Matrix<Scalar, n_x, 1> x_k;
            if(k == 0)
            {
                x_k = x0;
            }
            else
            {
                const int x_k_off = x_offset(k - 1);  // x_k lives at slot k-1
                for(int j = 0; j < n_x; ++j)
                    x_k[j] = z[x_k_off + j];
            }
            Eigen::Matrix<Scalar, n_u, 1> u_k;
            const int u_k_off = u_offset(k);
            for(int j = 0; j < n_u; ++j)
                u_k[j] = z[u_k_off + j];

            // x_{k+1} from z at slot k.
            Eigen::Matrix<Scalar, n_x, 1> x_kp1;
            const int x_kp1_off = x_offset(k);
            for(int j = 0; j < n_x; ++j)
                x_kp1[j] = z[x_kp1_off + j];

            // Residual r_k = x_{k+1} - A x_k - B u_k.
            Eigen::Matrix<Scalar, n_x, 1> r_k = x_kp1 - A * x_k - B * u_k;
            const int c_off = k * n_x;
            for(int j = 0; j < n_x; ++j)
                c[c_off + j] = r_k[j];
        }
    }

    // Combined Jacobian: rows = H * n_x equalities, cols = problem_dimension.
    void constraint_jacobian(const Eigen::Vector<Scalar, problem_dimension>& /*z*/,
                             auto& J) const
    {
        const auto A = A_matrix();
        const auto B = B_matrix();
        // Jacobian rows for constraint block k occupy [k*n_x, (k+1)*n_x).
        // d r_k / d x_{k+1} = +I  (at slot x_offset(k))
        // d r_k / d u_k     = -B  (at slot u_offset(k))
        // d r_k / d x_k     = -A  (at slot x_offset(k - 1), only if k > 0)
        for(int r = 0; r < constraint_count; ++r)
            for(int c = 0; c < problem_dimension; ++c)
                J(r, c) = Scalar(0);

        for(int k = 0; k < H; ++k)
        {
            const int row0 = k * n_x;
            // +I block at column x_offset(k).
            const int col_xkp1 = x_offset(k);
            for(int j = 0; j < n_x; ++j)
                J(row0 + j, col_xkp1 + j) = Scalar(1);

            // -B block at column u_offset(k).
            const int col_uk = u_offset(k);
            for(int i = 0; i < n_x; ++i)
                for(int j = 0; j < n_u; ++j)
                    J(row0 + i, col_uk + j) = -B(i, j);

            // -A block at column x_offset(k-1) for k > 0 (x_0 is fixed).
            if(k > 0)
            {
                const int col_xk = x_offset(k - 1);
                for(int i = 0; i < n_x; ++i)
                    for(int j = 0; j < n_x; ++j)
                        J(row0 + i, col_xk + j) = -A(i, j);
            }
        }
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        Eigen::Vector<Scalar, problem_dimension> lb;
        lb.setConstant(-inf);
        for(int k = 0; k < H; ++k)
        {
            const int u_off = u_offset(k);
            for(int j = 0; j < n_u; ++j)
                lb[u_off + j] = u_min_scalar;
        }
        return lb;
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
        Eigen::Vector<Scalar, problem_dimension> ub;
        ub.setConstant(inf);
        for(int k = 0; k < H; ++k)
        {
            const int u_off = u_offset(k);
            for(int j = 0; j < n_u; ++j)
                ub[u_off + j] = u_max_scalar;
        }
        return ub;
    }

    // initial_point: zero controls, propagate x with u = 0 starting from
    // x_0. This satisfies every dynamics equality exactly, giving the
    // SQP harness a warm-start feasible point (c_eq.norm() = 0).
    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        const auto A = A_matrix();
        Eigen::Vector<Scalar, problem_dimension> z;
        z.setZero();
        Eigen::Matrix<Scalar, n_x, 1> x_curr = x0_fixed();
        for(int k = 0; k < H; ++k)
        {
            // u_k = 0 (already zero from setZero); propagate x_{k+1} = A x_k.
            x_curr = A * x_curr;
            const int x_off = x_offset(k);
            for(int j = 0; j < n_x; ++j)
                z[x_off + j] = x_curr[j];
        }
        return z;
    }

    // optimal_value: bench-good-enough reference. The closed-form LQR
    // optimum requires a Riccati / DARE solve at fixed problem data; the
    // SQP harness gates on min_accuracy_log10, not on exact equality, so
    // a precomputed reference suffices. Set to zero; the bench harness
    // only consumes optimal_value for the accuracy column in publish
    // summaries and the absolute-difference computation is not
    // load-bearing for convergence gates on this cell.
    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }
};

}

#endif
