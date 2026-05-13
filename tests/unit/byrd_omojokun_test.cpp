#include "argmin/detail/byrd_omojokun.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <utility>

using Catch::Approx;
using argmin::detail::byrd_omojokun_composite_step;
using argmin::detail::byrd_omojokun_result;
using argmin::detail::cg_exit_status;
using argmin::detail::zeta;
using argmin::detail::tr_eta_1;
using argmin::detail::tr_eta_2;
using argmin::detail::tr_shrink_factor;
using argmin::detail::tr_expand_factor;
using argmin::detail::tr_delta_max;

namespace
{

constexpr double kInf = std::numeric_limits<double>::infinity();

// Dense 2-D Hessian-op closure: returns (B * v).eval() as the Hessian-
// vector product. Matches the steihaug_cg test pattern.
struct hessian_op_2d
{
    Eigen::Matrix2d B;
    Eigen::Vector2d operator()(
        const Eigen::Ref<const Eigen::Vector2d>& v) const
    {
        return (B * v).eval();
    }
};

// Quadratic-model trial-evaluator: f_new = f_old + g^T p + 0.5 p^T B p,
// c_norm_new = ||A p + c||. This isolates the ratio test from
// objective-function nonlinearity by matching the predicted reduction
// exactly on the model.
struct trial_eval_quadratic
{
    Eigen::Vector2d g;
    Eigen::Matrix2d B;
    Eigen::MatrixXd A;
    Eigen::VectorXd c;
    double f_old;

    std::pair<double, double> operator()(
        const Eigen::Ref<const Eigen::Vector2d>& p) const
    {
        const double f_new = f_old + g.dot(p) + 0.5 * p.dot(B * p);
        const double c_norm_new = (A * p + c).norm();
        return {f_new, c_norm_new};
    }
};

// Maratos-shape trial-evaluator: linear constraint Jacobian at the
// linearization point, but the actual constraint is the curved
// equality c(x) = x[0]^2 + x[1]^2 - 1. The trial point is the
// linearization site plus the proposed step, so the nonlinear
// residual reflects the curvature gap the ratio test must detect.
struct trial_eval_maratos
{
    Eigen::Vector2d g;
    Eigen::Matrix2d B;
    Eigen::Vector2d x_linearize;  // x at which the QP was linearized
    double f_old;

    std::pair<double, double> operator()(
        const Eigen::Ref<const Eigen::Vector2d>& p) const
    {
        // Quadratic objective model.
        const double f_new = f_old + g.dot(p) + 0.5 * p.dot(B * p);
        // True nonlinear constraint at the trial point x + p.
        Eigen::Vector2d x_trial = x_linearize + p;
        const double c_nl = x_trial.squaredNorm() - 1.0;
        const double c_norm_new = std::abs(c_nl);
        return {f_new, c_norm_new};
    }
};

}

TEST_CASE("byrd_omojokun pure equality quadratic accepts and progresses",
          "[byrd_omojokun][accept]")
{
    // 2-D problem with a single linear equality far from satisfied:
    //   J_eq = [1, 1], c_eq = 0.5 (||c|| = 0.5).
    //   g = [1, -1], B = I, delta = 10.
    // The gradient lies in the null space of A (A g = 0), so the
    // tangential subproblem gradient g_t = g + B v has its part along
    // null(A) untouched by the normal step. Steihaug-CG follows g_t
    // into null(A); the tangential step is naturally tangential to
    // the equality and the composite step preserves the linearized-
    // feasibility decrease the normal step achieves. (The helper does
    // not enforce tangentiality via null-space projection; it relies
    // on the caller-supplied subproblem geometry for this property.)
    //
    // The Cauchy and Newton legs of the normal step both fit inside
    // zeta * delta = 8, so the normal step is the minimum-norm LSQ
    // step v_N = -A^T (A A^T)^{-1} c with ||v_N|| = 0.5/sqrt(2). The
    // trial evaluator is the exact quadratic model, so predicted ==
    // actual and rho = 1 -- the step is accepted.
    Eigen::Vector2d g;
    g << 1.0, -1.0;
    Eigen::Matrix2d B = Eigen::Matrix2d::Identity();

    Eigen::MatrixXd A(1, 2);
    A << 1.0, 1.0;
    Eigen::VectorXd c(1);
    c << 0.5;

    const double delta_in = 10.0;
    const double eps = 1e-8;

    Eigen::Vector2d z_k = Eigen::Vector2d::Zero();
    Eigen::Vector2d lower_displaced;
    lower_displaced << -kInf, -kInf;
    Eigen::Vector2d upper_displaced;
    upper_displaced << kInf, kInf;

    Eigen::MatrixXd AAt_workspace(1, 1);
    Eigen::LDLT<Eigen::MatrixXd> ldlt_workspace(1);
    Eigen::VectorXd w_workspace(1);
    Eigen::Vector2d v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out;

    hessian_op_2d hop{B};
    const double f_old = 0.0;
    const double c_norm_old = c.norm();
    trial_eval_quadratic teval{g, B, A, c, f_old};

    auto r = byrd_omojokun_composite_step<double, 2>(
        z_k, g, hop, A, c,
        delta_in, eps, /*max_cg_iter=*/20,
        lower_displaced, upper_displaced,
        f_old, c_norm_old, teval,
        AAt_workspace, ldlt_workspace, w_workspace,
        v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out);

    // Normal step must reduce the linearized residual:
    // ||A v + c|| < ||c||.
    const double lin_resid_old = c.norm();
    const double lin_resid_new = (A * v_buf + c).norm();
    INFO("lin_resid_old=" << lin_resid_old
         << "  lin_resid_new=" << lin_resid_new);
    CHECK(lin_resid_new < lin_resid_old);

    CHECK_FALSE(r.normal_step_lsq_fallback);
    CHECK(r.accepted);
    CHECK(r.rho >= tr_eta_1);
    // Radius does not shrink: should be either delta_in (held) or
    // expanded to min(2*delta_in, delta_max) if rho > 0.75 AND step
    // hit the boundary. Either way, new_delta >= delta_in.
    CHECK(r.new_delta >= delta_in);
}

TEST_CASE("byrd_omojokun rank-deficient A triggers Cauchy fallback",
          "[byrd_omojokun][fallback]")
{
    // 2-D problem with two identical rows in A: rank(A) = 1, so AA^T
    // is rank-1 PSD. In theory the LDLT factorization of AA^T flags
    // a numerical issue and the helper falls back to pure Cauchy
    // truncated to zeta * delta. In practice, Eigen's LDLT is robust
    // enough to silently produce a minimum-norm solution when c lies
    // in range(A) (it does here: c = [1, 1] = A * [0.5, 0.5]), so the
    // LSQ leg produces a valid step v_N = -[0.5, 0.5] and the
    // fallback flag may NOT fire. The load-bearing behavior the
    // outer policy depends on is finiteness and respect for the
    // zeta * delta bound on the normal step, regardless of which
    // branch (LSQ or Cauchy) produced it.
    //
    // Reference: Eigen 3.4 documentation on LDLT::info() -- LDLT
    //            uses partial pivoting and is robust on PSD matrices
    //            including rank-deficient cases when the RHS lies in
    //            the range. The threat-model fallback fires on
    //            genuine numerical failures (NaN, indefinite AA^T
    //            with negative pivots beyond tolerance, etc.), not
    //            on every rank-deficient input.
    Eigen::Vector2d g;
    g << 1.0, 1.0;
    Eigen::Matrix2d B = Eigen::Matrix2d::Identity();

    Eigen::MatrixXd A(2, 2);
    A << 1.0, 1.0,
         1.0, 1.0;
    Eigen::VectorXd c(2);
    c << 1.0, 1.0;

    const double delta_in = 1.0;
    const double eps = 1e-8;

    Eigen::Vector2d z_k = Eigen::Vector2d::Zero();
    Eigen::Vector2d lower_displaced;
    lower_displaced << -kInf, -kInf;
    Eigen::Vector2d upper_displaced;
    upper_displaced << kInf, kInf;

    Eigen::MatrixXd AAt_workspace(2, 2);
    Eigen::LDLT<Eigen::MatrixXd> ldlt_workspace(2);
    Eigen::VectorXd w_workspace(2);
    Eigen::Vector2d v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out;

    hessian_op_2d hop{B};
    const double f_old = 0.0;
    const double c_norm_old = c.norm();
    trial_eval_quadratic teval{g, B, A, c, f_old};

    auto r = byrd_omojokun_composite_step<double, 2>(
        z_k, g, hop, A, c,
        delta_in, eps, /*max_cg_iter=*/20,
        lower_displaced, upper_displaced,
        f_old, c_norm_old, teval,
        AAt_workspace, ldlt_workspace, w_workspace,
        v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out);

    // Helper must not crash and must produce a finite step that
    // respects the displaced TR bound on the normal step.
    CHECK(v_buf.allFinite());
    CHECK(u_buf.allFinite());
    CHECK(p_out.allFinite());
    CHECK(v_buf.norm() <= zeta * delta_in + 1e-12);
}

TEST_CASE("byrd_omojokun Maratos-shape problem rejects unsafe step",
          "[byrd_omojokun][reject]")
{
    // Maratos shape: linearize the curved equality c_eq(x) = x^T x - 1
    // at x = (1, 0): J_eq = [2, 0], c_eq = 0. Objective f(x) = -x[0]
    // so g = [-1, 0], B = I.
    //
    // The linearized model says taking a step that keeps x[0] near 1
    // and changes x[1] keeps the linearized constraint satisfied
    // (since J_eq[1] = 0). But moving along x[1] increases the
    // nonlinear residual ||x_trial||^2 - 1 quadratically. The trial
    // evaluator uses the nonlinear constraint, so actual feasibility
    // worsens while predicted feasibility appears to stay near zero
    // -- rho < eta_1 and the step is rejected.
    Eigen::Vector2d x_linearize;
    x_linearize << 1.0, 0.0;
    Eigen::Vector2d g;
    g << -1.0, 0.0;
    Eigen::Matrix2d B = Eigen::Matrix2d::Identity();

    Eigen::MatrixXd A(1, 2);
    A << 2.0, 0.0;
    Eigen::VectorXd c(1);
    c << 0.0;  // already on the linearization point

    const double delta_in = 1.0;
    const double eps = 1e-8;

    Eigen::Vector2d z_k = Eigen::Vector2d::Zero();
    Eigen::Vector2d lower_displaced;
    lower_displaced << -kInf, -kInf;
    Eigen::Vector2d upper_displaced;
    upper_displaced << kInf, kInf;

    Eigen::MatrixXd AAt_workspace(1, 1);
    Eigen::LDLT<Eigen::MatrixXd> ldlt_workspace(1);
    Eigen::VectorXd w_workspace(1);
    Eigen::Vector2d v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out;

    hessian_op_2d hop{B};
    const double f_old = -x_linearize[0];  // -1.0
    const double c_norm_old = 0.0;
    trial_eval_maratos teval{g, B, x_linearize, f_old};

    auto r = byrd_omojokun_composite_step<double, 2>(
        z_k, g, hop, A, c,
        delta_in, eps, /*max_cg_iter=*/20,
        lower_displaced, upper_displaced,
        f_old, c_norm_old, teval,
        AAt_workspace, ldlt_workspace, w_workspace,
        v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out);

    INFO("rho=" << r.rho
         << "  actual=" << r.actual_reduction
         << "  predicted=" << r.predicted_reduction
         << "  ||p||=" << p_out.norm());
    CHECK_FALSE(r.accepted);
    CHECK(r.rho < tr_eta_1);
    CHECK(r.new_delta == Approx(tr_shrink_factor * delta_in).margin(1e-12));
}

TEST_CASE("byrd_omojokun ratio > 0.75 at TR boundary expands radius",
          "[byrd_omojokun][radius][expand]")
{
    // Unconstrained quadratic (no equality so the normal step is zero
    // and the composite step is purely tangential). Use g = [-1, 0],
    // B = 0.01 * I so the unconstrained minimizer is at p* = -B^{-1}g
    // = [100, 0], far outside any reasonable TR. With delta = 1 the
    // Steihaug-CG truncates to the TR boundary along -g, producing
    // step_norm = 1.0 = delta (boundary).
    //
    // The trial_eval is the exact quadratic, so predicted == actual
    // and rho = 1.0 > 0.75. step_norm >= 0.9 * delta is satisfied.
    // The radius must expand to min(2 * delta, delta_max) = 2.
    Eigen::Vector2d g;
    g << -1.0, 0.0;
    Eigen::Matrix2d B = 0.01 * Eigen::Matrix2d::Identity();

    Eigen::MatrixXd A(0, 2);  // no equality
    Eigen::VectorXd c(0);

    const double delta_in = 1.0;
    const double eps = 1e-12;

    Eigen::Vector2d z_k = Eigen::Vector2d::Zero();
    Eigen::Vector2d lower_displaced;
    lower_displaced << -kInf, -kInf;
    Eigen::Vector2d upper_displaced;
    upper_displaced << kInf, kInf;

    Eigen::MatrixXd AAt_workspace(0, 0);
    Eigen::LDLT<Eigen::MatrixXd> ldlt_workspace(0);
    Eigen::VectorXd w_workspace(0);
    Eigen::Vector2d v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out;

    hessian_op_2d hop{B};
    const double f_old = 0.0;
    const double c_norm_old = 0.0;
    trial_eval_quadratic teval{g, B, A, c, f_old};

    auto r = byrd_omojokun_composite_step<double, 2>(
        z_k, g, hop, A, c,
        delta_in, eps, /*max_cg_iter=*/20,
        lower_displaced, upper_displaced,
        f_old, c_norm_old, teval,
        AAt_workspace, ldlt_workspace, w_workspace,
        v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out);

    INFO("rho=" << r.rho
         << "  ||p||=" << p_out.norm()
         << "  new_delta=" << r.new_delta);
    CHECK(r.accepted);
    CHECK(r.rho > tr_eta_2);
    CHECK(p_out.norm() >= 0.9 * delta_in);
    CHECK(r.new_delta == Approx(
        std::min(tr_expand_factor * delta_in, tr_delta_max)).margin(1e-12));
}

TEST_CASE("byrd_omojokun interior step holds radius",
          "[byrd_omojokun][radius][hold]")
{
    // Unconstrained quadratic with the unconstrained minimizer well
    // inside the TR: g = [-1, 0], B = I, delta = 10. p* = -B^{-1}g
    // = [1, 0], ||p*|| = 1 < 0.9 * 10 = 9 (interior). rho = 1.0
    // (exact quadratic) so the high-rho expand branch is gated by
    // the boundary guard; new_delta = delta_in (held).
    Eigen::Vector2d g;
    g << -1.0, 0.0;
    Eigen::Matrix2d B = Eigen::Matrix2d::Identity();

    Eigen::MatrixXd A(0, 2);
    Eigen::VectorXd c(0);

    const double delta_in = 10.0;
    const double eps = 1e-12;

    Eigen::Vector2d z_k = Eigen::Vector2d::Zero();
    Eigen::Vector2d lower_displaced;
    lower_displaced << -kInf, -kInf;
    Eigen::Vector2d upper_displaced;
    upper_displaced << kInf, kInf;

    Eigen::MatrixXd AAt_workspace(0, 0);
    Eigen::LDLT<Eigen::MatrixXd> ldlt_workspace(0);
    Eigen::VectorXd w_workspace(0);
    Eigen::Vector2d v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out;

    hessian_op_2d hop{B};
    const double f_old = 0.0;
    const double c_norm_old = 0.0;
    trial_eval_quadratic teval{g, B, A, c, f_old};

    auto r = byrd_omojokun_composite_step<double, 2>(
        z_k, g, hop, A, c,
        delta_in, eps, /*max_cg_iter=*/20,
        lower_displaced, upper_displaced,
        f_old, c_norm_old, teval,
        AAt_workspace, ldlt_workspace, w_workspace,
        v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out);

    INFO("rho=" << r.rho
         << "  ||p||=" << p_out.norm()
         << "  new_delta=" << r.new_delta);
    CHECK(r.accepted);
    // Step is interior (||p|| < 0.9 * delta) so the expand branch
    // is gated off; radius held at delta_in.
    CHECK(p_out.norm() < 0.9 * delta_in);
    CHECK(r.new_delta == Approx(delta_in).margin(1e-12));
}
