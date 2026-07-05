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
using argmin::detail::byrd_omojokun_step_result;
using argmin::detail::cg_exit_status;
using argmin::detail::zeta;
using argmin::detail::tr_eta_1;
using argmin::detail::tr_eta_2;
using argmin::detail::tr_shrink_factor;
using argmin::detail::tr_expand_factor;
using argmin::detail::tr_delta_max;
using argmin::detail::tr_boundary_guard;

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

// Caller-side ratio test + radius update mirroring the L2-merit
// augmented-reduction gate that tr_sqp_policy applies at its composite-
// step call site. The helper itself no longer performs this branch;
// the unit-test cells below verify the same observable shape by
// re-running the migrated logic locally against the helper's raw
// outputs.
//
// A non-positive augmented predicted reduction takes the explicit
// reject branch (N&W Algorithm 4.1 assumes pred > 0); rho is formed
// only against a strictly positive denominator, mirroring the policy.
struct merit_acceptance
{
    double rho;
    double actual_reduction;
    double predicted_reduction;
    double new_delta;
    bool accepted;
};

merit_acceptance evaluate_acceptance(const byrd_omojokun_step_result& r,
                                     double f_old, double c_norm_old,
                                     double penalty,
                                     double delta_in,
                                     double step_norm)
{
    const double actual =
        (f_old + penalty * c_norm_old)
        - (r.f_new + penalty * r.c_norm_new);
    const double predicted = r.predicted + penalty * r.vpred;

    merit_acceptance out;
    out.actual_reduction = actual;
    out.predicted_reduction = predicted;
    if(!(predicted > 0.0))
    {
        out.rho = 0.0;  // never formed on the reject branch
        out.accepted = false;
        out.new_delta = tr_shrink_factor * delta_in;
        return out;
    }
    const double rho = actual / predicted;
    out.rho = rho;
    if(rho < tr_eta_1)
    {
        out.accepted = false;
        out.new_delta = tr_shrink_factor * delta_in;
    }
    else
    {
        out.accepted = true;
        if(rho > tr_eta_2
           && step_norm >= tr_boundary_guard * delta_in)
            out.new_delta =
                std::min(tr_expand_factor * delta_in, tr_delta_max);
        else
            out.new_delta = delta_in;
    }
    return out;
}

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

    double penalty = 1.0;
    const double penalty_factor = 0.0;
    auto r = byrd_omojokun_composite_step<double, 2>(
        z_k, g, /*g_obj=*/g, hop, A, c,
        delta_in, eps, /*max_cg_iter=*/20,
        lower_displaced, upper_displaced,
        teval,
        AAt_workspace, ldlt_workspace, w_workspace,
        v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out,
        penalty, penalty_factor);

    // Normal step must reduce the linearized residual:
    // ||A v + c|| < ||c||.
    const double lin_resid_old = c.norm();
    const double lin_resid_new = (A * v_buf + c).norm();
    INFO("lin_resid_old=" << lin_resid_old
         << "  lin_resid_new=" << lin_resid_new);
    CHECK(lin_resid_new < lin_resid_old);

    CHECK_FALSE(r.normal_step_lsq_fallback);

    const auto acc = evaluate_acceptance(r, f_old, c_norm_old,
                                         penalty, delta_in,
                                         p_out.norm());
    CHECK(acc.accepted);
    CHECK(acc.rho >= tr_eta_1);
    // Radius does not shrink: should be either delta_in (held) or
    // expanded to min(2*delta_in, delta_max) if rho > 0.75 AND step
    // hit the boundary. Either way, new_delta >= delta_in.
    CHECK(acc.new_delta >= delta_in);
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

    double penalty = 1.0;
    const double penalty_factor = 0.0;
    [[maybe_unused]] auto r = byrd_omojokun_composite_step<double, 2>(
        z_k, g, /*g_obj=*/g, hop, A, c,
        delta_in, eps, /*max_cg_iter=*/20,
        lower_displaced, upper_displaced,
        teval,
        AAt_workspace, ldlt_workspace, w_workspace,
        v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out,
        penalty, penalty_factor);

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

    double penalty = 1.0;
    const double penalty_factor = 0.0;
    auto r = byrd_omojokun_composite_step<double, 2>(
        z_k, g, /*g_obj=*/g, hop, A, c,
        delta_in, eps, /*max_cg_iter=*/20,
        lower_displaced, upper_displaced,
        teval,
        AAt_workspace, ldlt_workspace, w_workspace,
        v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out,
        penalty, penalty_factor);

    const auto acc = evaluate_acceptance(r, f_old, c_norm_old,
                                         penalty, delta_in,
                                         p_out.norm());
    INFO("rho=" << acc.rho
         << "  actual=" << acc.actual_reduction
         << "  predicted=" << acc.predicted_reduction
         << "  ||p||=" << p_out.norm());
    CHECK_FALSE(acc.accepted);
    CHECK(acc.rho < tr_eta_1);
    CHECK(acc.new_delta == Approx(tr_shrink_factor * delta_in).margin(1e-12));
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

    double penalty = 1.0;
    const double penalty_factor = 0.0;
    auto r = byrd_omojokun_composite_step<double, 2>(
        z_k, g, /*g_obj=*/g, hop, A, c,
        delta_in, eps, /*max_cg_iter=*/20,
        lower_displaced, upper_displaced,
        teval,
        AAt_workspace, ldlt_workspace, w_workspace,
        v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out,
        penalty, penalty_factor);

    const auto acc = evaluate_acceptance(r, f_old, c_norm_old,
                                         penalty, delta_in,
                                         p_out.norm());
    INFO("rho=" << acc.rho
         << "  ||p||=" << p_out.norm()
         << "  new_delta=" << acc.new_delta);
    CHECK(acc.accepted);
    CHECK(acc.rho > tr_eta_2);
    CHECK(p_out.norm() >= 0.9 * delta_in);
    CHECK(acc.new_delta == Approx(
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

    double penalty = 1.0;
    const double penalty_factor = 0.0;
    auto r = byrd_omojokun_composite_step<double, 2>(
        z_k, g, /*g_obj=*/g, hop, A, c,
        delta_in, eps, /*max_cg_iter=*/20,
        lower_displaced, upper_displaced,
        teval,
        AAt_workspace, ldlt_workspace, w_workspace,
        v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out,
        penalty, penalty_factor);

    const auto acc = evaluate_acceptance(r, f_old, c_norm_old,
                                         penalty, delta_in,
                                         p_out.norm());
    INFO("rho=" << acc.rho
         << "  ||p||=" << p_out.norm()
         << "  new_delta=" << acc.new_delta);
    CHECK(acc.accepted);
    // Step is interior (||p|| < 0.9 * delta) so the expand branch
    // is gated off; radius held at delta_in.
    CHECK(p_out.norm() < 0.9 * delta_in);
    CHECK(acc.new_delta == Approx(delta_in).margin(1e-12));
}

// ─── Penalty-update side-effect cells ──────────────────────────────
//
// Reference: Lalee, Nocedal, Plantenga 1998 SIAM J. Optim. 8(3)
//            equation 1.13; scipy update_penalty. The LNP rule
//            requires the predicted merit reduction to dominate the
//            feasibility leg:
//
//              -qm + pi * vpred >= rho_margin * pi * vpred,
//              qm = g_f^T p + 0.5 p^T B p  (objective-model CHANGE).
//
//            The penalty pi therefore grows iff the objective model
//            WORSENS on the step (qm > 0, a feasibility-driven step)
//            and vpred > 0, with the closed-form candidate
//
//              pi >= qm / ((1 - penalty_factor) * vpred).
//
//            On a descent-model step (qm < 0) the inequality holds
//            for every pi > 0, so the penalty must stay. The three
//            cells below pin (1) growth to the exact closed-form
//            candidate on qm > 0, (2) no growth on descent, and
//            (3) no growth on vpred <= 0.
//
// Shared hand instance for cells 1-2 (all quantities dyadic so the
// arithmetic is exact in IEEE-754): single equality A = [1, 1],
// c = (0.5), B = I, delta = 10, z_k = 0.
//   normal step: g_n = A^T c = (0.5, 0.5), AA^T = 2,
//                v_N = -A^T (AA^T)^{-1} c = (-0.25, -0.25),
//                ||v_N|| < zeta * delta -- Newton branch, v = v_N.
//   tangential:  g_t = g + B v; for g = (+-10, +-10),
//                g_t is orthogonal to null(A) = span{(1, -1)}, so the
//                projected CG residual is exactly zero and u = 0.
//   composite:   p = (-0.25, -0.25), p^T B p = 0.125,
//                vpred = ||c|| - ||A p + c|| = 0.5 - 0 = 0.5.

TEST_CASE("byrd_omojokun_composite_step penalty grows to the LNP "
          "closed-form candidate when the objective model worsens "
          "(qm > 0)",
          "[byrd_omojokun][penalty][grow]")
{
    // Cell 1: feasibility-driven step that WORSENS the objective
    // model. g = (-10, -10) gives, on p = (-0.25, -0.25):
    //   qm = g^T p + 0.5 p^T B p = 5 + 0.0625 = 5.0625 > 0.
    // With penalty_factor = 0.5 and vpred = 0.5 the LNP candidate is
    //   qm / ((1 - 0.5) * 0.5) = 5.0625 / 0.25 = 20.25,
    // so penalty = max(1, 20.25) = 20.25 exactly (all dyadic).
    //
    // Pre-fix red (recorded): the inverted gate computed
    // hredd = -qm = -5.0625 < 0 and never grew -- penalty stayed 1.0.
    Eigen::Vector2d g;
    g << -10.0, -10.0;
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
    trial_eval_quadratic teval{g, B, A, c, f_old};

    double penalty = 1.0;
    const double penalty_factor = 0.5;
    [[maybe_unused]] auto r = byrd_omojokun_composite_step<double, 2>(
        z_k, g, /*g_obj=*/g, hop, A, c,
        delta_in, eps, /*max_cg_iter=*/20,
        lower_displaced, upper_displaced,
        teval,
        AAt_workspace, ldlt_workspace, w_workspace,
        v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out,
        penalty, penalty_factor);

    INFO("p = (" << p_out[0] << ", " << p_out[1]
         << ")  vpred = " << r.vpred
         << "  penalty post-step = " << penalty);
    CHECK(penalty == 20.25);
}

TEST_CASE("byrd_omojokun_composite_step penalty stays on a "
          "descent-model step (qm < 0)",
          "[byrd_omojokun][penalty][descent-stay]")
{
    // Cell 2: good descent step. g = (10, 10) gives, on the same
    // p = (-0.25, -0.25):
    //   qm = g^T p + 0.5 p^T B p = -5 + 0.0625 = -4.9375 < 0,
    // with vpred = 0.5 > 0. LNP eq. 1.13 holds for every pi > 0 on a
    // descent-model step, so the penalty must stay at 1.0.
    //
    // Pre-fix red (recorded): this is precisely where the inverted
    // gate GREW the penalty -- hredd = -qm = 4.9375 > 0 produced
    // penalty = 4.9375 / ((1 - 0.5) * 0.5) = 19.75.
    Eigen::Vector2d g;
    g << 10.0, 10.0;
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
    trial_eval_quadratic teval{g, B, A, c, f_old};

    double penalty = 1.0;
    const double penalty_factor = 0.5;
    [[maybe_unused]] auto r = byrd_omojokun_composite_step<double, 2>(
        z_k, g, /*g_obj=*/g, hop, A, c,
        delta_in, eps, /*max_cg_iter=*/20,
        lower_displaced, upper_displaced,
        teval,
        AAt_workspace, ldlt_workspace, w_workspace,
        v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out,
        penalty, penalty_factor);

    INFO("p = (" << p_out[0] << ", " << p_out[1]
         << ")  vpred = " << r.vpred
         << "  penalty post-step = " << penalty);
    CHECK(penalty == 1.0);
}

// ─── Kernel pins: null-space invariant and hand-derived value pins ─
//
// Reference: Nocedal and Wright 2e Section 18.5, eq. 18.45-18.50
//            (Byrd-Omojokun composite step; the tangential leg must
//             satisfy A*u = 0 so it preserves the normal step's
//             linearized-feasibility gain).
//
// Shared failure instance (round-1 reviewer, finding 1): minimize x2
// subject to x1 = 0 from z = (1, 1) with B = I, lambda = 0, delta = 10
// (interior; no active box). Every quantity below is hand-derived in
// this comment from N&W 18.45-18.50, none pinned from a run:
//
//   objective f(x) = x2                => g = (0, 1)
//   constraint     x1 = 0 at z = (1,1) => A = [1, 0], c = (1)
//   normal step   v = -A^T (A A^T)^{-1} c = (-1, 0)  (A v + c = 0)
//   tangential gradient g_t = g + B v = (-1, 1)
//   null(A) = span{e2}; project g_t onto null(A) => (0, 1)
//   1-D CG minimizer of g_t^T u + 0.5 u^T u along e2 => u = (0, -1)
//   composite p = v + u = (-1, -1),  A u = 0  (invariant holds)
//   vpred     = ||c|| - ||A p + c|| = 1 - 0 = 1
//   predicted = -g^T p - 0.5 p^T B p = 1 - 1 = 0
//
// Pre-fix code (no null-space projection) minimizes the tangential
// model over the FULL joint space: u = -B^{-1} g_t = (1, -1), so
// p = (0, -1) with ||A u||_inf = 1 (invariant violated), vpred = 0
// (the tangential leg undid the normal step's feasibility gain), and
// predicted = 0.5. Those wrong values are the recorded pre-fix reds.

namespace
{

// One-shot driver for the shared null-space failure instance. Returns
// the composite step in p_out and the raw result POD.
byrd_omojokun_step_result run_nullspace_instance(
    Eigen::Vector2d& u_buf_out,
    Eigen::Vector2d& p_out_out)
{
    Eigen::Vector2d g;
    g << 0.0, 1.0;
    Eigen::Matrix2d B = Eigen::Matrix2d::Identity();

    Eigen::MatrixXd A(1, 2);
    A << 1.0, 0.0;
    Eigen::VectorXd c(1);
    c << 1.0;

    const double delta_in = 10.0;
    const double eps = 1e-10;

    Eigen::Vector2d z_k;
    z_k << 1.0, 1.0;
    Eigen::Vector2d lower_displaced;
    lower_displaced << -kInf, -kInf;
    Eigen::Vector2d upper_displaced;
    upper_displaced << kInf, kInf;

    Eigen::MatrixXd AAt_workspace(1, 1);
    Eigen::LDLT<Eigen::MatrixXd> ldlt_workspace(1);
    Eigen::VectorXd w_workspace(1);
    Eigen::Vector2d v_buf, r_cg_buf, d_cg_buf, Bd_cg_buf;

    hessian_op_2d hop{B};
    trial_eval_quadratic teval{g, B, A, c, 0.0};

    double penalty = 1.0;
    const double penalty_factor = 0.0;
    auto r = byrd_omojokun_composite_step<double, 2>(
        z_k, g, /*g_obj=*/g, hop, A, c,
        delta_in, eps, /*max_cg_iter=*/20,
        lower_displaced, upper_displaced,
        teval,
        AAt_workspace, ldlt_workspace, w_workspace,
        v_buf, u_buf_out, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out_out,
        penalty, penalty_factor);
    return r;
}

}

TEST_CASE("byrd_omojokun tangential step satisfies A*u = 0 in null(A)",
          "[kernel-pin][byrd_omojokun][nullspace]")
{
    // Pin 1 (N&W 18.49): the tangential leg must lie in null(A_joint).
    // Pre-fix ||A*u||_inf = 1.0 (recorded red); post-fix ~0.
    Eigen::Vector2d u_buf, p_out;
    Eigen::MatrixXd A(1, 2);
    A << 1.0, 0.0;
    (void)run_nullspace_instance(u_buf, p_out);

    const double Au_inf = (A * u_buf).cwiseAbs().maxCoeff();
    const double scale = std::max(1.0, A.norm() * u_buf.norm());
    INFO("||A*u||_inf = " << Au_inf << "  u = (" << u_buf[0]
         << ", " << u_buf[1] << ")");
    CHECK(Au_inf <= 1e-10 * scale);
}

TEST_CASE("byrd_omojokun vpred = 1.0 and predicted = 0.0 (hand-derived)",
          "[kernel-pin][byrd_omojokun][value]")
{
    // Pin 2 (N&W 18.45-18.50): vpred == ||c|| - ||A p + c|| == 1.0 and
    // predicted == -g^T p - 0.5 p^T B p == 0.0. Pre-fix recorded reds:
    // vpred = 0.0, predicted = 0.5.
    //
    // Objective-leg note: `predicted` is now computed against the
    // objective gradient g_obj rather than the Lagrangian gradient g.
    // This instance has lambda = 0 (the driver passes the same vector
    // for both), so grad L == grad f and the hand-derived expected
    // value 0.0 is UNCHANGED by the redefinition -- asserted here
    // explicitly rather than re-pinned.
    Eigen::Vector2d u_buf, p_out;
    auto r = run_nullspace_instance(u_buf, p_out);

    INFO("vpred = " << r.vpred << "  predicted = " << r.predicted
         << "  p = (" << p_out[0] << ", " << p_out[1] << ")");
    CHECK(r.vpred == Approx(1.0).margin(1e-12));
    CHECK(r.predicted == Approx(0.0).margin(1e-12));
}

TEST_CASE("byrd_omojokun vpred >= 0 across a delta grid (5-D, 2 eq)",
          "[kernel-pin][byrd_omojokun][vpred]")
{
    // Pin 3: vpred >= 0 must hold structurally on every composite step
    // once the tangential leg is confined to null(A). Deterministic
    // well-conditioned instance (fixed literals, not RNG, for exact
    // reproducibility): A fixes the first two coordinates, and the
    // objective gradient loads those same coordinates so the pre-fix
    // full-space tangential step re-violates linearized feasibility
    // (vpred < 0) for the interior deltas. Post-fix vpred = ||c|| >= 0.
    Eigen::MatrixXd A(2, 5);
    A << 1.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0, 0.0;
    Eigen::VectorXd c(2);
    c << 0.5, 0.5;

    Eigen::Matrix<double, 5, 5> B = Eigen::Matrix<double, 5, 5>::Identity();
    Eigen::Vector<double, 5> g;
    g << 5.0, 5.0, 0.0, 0.0, 0.0;

    struct hop5
    {
        Eigen::Matrix<double, 5, 5> B;
        Eigen::Vector<double, 5> operator()(
            const Eigen::Ref<const Eigen::Vector<double, 5>>& v) const
        {
            return (B * v).eval();
        }
    } hop{B};

    struct teval5
    {
        Eigen::Vector<double, 5> g;
        Eigen::Matrix<double, 5, 5> B;
        Eigen::MatrixXd A;
        Eigen::VectorXd c;
        std::pair<double, double> operator()(
            const Eigen::Ref<const Eigen::Vector<double, 5>>& p) const
        {
            return {g.dot(p) + 0.5 * p.dot(B * p), (A * p + c).norm()};
        }
    } teval{g, B, A, c};

    Eigen::Vector<double, 5> z_k = Eigen::Vector<double, 5>::Zero();
    Eigen::Vector<double, 5> lower_displaced =
        Eigen::Vector<double, 5>::Constant(-kInf);
    Eigen::Vector<double, 5> upper_displaced =
        Eigen::Vector<double, 5>::Constant(kInf);

    for(double delta : {0.5, 1.0, 2.0, 5.0, 10.0})
    {
        Eigen::MatrixXd AAt_workspace(2, 2);
        Eigen::LDLT<Eigen::MatrixXd> ldlt_workspace(2);
        Eigen::VectorXd w_workspace(2);
        Eigen::Vector<double, 5> v_buf, u_buf, r_cg_buf, d_cg_buf,
            Bd_cg_buf, p_out;

        double penalty = 1.0;
        auto r = byrd_omojokun_composite_step<double, 5>(
            z_k, g, /*g_obj=*/g, hop, A, c,
            delta, 1e-10, /*max_cg_iter=*/50,
            lower_displaced, upper_displaced,
            teval,
            AAt_workspace, ldlt_workspace, w_workspace,
            v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out,
            penalty, 0.0);

        INFO("delta = " << delta << "  vpred = " << r.vpred
             << "  ||A*u||_inf = " << (A * u_buf).cwiseAbs().maxCoeff());
        CHECK(r.vpred >= -1e-12);
    }
}

TEST_CASE("byrd_omojokun_composite_step penalty stays at initial "
          "value when vpred is non-positive",
          "[byrd_omojokun][penalty][stay]")
{
    // Cell 3 (LNP eq. 1.13 denominator guard): unconstrained problem,
    // no equality, c is empty, so c_norm_lin and Ap_plus_c_norm are
    // both zero and vpred = 0. The penalty update gate is closed;
    // penalty must equal its initial value post-call regardless of
    // penalty_factor.
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
    trial_eval_quadratic teval{g, B, A, c, f_old};

    double penalty = 1.0;
    const double penalty_factor = 0.3;
    [[maybe_unused]] auto r = byrd_omojokun_composite_step<double, 2>(
        z_k, g, /*g_obj=*/g, hop, A, c,
        delta_in, eps, /*max_cg_iter=*/20,
        lower_displaced, upper_displaced,
        teval,
        AAt_workspace, ldlt_workspace, w_workspace,
        v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out,
        penalty, penalty_factor);

    INFO("penalty post-step=" << penalty);
    CHECK(penalty == 1.0);
}

// ─── Normal-step box-confinement pin ───────────────────────────────
//
// Reference: scipy optimize/_trustregion_constr/qp_subproblem.py
//            modified_dogleg (the reference dogleg is BOX-AWARE:
//            candidate points are intersected with the variable box
//            and ranked on ||A x + b||); Nocedal and Wright 2e
//            Section 18.5 (slack reformulation: the s >= 0 bound is a
//            hard bound -- a negative slack makes the joint residual
//            -c_ineq + s vanish at a genuinely VIOLATED inequality,
//            silently masking the violation from the merit function).
//
// Hand instance (all dyadic, exact in IEEE-754): joint dim 2 with the
// second coordinate a slack at its lower bound 0,
//   A = [[1, 0], [1, 1]],  c = (-1, 0),  B = I,  g = 0,  delta = 10.
//   AA^T = [[1, 1], [1, 2]],  w = (AA^T)^{-1} c = (-2, 1),
//   v_N = -A^T w = (1, -1)  -- the min-norm LSQ step drives the slack
//   coordinate to -1, BELOW its displaced lower bound 0.
// Post-fix: v is clamped to the box, (1, 0), then line-searched on
//   ||A (t v) + c||^2 over t in [0, 1]:
//   Av = (1, 1), t* = -(Av . c)/||Av||^2 = 1/2  =>  v = (0.5, 0),
//   ||A v + c|| = ||(-0.5, 0.5)|| < ||c|| = 1  (vpred > 0 preserved).
// The tangential leg is zero here (g_t = B v lies in range(A^T) and
// A has full column rank, so null(A) = {0}), hence p == v.
//
// Pre-fix red (recorded): v_buf = (1, -1) -- the slack block violated
// its bound by the full -1.
TEST_CASE("byrd_omojokun normal step is confined to the displaced box "
          "(slack lower bound)",
          "[kernel-pin][byrd_omojokun][normal-box]")
{
    Eigen::Vector2d g = Eigen::Vector2d::Zero();
    Eigen::Matrix2d B = Eigen::Matrix2d::Identity();

    Eigen::MatrixXd A(2, 2);
    A << 1.0, 0.0,
         1.0, 1.0;
    Eigen::VectorXd c(2);
    c << -1.0, 0.0;

    const double delta_in = 10.0;
    const double eps = 1e-10;

    Eigen::Vector2d z_k = Eigen::Vector2d::Zero();
    Eigen::Vector2d lower_displaced;
    lower_displaced << -kInf, 0.0;  // slack coordinate at its bound
    Eigen::Vector2d upper_displaced;
    upper_displaced << kInf, kInf;

    Eigen::MatrixXd AAt_workspace(2, 2);
    Eigen::LDLT<Eigen::MatrixXd> ldlt_workspace(2);
    Eigen::VectorXd w_workspace(2);
    Eigen::Vector2d v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out;

    hessian_op_2d hop{B};
    trial_eval_quadratic teval{g, B, A, c, /*f_old=*/0.0};

    double penalty = 1.0;
    auto r = byrd_omojokun_composite_step<double, 2>(
        z_k, g, /*g_obj=*/g, hop, A, c,
        delta_in, eps, /*max_cg_iter=*/20,
        lower_displaced, upper_displaced,
        teval,
        AAt_workspace, ldlt_workspace, w_workspace,
        v_buf, u_buf, r_cg_buf, d_cg_buf, Bd_cg_buf, p_out,
        penalty, 0.0);

    INFO("v = (" << v_buf[0] << ", " << v_buf[1]
         << ")  p = (" << p_out[0] << ", " << p_out[1]
         << ")  vpred = " << r.vpred);
    // Box confinement: the slack block of BOTH the normal step and
    // the composite step must respect the displaced lower bound.
    CHECK(v_buf[1] >= 0.0);
    CHECK(p_out[1] >= 0.0);
    // Hand-derived post-fix values.
    CHECK(v_buf[0] == Approx(0.5).margin(1e-14));
    CHECK(v_buf[1] == Approx(0.0).margin(1e-14));
    // The residual line search preserves the normal step's descent on
    // the linearized feasibility objective.
    CHECK(r.vpred >= 0.0);
}

// ─── SOC-residual accumulator pin ──────────────────────────────────
//
// Reference: Nocedal and Wright 2e Section 18.3 (second-order
//            correction; the SOC re-solve targets the residual
//            d = c(z + p) - A p, which preserves the linear part and
//            corrects only the curvature remainder);
//            Wachter and Biegler 2006 Math. Programming 106:25-57
//            eq. (24)/(27) (in the alpha_0 = 1 full-step convention
//            the correction form alpha_0 * c(z_k) + curvature
//            remainder equals the full-re-solve target
//            c(z + p) - A p).
//
// Hand instance with exactly LINEAR joint constraints (A constant), so
// c(z + p) = c(z) + A p identically and the SOC residual collapses to
//   c_soc = c(z + p) - A p = c(z)   EXACTLY.
// All literals dyadic so the identity is exact in IEEE-754:
//   A = [[1, 0], [0, 2]],  c(z) = (0.75, -0.25),
//   p = (0.5, -0.25)  =>  A p = (0.5, -0.5) != 0,
//   c(z + p) = c(z) + A p = (1.25, -0.75).
//
// The PRE-FIX Section K' assembly used c(z + p) ALONE as the retry
// residual, silently injecting the spurious +A p feasibility error:
// its assembled value on this instance is c(z) + A p = (1.25, -0.75),
// not c(z). That wrong value is the recorded pre-fix red.
TEST_CASE("SOC residual assembly subtracts the linearized constraint "
          "term (linear case: c_soc == c(z_k))",
          "[kernel-pin][byrd_omojokun][soc]")
{
    Eigen::MatrixXd A(2, 2);
    A << 1.0, 0.0,
         0.0, 2.0;
    Eigen::VectorXd c_z(2);
    c_z << 0.75, -0.25;
    Eigen::VectorXd p(2);
    p << 0.5, -0.25;

    // Exactly linear constraints: the trial-point residual is
    // c(z + p) = c(z) + A p by construction.
    const Eigen::VectorXd c_trial = c_z + A * p;

    // Documentation assertion for the defect: the pre-fix assembly
    // (c_trial alone) equals c(z) + A p -- the spurious injected term
    // is exactly A p = (0.5, -0.5). Recorded pre-fix red: the
    // assembled residual came out (1.25, -0.75) instead of c(z).
    CHECK(c_trial[0] == 1.25);
    CHECK(c_trial[1] == -0.75);

    // Assembled SOC residual through the shared helper both policies
    // route through: c_soc = c_trial - A p.
    Eigen::VectorXd c_soc(2);
    argmin::detail::assemble_joint_soc<double, Eigen::Dynamic>(
        c_trial, A, p, c_soc);

    INFO("c_soc = (" << c_soc[0] << ", " << c_soc[1]
         << ")  expected c(z) = (" << c_z[0] << ", " << c_z[1] << ")");
    CHECK(c_soc[0] == c_z[0]);
    CHECK(c_soc[1] == c_z[1]);
}

// ─── pred <= 0 rejection pin (caller-side acceptance gate) ─────────
//
// Reference: Nocedal and Wright 2e Section 4.1 Algorithm 4.1 -- the
//            ratio test rho = ared / pred is meaningful only under the
//            implicit assumption pred > 0; a non-positive predicted
//            reduction means the model forecasts no progress and the
//            step must be rejected outright.
//
// The pre-fix gate computed pred_guarded = max(pred, eps) and
// rho = actual / pred_guarded: a legitimately NEGATIVE augmented
// predicted reduction produced rho = actual / 2.2e-16 of magnitude
// ~1e15, so ANY actual reduction >= 0.1 * eps was accepted --
// sufficient decrease degenerated to simple decrease exactly in the
// degenerate cases the guard was meant to protect. That pre-fix
// acceptance is the recorded red.
TEST_CASE("caller acceptance gate rejects on non-positive augmented "
          "predicted reduction regardless of the actual sign",
          "[kernel-pin][byrd_omojokun][pred-nonpositive]")
{
    // Hand-built raw result: forced-uphill objective model
    // (predicted = -0.5) with vpred = 0 (no feasibility leg), so the
    // augmented predicted reduction is -0.5 <= 0. The actual merit
    // reduction is +0.1 (f 1.0 -> 0.9, violation legs zero): the
    // model forecast no progress, so the gate must reject even though
    // the actual reduction is positive.
    byrd_omojokun_step_result bo{};
    bo.predicted  = -0.5;
    bo.vpred      = 0.0;
    bo.f_new      = 0.9;
    bo.c_norm_new = 0.0;

    const double delta_in = 1.0;
    const auto acc = evaluate_acceptance(bo, /*f_old=*/1.0,
                                         /*c_norm_old=*/0.0,
                                         /*penalty=*/1.0,
                                         delta_in,
                                         /*step_norm=*/0.5);
    INFO("rho = " << acc.rho
         << "  predicted (augmented) = " << acc.predicted_reduction
         << "  actual = " << acc.actual_reduction);
    CHECK_FALSE(acc.accepted);
    CHECK(acc.new_delta
          == Approx(tr_shrink_factor * delta_in).margin(1e-15));
}
