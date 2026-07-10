// Joint-space (x, s) accumulator pins for the trust-region SQP family.
//
// Pins the two coupled quantities that feed the shared joint BFGS
// accumulator at the value level (not through E2E verdicts):
//
//   1. The joint BFGS curvature pair per N&W 2e eq. 18.13 with the NEW
//      multipliers held fixed at BOTH points -- including the
//      structural fact that the slack block of y is exactly zero.
//   2. The joint Lagrangian gradient slack block, which equals +mu at
//      a KKT point (the slack-bound multiplier zeta = mu), NOT zero
//      and NOT -mu. The historical in-code claim that this block
//      "vanishes at KKT by complementarity" is false for either sign.
//   3. The box-projected stationarity measure, which IS near-zero at
//      that KKT point while the raw infinity norm equals mu -- the
//      pair of facts that justifies keying inexact-Newton forcing and
//      reported stationarity on the projected measure.
//
// The mixed-multiplier / -mu inline variant is reproduced in-test and
// its divergence from the correct pair is asserted term-by-term: the
// two defects feed the same accumulator and can partially cancel on
// benign problems, so the compensation structure is documented at the
// accumulator level where it cannot hide.
//
// Reference: Nocedal and Wright 2e eq. 18.13 (fixed lambda_{k+1} SQP
//            curvature pair); Section 16.7 (projected gradient
//            optimality); Section 18.5 (slack reformulation);
//            Lalee, Nocedal, Plantenga 1998 SIAM J. Optim. 8(3)
//            Section 3.1.

#include "argmin/detail/sqp_common.h"
#include "argmin/detail/kkt_residual.h"
#include "argmin/solver/options.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/filter_trsqp_policy.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <cmath>
#include <limits>

using namespace argmin;

namespace
{

// Hand instance for the joint-space pins:
//
//     min 0.5 ||x||^2   s.t.   c(x) = x_0^2 + x_1 - 1 >= 0.
//
// KKT derivation: L = 0.5 ||x||^2 - mu (x_0^2 + x_1 - 1).
//   stationarity:  x_0 (1 - 2 mu) = 0,   x_1 - mu = 0.
//   The x_0 != 0 branch forces mu = 1/2, hence x_1 = 1/2; the active
//   constraint x_0^2 + x_1 - 1 = 0 gives x_0 = 1/sqrt(2) and
//   f = 0.5 (1/2 + 1/4) = 3/8. The x_0 = 0 branch gives x_1 = 1,
//   mu = 1, f = 1/2 > 3/8. Minimizer:
//     x* = (1/sqrt(2), 1/2),  mu* = 1/2,  constraint active (s* = 0).
struct joint_pin_problem
{
    static constexpr int problem_dimension = 2;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 1; }

    [[nodiscard]] double value(const Eigen::Vector<double, 2>& x) const
    {
        return 0.5 * x.squaredNorm();
    }

    void gradient(const Eigen::Vector<double, 2>& x,
                  Eigen::Vector<double, 2>& g) const
    {
        g = x;
    }

    void constraints(const Eigen::Vector<double, 2>& x, auto& c) const
    {
        c[0] = x[0] * x[0] + x[1] - 1.0;
    }

    void constraint_jacobian(const Eigen::Vector<double, 2>& x,
                             auto& J) const
    {
        J(0, 0) = 2.0 * x[0];
        J(0, 1) = 1.0;
    }
};

// Bound-active convergence instance:
//
//     min 0.5 ||x - (2, 0)||^2   s.t.   x in [-1, 1]^2,
//                                       c(x) = x_1 + 10 >= 0 (inactive).
//
// Optimum x* = (1, 0) with the upper bound on x_0 active and bound
// multiplier zeta = 1 (stationarity: grad_f(x*) - (-zeta) e_0 = 0 with
// grad_f(x*) = (-1, 0)). The raw composite KKT residual at x* equals
// |zeta| = 1 and can never reach a convergence threshold; the
// box-projected composite is exactly zero there.
struct bound_active_problem
{
    static constexpr int problem_dimension = 2;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 1; }

    [[nodiscard]] double value(const Eigen::Vector<double, 2>& x) const
    {
        const Eigen::Vector<double, 2> d{{x[0] - 2.0, x[1]}};
        return 0.5 * d.squaredNorm();
    }

    void gradient(const Eigen::Vector<double, 2>& x,
                  Eigen::Vector<double, 2>& g) const
    {
        g[0] = x[0] - 2.0;
        g[1] = x[1];
    }

    void constraints(const Eigen::Vector<double, 2>& x, auto& c) const
    {
        c[0] = x[1] + 10.0;
    }

    void constraint_jacobian(const Eigen::Vector<double, 2>& /*x*/,
                             auto& J) const
    {
        J(0, 0) = 0.0;
        J(0, 1) = 1.0;
    }

    [[nodiscard]] Eigen::Vector<double, 2> lower_bounds() const
    {
        return Eigen::Vector<double, 2>::Constant(-1.0);
    }

    [[nodiscard]] Eigen::Vector<double, 2> upper_bounds() const
    {
        return Eigen::Vector<double, 2>::Constant(1.0);
    }
};

}

TEST_CASE("compute_joint_bfgs_pair matches the hand-derived fixed-multiplier pair",
          "[sqp_common_joint]")
{
    // n = 2, n_eq = 0, n_ineq = 1 on the joint_pin_problem instance
    // (nonlinear constraint so J(x_old) != J(x_new) and the per-point
    // Jacobian evaluation is exercised, not just the multiplier
    // freeze).
    //
    // Old point x  = (1.0, 0.5):  g_old = x_old,  J_old = [2.0  1.0].
    // New point x+ = (0.8, 0.4):  g_new = x_new,  J_new = [1.6  1.0].
    // Multipliers: mu_old = 0.3 (stale), mu_plus = 0.7 (post-step).
    // Slacks: s_old = 0.5, s_new = 0.2 (gradients are z-independent
    // on the slack axis, so any s >= 0 works).
    //
    // N&W 2e eq. 18.13 with mu_plus fixed at BOTH points:
    //   grad_L(x+, mu+) x-block = g_new - J_new^T mu+
    //                           = (0.8 - 0.7 * 1.6, 0.4 - 0.7 * 1.0)
    //                           = (-0.32, -0.30)
    //   grad_L(x , mu+) x-block = g_old - J_old^T mu+
    //                           = (1.0 - 0.7 * 2.0, 0.5 - 0.7 * 1.0)
    //                           = (-0.40, -0.20)
    //   y_x = (-0.32 + 0.40, -0.30 + 0.20) = (0.08, -0.10)
    //   y_s = (+mu+) - (+mu+) = 0 exactly (grad_s L = +mu does not
    //         depend on z; the zero is structural, not numerical).
    const int n = 2;
    const int n_ineq = 1;

    Eigen::VectorXd g_old(2);
    g_old << 1.0, 0.5;
    Eigen::VectorXd g_new(2);
    g_new << 0.8, 0.4;
    Eigen::MatrixXd J_eq_old(0, 2);
    Eigen::MatrixXd J_eq_new(0, 2);
    Eigen::MatrixXd J_ineq_old(1, 2);
    J_ineq_old << 2.0, 1.0;
    Eigen::MatrixXd J_ineq_new(1, 2);
    J_ineq_new << 1.6, 1.0;
    Eigen::VectorXd lambda_eq(0);
    Eigen::VectorXd mu_plus(1);
    mu_plus << 0.7;
    Eigen::VectorXd z_old(3);
    z_old << 1.0, 0.5, 0.5;
    Eigen::VectorXd z_new(3);
    z_new << 0.8, 0.4, 0.2;

    Eigen::VectorXd grad_L_old(3);
    Eigen::VectorXd grad_L_new(3);
    Eigen::VectorXd sk(3);
    Eigen::VectorXd yk(3);

    detail::compute_joint_bfgs_pair<double, Eigen::Dynamic>(
        g_old, g_new, J_eq_old, J_ineq_old, J_eq_new, J_ineq_new,
        lambda_eq, mu_plus, z_new, z_old, n, n_ineq,
        grad_L_old, grad_L_new, sk, yk);

    // Hand values to 1e-12; the slack block exactly zero.
    CHECK(std::abs(yk[0] - 0.08) < 1e-12);
    CHECK(std::abs(yk[1] - (-0.10)) < 1e-12);
    CHECK(yk[2] == 0.0);

    // sk = z_new - z_old on the joint space.
    CHECK(std::abs(sk[0] - (-0.2)) < 1e-15);
    CHECK(std::abs(sk[1] - (-0.1)) < 1e-15);
    CHECK(std::abs(sk[2] - (-0.3)) < 1e-15);

    // Both gradient legs were assembled with mu_plus and the +mu slack
    // block (links this pin to the +mu invariant pin below).
    CHECK(std::abs(grad_L_old[0] - (-0.40)) < 1e-12);
    CHECK(std::abs(grad_L_old[1] - (-0.20)) < 1e-12);
    CHECK(grad_L_old[2] == 0.7);
    CHECK(std::abs(grad_L_new[0] - (-0.32)) < 1e-12);
    CHECK(std::abs(grad_L_new[1] - (-0.30)) < 1e-12);
    CHECK(grad_L_new[2] == 0.7);
}

TEST_CASE("mixed-multiplier / -mu inline pair diverges from the fixed pair by "
          "the documented compensation structure",
          "[sqp_common_joint]")
{
    // Reproduces the historical inline computation (old gradient with
    // OLD multipliers; slack block -mu at both points):
    //
    //   y_prefix_x = [g_new - J_new^T mu+] - [g_old - J_old^T mu_old]
    //   y_prefix_s = (-mu+) - (-mu_old)
    //
    // and pins its divergence from the N&W 18.13 pair:
    //
    //   y_correct_x - y_prefix_x
    //     = [g_old - J_old^T mu_old] - [g_old - J_old^T mu+]
    //     = +J_old^T (mu+ - mu_old)
    //       (the non-curvature multiplier-motion term the mixed pair
    //        injects into the secant);
    //   y_correct_s - y_prefix_s = +(mu+ - mu_old)
    //       (the slack pollution; scale ~ the multiplier motion, on
    //        top of the 2 mu gradient-level error of the -mu sign).
    //
    // The two error terms enter the SAME accumulator with structured
    // signs, which is why they can partially cancel on benign
    // problems (a green E2E cell does not certify this path).
    const Eigen::Vector2d g_old(1.0, 0.5);
    const Eigen::Vector2d g_new(0.8, 0.4);
    const Eigen::Vector2d J_old_row(2.0, 1.0);
    const Eigen::Vector2d J_new_row(1.6, 1.0);
    const double mu_old = 0.3;
    const double mu_plus = 0.7;

    // Correct fixed-multiplier pair (hand values from the pin above).
    const Eigen::Vector2d y_correct_x(0.08, -0.10);
    const double y_correct_s = 0.0;

    // Pre-fix inline reproduction.
    const Eigen::Vector2d y_prefix_x =
        (g_new - J_new_row * mu_plus) - (g_old - J_old_row * mu_old);
    const double y_prefix_s = (-mu_plus) - (-mu_old);

    const Eigen::Vector2d x_block_error =
        y_correct_x - y_prefix_x;
    const Eigen::Vector2d expected_x_error =
        J_old_row * (mu_plus - mu_old);
    CHECK(std::abs(x_block_error[0] - expected_x_error[0]) < 1e-12);
    CHECK(std::abs(x_block_error[1] - expected_x_error[1]) < 1e-12);

    const double s_block_error = y_correct_s - y_prefix_s;
    CHECK(std::abs(s_block_error - (mu_plus - mu_old)) < 1e-15);

    // The -mu slack sign alone is a 2 mu error at the gradient level:
    // the assembled slack block is +mu, the historical one was -mu.
    CHECK(std::abs((+mu_plus) - (-mu_plus) - 2.0 * mu_plus) < 1e-15);
}

TEST_CASE("assemble_joint_lagrangian_gradient slack block equals +mu at a KKT "
          "point",
          "[sqp_common_joint]")
{
    // At the joint_pin_problem KKT point x* = (1/sqrt(2), 1/2),
    // mu* = 1/2 (derivation in the fixture comment): the x-block of
    // the joint gradient vanishes, and the slack block equals +mu*
    // = +0.5 -- it does NOT vanish "by complementarity" (it equals
    // the slack-bound multiplier zeta = mu*), and it is not -mu*.
    const double x0 = 1.0 / std::sqrt(2.0);
    const double mu_star = 0.5;

    Eigen::VectorXd g(2);
    g << x0, 0.5;
    Eigen::MatrixXd J_eq(0, 2);
    Eigen::MatrixXd J_ineq(1, 2);
    J_ineq << 2.0 * x0, 1.0;
    Eigen::VectorXd lambda_eq(0);
    Eigen::VectorXd mu(1);
    mu << mu_star;

    Eigen::VectorXd grad_L(3);
    detail::assemble_joint_lagrangian_gradient<double, Eigen::Dynamic>(
        g, J_eq, J_ineq, lambda_eq, mu, 2, 1, grad_L);

    CHECK(std::abs(grad_L[0]) < 1e-15);
    CHECK(std::abs(grad_L[1]) < 1e-15);
    CHECK(grad_L[2] == +mu_star);
    CHECK(grad_L[2] != 0.0);
    CHECK(grad_L[2] != -mu_star);
}

TEST_CASE("box-projected stationarity is near-zero at the KKT point while the "
          "raw norm equals mu",
          "[sqp_common_joint]")
{
    // Same KKT point, joint primal z* = (x*, s* = 0) with the slack at
    // its lower bound. The raw infinity norm of the joint gradient is
    // mu* = 0.5 (the +mu slack block), so any raw-norm-keyed forcing
    // sequence or convergence gate is capped at sqrt(mu) / mu on
    // inequality-active problems. The box-projected measure
    // ||P(z - grad_L, l, u) - z||_inf is zero: the slack sits at its
    // bound with inward gradient +mu > 0, projecting to a zero step.
    constexpr double inf = std::numeric_limits<double>::infinity();
    const double x0 = 1.0 / std::sqrt(2.0);
    const double mu_star = 0.5;

    Eigen::VectorXd z(3);
    z << x0, 0.5, 0.0;
    Eigen::VectorXd grad_L(3);
    grad_L << 0.0, 0.0, mu_star;
    Eigen::VectorXd lower(3);
    lower << -inf, -inf, 0.0;
    Eigen::VectorXd upper(3);
    upper << inf, inf, inf;

    // Raw measure: exactly mu*.
    CHECK(grad_L.lpNorm<Eigen::Infinity>() == mu_star);

    // Projected measure (absolute-bound form): zero.
    CHECK(detail::kkt_residual_bound<double, Eigen::Dynamic>(
              z, grad_L, lower, upper) < 1e-15);

    // Displaced-bound form (the policies' Steihaug-CG box) agrees.
    const Eigen::VectorXd lower_disp = lower - z;
    const Eigen::VectorXd upper_disp = upper - z;
    CHECK(detail::projected_stationarity_displaced<double>(
              grad_L, lower_disp, upper_disp) < 1e-15);
}

TEST_CASE("tr_sqp accepted step produces an exactly-zero slack block in the "
          "joint BFGS y",
          "[sqp_common_joint][tr_sqp]")
{
    // Policy-level accumulator pin: after any accepted trust-region
    // SQP step the state's joint curvature vector y must carry an
    // exactly-zero slack block (N&W 18.13 with fixed new multipliers
    // makes grad_s L = +mu identical at both points). A stale nonzero
    // multiplier estimate is seeded so a mixed-multiplier pair --
    // y = grad_L(z+, mu+) - grad_L(z, mu_old) -- would leave a
    // nonzero slack block of magnitude |mu+ - mu_old| and fail this
    // check.
    joint_pin_problem problem;
    Eigen::Vector<double, 2> x0{{1.0, 0.5}};
    solver_options opts;
    opts.max_iterations = 50;

    tr_sqp_policy_accurate<2> policy;
    auto s = policy.init(problem, x0, opts);

    // Stale multiplier estimate as left behind by a prior iteration's
    // active-set re-estimation. Small enough that the projected model
    // direction still descends on the merit (so the step is accepted
    // and Sections P/Q run), large enough that a mixed-multiplier
    // pair leaves a visibly nonzero slack block: the re-estimated
    // multiplier at the strictly-feasible new iterate is 0, so the
    // mixed pair's slack block is (-mu_new) - (-mu_old) = +0.05.
    s.bufs.kkt_mu_ineq_buf[0] = 0.05;

    bool saw_accepted = false;
    for(int k = 0; k < 25 && !saw_accepted; ++k)
    {
        const auto r = policy.step(s);
        if(!r.is_null_step)
            saw_accepted = true;
    }
    REQUIRE(saw_accepted);

    INFO("joint yk after first accepted step: "
         << s.joint_yk_buf.transpose());
    CHECK(s.joint_yk_buf.tail(1)[0] == 0.0);
}

TEMPLATE_TEST_CASE(
    "kkt-gated convergence fires at a solution with an active box bound",
    "[sqp_common_joint][tr_sqp][filter_trsqp]",
    tr_sqp_policy_accurate<2>,
    filter_trsqp_policy_accurate<2>)
{
    using Policy = TestType;
    // Convergence-path check for the bound-aware composite residual:
    // on bound_active_problem the solver reaches x* = (1, 0) with the
    // upper bound on x_0 active and bound multiplier zeta = 1. The
    // raw-stationarity composite equals |zeta| = 1 there (see the
    // matching kkt_residual_test defect-documentation case), so the
    // kkt-gated criterion could structurally never fire and
    // termination happened only through the stall window with a
    // rejected-step status. With the box-projected stationarity leg
    // the residual is zero at x* and the criterion fires.
    bound_active_problem problem;
    Eigen::Vector<double, 2> x0{{0.0, 0.0}};
    solver_options opts;
    opts.max_iterations = 100;
    opts.set_gradient_threshold(Policy::default_gradient_tolerance);
    opts.set_step_threshold(Policy::default_step_tolerance_rel);
    opts.constraint_tolerance = Policy::default_feasibility_tolerance;

    step_budget_solver solver{Policy{}, problem, x0, opts};
    const auto result = solver.solve(opts);

    CHECK(result.status == solver_status::converged);
    CHECK(std::abs(result.x[0] - 1.0) < 1e-10);
    CHECK(std::abs(result.x[1]) < 1e-10);
}
