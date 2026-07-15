// Robustness: the filter-SQP feasibility-restoration strategies.
//
// When a filter rejects every trial step the iterate is stuck and a
// restoration leg tries to reduce the constraint violation h(x) back into
// the filter-acceptable region. These cells drive the two restoration
// helpers (restore_l1 steepest descent and restore_feasibility_qp) into
// each of their guard branches and assert the observable outcome:
//
//   restore_l1:
//     feasible-entry        -- h < tol on entry: success, zero steps.
//     descent-to-feasible   -- a violated equality / inequality is driven
//                             feasible; success with h -> 0.
//     stall-detection       -- progress slower than the stall ratio over
//                             the window returns unsuccessfully (the hybrid
//                             hook for falling back to the QP leg).
//     zero-subgradient      -- a vanishing descent direction (rank-deficient
//                             linearization) returns unsuccessfully instead
//                             of dividing by a ~0 norm.
//     backtrack-exhausted   -- a direction that never reduces h backtracks
//                             to alpha_min and returns unsuccessfully.
//
//   restore_feasibility_qp:
//     feasible-entry        -- h < tol on entry: success, zero steps.
//     qp-step-to-feasible   -- a min-norm QP step drives h -> 0.
//     zero-qp-step          -- a ~0 QP step (degenerate linearization)
//                             returns unsuccessfully.
//     step-budget-exhausted -- a too-small step makes partial progress and
//                             the leg returns unsuccessfully at max_steps.
//
// The QP leg is driven through a minimal min-norm-step mock solver so the
// restoration control flow is exercised independently of any particular
// production QP backend (the routing to a real QP solver is identical).
//
// Reference: Wachter and Biegler 2006 Section 3 (feasibility restoration);
//            Fletcher, Leyffer and Toint 2002 (filter-SQP).

#include "argmin/detail/filter_restoration.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using argmin::detail::restore_l1;
using argmin::detail::restore_feasibility_qp;

namespace
{

// Configurable affine constraint set c_eq = A_eq x - b_eq,
// c_ineq = A_ineq x - b_ineq (argmin convention: c_ineq >= 0 feasible).
// Empty A_* / b_* means "no constraints of that kind".
struct affine_problem
{
    Eigen::MatrixXd A_eq;
    Eigen::VectorXd b_eq;
    Eigen::MatrixXd A_ineq;
    Eigen::VectorXd b_ineq;

    void eval_constraints(const Eigen::Vector<double, 2>& x,
                          Eigen::VectorXd& c_eq,
                          Eigen::VectorXd& c_ineq) const
    {
        c_eq = A_eq.rows() > 0 ? (A_eq * x - b_eq).eval() : Eigen::VectorXd();
        c_ineq = A_ineq.rows() > 0 ? (A_ineq * x - b_ineq).eval()
                                   : Eigen::VectorXd();
    }

    void eval_constraint_jacobians(const Eigen::Vector<double, 2>& /*x*/,
                                   Eigen::MatrixXd& J_eq,
                                   Eigen::MatrixXd& J_ineq) const
    {
        J_eq = A_eq;
        J_ineq = A_ineq;
    }
};

// Minimal min-norm-step QP mock: returns p = scale * (J_eq^T rhs_eq +
// J_ineq^T max(rhs_ineq, 0)). For a full-rank single equality this is the
// exact least-norm feasibility step; a zero Jacobian yields p = 0, which
// drives restore_feasibility_qp's zero-step guard. `scale` throttles
// progress to reach the step-budget-exhausted branch.
struct min_norm_qp_mock
{
    double scale{1.0};

    struct result_type
    {
        Eigen::VectorXd x;
    };

    result_type solve(const auto& H, const auto& /*g*/,
                      const auto& J_eq, const auto& rhs_eq,
                      const auto& J_ineq, const auto& rhs_ineq,
                      const auto& /*p_lo*/, const auto& /*p_hi*/) const
    {
        const int n = static_cast<int>(H.rows());
        Eigen::VectorXd p = Eigen::VectorXd::Zero(n);
        if(J_eq.rows() > 0)
            p.noalias() += J_eq.transpose() * rhs_eq;
        if(J_ineq.rows() > 0)
            p.noalias() += J_ineq.transpose()
                * rhs_ineq.cwiseMax(0.0).matrix();
        p *= scale;
        return result_type{p};
    }
};

const Eigen::Vector<double, 2> kZeroGrad = Eigen::Vector<double, 2>::Zero();
const Eigen::Vector<double, 2> kLower{{-100.0, -100.0}};
const Eigen::Vector<double, 2> kUpper{{100.0, 100.0}};

affine_problem eq_problem(double coeff0, double rhs)
{
    affine_problem p;
    p.A_eq.resize(1, 2);
    p.A_eq << coeff0, 0.0;
    p.b_eq.resize(1);
    p.b_eq << rhs;
    return p;
}

}  // namespace

TEST_CASE("restore_l1 returns success on an already-feasible entry",
          "[robustness][filter-restoration][l1][entry]")
{
    // c_eq = x0 - 0 == 0 at x0 = 0: feasible on entry.
    affine_problem prob = eq_problem(1.0, 0.0);
    Eigen::Vector<double, 2> x0{{0.0, 0.0}};

    auto r = restore_l1<double, 2>(prob, x0, kZeroGrad, kLower, kUpper);

    CHECK(r.success);
    CHECK(r.constraint_violation < 1e-12);
    CHECK(std::isfinite(r.x[0]));
    CHECK(r.iterations_used == 0);
}

TEST_CASE("restore_l1 descends a violated equality to feasibility",
          "[robustness][filter-restoration][l1][descent]")
{
    // c_eq = x0 - 3: violated at x0 = 0, steepest descent closes it.
    affine_problem prob = eq_problem(1.0, 3.0);
    Eigen::Vector<double, 2> x0{{0.0, 0.0}};

    auto r = restore_l1<double, 2>(prob, x0, kZeroGrad, kLower, kUpper);

    CHECK(r.success);
    CHECK(r.constraint_violation < 1e-12);
    CHECK(r.x[0] == Approx(3.0).margin(1e-9));
}

TEST_CASE("restore_l1 descends a violated inequality to feasibility",
          "[robustness][filter-restoration][l1][ineq]")
{
    // c_ineq = x0 (>= 0 feasible): violated at x0 = -5, driven to 0.
    affine_problem prob;
    prob.A_ineq.resize(1, 2);
    prob.A_ineq << 1.0, 0.0;
    prob.b_ineq.resize(1);
    prob.b_ineq << 0.0;

    Eigen::Vector<double, 2> x0{{-5.0, 0.0}};

    auto r = restore_l1<double, 2>(prob, x0, kZeroGrad, kLower, kUpper);

    CHECK(r.success);
    CHECK(r.constraint_violation < 1e-12);
    CHECK(r.x[0] >= -1e-9);
}

TEST_CASE("restore_l1 returns unsuccessfully on stalled progress",
          "[robustness][filter-restoration][l1][stall]")
{
    // c_eq = x0 - 1000: unit-step steepest descent reduces h by ~0.1% per
    // step, so over the 5-step window progress stays above the 0.99 stall
    // ratio and the stall guard fires before feasibility.
    affine_problem prob = eq_problem(1.0, 1000.0);
    Eigen::Vector<double, 2> x0{{0.0, 0.0}};

    auto r = restore_l1<double, 2>(prob, x0, kZeroGrad, kLower, kUpper,
                                   /*sigma_restore=*/1e4,
                                   /*max_steps=*/20,
                                   /*stall_ratio=*/0.99,
                                   /*stall_window=*/5);

    CHECK_FALSE(r.success);
    CHECK(r.constraint_violation > 1.0);
    CHECK(std::isfinite(r.constraint_violation));
}

TEST_CASE("restore_l1 returns unsuccessfully on a vanishing subgradient",
          "[robustness][filter-restoration][l1][degenerate]")
{
    // Zero Jacobian with a nonzero constant residual: the L1 subgradient
    // direction is ~0, so no descent step exists.
    affine_problem prob;
    prob.A_eq = Eigen::MatrixXd::Zero(1, 2);
    prob.b_eq.resize(1);
    prob.b_eq << -1.0;  // c_eq = 0*x - (-1) = 1, constant & infeasible

    Eigen::Vector<double, 2> x0{{0.5, -0.5}};

    auto r = restore_l1<double, 2>(prob, x0, kZeroGrad, kLower, kUpper);

    CHECK_FALSE(r.success);
    CHECK(r.constraint_violation == Approx(1.0));
    CHECK(std::isfinite(r.x[0]));
}

TEST_CASE("restore_l1 returns unsuccessfully when backtracking is exhausted",
          "[robustness][filter-restoration][l1][backtrack]")
{
    // Nonzero Jacobian but a residual that does not respond to x (the
    // Jacobian is a linearization lie): the descent direction is nonzero
    // yet no step reduces h, so backtracking runs to alpha_min.
    struct frozen_problem
    {
        void eval_constraints(const Eigen::Vector<double, 2>& /*x*/,
                              Eigen::VectorXd& c_eq,
                              Eigen::VectorXd& c_ineq) const
        {
            c_eq.resize(1);
            c_eq << 1.0;  // constant, never improves
            c_ineq = Eigen::VectorXd();
        }
        void eval_constraint_jacobians(const Eigen::Vector<double, 2>& /*x*/,
                                       Eigen::MatrixXd& J_eq,
                                       Eigen::MatrixXd& J_ineq) const
        {
            J_eq.resize(1, 2);
            J_eq << 1.0, 0.0;  // nonzero -> subgradient guard passes
            J_ineq = Eigen::MatrixXd();
        }
    } prob;

    Eigen::Vector<double, 2> x0{{0.0, 0.0}};

    auto r = restore_l1<double, 2>(prob, x0, kZeroGrad, kLower, kUpper);

    CHECK_FALSE(r.success);
    CHECK(r.constraint_violation == Approx(1.0));
    CHECK(std::isfinite(r.x[0]));
    CHECK(std::isfinite(r.x[1]));
}

TEST_CASE("restore_feasibility_qp returns success on an already-feasible entry",
          "[robustness][filter-restoration][qp][entry]")
{
    affine_problem prob = eq_problem(1.0, 0.0);
    Eigen::Vector<double, 2> x0{{0.0, 0.0}};
    min_norm_qp_mock qp;

    auto r = restore_feasibility_qp<double, 2>(prob, qp, x0, kLower, kUpper);

    CHECK(r.success);
    CHECK(r.constraint_violation < 1e-12);
    CHECK(r.iterations_used == 0);
}

TEST_CASE("restore_feasibility_qp drives a violated equality feasible via the QP step",
          "[robustness][filter-restoration][qp][step]")
{
    affine_problem prob = eq_problem(1.0, 3.0);
    Eigen::Vector<double, 2> x0{{0.0, 0.0}};
    min_norm_qp_mock qp{.scale = 1.0};  // exact least-norm step

    auto r = restore_feasibility_qp<double, 2>(prob, qp, x0, kLower, kUpper);

    CHECK(r.success);
    CHECK(r.constraint_violation < 1e-12);
    CHECK(r.x[0] == Approx(3.0).margin(1e-9));
}

TEST_CASE("restore_feasibility_qp returns unsuccessfully on a ~zero QP step",
          "[robustness][filter-restoration][qp][zero-step]")
{
    // Zero Jacobian -> the mock's min-norm step is exactly zero.
    affine_problem prob;
    prob.A_eq = Eigen::MatrixXd::Zero(1, 2);
    prob.b_eq.resize(1);
    prob.b_eq << -1.0;  // c_eq = 1, infeasible

    Eigen::Vector<double, 2> x0{{0.0, 0.0}};
    min_norm_qp_mock qp;

    auto r = restore_feasibility_qp<double, 2>(prob, qp, x0, kLower, kUpper);

    CHECK_FALSE(r.success);
    CHECK(r.constraint_violation == Approx(1.0));
    CHECK(std::isfinite(r.x[0]));
}

TEST_CASE("restore_feasibility_qp returns unsuccessfully at the step budget",
          "[robustness][filter-restoration][qp][budget]")
{
    // A throttled QP step makes ~1% progress per step, so the max_steps
    // budget is reached before feasibility -- the leg reports failure but
    // has still reduced the violation.
    affine_problem prob = eq_problem(1.0, 3.0);
    Eigen::Vector<double, 2> x0{{0.0, 0.0}};
    min_norm_qp_mock qp{.scale = 0.01};

    const double h0 = 3.0;
    auto r = restore_feasibility_qp<double, 2>(prob, qp, x0, kLower, kUpper,
                                               /*max_steps=*/3);

    CHECK_FALSE(r.success);
    CHECK(r.constraint_violation < h0);      // progress was made
    CHECK(r.constraint_violation > 1e-6);    // but not to feasibility
    CHECK(std::isfinite(r.x[0]));
}
