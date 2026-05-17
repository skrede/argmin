// Unit cells for the Levenberg-Marquardt feasibility-restoration
// helper.
//
// Reference: Nocedal and Wright 2e Section 10.3 (Levenberg-Marquardt
//            for least-squares);
//            Wachter and Biegler 2006 Math. Programming 106:25-57
//            Section 3.3 (IPOPT restoration phase; the prototype LM
//            cell is a structural simplification of the full
//            restoration leg).
//
// argmin variant: the helper minimizes 1/2 ||c(x)||_2^2 subject to
//                 lower <= x <= upper via the LM update
//                 (J^T J + lambda I) dx = -J^T c with adaptive lambda
//                 and box projection. These cells drive the helper
//                 directly with hand-rolled constraint and Jacobian
//                 callbacks; the policy-integration path is exercised
//                 separately by the filter_trsqp unit cells through
//                 the runtime opt-in restoration knob.

#include "argmin/detail/restoration.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>

using argmin::detail::feasibility_restoration;
using argmin::detail::lm_restoration_result;
using argmin::detail::restoration_status;
using argmin::detail::restoration_lambda_min;
using argmin::detail::restoration_lambda_max;

namespace
{

// Linear constraint c(x) = x[0] + x[1] - 1 with constant Jacobian
// [1, 1]. A single Newton-style step from any interior point closes
// the constraint exactly; LM with small lambda effectively reproduces
// the Newton step.
struct linear_problem
{
    void constraints(const Eigen::Vector2d& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c(0) = x(0) + x(1) - 1.0;
    }
    void jacobian(const Eigen::Vector2d& /*x*/, Eigen::MatrixXd& J) const
    {
        J.resize(1, 2);
        J(0, 0) = 1.0;
        J(0, 1) = 1.0;
    }
};

// Quadratic constraint c(x) = x[0]^2 + x[1] - 1 with Jacobian
// [2*x[0], 1]. Multi-iter to feasibility; LM lambda adapts as the
// model improves locally near the feasible manifold.
struct quadratic_problem
{
    void constraints(const Eigen::Vector2d& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c(0) = x(0) * x(0) + x(1) - 1.0;
    }
    void jacobian(const Eigen::Vector2d& x, Eigen::MatrixXd& J) const
    {
        J.resize(1, 2);
        J(0, 0) = 2.0 * x(0);
        J(0, 1) = 1.0;
    }
};

// Rank-deficient Jacobian: two rows that are proportional (the second
// is twice the first), so J has rank 1 in a 2x2 system. J^T J is
// singular; LM regularization is what keeps the step bounded and
// drives x[0] toward the feasibility manifold.
struct rank_deficient_problem
{
    void constraints(const Eigen::Vector2d& x, Eigen::VectorXd& c) const
    {
        c.resize(2);
        c(0) = x(0) - 1.0;
        c(1) = 2.0 * (x(0) - 1.0);
    }
    void jacobian(const Eigen::Vector2d& /*x*/, Eigen::MatrixXd& J) const
    {
        J.resize(2, 2);
        J(0, 0) = 1.0;
        J(0, 1) = 0.0;
        J(1, 0) = 2.0;
        J(1, 1) = 0.0;
    }
};

// Workspace bundle. Mirrors the policy-side state buffers: one
// allocation block per scenario, no allocation inside the helper.
struct workspace_2d
{
    Eigen::VectorXd              c_buf;
    Eigen::VectorXd              c_trial_buf;
    Eigen::MatrixXd              J_buf;
    Eigen::MatrixXd              JtJ_buf;
    Eigen::Vector2d              rhs_buf;
    Eigen::Vector2d              dx_buf;
    Eigen::Vector2d              x_trial_buf;
    Eigen::LDLT<Eigen::MatrixXd> ldlt_buf;

    workspace_2d(int m) : ldlt_buf(2)
    {
        c_buf.resize(m);
        c_trial_buf.resize(m);
        J_buf.resize(m, 2);
        JtJ_buf.resize(2, 2);
    }
};

}  // namespace

TEST_CASE("feasibility_restoration linear constraint converges",
          "[restoration][lm][linear]")
{
    linear_problem prob;
    auto cfn = [&](const Eigen::Vector2d& xv, Eigen::VectorXd& cv) {
        prob.constraints(xv, cv);
    };
    auto jfn = [&](const Eigen::Vector2d& xv, Eigen::MatrixXd& Jv) {
        prob.jacobian(xv, Jv);
    };

    Eigen::Vector2d x(0.0, 0.0);
    Eigen::Vector2d lower(-10.0, -10.0);
    Eigen::Vector2d upper(10.0, 10.0);

    workspace_2d wk(1);

    lm_restoration_result result = feasibility_restoration<double, 2>(
        x, cfn, jfn, lower, upper,
        /*max_iter=*/std::size_t{5},
        /*lambda_init=*/1e-3,
        /*feasibility_tol=*/1e-10,
        wk.c_buf, wk.c_trial_buf,
        wk.J_buf, wk.JtJ_buf,
        wk.rhs_buf, wk.dx_buf, wk.x_trial_buf,
        wk.ldlt_buf);

    REQUIRE(result.status == restoration_status::converged);
    // LM with lambda=1e-3 damps the Gauss-Newton step on the linear
    // residual; the damped sequence converges in a small handful of
    // iters as lambda shrinks on consecutive accepts.
    REQUIRE(result.iterations_used <= std::size_t{5});
    REQUIRE(result.final_c_norm_l2 < 1e-10);
    REQUIRE(std::isfinite(x(0)));
    REQUIRE(std::isfinite(x(1)));
    // Feasibility manifold is x[0] + x[1] == 1; verify directly.
    REQUIRE(std::abs(x(0) + x(1) - 1.0) < 1e-10);
}

TEST_CASE("feasibility_restoration quadratic constraint converges",
          "[restoration][lm][quadratic]")
{
    quadratic_problem prob;
    auto cfn = [&](const Eigen::Vector2d& xv, Eigen::VectorXd& cv) {
        prob.constraints(xv, cv);
    };
    auto jfn = [&](const Eigen::Vector2d& xv, Eigen::MatrixXd& Jv) {
        prob.jacobian(xv, Jv);
    };

    Eigen::Vector2d x(2.0, 0.0);  // c(x_0) = 4 + 0 - 1 = 3 (infeasible)
    Eigen::Vector2d lower(-10.0, -10.0);
    Eigen::Vector2d upper(10.0, 10.0);

    workspace_2d wk(1);

    lm_restoration_result result = feasibility_restoration<double, 2>(
        x, cfn, jfn, lower, upper,
        /*max_iter=*/std::size_t{50},
        /*lambda_init=*/1e-3,
        /*feasibility_tol=*/1e-10,
        wk.c_buf, wk.c_trial_buf,
        wk.J_buf, wk.JtJ_buf,
        wk.rhs_buf, wk.dx_buf, wk.x_trial_buf,
        wk.ldlt_buf);

    REQUIRE(result.status == restoration_status::converged);
    REQUIRE(result.iterations_used <= std::size_t{20});
    REQUIRE(result.final_c_norm_l2 < 1e-10);
    REQUIRE(std::isfinite(x(0)));
    REQUIRE(std::isfinite(x(1)));
    // Verify the manifold: x[0]^2 + x[1] - 1 == 0.
    REQUIRE(std::abs(x(0) * x(0) + x(1) - 1.0) < 1e-10);
}

TEST_CASE("feasibility_restoration rank-deficient Jacobian stabilizes",
          "[restoration][lm][rank-deficient]")
{
    rank_deficient_problem prob;
    auto cfn = [&](const Eigen::Vector2d& xv, Eigen::VectorXd& cv) {
        prob.constraints(xv, cv);
    };
    auto jfn = [&](const Eigen::Vector2d& xv, Eigen::MatrixXd& Jv) {
        prob.jacobian(xv, Jv);
    };

    Eigen::Vector2d x(5.0, 0.0);
    Eigen::Vector2d lower(-10.0, -10.0);
    Eigen::Vector2d upper(10.0, 10.0);

    // Initial residual norm: c = [4, 8], ||c||_2 = sqrt(80).
    Eigen::VectorXd c_initial(2);
    prob.constraints(x, c_initial);
    const double c_norm_initial = c_initial.norm();

    workspace_2d wk(2);

    lm_restoration_result result = feasibility_restoration<double, 2>(
        x, cfn, jfn, lower, upper,
        /*max_iter=*/std::size_t{100},
        /*lambda_init=*/1e-3,
        /*feasibility_tol=*/1e-10,
        wk.c_buf, wk.c_trial_buf,
        wk.J_buf, wk.JtJ_buf,
        wk.rhs_buf, wk.dx_buf, wk.x_trial_buf,
        wk.ldlt_buf);

    // Either we converged (LM regularization kept the step bounded and
    // drove x[0] to 1) or lambda grew unbounded; in either case x must
    // be finite and the residual must not increase.
    const bool ok_status =
        (result.status == restoration_status::converged
         || result.status == restoration_status::lambda_grew_unbounded
         || result.status == restoration_status::max_iter_reached);
    REQUIRE(ok_status);
    REQUIRE(std::isfinite(x(0)));
    REQUIRE(std::isfinite(x(1)));
    REQUIRE(std::isfinite(result.final_c_norm_l2));
    REQUIRE(result.final_c_norm_l2 < c_norm_initial);
    REQUIRE(result.final_lambda >= restoration_lambda_min);
    REQUIRE(result.final_lambda <= restoration_lambda_max);
}
