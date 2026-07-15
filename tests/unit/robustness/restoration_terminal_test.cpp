// Robustness: terminal-status guards of the Levenberg-Marquardt
// feasibility-restoration helper.
//
// restoration_test.cpp exercises the happy paths (linear / quadratic /
// rank-deficient descent to feasibility). These cells drive the helper
// into each NON-converging terminal branch and assert the correct
// classification, that x stays finite, and that no NaN/Inf escapes:
//
//   converged-on-entry     -- an already-feasible iterate returns
//                             converged with zero inner iterations.
//   degenerate_no_progress -- J^T c == 0 with c != 0: the regularized
//                             solve can only return dx = 0, so the helper
//                             short-circuits instead of spinning.
//   lambda_grew_unbounded  -- every trial is rejected (constant residual)
//                             until lambda hits the cap; and the non-finite
//                             step guard (NaN / Inf Jacobian) routes through
//                             the same terminal without letting a NaN
//                             iterate escape.
//
// Reference: Nocedal and Wright 2e Section 10.3 (Levenberg-Marquardt
//            damping bracket and reject schedule).

#include "argmin/detail/restoration.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <limits>

using argmin::detail::feasibility_restoration;
using argmin::detail::lm_restoration_result;
using argmin::detail::restoration_status;

namespace
{

// One allocation block per scenario; the helper allocates nothing.
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

    explicit workspace_2d(int m) : ldlt_buf(2)
    {
        c_buf.resize(m);
        c_trial_buf.resize(m);
        J_buf.resize(m, 2);
        JtJ_buf.resize(2, 2);
    }
};

// Runs the helper with the given callbacks from x, returning the result.
template <typename ConstraintFn, typename JacobianFn>
lm_restoration_result run_restoration(
    Eigen::Vector2d& x, ConstraintFn&& cfn, JacobianFn&& jfn,
    workspace_2d& wk, std::size_t max_iter, double lambda_init)
{
    Eigen::Vector2d lower(-10.0, -10.0);
    Eigen::Vector2d upper(10.0, 10.0);
    return feasibility_restoration<double, 2>(
        x, std::forward<ConstraintFn>(cfn), std::forward<JacobianFn>(jfn),
        lower, upper, max_iter, lambda_init, /*feasibility_tol=*/1e-10,
        wk.c_buf, wk.c_trial_buf, wk.J_buf, wk.JtJ_buf,
        wk.rhs_buf, wk.dx_buf, wk.x_trial_buf, wk.ldlt_buf);
}

}  // namespace

TEST_CASE("feasibility_restoration returns converged on an already-feasible entry",
          "[robustness][restoration][entry]")
{
    // c(x) = x[0] identically small: the entry feasibility check trips
    // before any LM iteration runs.
    auto cfn = [](const Eigen::Vector2d& xv, Eigen::VectorXd& cv) {
        cv.resize(1);
        cv(0) = xv(0);
    };
    auto jfn = [](const Eigen::Vector2d& /*xv*/, Eigen::MatrixXd& Jv) {
        Jv.resize(1, 2);
        Jv << 1.0, 0.0;
    };

    Eigen::Vector2d x(0.0, 3.0);  // c = 0 -> feasible on entry
    workspace_2d wk(1);

    auto result = run_restoration(x, cfn, jfn, wk, /*max_iter=*/5, 1e-3);

    REQUIRE(result.status == restoration_status::converged);
    CHECK(result.iterations_used == std::size_t{0});
    CHECK(result.final_c_norm_l2 < 1e-10);
    CHECK(std::isfinite(x(0)));
    CHECK(std::isfinite(x(1)));
}

TEST_CASE("feasibility_restoration flags degenerate_no_progress when J^T c vanishes",
          "[robustness][restoration][degenerate]")
{
    // Nonzero constant residual with a ZERO Jacobian: J^T c == 0, so the
    // LM step is identically zero and no reduction in ||c|| is possible.
    auto cfn = [](const Eigen::Vector2d& /*xv*/, Eigen::VectorXd& cv) {
        cv.resize(1);
        cv(0) = 1.0;  // infeasible, constant
    };
    auto jfn = [](const Eigen::Vector2d& /*xv*/, Eigen::MatrixXd& Jv) {
        Jv.resize(1, 2);
        Jv.setZero();
    };

    Eigen::Vector2d x(0.5, -0.5);
    workspace_2d wk(1);

    auto result = run_restoration(x, cfn, jfn, wk, /*max_iter=*/10, 1e-3);

    REQUIRE(result.status == restoration_status::degenerate_no_progress);
    CHECK(result.iterations_used == std::size_t{0});
    CHECK(result.final_c_norm_l2 == 1.0);
    CHECK(std::isfinite(x(0)));
    CHECK(std::isfinite(x(1)));
}

TEST_CASE("feasibility_restoration grows lambda to the cap when every trial is rejected",
          "[robustness][restoration][lambda-cap]")
{
    // A constant residual with a NONZERO Jacobian: rhs != 0 (so the
    // degeneracy short-circuit is skipped), but moving x never changes
    // ||c||, so every trial is rejected and lambda is grown until it hits
    // the cap -- the lambda_grew_unbounded terminal.
    auto cfn = [](const Eigen::Vector2d& /*xv*/, Eigen::VectorXd& cv) {
        cv.resize(1);
        cv(0) = 1.0;  // constant despite a nonzero Jacobian
    };
    auto jfn = [](const Eigen::Vector2d& /*xv*/, Eigen::MatrixXd& Jv) {
        Jv.resize(1, 2);
        Jv << 1.0, 0.0;
    };

    Eigen::Vector2d x(0.0, 0.0);
    workspace_2d wk(1);

    auto result = run_restoration(x, cfn, jfn, wk, /*max_iter=*/1, 1e-3);

    REQUIRE(result.status == restoration_status::lambda_grew_unbounded);
    CHECK(std::isfinite(x(0)));
    CHECK(std::isfinite(x(1)));
    CHECK(std::isfinite(result.final_c_norm_l2));
}

TEST_CASE("feasibility_restoration guards a non-finite LM step -- NaN Jacobian",
          "[robustness][restoration][nan]")
{
    // A NaN Jacobian pushes NaN into JtJ and hence the solved step; the
    // helper's non-finite guard must reject it and reach a terminal
    // status without letting the NaN corrupt x.
    auto cfn = [](const Eigen::Vector2d& /*xv*/, Eigen::VectorXd& cv) {
        cv.resize(1);
        cv(0) = 1.0;
    };
    auto jfn = [](const Eigen::Vector2d& /*xv*/, Eigen::MatrixXd& Jv) {
        Jv.resize(1, 2);
        Jv << std::numeric_limits<double>::quiet_NaN(), 0.0;
    };

    Eigen::Vector2d x(0.0, 0.0);
    workspace_2d wk(1);

    auto result = run_restoration(x, cfn, jfn, wk, /*max_iter=*/1, 1e-3);

    // The step is never accepted (it is non-finite), so the helper grows
    // lambda to the cap and reports lambda_grew_unbounded. The observable
    // that matters: x did not absorb the NaN step.
    CHECK(result.status == restoration_status::lambda_grew_unbounded);
    CHECK(std::isfinite(x(0)));
    CHECK(std::isfinite(x(1)));
}

TEST_CASE("feasibility_restoration guards a non-finite LM step -- Inf Jacobian",
          "[robustness][restoration][inf]")
{
    // An Inf Jacobian drives the LDLT factorization of the regularized
    // normal equations to a non-success state (or a non-finite solve);
    // either way the terminal must be lambda_grew_unbounded with a finite
    // returned iterate.
    auto cfn = [](const Eigen::Vector2d& /*xv*/, Eigen::VectorXd& cv) {
        cv.resize(1);
        cv(0) = 1.0;
    };
    auto jfn = [](const Eigen::Vector2d& /*xv*/, Eigen::MatrixXd& Jv) {
        Jv.resize(1, 2);
        Jv << std::numeric_limits<double>::infinity(), 0.0;
    };

    Eigen::Vector2d x(0.0, 0.0);
    workspace_2d wk(1);

    auto result = run_restoration(x, cfn, jfn, wk, /*max_iter=*/1, 1e-3);

    CHECK(result.status == restoration_status::lambda_grew_unbounded);
    CHECK(std::isfinite(x(0)));
    CHECK(std::isfinite(x(1)));
}
