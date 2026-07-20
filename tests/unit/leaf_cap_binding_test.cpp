// Cap-exhaustion honesty gates for the four intrinsic inner leaves.
//
// A bounded-iteration guarantee is only meaningful if HITTING the cap
// surfaces an honest non-converged status rather than a silent accept. Each
// case here drives one leaf (line search, NNLS, Steihaug-CG, restoration)
// into its iteration cap and asserts the leaf reports the honest exhaustion
// enumerant and still returns a finite, in-range artifact. A bare
// constant-equals-its-literal assertion is deliberately avoided: a check that
// cannot fail launders an unverified claim.

#include "argmin/detail/nnls.h"
#include "argmin/detail/restoration.h"
#include "argmin/detail/steihaug_cg.h"

#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/filter_trsqp_policy.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <cstddef>

using namespace argmin;

namespace
{

// Differentiable equality-constrained problem whose objective is finite only
// at the start iterate and NaN at every probed trial point. The QP still
// produces a well-defined descent direction from the analytic gradient and
// Jacobian, but the line search can never accept a step, so the BFGS-reset
// cap is forced to bind -- the deterministic driver for line-search
// exhaustion (the natural analog of the restoration NaN-guard fixtures).
struct nan_trial_problem
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    Eigen::VectorXd start;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        if(x == start)
            return 0.0;
        return std::numeric_limits<double>::quiet_NaN();
    }

    void gradient(const Eigen::VectorXd& /*x*/, Eigen::VectorXd& g) const
    {
        g.resize(2);
        g << 1.0, 1.0;
    }

    int num_equality() const { return 1; }
    int num_inequality() const { return 0; }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = x[0] + x[1] - 1.0;
    }

    void constraint_jacobian(const Eigen::VectorXd& /*x*/, Eigen::MatrixXd& J) const
    {
        J.resize(1, 2);
        J << 1.0, 1.0;
    }
};

struct dense_hessian_op
{
    Eigen::Matrix2d B;
    void operator()(const Eigen::Ref<const Eigen::Vector2d>& v,
                    Eigen::Ref<Eigen::Vector2d> out) const
    {
        out.noalias() = B * v;
    }
};

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

template <typename ConstraintFn, typename JacobianFn>
argmin::detail::lm_restoration_result run_restoration(
    Eigen::Vector2d& x, ConstraintFn&& cfn, JacobianFn&& jfn,
    workspace_2d& wk, std::size_t max_iter, double lambda_init)
{
    Eigen::Vector2d lower(-10.0, -10.0);
    Eigen::Vector2d upper(10.0, 10.0);
    return argmin::detail::feasibility_restoration<double, 2>(
        x, std::forward<ConstraintFn>(cfn), std::forward<JacobianFn>(jfn),
        lower, upper, max_iter, lambda_init, /*feasibility_tol=*/1e-10,
        wk.c_buf, wk.c_trial_buf, wk.J_buf, wk.JtJ_buf,
        wk.rhs_buf, wk.dx_buf, wk.x_trial_buf, wk.ldlt_buf);
}

}

TEST_CASE("line search exhaustion surfaces a null step rather than a silent accept",
          "[leaf-cap][line-search]")
{
    Eigen::VectorXd x0{{2.0, 2.0}};
    nan_trial_problem problem;
    problem.start = x0;

    nw_sqp_policy<> policy;
    policy.options.bfgs_reset_max = 1;

    solver_options<> opts;
    opts.max_iterations = 10;

    step_budget_solver solver{policy, problem, x0, opts};
    auto sr = solver.step();

    const bool reported_converged =
        sr.policy_status && *sr.policy_status == solver_status::converged;

    CHECK(sr.is_null_step);
    CHECK(sr.diagnostics.bfgs_reset_count >= std::size_t{1});
    CHECK_FALSE(reported_converged);
    CHECK(std::isfinite(sr.objective_value));
    CHECK(solver.state().x == x0);
}

TEST_CASE("nnls exhaustion returns mode 3 rather than a silent accept",
          "[leaf-cap][nnls]")
{
    using argmin::detail::nnls;
    using argmin::detail::nnls_max_iter_factor;
    using argmin::detail::nnls_workspace;

    // Pins the default cap multiplier. An exact behavioral pin of 3*n is
    // infeasible -- Lawson-Hanson terminates in <= n steps, so 3n is never
    // reached on a natural input -- so this cross-TU pin goes red only if the
    // source multiplier drifts.
    static_assert(nnls_max_iter_factor == 3);

    // A 3x3 identity system whose non-negative optimum has three positive
    // components needs three column-adds; one outer iteration cannot finish
    // it, so the forced cap exhausts and returns mode 3.
    constexpr int n = 3;
    Eigen::Matrix<double, n, n> A = Eigen::Matrix<double, n, n>::Identity();
    Eigen::Vector<double, n> b{{1.0, 2.0, 3.0}};
    Eigen::Vector<double, n> x = Eigen::Vector<double, n>::Zero();
    Eigen::Vector<double, n> w = Eigen::Vector<double, n>::Zero();

    nnls_workspace<double> ws;
    auto capped = nnls<double, n, n>(A, b, x, w, ws, n, n, /*max_iter_override=*/1);
    CHECK(capped.mode == 3);

    // With the override unset the default 3*n cap far exceeds the natural
    // termination, so an ordinary fixture converges (mode 1) -- confirming
    // the exhaustion above is the forced cap, not a broken solve.
    constexpr int n2 = 4;
    Eigen::Matrix<double, n2, n2> A2 = Eigen::Matrix<double, n2, n2>::Identity();
    Eigen::Vector<double, n2> b2{{1.0, 2.0, 3.0, 4.0}};
    Eigen::Vector<double, n2> x2 = Eigen::Vector<double, n2>::Zero();
    Eigen::Vector<double, n2> w2 = Eigen::Vector<double, n2>::Zero();

    nnls_workspace<double> ws2;
    auto natural = nnls<double, n2, n2>(A2, b2, x2, w2, ws2, n2, n2);
    CHECK(natural.mode == 1);
}

TEST_CASE("steihaug-cg exhaustion returns max_iterations rather than a silent accept",
          "[leaf-cap][steihaug-cg]")
{
    using argmin::detail::cg_exit_status;
    using argmin::detail::steihaug_cg;

    constexpr double inf = std::numeric_limits<double>::infinity();

    // Ill-conditioned SPD B = diag(1, 10) with an off-axis gradient: CG needs
    // two steps on this 2-D system, so max_iter = 1 exhausts the budget
    // mid-solve with the iterate still inside the trust region.
    Eigen::Vector2d g;
    g << 1.0, 1.0;
    Eigen::Matrix2d B;
    B << 1.0, 0.0,
         0.0, 10.0;

    Eigen::Vector2d lower(-inf, -inf);
    Eigen::Vector2d upper(inf, inf);
    Eigen::Vector2d p_out, r_buf, d_buf, Bd_buf;
    dense_hessian_op op{B};

    auto status = steihaug_cg<double, 2>(
        g, op, /*delta=*/100.0, /*eps=*/1e-14,
        lower, upper, /*max_iter=*/1,
        p_out, r_buf, d_buf, Bd_buf);

    CHECK(status == cg_exit_status::max_iterations);
    CHECK(std::isfinite(p_out[0]));
    CHECK(std::isfinite(p_out[1]));
    CHECK(p_out.norm() <= 100.0 + 1e-12);
}

TEST_CASE("restoration exhaustion returns max_iter_reached rather than a silent accept",
          "[leaf-cap][restoration]")
{
    using argmin::detail::restoration_status;

    // A well-posed linear feasibility target (x0 = 5) with a large initial LM
    // damping so each accepted step shrinks ||c|| only slightly. Capped at one
    // iteration it makes progress but cannot reach the feasibility tolerance,
    // so it exits at the iteration cap -- not converged, not lambda-unbounded.
    auto cfn = [](const Eigen::Vector2d& xv, Eigen::VectorXd& cv) {
        cv.resize(1);
        cv(0) = xv(0) - 5.0;
    };
    auto jfn = [](const Eigen::Vector2d& /*xv*/, Eigen::MatrixXd& Jv) {
        Jv.resize(1, 2);
        Jv << 1.0, 0.0;
    };

    Eigen::Vector2d x(0.0, 0.0);
    workspace_2d wk(1);
    auto result = run_restoration(x, cfn, jfn, wk, /*max_iter=*/1, /*lambda_init=*/1e3);

    REQUIRE(result.status == restoration_status::max_iter_reached);
    CHECK(std::isfinite(x(0)));
    CHECK(std::isfinite(x(1)));

    // Trace the restoration cap that binds at runtime and reconcile it against
    // the step_result diagnostics comment: filter_trsqp installs 10 by
    // default, not the previously documented 0.
    const filter_trsqp_policy<>::options_type restoration_defaults{};
    CHECK(restoration_defaults.restoration_max_iter == std::size_t{10});
}
