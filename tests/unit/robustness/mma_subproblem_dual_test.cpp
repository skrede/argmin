// Robustness: the CCSA subproblem dual Newton solve on an ACTIVE
// constraint.
//
// zero_copy_test.cpp exercises mma_subproblem_solver only with a satisfied
// constraint (fc < 0), so the projected-gradient convergence check trips on
// the first iteration and the dual Newton loop -- the Hessian assembly, the
// LDLT solve, the y >= 0 backtrack, and the primal clamping guards -- never
// runs. These cells drive the solver with a VIOLATED approximation
// constraint so the dual multiplier must be raised, and assert the
// observable KKT-consistent outcome:
//
//   active-constraint solve  -- a violated fc > 0 forces >= 1 dual Newton
//                             iteration; the returned multiplier is >= 0 and
//                             the approximation constraint is driven to
//                             (near) satisfaction.
//   bound-clamped primal     -- tight box bounds force the primal x(y) onto
//                             a bound, exercising the clamp guards inside
//                             eval_primal and the clamped-variable skip in
//                             the Hessian assembly.
//   unconstrained passthrough-- m = 0 returns the analytic primal directly.
//
// Reference: Svanberg 2002, SIAM J. Optim. 12(2):555-573 (CCSA);
//            NLopt ccsa_quadratic.c dual_func() (Steven G. Johnson).

#include "argmin/detail/mma_subproblem.h"
#include "argmin/options/mma_subproblem_options.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using argmin::detail::mma_subproblem_solver;

TEST_CASE("mma_subproblem raises the dual multiplier on an active constraint",
          "[robustness][mma-subproblem][dual]")
{
    // n = 2, m = 1. The objective gradient pulls dx in the +ve direction
    // (grad_f = -1 => unconstrained dx = +1), which drives the linearized
    // constraint MORE positive (dfc = +1). So at the y = 0 primal the
    // constraint is still violated and the dual solve must raise y (pulling
    // dx back negative) until the CCSA approximation constraint is (near)
    // satisfied -- exercising the projected-Newton dual loop.
    constexpr int N = 2;
    const int m = 1;
    mma_subproblem_solver<double, N> sub{N, m};

    Eigen::Vector<double, N> x{{0.0, 0.0}};
    const double f = 1.0;
    Eigen::Vector<double, N> grad_f{{-1.0, -1.0}};

    Eigen::VectorXd fc(m);
    fc << 0.5;  // constraint approximation already violated at dx = 0
    Eigen::Matrix<double, Eigen::Dynamic, N> dfc(m, N);
    dfc << 1.0, 1.0;  // the objective's +ve step makes this constraint worse

    Eigen::Vector<double, N> sigma{{1.0, 1.0}};
    const double rho = 1.0;
    Eigen::VectorXd rhoc = Eigen::VectorXd::Ones(m);
    Eigen::Vector<double, N> lb{{-10.0, -10.0}};
    Eigen::Vector<double, N> ub{{10.0, 10.0}};

    const auto x_opt = sub.solve(x, f, grad_f, fc, dfc, sigma, rho, rhoc, lb, ub);

    // A raised, sign-correct multiplier and a satisfied approximation
    // constraint: the dual Newton loop actually ran and converged.
    CHECK(sub.multipliers()[0] >= 0.0);
    CHECK(sub.multipliers()[0] > 0.0);
    CHECK(sub.gcval()[0] <= 1e-6);
    CHECK(std::isfinite(x_opt[0]));
    CHECK(std::isfinite(x_opt[1]));
}

TEST_CASE("mma_subproblem clamps the primal onto tight bounds",
          "[robustness][mma-subproblem][clamp]")
{
    // Same violated constraint, but the box is tight enough that the primal
    // step saturates a bound -- exercising the clamp guards and the
    // clamped-variable skip in the Hessian assembly.
    constexpr int N = 2;
    const int m = 1;
    mma_subproblem_solver<double, N> sub{N, m};

    Eigen::Vector<double, N> x{{0.0, 0.0}};
    const double f = 1.0;
    Eigen::Vector<double, N> grad_f{{1.0, 1.0}};

    Eigen::VectorXd fc(m);
    fc << 2.0;
    Eigen::Matrix<double, Eigen::Dynamic, N> dfc(m, N);
    dfc << 1.0, 1.0;

    Eigen::Vector<double, N> sigma{{5.0, 5.0}};  // large -> step wants to overshoot
    const double rho = 1.0;
    Eigen::VectorXd rhoc = Eigen::VectorXd::Ones(m);
    Eigen::Vector<double, N> lb{{-0.05, -0.05}};  // tight lower bound
    Eigen::Vector<double, N> ub{{10.0, 10.0}};

    const auto x_opt = sub.solve(x, f, grad_f, fc, dfc, sigma, rho, rhoc, lb, ub);

    // The primal is clamped to the tight lower bound and stays finite; the
    // multiplier is non-negative.
    CHECK(x_opt[0] >= lb[0] - 1e-12);
    CHECK(x_opt[1] >= lb[1] - 1e-12);
    CHECK(x_opt[0] == Approx(lb[0]).margin(1e-9));
    CHECK(sub.multipliers()[0] >= 0.0);
    CHECK(std::isfinite(x_opt[0]));
}

TEST_CASE("mma_subproblem returns the analytic primal for the unconstrained case",
          "[robustness][mma-subproblem][unconstrained]")
{
    // m = 0: no dual, the solver returns the analytic primal x(y=0).
    constexpr int N = 2;
    const int m = 0;
    mma_subproblem_solver<double, N> sub{N, m};

    Eigen::Vector<double, N> x{{0.0, 0.0}};
    const double f = 1.0;
    Eigen::Vector<double, N> grad_f{{1.0, -1.0}};

    Eigen::VectorXd fc(0);
    Eigen::Matrix<double, Eigen::Dynamic, N> dfc(0, N);
    Eigen::Vector<double, N> sigma{{1.0, 1.0}};
    const double rho = 1.0;
    Eigen::VectorXd rhoc(0);
    Eigen::Vector<double, N> lb{{-10.0, -10.0}};
    Eigen::Vector<double, N> ub{{10.0, 10.0}};

    const auto x_opt = sub.solve(x, f, grad_f, fc, dfc, sigma, rho, rhoc, lb, ub);

    // With rho = 1, sigma = 1: dx_j = -grad_f_j, so x moves opposite the
    // gradient and stays finite.
    CHECK(x_opt[0] == Approx(-1.0).margin(1e-9));
    CHECK(x_opt[1] == Approx(1.0).margin(1e-9));
}
