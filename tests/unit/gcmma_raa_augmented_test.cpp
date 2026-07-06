// Tests for the strict Svanberg 2002 raa-augmented GCMMA variant
// (argmin::alternative::gcmma::raa_augmented_policy).

#include "argmin/detail/mma_raa_augmented_dual_problem.h"
#include "argmin/solver/alternative/gcmma/raa_augmented_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using namespace argmin;

TEST_CASE("gcmma raa-augmented converges on HS024", "[gcmma_raa]")
{
    hs024 problem;
    Eigen::VectorXd x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-5);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);
    basic_solver solver{
        alternative::gcmma::raa_augmented_policy<
            hs024<>::problem_dimension>{},
        problem, x0, opts};
    const auto result = solver.solve(opts);
    CHECK(result.objective_value == Approx(-1.0).margin(0.05));
}

TEST_CASE("gcmma raa-augmented converges on HS035", "[gcmma_raa]")
{
    hs035 problem;
    Eigen::VectorXd x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-5);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);
    basic_solver solver{
        alternative::gcmma::raa_augmented_policy<
            hs035<>::problem_dimension>{},
        problem, x0, opts};
    const auto result = solver.solve(opts);
    CHECK(result.objective_value == Approx(1.0 / 9.0).margin(0.05));
}

TEST_CASE("gcmma raa-augmented converges on HS076", "[gcmma_raa]")
{
    hs076 problem;
    Eigen::VectorXd x0 = problem.initial_point();
    solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-5);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);
    basic_solver solver{
        alternative::gcmma::raa_augmented_policy<
            hs076<>::problem_dimension>{},
        problem, x0, opts};
    const auto result = solver.solve(opts);
    CHECK(result.objective_value == Approx(-4.6818).margin(0.05));
}

// The raa-augmented dual solves its per-component primal by Newton on the
// augmented first-order condition
//   F(x) = P/(U-x)^2 - Q/(x-L)^2 + R * d'_j(x) = 0,
// using the exact analytic slope F'(x) = 2P/(U-x)^3 + 2Q/(x-L)^3 +
// R * d''_j(x) (previously a finite difference). A correct slope drives
// the FOC residual to ~0 at an interior root well inside the Newton cap.
// This pins the analytic slope by its effect: with a strong augmentation
// (large raa_obj) the root is interior and the residual must vanish.
TEST_CASE("gcmma raa-augmented primal satisfies the augmented FOC",
          "[gcmma_raa]")
{
    Eigen::VectorXd L{{0.0}}, U{{2.0}}, xk{{0.6}};
    Eigen::VectorXd alpha{{0.01}}, beta{{1.99}};
    // Asymmetric p/q so the un-augmented minimizer is off-center; a strong
    // raa penalty then pulls the interior root back toward x_k.
    Eigen::VectorXd p0{{4.0}}, q0{{0.25}};
    Eigen::MatrixXd pc(0, 1), qc(0, 1);
    Eigen::VectorXd rc(0), raac(0);

    detail::mma_raa_augmented_dual_problem<double, Eigen::Dynamic,
                                           Eigen::Dynamic> dual;
    dual.L_out = &L;
    dual.U_out = &U;
    dual.x_k_out = &xk;
    dual.alpha_out = &alpha;
    dual.beta_out = &beta;
    dual.p_obj_out = &p0;
    dual.q_obj_out = &q0;
    dual.p_con_out = &pc;
    dual.q_con_out = &qc;
    dual.r_obj = 0.0;
    dual.r_con_out = &rc;
    dual.raa_obj = 3.0;   // strong augmentation -> interior root
    dual.raa_con_out = &raac;
    dual.n_primal = 1;
    dual.m_dual = 0;

    Eigen::VectorXd y(0);
    (void)dual.value(y);
    const double x = dual.x_primal[0];

    // Premise: the root is interior (not on a clamp), so the FOC -- not a
    // clamp -- determines x and the Newton slope is what found it.
    REQUIRE(x > alpha[0] + 1e-6);
    REQUIRE(x < beta[0] - 1e-6);

    // Augmented FOC residual at the solved primal must vanish.
    const double a = x - xk[0];
    const double u = U[0] - x;
    const double l = x - L[0];
    const double d_deriv = (U[0] - L[0])
        * (2.0 * a / (u * l)
           - a * a * (U[0] + L[0] - 2.0 * x) / (u * u * l * l));
    const double foc = p0[0] / (u * u) - q0[0] / (l * l)
                     + dual.raa_obj * d_deriv;
    CHECK(foc == Approx(0.0).margin(1e-9));
}
