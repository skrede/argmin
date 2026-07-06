// MMA / GCMMA approximation-coefficient and conservativity oracle pins.
//
// Pin 1 -- Svanberg p/q/r coefficients. Svanberg (2007), "MMA and
// GCMMA -- two methods for nonlinear optimization", KTH notes,
// eqs. (2.3)-(2.5):
//   p_ij = (U_j - x_j)^2 (1.001 (dg_i/dx_j)^+ + 0.001 (dg_i/dx_j)^- + raai/range_j)
//   q_ij = (x_j - L_j)^2 (0.001 (dg_i/dx_j)^+ + 1.001 (dg_i/dx_j)^- + raai/range_j)
//   r_i  = g_i(x_k) - sum_j [ p_ij/(U_j - x_j) + q_ij/(x_j - L_j) ]
// with raai = 1e-5 and range_j = xmax_j - xmin_j. Golden values for a
// fixed worked case are hand-derived in
// tests/oracles/mma_svanberg_coefficients.csv (derivation in the CSV
// header). Red-state proof (pre-fix): the implementation uses a flat
// 1e-10 additive epsilon with no 1.001/0.001 gradient mixing and no
// raai/range term; observed p_obj = (2.0000000001, 1e-10) vs the
// Svanberg oracle (2.002005, 0.01201), q_obj = (1e-10, 12.0000000001)
// vs (0.002005, 12.01201), r_obj = -13.0000000003 vs -13.01602, and
// the constraint row deviates identically.
//
// Pin 2 -- dual-gradient finite-difference check (reference leg,
// green): the reciprocal dual exposes grad_i W(y) = -g_tilde_i(x(y))
// by the envelope theorem (Danskin: exact also when a coordinate of
// x(y) sits on its alpha/beta clamp over a neighborhood of y). The
// analytic gradient of the production dual problem must match a
// central finite difference of its value at an interior y.
//
// Pin 3 -- quadratic-penalty mass per unit rho. The rho-augmented
// approximation (NLopt mma.c dual_func form) is
//   g_tilde(x) = r + sum_j [ p_j/(U_j - x_j) + q_j/(x_j - L_j) ]
//              + rho * sum_j 0.5 w_j (x_j - x_kj)^2
// so its rho-linear coefficient -- the penalty mass per unit rho, and
// therefore the denominator of the minimal conservative rho increment
//   delta_min = (f(x_trial) - g_tilde(x_trial)) / mass,
//   mass = sum_j 0.5 w_j (x_trial_j - x_kj)^2
// -- must be what the dual problem reports as wval (NLopt mma.c:
// gval += rho * dx2sig; wval += dx2sig with dx2sig = 0.5 dx^2/sigma^2).
// Worked case (n = 1, m = 0, rho = 0 so the primal is the closed-form
// reciprocal minimizer): L = 0, U = 2, x_k = 1, alpha = 0.1,
// beta = 1.9, w = 2, p = 1, q = 4, r = 0.5:
//   x* = (sqrt(1)*0 + sqrt(4)*2)/(sqrt(1) + sqrt(4)) = 4/3, dx = 1/3
//   g_tilde(x*) = 0.5 + 1/(2/3) + 4/(4/3) = 5.0
//   mass = 0.5 * 2 * (1/3)^2 = 1/9
// Red-state proof (pre-fix): the implementation accumulates
// wval = sum_j w_j dx^2 = 2/9, i.e. exactly 2x the penalty mass.
//
// Pin 4 -- minimal conservative rho increment (closed form). After a
// non-conservative trial with shortfall S = f(x_trial) - g_tilde(x_trial)
// > 0, the grown rho must satisfy g_tilde_{rho'}(x_trial) >= f(x_trial)
// at the same trial point, i.e. rho' >= rho + S/mass, whenever the 10x
// growth cap is not binding (NLopt mma.c: rho' = min(10 rho,
// 1.1 (rho + S/wval)) with wval = mass; 1.1 (rho + S/mass) >= rho +
// S/mass always). Worked case (n = 1, m = 0): f(x) = 50 (x - 1.5)^2 on
// [0, 2], x_k = 1, f(x_k) = 12.5, f'(x_k) = -50; approximation_epsilon
// = 0 so with the pre-fix coefficients p = 0, q = 50 (x_k - L)^2 = 50,
// r = 12.5 - 50 = -37.5; asymptotes L = 0, U = 2, alpha = 0.1,
// beta = 1.9, w = (U - L)/((U - x_k)(x_k - L)) = 2; rho_init = 3,
// max_inner_iterations = 1, rho_decay = 1 (so the post-step state
// field holds exactly the once-grown rho). The augmented first-order
// condition -q/x^2 + rho w (x - 1) = 0 has no root in (0, 2)
// (-50/x^2 + 6(x-1) < 0 everywhere there), so the per-component
// Newton walks up and the returned trial clamps at beta:
//   x_trial = 1.9 exactly, dx = 0.9, f_trial = 50 (0.4)^2 = 8
//   g_tilde(1.9) = -37.5 + 50/1.9 + 3 * 0.5 * 2 * 0.81
//                = -8.754210526315789
//   S = 16.754210526315789, mass = 0.81
//   delta_min = S/mass = 20.684210526315789
//   required rho' >= 3 + 20.684210526315789 = 23.684210526315789
//   cap premise: 1.1 (3 + delta_min) = 26.0526... <= 10 rho = 30
// Red-state proof (pre-fix): the growth divides S by wval = 2 mass,
// giving rho' = 1.1 (3 + 10.342105...) = 14.676315789473684 <
// 23.684..., i.e. the increment is half the minimal conservative
// increment and the grown approximation still under-estimates f at
// the very trial that triggered the growth.

#include "argmin/detail/mma_augmented_dual_problem.h"
#include "argmin/detail/mma_reciprocal_dual_problem.h"
#include "argmin/solver/options.h"
#include "argmin/solver/mma_policy.h"
#include "argmin/solver/alternative/gcmma/rho_wval_policy.h"
#include "argmin/types.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <map>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

using Catch::Approx;
using namespace argmin;

namespace
{

// Worked case for the coefficient pin (matches the oracle CSV header):
// f(x) = x1^2 - 3 x2, grad f(1,2) = (2, -3);
// c(x) = 6 - x1 - 2 x2 >= 0 (argmin convention), so in MMA convention
// g_1 = -c with g_1(1,2) = -1 and grad g_1 = (1, 2);
// bounds [0,2] x [0,4], x0 = (1, 2).
struct coeff_problem
{
    static constexpr int problem_dimension = dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        return x[0] * x[0] - 3.0 * x[1];
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g.resize(2);
        g[0] = 2.0 * x[0];
        g[1] = -3.0;
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = 6.0 - x[0] - 2.0 * x[1];
    }

    void constraint_jacobian(const Eigen::VectorXd&, Eigen::MatrixXd& J) const
    {
        J.resize(1, 2);
        J(0, 0) = -1.0;
        J(0, 1) = -2.0;
    }

    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd{{0.0, 0.0}}; }
    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd{{2.0, 4.0}}; }
};

// Worked case for the rho-increment pin: f(x) = 50 (x - 1.5)^2 on
// [0, 2], no inequality constraints (m = 0; the conservativity loop
// then tests the objective approximation only).
struct rho_problem
{
    static constexpr int problem_dimension = dynamic_dimension;

    int dimension() const { return 1; }

    double value(const Eigen::VectorXd& x) const
    {
        const double d = x[0] - 1.5;
        return 50.0 * d * d;
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g.resize(1);
        g[0] = 100.0 * (x[0] - 1.5);
    }

    void constraints(const Eigen::VectorXd&, Eigen::VectorXd& c) const
    {
        c.resize(0);
    }

    void constraint_jacobian(const Eigen::VectorXd&, Eigen::MatrixXd& J) const
    {
        J.resize(0, 1);
    }

    int num_equality() const { return 0; }
    int num_inequality() const { return 0; }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd{{0.0}}; }
    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd{{2.0}}; }
};

// Unbounded 1-D problem (no box) for the move-bound cap: without a finite
// range the move limit must fall back to the scale proxy max(|x_k|, 1) so
// the trial window stays bounded. Not bound_constrained, so the policy
// runs with +/-inf bounds.
struct unbounded_1d
{
    static constexpr int problem_dimension = dynamic_dimension;

    int dimension() const { return 1; }
    int num_equality() const { return 0; }
    int num_inequality() const { return 0; }

    double value(const Eigen::VectorXd& x) const { return x[0] * x[0]; }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g.resize(1);
        g[0] = 2.0 * x[0];
    }

    void constraints(const Eigen::VectorXd&, Eigen::VectorXd& c) const
    {
        c.resize(0);
    }

    void constraint_jacobian(const Eigen::VectorXd&, Eigen::MatrixXd& J) const
    {
        J.resize(0, 1);
    }
};

// Minimal oracle CSV reader: '#' lines are comments; data lines are
// "name,v1,v2,...". Returns name -> values.
std::map<std::string, std::vector<double>> load_oracle(const std::string& path)
{
    std::map<std::string, std::vector<double>> rows;
    std::ifstream in(path);
    if(!in.is_open())
        return rows;
    std::string line;
    while(std::getline(in, line))
    {
        if(line.empty() || line.front() == '#')
            continue;
        std::istringstream ss(line);
        std::string name;
        std::getline(ss, name, ',');
        std::vector<double> vals;
        std::string tok;
        while(std::getline(ss, tok, ','))
            vals.push_back(std::stod(tok));
        rows.emplace(name, std::move(vals));
    }
    return rows;
}

}

// The Svanberg p/q/r formula -- the 1.001/0.001 gradient mixing plus the
// distance-scaled raai/range regularizer (in place of the former flat
// additive epsilon, which collapsed the reciprocal minimizer onto the
// window midpoint) -- reproduces the hand-derived oracle coefficients.
// raai is set to the oracle's 1e-5 so this pins the formula independent of
// the shipped (sweep-derived) default.
TEST_CASE("mma: p/q/r coefficients match the Svanberg worked values",
          "[mma][oracle-pin]")
{
    const auto oracle = load_oracle("oracles/mma_svanberg_coefficients.csv");
    REQUIRE(oracle.contains("p_obj"));

    coeff_problem problem;
    Eigen::VectorXd x0{{1.0, 2.0}};
    solver_options opts;
    mma_policy<> policy;
    mma_policy<>::options_type popts;
    popts.raai = 1e-5;
    auto s = policy.init(problem, x0, opts, popts);
    (void)policy.step(s);

    // The first step computes the approximation at x0 with the init
    // asymptotes L = x0 - 0.5 range, U = x0 + 0.5 range; premise-check
    // the asymptote geometry the oracle was derived for.
    REQUIRE(s.L[0] == Approx(0.0).margin(1e-15));
    REQUIRE(s.L[1] == Approx(0.0).margin(1e-15));
    REQUIRE(s.U[0] == Approx(2.0).margin(1e-15));
    REQUIRE(s.U[1] == Approx(4.0).margin(1e-15));

    const auto& p_obj = oracle.at("p_obj");
    const auto& q_obj = oracle.at("q_obj");
    const auto& p_con = oracle.at("p_con");
    const auto& q_con = oracle.at("q_con");

    INFO("p_obj observed (" << s.p_obj[0] << ", " << s.p_obj[1]
         << ") pinned (" << p_obj[0] << ", " << p_obj[1] << ")");
    CHECK(s.p_obj[0] == Approx(p_obj[0]).epsilon(1e-9).margin(1e-9));
    CHECK(s.p_obj[1] == Approx(p_obj[1]).epsilon(1e-9).margin(1e-9));
    CHECK(s.q_obj[0] == Approx(q_obj[0]).epsilon(1e-9).margin(1e-9));
    CHECK(s.q_obj[1] == Approx(q_obj[1]).epsilon(1e-9).margin(1e-9));
    CHECK(s.r_obj == Approx(oracle.at("r_obj")[0]).epsilon(1e-9).margin(1e-9));

    CHECK(s.p_con(0, 0) == Approx(p_con[0]).epsilon(1e-9).margin(1e-9));
    CHECK(s.p_con(0, 1) == Approx(p_con[1]).epsilon(1e-9).margin(1e-9));
    CHECK(s.q_con(0, 0) == Approx(q_con[0]).epsilon(1e-9).margin(1e-9));
    CHECK(s.q_con(0, 1) == Approx(q_con[1]).epsilon(1e-9).margin(1e-9));
    CHECK(s.r_con[0]
          == Approx(oracle.at("r_con")[0]).epsilon(1e-9).margin(1e-9));
}

TEST_CASE("mma: analytic dual gradient matches a finite difference",
          "[mma][oracle-pin]")
{
    // Production dual problem loaded with the Svanberg oracle
    // coefficients; the analytic gradient -g_tilde_1(x(y)) must match
    // a central finite difference of the dual value at interior y.
    const auto oracle = load_oracle("oracles/mma_svanberg_coefficients.csv");
    REQUIRE(oracle.contains("p_obj"));

    Eigen::VectorXd L{{0.0, 0.0}}, U{{2.0, 4.0}};
    Eigen::VectorXd alpha{{0.1, 0.2}}, beta{{1.9, 3.8}};
    Eigen::VectorXd p0{{oracle.at("p_obj")[0], oracle.at("p_obj")[1]}};
    Eigen::VectorXd q0{{oracle.at("q_obj")[0], oracle.at("q_obj")[1]}};
    Eigen::MatrixXd pc(1, 2), qc(1, 2);
    pc << oracle.at("p_con")[0], oracle.at("p_con")[1];
    qc << oracle.at("q_con")[0], oracle.at("q_con")[1];
    Eigen::VectorXd rc{{oracle.at("r_con")[0]}};

    detail::mma_reciprocal_dual_problem<double, Eigen::Dynamic,
                                        Eigen::Dynamic> dual;
    dual.L_out = &L;
    dual.U_out = &U;
    dual.alpha_out = &alpha;
    dual.beta_out = &beta;
    dual.p_obj_out = &p0;
    dual.q_obj_out = &q0;
    dual.p_con_out = &pc;
    dual.q_con_out = &qc;
    dual.r_obj = oracle.at("r_obj")[0];
    dual.r_con_out = &rc;
    dual.n_primal = 2;
    dual.m_dual = 1;

    Eigen::VectorXd y{{0.5}};
    Eigen::VectorXd g(1);
    dual.gradient(y, g);

    // Premise: x_2(y) is strictly interior; x_1(y) sits on its alpha
    // clamp for the whole FD stencil (both are envelope-exact regimes,
    // and neither switches within +/- h of y).
    CHECK(dual.x_primal[0] == Approx(alpha[0]).margin(1e-12));
    REQUIRE(dual.x_primal[1] > alpha[1] + 0.1);
    REQUIRE(dual.x_primal[1] < beta[1] - 0.1);

    const double h = 1e-6;
    Eigen::VectorXd yp{{y[0] + h}}, ym{{y[0] - h}};
    const double fd = (dual.value(yp) - dual.value(ym)) / (2.0 * h);

    INFO("analytic " << g[0] << " vs central FD " << fd);
    CHECK(g[0] == Approx(fd).epsilon(0).margin(1e-6));
}

// The dual problem reports wval as the 0.5-weighted penalty mass per
// unit rho (sum_j 0.5 w_j dx^2), the denominator of the minimal
// conservative rho increment. (Pre-fix it accumulated sum_j w_j dx^2,
// exactly twice the mass, which halved every rho increment; that defect
// is corrected and this pin is now green.)
TEST_CASE("gcmma: reported wval is the penalty mass per unit rho",
          "[gcmma][oracle-pin]")
{
    // Hand-worked case from the header comment: rho = 0 makes the
    // primal the closed-form reciprocal minimizer x* = 4/3, and the
    // rho-linear coefficient of the augmented approximation at x* is
    // mass = 0.5 * w * dx^2 = 1/9. wval must report that mass -- it is
    // the denominator of the minimal conservative rho increment.
    Eigen::VectorXd L{{0.0}}, U{{2.0}}, xk{{1.0}};
    Eigen::VectorXd alpha{{0.1}}, beta{{1.9}}, w{{2.0}};
    Eigen::VectorXd p{{1.0}}, q{{4.0}};
    Eigen::MatrixXd pc(0, 1), qc(0, 1);
    Eigen::VectorXd rc(0), rhoc(0);

    detail::mma_augmented_dual_problem<double, Eigen::Dynamic,
                                       Eigen::Dynamic> dual;
    dual.L_out = &L;
    dual.U_out = &U;
    dual.x_k_out = &xk;
    dual.alpha_out = &alpha;
    dual.beta_out = &beta;
    dual.w_out = &w;
    dual.p_obj_out = &p;
    dual.q_obj_out = &q;
    dual.p_con_out = &pc;
    dual.q_con_out = &qc;
    dual.r_obj = 0.5;
    dual.r_con_out = &rc;
    dual.rho_obj = 0.0;
    dual.rho_con_out = &rhoc;
    dual.n_primal = 1;
    dual.m_dual = 0;

    Eigen::VectorXd y(0);
    (void)dual.value(y);

    // Green premises: closed-form primal and approximation value.
    REQUIRE(dual.x_primal[0] == Approx(4.0 / 3.0).epsilon(0).margin(1e-12));
    CHECK(dual.gval == Approx(5.0).epsilon(0).margin(1e-12));

    // Penalty-mass pin: mass = 0.5 * 2 * (1/3)^2 = 1/9. The pre-fix
    // implementation reports 2/9 (twice the mass).
    INFO("wval observed " << dual.wval << " pinned " << 1.0 / 9.0);
    CHECK(dual.wval == Approx(1.0 / 9.0).epsilon(0).margin(1e-12));
}

// Green leg: the trial position, non-conservative shortfall, and
// growth-cap premises depend only on the coefficient formula and the
// initial rho -- not on the rho-growth-increment defect pinned below --
// so this stays a normally-passing regression guard on the composite-
// step reconstruction regardless of that defect's fix state.
TEST_CASE("gcmma: rho growth trial reconstruction matches the closed form",
          "[gcmma][oracle-pin]")
{
    rho_problem problem;
    Eigen::VectorXd x0{{1.0}};
    solver_options opts;

    alternative::gcmma::rho_wval_policy<> policy;
    alternative::gcmma::rho_wval_policy<>::options_type popts;
    popts.rho_init = 3.0;
    // One inner iteration: trial -> conservativity test -> one growth.
    popts.max_inner_iterations = 1;
    // Identity inter-outer decay so the state field holds exactly the
    // once-grown rho.
    popts.rho_decay = 1.0;
    popts.raai = 0.0;

    auto s = policy.init(problem, x0, opts, popts);
    const double rho0 = 3.0;
    (void)policy.step(s);

    // Premise: the committed trial is the beta clamp (see the header
    // derivation: the augmented first-order condition has no root in
    // (0, 2), so the Newton solve runs into the upper move limit).
    REQUIRE(s.x[0] == Approx(1.9).epsilon(0).margin(1e-12));

    // Reconstruct the approximation value and penalty mass at the
    // committed trial from the production state (p/q/r/w/L/U are those
    // of the step just taken; rho at trial time was rho_init since the
    // single inner iteration grows rho only after the trial).
    const double xt = s.x[0];
    const double xk = x0[0];
    const double dxU = s.U[0] - xt;
    const double dxL = xt - s.L[0];
    const double dx = xt - xk;
    const double mass = 0.5 * s.w[0] * dx * dx;
    const double gval = s.r_obj + s.p_obj[0] / dxU + s.q_obj[0] / dxL
                      + rho0 * mass;
    const double f_trial = problem.value(s.x);

    // Premise: the trial was non-conservative, so a growth fired.
    REQUIRE(f_trial > gval);
    const double delta_min = (f_trial - gval) / mass;

    // Premise: the 10x growth cap is not binding, so a correct growth
    // rho' = 1.1 (rho + S/mass) >= rho + delta_min must cover the
    // minimal conservative increment.
    REQUIRE(1.1 * (rho0 + delta_min) <= 10.0 * rho0 + 1e-9);

    // Hand cross-check of the reconstruction, valid for the pre-fix
    // flat-epsilon coefficients (p = 0, q = 50, r = -37.5): g_tilde(1.9)
    // = -8.754210526315789 and delta_min = 20.684210526315789. Guarded
    // so the closed-form pin below stays meaningful if the coefficient
    // formula changes.
    if(s.p_obj[0] == 0.0)
    {
        CHECK(gval == Approx(-8.754210526315789).epsilon(0).margin(1e-9));
        CHECK(delta_min
              == Approx(20.684210526315789).epsilon(0).margin(1e-6));
    }
}

// The rho growth divides the shortfall by the corrected penalty mass, so
// the grown approximation covers f at the trial that triggered the
// growth: rho' = 1.1 (rho + S/mass) = 26.05... >= rho + delta_min =
// 23.684.... (Pre-fix the growth divided by twice the mass and landed at
// 1.1 (rho + delta_min/2) = 14.676... < 23.684..., under-covering the
// very trial that fired the growth; that defect is corrected and this pin
// is now green.)
TEST_CASE("gcmma: rho growth covers the minimal conservative increment",
          "[gcmma][oracle-pin]")
{
    rho_problem problem;
    Eigen::VectorXd x0{{1.0}};
    solver_options opts;

    alternative::gcmma::rho_wval_policy<> policy;
    alternative::gcmma::rho_wval_policy<>::options_type popts;
    popts.rho_init = 3.0;
    // One inner iteration: trial -> conservativity test -> one growth.
    popts.max_inner_iterations = 1;
    // Identity inter-outer decay so the state field holds exactly the
    // once-grown rho.
    popts.rho_decay = 1.0;
    popts.raai = 0.0;

    auto s = policy.init(problem, x0, opts, popts);
    const double rho0 = 3.0;
    (void)policy.step(s);

    // Premise: identical trial-reconstruction setup as the green-leg
    // case above.
    REQUIRE(s.x[0] == Approx(1.9).epsilon(0).margin(1e-12));

    const double xt = s.x[0];
    const double xk = x0[0];
    const double dxU = s.U[0] - xt;
    const double dxL = xt - s.L[0];
    const double dx = xt - xk;
    const double mass = 0.5 * s.w[0] * dx * dx;
    const double gval = s.r_obj + s.p_obj[0] / dxU + s.q_obj[0] / dxL
                      + rho0 * mass;
    const double f_trial = problem.value(s.x);

    REQUIRE(f_trial > gval);
    const double delta_min = (f_trial - gval) / mass;
    REQUIRE(1.1 * (rho0 + delta_min) <= 10.0 * rho0 + 1e-9);

    // Closed-form conservativity pin: the grown rho must make the
    // augmented approximation cover f at the trial that triggered the
    // growth, rho' >= rho + delta_min. The pre-fix growth divides the
    // shortfall by twice the penalty mass and lands at
    // 1.1 (rho + delta_min/2) = 14.676... < 23.684... (red).
    INFO("rho after growth " << s.rho_obj << ", required at least "
         << rho0 + delta_min);
    CHECK(s.rho_obj >= rho0 + delta_min - 1e-9);
}

// Move-limit move bound (Svanberg 2007 notes eqs. 2.8-2.9). alpha_j/beta_j
// must include the XXMOVE = 0.5 term x_k +/- 0.5*range_j in addition to the
// box and the 0.1 asymptote buffer. For an unbounded variable range_j is the
// box-width analog 2*max(|x_k|, 1). The factor 2 makes the bound coincide
// with the initial asymptote so it does not over-constrain the first step;
// it engages only once the asymptote has inflated. This case forces that
// inflated regime with a large initial asymptote and checks the bound then
// caps the step where the two-term (box + 0.1 buffer) form would not.
//
// Worked case: unbounded x, x_k = 3, asymptote_init_unbounded = 5 so the
// init half-width is 5*max(|3|,1) = 15, giving L = -12, U = 18. The move
// bound uses range = 2*max(|3|,1) = 6, move bound = 0.5*6 = 3:
//   alpha = max(-inf, -12 + 0.1*15, 3 - 3) = max(-10.5, 0) = 0
//   beta  = min(+inf, 18 - 0.1*15, 3 + 3) = min(16.5, 6) = 6
// The two-term form would give alpha = -10.5, beta = 16.5; the move bound
// caps the inflated window to [0, 6].
TEST_CASE("mma: alpha/beta include the XXMOVE move bound (unbounded)",
          "[mma][oracle-pin]")
{
    unbounded_1d problem;
    Eigen::VectorXd x0{{3.0}};
    solver_options opts;

    mma_policy<> policy;
    mma_policy<>::options_type popts;
    popts.asymptote_init_unbounded = 5.0;   // force an inflated asymptote
    auto s = policy.init(problem, x0, opts, popts);

    // Premise: inflated init asymptotes L = -12, U = 18.
    REQUIRE(s.L[0] == Approx(-12.0).margin(1e-12));
    REQUIRE(s.U[0] == Approx(18.0).margin(1e-12));

    (void)policy.step(s);

    // The move bound caps both sides of the inflated window.
    CHECK(s.alpha[0] == Approx(0.0).margin(1e-12));
    CHECK(s.beta[0] == Approx(6.0).margin(1e-12));
    // And it is strictly tighter than the two-term (box + 0.1 buffer)
    // window that would otherwise leave alpha at -10.5 and beta at 16.5.
    CHECK(s.beta[0] < 16.5 - 1e-9);
    CHECK(s.alpha[0] > -10.5 + 1e-9);
}
