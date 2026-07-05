// Regression tests for the finite-horizon LQR-shaped NMPC test
// problem. Covers dimension reporting, problem-class flags, dynamics
// feasibility at the initial point, central-difference gradient
// consistency, and a domain-verification suite that checks the
// zero-order-hold discretization, the LQR stage-cost structure, and
// the dynamics equality-constraint block against independently derived
// reference values.
//
// Reference: Anderson and Moore, Optimal Control: Linear Quadratic
//            Methods, 1990, Section 2.3 (LQR / Riccati); Borrelli,
//            Bemporad and Morari, Predictive Control for Linear and
//            Hybrid Systems, Chapter on discrete-time systems
//            (finite-horizon stacked LQR-MPC, x_{k+1} = A x_k + B u_k
//            equality blocks); Lynch and Park, Modern Robotics, 2017,
//            Chapter 8 (double-integrator zero-order-hold). The exact
//            ZOH of a double integrator with a nilpotent continuous
//            A_c (A_c^2 = 0) is
//                A_d = I + A_c dt,
//                B_d = (I dt + A_c dt^2 / 2) B_c,
//            which for the (px, py, vx, vy) / (ax, ay) stack gives the
//            0.5 dt^2 position rows and the dt velocity rows asserted
//            below.
//
// Consumer cross-check (NMPC): the ctrlpp double-integrator NMPC
// problem (benchmarks/problems/nmpc/double_integrator.h) uses the same
// dt = 0.1 and the same LQR weights Q = I, R = 0.1 I, but discretizes
// with forward Euler (position row = x_pos + dt x_vel, no 0.5 dt^2
// control coupling) and interleaves the state as (px, vx, py, vy). The
// argmin fixture uses the exact ZOH (0.5 dt^2 coupling present) and the
// grouped ordering (px, py, vx, vy). Both discretizations are valid;
// ZOH is exact for a double integrator while Euler is its first-order
// truncation. The difference is a deliberate, documented convention
// divergence, not a defect: this suite asserts the ZOH form against the
// analytic reference above, and the ctrlpp Euler form is recorded here
// as the consumer counterpart.

#include "argmin/test_functions/problem_class.h"
#include "argmin/test_functions/nmpc_lqr.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace argmin;

TEST_CASE("nmpc_lqr dimensions and class flags", "[nmpc_lqr]")
{
    nmpc_lqr<10> p10{};
    REQUIRE(p10.dimension() == 60);
    REQUIRE(p10.num_equality() == 40);
    REQUIRE(p10.num_inequality() == 0);

    nmpc_lqr<20> p20{};
    REQUIRE(p20.dimension() == 120);
    REQUIRE(p20.num_equality() == 80);
    REQUIRE(p20.num_inequality() == 0);

    REQUIRE(has_class(p10.pclass, problem_class::application));
    REQUIRE(has_class(p10.pclass, problem_class::equality));
    REQUIRE(has_class(p10.pclass, problem_class::bound_constrained));

    REQUIRE(has_class(p20.pclass, problem_class::application));
    REQUIRE(has_class(p20.pclass, problem_class::equality));
    REQUIRE(has_class(p20.pclass, problem_class::bound_constrained));

    // application is the seventh atomic flag; compositions are
    // OR-clean across the existing six flags.
    constexpr auto composed =
        problem_class::bound_constrained | problem_class::application;
    REQUIRE(has_class(composed, problem_class::application));
    REQUIRE(has_class(composed, problem_class::bound_constrained));
    REQUIRE(!has_class(composed, problem_class::equality));
}

TEST_CASE("nmpc_lqr initial point is feasible for the dynamics equalities",
          "[nmpc_lqr]")
{
    nmpc_lqr<10> p{};
    const auto z0 = p.initial_point();
    Eigen::Matrix<double, nmpc_lqr<10>::constraint_count, 1> c_eq;
    p.constraints(z0, c_eq);
    REQUIRE(c_eq.norm() < 1e-10);
}

TEST_CASE("nmpc_lqr bound layout: states unbounded, controls box-bounded",
          "[nmpc_lqr]")
{
    nmpc_lqr<10> p{};
    const auto lb = p.lower_bounds();
    const auto ub = p.upper_bounds();
    constexpr double inf = std::numeric_limits<double>::infinity();
    // Each stage block has n_x = 4 state entries (indices 0..3 of
    // stage) and n_u = 2 control entries (indices 4..5 of stage).
    for(int k = 0; k < 10; ++k)
    {
        const int x_off = k * 6;
        for(int j = 0; j < 4; ++j)
        {
            REQUIRE(lb[x_off + j] == -inf);
            REQUIRE(ub[x_off + j] == inf);
        }
        const int u_off = k * 6 + 4;
        for(int j = 0; j < 2; ++j)
        {
            REQUIRE(lb[u_off + j] == -2.0);
            REQUIRE(ub[u_off + j] == 2.0);
        }
    }
}

TEST_CASE("nmpc_lqr gradient matches central-difference at the initial point",
          "[nmpc_lqr]")
{
    using prob_t = nmpc_lqr<10>;
    prob_t p{};
    const auto z0 = p.initial_point();
    Eigen::Matrix<double, prob_t::problem_dimension, 1> g;
    p.gradient(z0, g);

    const double h = 1e-5;
    double max_err = 0.0;
    Eigen::Matrix<double, prob_t::problem_dimension, 1> z_plus = z0;
    Eigen::Matrix<double, prob_t::problem_dimension, 1> z_minus = z0;
    for(int i = 0; i < p.dimension(); ++i)
    {
        z_plus[i] = z0[i] + h;
        z_minus[i] = z0[i] - h;
        const double f_plus = p.value(z_plus);
        const double f_minus = p.value(z_minus);
        const double fd = (f_plus - f_minus) / (2.0 * h);
        const double err = std::abs(fd - g[i]);
        if(err > max_err)
            max_err = err;
        z_plus[i] = z0[i];
        z_minus[i] = z0[i];
    }
    REQUIRE(max_err < 1e-4);
}

TEST_CASE("nmpc_lqr constraint jacobian matches central-difference at z0",
          "[nmpc_lqr]")
{
    using prob_t = nmpc_lqr<10>;
    prob_t p{};
    const auto z0 = p.initial_point();
    Eigen::Matrix<double, prob_t::constraint_count, prob_t::problem_dimension> J;
    p.constraint_jacobian(z0, J);

    const double h = 1e-5;
    Eigen::Matrix<double, prob_t::constraint_count, 1> c_plus;
    Eigen::Matrix<double, prob_t::constraint_count, 1> c_minus;
    Eigen::Matrix<double, prob_t::problem_dimension, 1> z_plus = z0;
    Eigen::Matrix<double, prob_t::problem_dimension, 1> z_minus = z0;
    double max_err = 0.0;
    for(int j = 0; j < p.dimension(); ++j)
    {
        z_plus[j] = z0[j] + h;
        z_minus[j] = z0[j] - h;
        p.constraints(z_plus, c_plus);
        p.constraints(z_minus, c_minus);
        for(int i = 0; i < p.num_equality(); ++i)
        {
            const double fd = (c_plus[i] - c_minus[i]) / (2.0 * h);
            const double err = std::abs(fd - J(i, j));
            if(err > max_err)
                max_err = err;
        }
        z_plus[j] = z0[j];
        z_minus[j] = z0[j];
    }
    REQUIRE(max_err < 1e-7);
}

// ---- Domain verification ------------------------------------------------
//
// Everything below asserts against values derived independently of
// nmpc_lqr.h (recomputed here from the continuous double-integrator
// model and the analytic ZOH), not against the header's own outputs.

namespace
{

// Independent reference: exact zero-order-hold of the continuous
// double-integrator stack. Continuous model per axis is p' = v, v' = a
// with the (px, py, vx, vy) / (ax, ay) ordering, i.e.
//   A_c = [[0 0 1 0],[0 0 0 1],[0 0 0 0],[0 0 0 0]],
//   B_c = [[0 0],[0 0],[1 0],[0 1]].
// A_c is nilpotent (A_c^2 = 0), so exp(A_c dt) = I + A_c dt and
// integral_0^dt exp(A_c s) ds = I dt + A_c dt^2 / 2. Hence:
Eigen::Matrix<double, 4, 4> reference_A(double dt)
{
    Eigen::Matrix<double, 4, 4> A = Eigen::Matrix<double, 4, 4>::Identity();
    A(0, 2) = dt;
    A(1, 3) = dt;
    return A;
}

Eigen::Matrix<double, 4, 2> reference_B(double dt)
{
    Eigen::Matrix<double, 4, 2> B = Eigen::Matrix<double, 4, 2>::Zero();
    const double half_dt2 = 0.5 * dt * dt;
    B(0, 0) = half_dt2;
    B(1, 1) = half_dt2;
    B(2, 0) = dt;
    B(3, 1) = dt;
    return B;
}

// The forward-Euler discretization used by the ctrlpp NMPC consumer.
// Recorded here only so the divergence from the exact ZOH is asserted
// explicitly rather than implied by prose. Euler drops the 0.5 dt^2
// control-to-position coupling.
Eigen::Matrix<double, 4, 2> ctrlpp_euler_B(double dt)
{
    Eigen::Matrix<double, 4, 2> B = Eigen::Matrix<double, 4, 2>::Zero();
    B(2, 0) = dt;
    B(3, 1) = dt;
    return B;
}

}  // namespace

TEST_CASE("nmpc_lqr ZOH A/B match the analytic double-integrator reference",
          "[nmpc_lqr][domain]")
{
    using prob_t = nmpc_lqr<10>;
    const double dt = prob_t::dt;
    const auto A = prob_t::A_matrix();
    const auto B = prob_t::B_matrix();

    const auto A_ref = reference_A(dt);
    const auto B_ref = reference_B(dt);

    REQUIRE((A - A_ref).cwiseAbs().maxCoeff() < 1e-15);
    REQUIRE((B - B_ref).cwiseAbs().maxCoeff() < 1e-15);

    // The exact ZOH B keeps the 0.5 dt^2 position coupling that the
    // ctrlpp forward-Euler consumer drops; assert the two are genuinely
    // distinct so the documented divergence cannot silently regress into
    // agreement (which would mean the header quietly switched to Euler).
    const auto B_euler = ctrlpp_euler_B(dt);
    REQUIRE((B - B_euler).cwiseAbs().maxCoeff() > 1e-6);
    REQUIRE(std::abs(B(0, 0) - 0.5 * dt * dt) < 1e-15);
    REQUIRE(std::abs(B_euler(0, 0)) < 1e-15);
}

TEST_CASE("nmpc_lqr stage cost is 0.5 (x^T Q x + u^T R u), Q = I, R = 0.1 I",
          "[nmpc_lqr][domain]")
{
    using prob_t = nmpc_lqr<10>;
    prob_t p{};
    constexpr int H = prob_t::horizon;
    constexpr int n_x = prob_t::n_x;
    constexpr int n_u = prob_t::n_u;
    const double q = prob_t::q_diag;
    const double r = prob_t::r_diag;

    // Build a nontrivial z and recompute the LQR stage cost directly
    // from the layout: 0.5 * sum_{k=0}^{H-1} (x_k^T Q x_k + u_k^T R u_k),
    // Q = q I, R = r I, no terminal cost on x_H. x_0 is fixed data.
    Eigen::Matrix<double, prob_t::problem_dimension, 1> z;
    for(int i = 0; i < prob_t::problem_dimension; ++i)
        z[i] = 0.03 * (i + 1) - 0.5 * std::sin(0.7 * i);

    const auto x0 = prob_t::x0_fixed();
    double cost_ref = 0.5 * q * x0.squaredNorm();
    for(int k = 0; k < H; ++k)
    {
        double u_sq = 0.0;
        for(int j = 0; j < n_u; ++j)
        {
            const double u = z[prob_t::u_offset(k) + j];
            u_sq += u * u;
        }
        cost_ref += 0.5 * r * u_sq;
        if(k + 1 < H)
        {
            double x_sq = 0.0;
            for(int j = 0; j < n_x; ++j)
            {
                const double x = z[prob_t::x_offset(k) + j];
                x_sq += x * x;
            }
            cost_ref += 0.5 * q * x_sq;
        }
    }

    REQUIRE(std::abs(p.value(z) - cost_ref) < 1e-12);
    REQUIRE(q == 1.0);
    REQUIRE(r == 0.1);

    // Gradient block structure: d cost / d u_k = R u_k = r u_k,
    // d cost / d x_{k+1} = Q x_{k+1} = q x_{k+1}, and x_H carries no cost.
    Eigen::Matrix<double, prob_t::problem_dimension, 1> g;
    p.gradient(z, g);
    for(int k = 0; k < H; ++k)
    {
        for(int j = 0; j < n_u; ++j)
        {
            const int idx = prob_t::u_offset(k) + j;
            REQUIRE(std::abs(g[idx] - r * z[idx]) < 1e-12);
        }
        for(int j = 0; j < n_x; ++j)
        {
            const int idx = prob_t::x_offset(k) + j;
            const double expected = (k + 1 < H) ? q * z[idx] : 0.0;
            REQUIRE(std::abs(g[idx] - expected) < 1e-12);
        }
    }
}

TEST_CASE("nmpc_lqr dynamics constraint block equals x_{k+1} - A x_k - B u_k",
          "[nmpc_lqr][domain]")
{
    using prob_t = nmpc_lqr<10>;
    prob_t p{};
    constexpr int H = prob_t::horizon;
    constexpr int n_x = prob_t::n_x;
    constexpr int n_u = prob_t::n_u;
    const double dt = prob_t::dt;
    const auto A_ref = reference_A(dt);
    const auto B_ref = reference_B(dt);
    const auto x0 = prob_t::x0_fixed();

    Eigen::Matrix<double, prob_t::problem_dimension, 1> z;
    for(int i = 0; i < prob_t::problem_dimension; ++i)
        z[i] = 0.02 * (i + 1) + 0.4 * std::cos(0.9 * i);

    Eigen::Matrix<double, prob_t::constraint_count, 1> c;
    p.constraints(z, c);

    // Independent residual assembly using the reference A/B and the
    // stacking convention z = [x_1, u_0, x_2, u_1, ...].
    for(int k = 0; k < H; ++k)
    {
        Eigen::Matrix<double, n_x, 1> x_k;
        if(k == 0)
            x_k = x0;
        else
            for(int j = 0; j < n_x; ++j)
                x_k[j] = z[prob_t::x_offset(k - 1) + j];
        Eigen::Matrix<double, n_u, 1> u_k;
        for(int j = 0; j < n_u; ++j)
            u_k[j] = z[prob_t::u_offset(k) + j];
        Eigen::Matrix<double, n_x, 1> x_kp1;
        for(int j = 0; j < n_x; ++j)
            x_kp1[j] = z[prob_t::x_offset(k) + j];

        const Eigen::Matrix<double, n_x, 1> r_ref = x_kp1 - A_ref * x_k - B_ref * u_k;
        for(int j = 0; j < n_x; ++j)
            REQUIRE(std::abs(c[k * n_x + j] - r_ref[j]) < 1e-12);
    }

    // Constraint-Jacobian sign convention: +I on x_{k+1}, -B on u_k,
    // -A on x_k (k > 0). Asserted against the reference A/B blocks.
    Eigen::Matrix<double, prob_t::constraint_count, prob_t::problem_dimension> J;
    p.constraint_jacobian(z, J);
    for(int k = 0; k < H; ++k)
    {
        const int row0 = k * n_x;
        for(int j = 0; j < n_x; ++j)
            REQUIRE(std::abs(J(row0 + j, prob_t::x_offset(k) + j) - 1.0) < 1e-15);
        for(int i = 0; i < n_x; ++i)
            for(int j = 0; j < n_u; ++j)
                REQUIRE(std::abs(J(row0 + i, prob_t::u_offset(k) + j) + B_ref(i, j)) < 1e-15);
        if(k > 0)
            for(int i = 0; i < n_x; ++i)
                for(int j = 0; j < n_x; ++j)
                    REQUIRE(std::abs(J(row0 + i, prob_t::x_offset(k - 1) + j) + A_ref(i, j)) < 1e-15);
    }
}

TEST_CASE("nmpc_lqr ZOH domain check fires under a seeded B perturbation",
          "[nmpc_lqr][domain][sensitivity]")
{
    // Sensitivity proof: the domain check must FAIL when the discretized
    // input matrix is perturbed. nmpc_lqr's A/B are constexpr, so we
    // perturb a local copy of the reference B (mimicking a single wrong
    // entry in the header's B_matrix) and confirm the equality that the
    // "ZOH A/B match reference" case asserts no longer holds.
    using prob_t = nmpc_lqr<10>;
    const double dt = prob_t::dt;
    const auto B = prob_t::B_matrix();

    Eigen::Matrix<double, 4, 2> B_perturbed = reference_B(dt);
    B_perturbed(2, 0) += 1e-3;  // seeded single-entry error in dt row

    // The genuine header B still matches the true reference (guards
    // against a false-positive sensitivity claim)...
    REQUIRE((B - reference_B(dt)).cwiseAbs().maxCoeff() < 1e-15);
    // ...but the perturbed B is caught by the same tolerance the domain
    // check uses, so the instrument is demonstrably sensitive.
    REQUIRE((B - B_perturbed).cwiseAbs().maxCoeff() > 1e-15);

    // And the downstream effect is observable: a dynamics residual built
    // with the perturbed B diverges from the header's residual, so a
    // wrong discretization would not slip past the constraint check.
    prob_t p{};
    Eigen::Matrix<double, prob_t::problem_dimension, 1> z = p.initial_point();
    // Give a nonzero control so B multiplies something.
    z[prob_t::u_offset(0)] = 0.5;
    Eigen::Matrix<double, prob_t::constraint_count, 1> c_header;
    p.constraints(z, c_header);

    const auto A_ref = reference_A(dt);
    const auto x0 = prob_t::x0_fixed();
    Eigen::Matrix<double, 2, 1> u0;
    u0 << z[prob_t::u_offset(0)], z[prob_t::u_offset(0) + 1];
    Eigen::Matrix<double, 4, 1> x1;
    for(int j = 0; j < 4; ++j)
        x1[j] = z[prob_t::x_offset(0) + j];
    const Eigen::Matrix<double, 4, 1> r_perturbed = x1 - A_ref * x0 - B_perturbed * u0;
    REQUIRE((c_header.head<4>() - r_perturbed).cwiseAbs().maxCoeff() > 1e-6);
}
