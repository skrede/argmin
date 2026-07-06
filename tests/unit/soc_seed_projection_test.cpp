// Hand-computed pins for the second-order-correction warm-start
// projection (detail::soc_seed_projection).
//
// The SOC QP re-solve warm-starts at the rejected primary direction p,
// which is a stationary point of the unchanged QP objective. When p
// violates the corrected inequality RHS, active_set_qp_solver (whose
// phase-1 restores equality feasibility only) terminates at the warm
// start without consulting that RHS. The projection restores the
// solver's feasible-start precondition by the minimum-norm LDP shift
//
//   min ||q||  s.t.  J_ineq * q >= b_ineq_soc - J_ineq * p
//                    (plus finite bound rows p_lo <= p + q <= p_hi)
//
// seeding the re-solve at p + q. Each case below pins the projected
// seed against a hand-derived KKT solution of that least-distance
// program.
//
// Reference: Lawson & Hanson 1974 Ch. 23.4 (LDP via dual NNLS);
//            N&W 2e Section 16.1 Algorithm 16.1 (active-set
//            feasibility invariant), Section 18.3 (second-order
//            correction).

#include "argmin/detail/sqp_common.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <limits>

using Catch::Approx;
using namespace argmin;

namespace
{

constexpr double inf = std::numeric_limits<double>::infinity();

detail::sqp_state_buffers<double, dynamic_dimension> make_buffers(
    int n, int n_ineq)
{
    detail::sqp_state_buffers<double, dynamic_dimension> bufs;
    bufs.resize(n, 0, n_ineq);
    return bufs;
}

}  // namespace

TEST_CASE("soc_seed_projection shifts an infeasible warm start onto the "
          "corrected inequality polyhedron by the minimum-norm correction",
          "[sqp_common][soc][ldp]")
{
    SECTION("single violated row: axis-aligned min-norm shift")
    {
        // min ||q|| s.t. q0 >= 1: q* = (1, 0), seed = p + q* = (1, 0).
        // Any feasible q has q0 >= 1, hence ||q|| >= 1; the KKT point
        // q = G^T lambda / 2-form gives exactly (1, 0).
        auto bufs = make_buffers(2, 1);
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> J(1, 2);
        J << 1.0, 0.0;
        Eigen::VectorXd b(1);
        b << 1.0;
        Eigen::VectorXd seed = Eigen::VectorXd::Zero(2);
        Eigen::VectorXd unused_lo, unused_hi;

        const bool projected =
            detail::soc_seed_projection<double, dynamic_dimension>(
                J, b, unused_lo, unused_hi, false, bufs, seed);

        CHECK(projected);
        CHECK(seed[0] == Approx(1.0).margin(1e-9));
        CHECK(seed[1] == Approx(0.0).margin(1e-9));
        CHECK((J * seed - b).minCoeff() >= -1e-9);
    }

    SECTION("two orthogonal violated rows: componentwise shift")
    {
        // min ||q|| s.t. q0 >= 1, q1 >= 2: q* = (1, 2).
        auto bufs = make_buffers(2, 2);
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> J(2, 2);
        J << 1.0, 0.0,
             0.0, 1.0;
        Eigen::VectorXd b(2);
        b << 1.0, 2.0;
        Eigen::VectorXd seed = Eigen::VectorXd::Zero(2);
        Eigen::VectorXd unused_lo, unused_hi;

        const bool projected =
            detail::soc_seed_projection<double, dynamic_dimension>(
                J, b, unused_lo, unused_hi, false, bufs, seed);

        CHECK(projected);
        CHECK(seed[0] == Approx(1.0).margin(1e-9));
        CHECK(seed[1] == Approx(2.0).margin(1e-9));
    }

    SECTION("nonzero warm start: only the violation gap is corrected")
    {
        // p = (0.5, 3.0), row q0 >= 1 - 0.5 = 0.5 in the shifted
        // variable: q* = (0.5, 0), seed = (1, 3).
        auto bufs = make_buffers(2, 1);
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> J(1, 2);
        J << 1.0, 0.0;
        Eigen::VectorXd b(1);
        b << 1.0;
        Eigen::VectorXd seed(2);
        seed << 0.5, 3.0;
        Eigen::VectorXd unused_lo, unused_hi;

        const bool projected =
            detail::soc_seed_projection<double, dynamic_dimension>(
                J, b, unused_lo, unused_hi, false, bufs, seed);

        CHECK(projected);
        CHECK(seed[0] == Approx(1.0).margin(1e-9));
        CHECK(seed[1] == Approx(3.0).margin(1e-9));
    }
}

TEST_CASE("soc_seed_projection stacks finite bound rows and skips infinite "
          "ones",
          "[sqp_common][soc][ldp]")
{
    // min ||q|| s.t. q0 + q1 >= 2 and p + q <= p_hi with
    // p_hi = (0.5, inf), p = 0. Unconstrained-by-bounds min-norm is
    // q = (1, 1), which violates q0 <= 0.5; with both rows active the
    // KKT system q = (lambda_1 (1,1) + lambda_2 (-1,0)) / 2 gives
    // q* = (0.5, 1.5) with lambda = (3, 2) >= 0.
    auto bufs = make_buffers(2, 1);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> J(1, 2);
    J << 1.0, 1.0;
    Eigen::VectorXd b(1);
    b << 2.0;
    Eigen::VectorXd seed = Eigen::VectorXd::Zero(2);
    Eigen::VectorXd p_lo(2), p_hi(2);
    p_lo << -inf, -inf;
    p_hi << 0.5, inf;

    const bool projected =
        detail::soc_seed_projection<double, dynamic_dimension>(
            J, b, p_lo, p_hi, true, bufs, seed);

    CHECK(projected);
    CHECK(seed[0] == Approx(0.5).margin(1e-9));
    CHECK(seed[1] == Approx(1.5).margin(1e-9));
    CHECK((J * seed - b).minCoeff() >= -1e-9);
    CHECK(seed[0] <= 0.5 + 1e-9);
}

TEST_CASE("soc_seed_projection leaves feasible, incompatible, and "
          "equality-only warm starts untouched",
          "[sqp_common][soc][ldp]")
{
    SECTION("already-feasible warm start is a no-op")
    {
        auto bufs = make_buffers(2, 1);
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> J(1, 2);
        J << 1.0, 0.0;
        Eigen::VectorXd b(1);
        b << -1.0;  // q0 >= -1 holds at q = 0.
        Eigen::VectorXd seed = Eigen::VectorXd::Zero(2);
        Eigen::VectorXd unused_lo, unused_hi;

        const bool projected =
            detail::soc_seed_projection<double, dynamic_dimension>(
                J, b, unused_lo, unused_hi, false, bufs, seed);

        CHECK_FALSE(projected);
        CHECK(seed[0] == 0.0);
        CHECK(seed[1] == 0.0);
    }

    SECTION("incompatible rows fall back to the unprojected warm start")
    {
        // q0 >= 1 and -q0 >= 1 have empty intersection: the LDP
        // reports mode 4 and the seed must stay at p (strictly no
        // worse than the pre-projection behavior).
        auto bufs = make_buffers(2, 2);
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> J(2, 2);
        J << 1.0, 0.0,
             -1.0, 0.0;
        Eigen::VectorXd b(2);
        b << 1.0, 1.0;
        Eigen::VectorXd seed(2);
        seed << 0.25, -0.75;
        const Eigen::VectorXd seed_before = seed;
        Eigen::VectorXd unused_lo, unused_hi;

        const bool projected =
            detail::soc_seed_projection<double, dynamic_dimension>(
                J, b, unused_lo, unused_hi, false, bufs, seed);

        CHECK_FALSE(projected);
        CHECK(seed[0] == seed_before[0]);
        CHECK(seed[1] == seed_before[1]);
    }

    SECTION("equality-only problems (no inequality rows) are untouched")
    {
        auto bufs = make_buffers(2, 0);
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> J(0, 2);
        Eigen::VectorXd b(0);
        Eigen::VectorXd seed(2);
        seed << -0.5, 2.0;
        Eigen::VectorXd unused_lo, unused_hi;

        const bool projected =
            detail::soc_seed_projection<double, dynamic_dimension>(
                J, b, unused_lo, unused_hi, false, bufs, seed);

        CHECK_FALSE(projected);
        CHECK(seed[0] == -0.5);
        CHECK(seed[1] == 2.0);
    }
}
