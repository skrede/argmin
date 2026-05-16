#ifndef HPP_GUARD_ARGMIN_TEST_FUNCTIONS_IK_POSE_BATCH_H
#define HPP_GUARD_ARGMIN_TEST_FUNCTIONS_IK_POSE_BATCH_H

// Pose-batch inverse-kinematics test problem.
//
// 6-DOF revolute serial arm with Denavit-Hartenberg parameters. For
// each of K target positions p_target^(k) the decision vector holds a
// joint configuration theta^(k) in R^NJoints; the K joint vectors are
// stacked into z in R^{NJoints * K}. The objective is the sum of
// squared end-effector position errors:
//
//   f(z) = sum_{k=0..K-1} || p_ee(theta^(k)) - p_target^(k) ||^2
//
// p_ee is the end-effector position computed by walking the DH chain
// of 4x4 homogeneous SE(3) transforms. The K targets are positioned
// inside the manipulator's reachable workspace (well within sum of
// link lengths), so the loss is achievable to within
// bench-accuracy gates.
//
// The gradient uses the geometric Jacobian J_p (3 x NJoints) at each
// target's joint vector; for revolute joints column i is
//   J_p,i = z_axis_i x (p_ee - p_i)
// where z_axis_i is the rotation axis of joint i (third column of the
// cumulative rotation matrix up to and including joint i's transform)
// and p_i is the origin of joint i's frame.
//
// Joint bounds: -pi <= theta_j <= +pi for every primal entry (the K
// joint vectors are stacked but the bounds are uniform).
//
// SE(3) forward kinematics: Craig, Introduction to Robotics, 3rd ed.,
// Section 3.6 (Denavit-Hartenberg convention).
// Geometric Jacobian: Craig Section 5.5 (cross-product form for
// revolute joints).
// No Lie-algebra dependency; pure 4x4 homogeneous transforms via
// Eigen::Matrix primitives.

#include "argmin/test_functions/problem_class.h"

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <limits>
#include <numbers>

namespace argmin
{

template <int NJoints = 6, int K = 5, typename Scalar = double>
struct ik_pose_batch
{
    static constexpr int joint_count = NJoints;
    static constexpr int target_count = K;
    static constexpr int problem_dimension = NJoints * K;
    static constexpr int constraint_count = 0;
    static constexpr problem_class pclass =
          problem_class::bound_constrained
        | problem_class::application;

    // Single Denavit-Hartenberg row: (a, alpha, d, theta_offset).
    struct dh_row
    {
        Scalar a;
        Scalar alpha;
        Scalar d;
        Scalar theta_offset;
    };

    // 6-DOF revolute-arm DH table (generic well-conditioned shape, no
    // singularities at the chosen home/target configurations). Link
    // lengths and offsets are dimensionless; the workspace spans a
    // sphere of radius ~ sum(|a_i|) + sum(|d_i|) about the base.
    [[nodiscard]] static std::array<dh_row, NJoints> dh_table() noexcept
    {
        static_assert(NJoints == 6,
                      "ik_pose_batch currently ships a 6-DOF DH table; "
                      "templating on NJoints is reserved for future "
                      "kinematic shapes.");
        std::array<dh_row, NJoints> table{};
        constexpr Scalar pi = std::numbers::pi_v<Scalar>;
        table[0] = dh_row{Scalar(0),    pi / Scalar(2),  Scalar(1.0),  Scalar(0)};
        table[1] = dh_row{Scalar(1.0),  Scalar(0),       Scalar(0),    Scalar(0)};
        table[2] = dh_row{Scalar(1.0),  Scalar(0),       Scalar(0),    Scalar(0)};
        table[3] = dh_row{Scalar(0),    pi / Scalar(2), Scalar(0.5),  Scalar(0)};
        table[4] = dh_row{Scalar(0),   -pi / Scalar(2), Scalar(0),    Scalar(0)};
        table[5] = dh_row{Scalar(0),    Scalar(0),       Scalar(0.3),  Scalar(0)};
        return table;
    }

    // K target positions, distributed on a circle in the xy plane at
    // z = 0.8 with radius 1.2 about the base. All inside the reachable
    // workspace (sum of link lengths is approximately 2.8).
    [[nodiscard]] static std::array<Eigen::Matrix<Scalar, 3, 1>, K> targets() noexcept
    {
        std::array<Eigen::Matrix<Scalar, 3, 1>, K> p_targets{};
        constexpr Scalar pi = std::numbers::pi_v<Scalar>;
        constexpr Scalar radius = Scalar(1.2);
        constexpr Scalar z_target = Scalar(0.8);
        for(int k = 0; k < K; ++k)
        {
            const Scalar angle = Scalar(2) * pi * Scalar(k) / Scalar(K);
            p_targets[k][0] = radius * std::cos(angle);
            p_targets[k][1] = radius * std::sin(angle);
            p_targets[k][2] = z_target;
        }
        return p_targets;
    }

    // Home configuration shared by every target's initial joint vector.
    [[nodiscard]] static Eigen::Matrix<Scalar, NJoints, 1> theta_home() noexcept
    {
        constexpr Scalar pi = std::numbers::pi_v<Scalar>;
        Eigen::Matrix<Scalar, NJoints, 1> theta;
        theta[0] = Scalar(0);
        theta[1] = -pi / Scalar(4);
        theta[2] = pi / Scalar(4);
        theta[3] = Scalar(0);
        theta[4] = pi / Scalar(4);
        theta[5] = Scalar(0);
        return theta;
    }

    // Fixed (alpha, a) prefix of a DH transform; independent of
    // theta_i. T_i_fixed = Rot_x(alpha_i) * Trans_x(a_i).
    [[nodiscard]] static Eigen::Matrix<Scalar, 4, 4>
    dh_fixed_prefix(const dh_row& row) noexcept
    {
        const Scalar ca = std::cos(row.alpha);
        const Scalar sa = std::sin(row.alpha);
        Eigen::Matrix<Scalar, 4, 4> F;
        F(0, 0) = Scalar(1);  F(0, 1) = Scalar(0);  F(0, 2) = Scalar(0);  F(0, 3) = row.a;
        F(1, 0) = Scalar(0);  F(1, 1) = ca;         F(1, 2) = -sa;        F(1, 3) = Scalar(0);
        F(2, 0) = Scalar(0);  F(2, 1) = sa;         F(2, 2) = ca;         F(2, 3) = Scalar(0);
        F(3, 0) = Scalar(0);  F(3, 1) = Scalar(0);  F(3, 2) = Scalar(0);  F(3, 3) = Scalar(1);
        return F;
    }

    // Theta-dependent suffix of a DH transform; T_i_var(theta) =
    // Rot_z(theta + offset) * Trans_z(d).
    [[nodiscard]] static Eigen::Matrix<Scalar, 4, 4>
    dh_theta_suffix(const dh_row& row, Scalar theta_in) noexcept
    {
        const Scalar theta = theta_in + row.theta_offset;
        const Scalar ct = std::cos(theta);
        const Scalar st = std::sin(theta);
        Eigen::Matrix<Scalar, 4, 4> R;
        R(0, 0) = ct;         R(0, 1) = -st;        R(0, 2) = Scalar(0);  R(0, 3) = Scalar(0);
        R(1, 0) = st;         R(1, 1) = ct;         R(1, 2) = Scalar(0);  R(1, 3) = Scalar(0);
        R(2, 0) = Scalar(0);  R(2, 1) = Scalar(0);  R(2, 2) = Scalar(1);  R(2, 3) = row.d;
        R(3, 0) = Scalar(0);  R(3, 1) = Scalar(0);  R(3, 2) = Scalar(0);  R(3, 3) = Scalar(1);
        return R;
    }

    // Full single DH transform per Craig Section 3.6 (modified DH):
    // T_i = Rot_x(alpha_i) * Trans_x(a_i) * Rot_z(theta_i + offset_i) * Trans_z(d_i).
    [[nodiscard]] static Eigen::Matrix<Scalar, 4, 4>
    dh_transform(const dh_row& row, Scalar theta_in) noexcept
    {
        return dh_fixed_prefix(row) * dh_theta_suffix(row, theta_in);
    }

    // Forward kinematics: computes two interleaved cumulative
    // transform sequences:
    //   cumulative[i]      = T_0 * T_1 * ... * T_{i-1}   (frame i)
    //   pre_joint[i]       = cumulative[i] * F_i         (frame in which
    //                                                     joint i's
    //                                                     z-axis rotates)
    // Returns the end-effector transform cumulative[NJoints].
    [[nodiscard]] static Eigen::Matrix<Scalar, 4, 4> forward_kinematics(
        const Eigen::Matrix<Scalar, NJoints, 1>& theta,
        std::array<Eigen::Matrix<Scalar, 4, 4>, NJoints + 1>& cumulative,
        std::array<Eigen::Matrix<Scalar, 4, 4>, NJoints>& pre_joint) noexcept
    {
        const auto table = dh_table();
        cumulative[0].setIdentity();
        for(int i = 0; i < NJoints; ++i)
        {
            const auto F_i = dh_fixed_prefix(table[i]);
            pre_joint[i] = cumulative[i] * F_i;
            const auto R_i = dh_theta_suffix(table[i], theta[i]);
            cumulative[i + 1] = pre_joint[i] * R_i;
        }
        return cumulative[NJoints];
    }

    // Geometric Jacobian J_p (3 x NJoints) of end-effector position
    // wrt joint angles, via Craig Section 5.5 cross-product form for
    // revolute joints. Each joint i rotates about the z-axis of the
    // pre_joint[i] frame; that axis and origin feed the cross product
    //   J_p.col(i) = z_axis_i x (p_ee - p_pre_i).
    [[nodiscard]] static Eigen::Matrix<Scalar, 3, NJoints> position_jacobian(
        const std::array<Eigen::Matrix<Scalar, 4, 4>, NJoints + 1>& cumulative,
        const std::array<Eigen::Matrix<Scalar, 4, 4>, NJoints>& pre_joint) noexcept
    {
        const Eigen::Matrix<Scalar, 3, 1> p_ee =
            cumulative[NJoints].template block<3, 1>(0, 3);
        Eigen::Matrix<Scalar, 3, NJoints> J;
        for(int i = 0; i < NJoints; ++i)
        {
            const Eigen::Matrix<Scalar, 3, 1> z_axis =
                pre_joint[i].template block<3, 1>(0, 2);
            const Eigen::Matrix<Scalar, 3, 1> p_pre =
                pre_joint[i].template block<3, 1>(0, 3);
            const Eigen::Matrix<Scalar, 3, 1> delta = p_ee - p_pre;
            J(0, i) = z_axis[1] * delta[2] - z_axis[2] * delta[1];
            J(1, i) = z_axis[2] * delta[0] - z_axis[0] * delta[2];
            J(2, i) = z_axis[0] * delta[1] - z_axis[1] * delta[0];
        }
        return J;
    }

    // ---- Public API -----------------------------------------------------

    [[nodiscard]] int dimension() const { return problem_dimension; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] Scalar value(const Eigen::Vector<Scalar, problem_dimension>& z) const
    {
        const auto p_targets = targets();
        Scalar acc = Scalar(0);
        std::array<Eigen::Matrix<Scalar, 4, 4>, NJoints + 1> cumulative;
        std::array<Eigen::Matrix<Scalar, 4, 4>, NJoints> pre_joint;
        Eigen::Matrix<Scalar, NJoints, 1> theta_k;
        for(int k = 0; k < K; ++k)
        {
            const int off = k * NJoints;
            for(int j = 0; j < NJoints; ++j)
                theta_k[j] = z[off + j];
            const auto T_ee = forward_kinematics(theta_k, cumulative, pre_joint);
            const Eigen::Matrix<Scalar, 3, 1> p_ee =
                T_ee.template block<3, 1>(0, 3);
            const Eigen::Matrix<Scalar, 3, 1> r = p_ee - p_targets[k];
            acc += r.squaredNorm();
        }
        return acc;
    }

    void gradient(const Eigen::Vector<Scalar, problem_dimension>& z,
                  Eigen::Vector<Scalar, problem_dimension>& g) const
    {
        const auto p_targets = targets();
        g.setZero();
        std::array<Eigen::Matrix<Scalar, 4, 4>, NJoints + 1> cumulative;
        std::array<Eigen::Matrix<Scalar, 4, 4>, NJoints> pre_joint;
        Eigen::Matrix<Scalar, NJoints, 1> theta_k;
        for(int k = 0; k < K; ++k)
        {
            const int off = k * NJoints;
            for(int j = 0; j < NJoints; ++j)
                theta_k[j] = z[off + j];
            (void)forward_kinematics(theta_k, cumulative, pre_joint);
            const Eigen::Matrix<Scalar, 3, 1> p_ee =
                cumulative[NJoints].template block<3, 1>(0, 3);
            const Eigen::Matrix<Scalar, 3, 1> r = p_ee - p_targets[k];
            const auto J = position_jacobian(cumulative, pre_joint);
            // d(||r||^2) / d(theta_k) = 2 * J^T * r.
            const Eigen::Matrix<Scalar, NJoints, 1> grad_k =
                Scalar(2) * (J.transpose() * r);
            for(int j = 0; j < NJoints; ++j)
                g[off + j] = grad_k[j];
        }
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> lower_bounds() const
    {
        constexpr Scalar pi = std::numbers::pi_v<Scalar>;
        Eigen::Vector<Scalar, problem_dimension> lb;
        lb.setConstant(-pi);
        return lb;
    }

    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> upper_bounds() const
    {
        constexpr Scalar pi = std::numbers::pi_v<Scalar>;
        Eigen::Vector<Scalar, problem_dimension> ub;
        ub.setConstant(pi);
        return ub;
    }

    // initial_point: deterministic per-target perturbation of the home
    // configuration. No runtime RNG so the bench cells are
    // reproducible. The offset scale (0.1 rad) is small enough to keep
    // the start well inside the joint box but large enough to avoid
    // exact-zero starting gradients across all K targets.
    [[nodiscard]] Eigen::Vector<Scalar, problem_dimension> initial_point() const
    {
        const auto home = theta_home();
        Eigen::Vector<Scalar, problem_dimension> z;
        for(int k = 0; k < K; ++k)
        {
            const int off = k * NJoints;
            for(int j = 0; j < NJoints; ++j)
            {
                // Deterministic interleaved offset; varies with both
                // target index k and joint index j.
                const Scalar offset =
                    Scalar(0.1) * std::sin(Scalar(7) * Scalar(k)
                                            + Scalar(3) * Scalar(j) + Scalar(1));
                z[off + j] = home[j] + offset;
            }
        }
        return z;
    }

    // optimal_value: bench-good-enough reference. The targets are
    // chosen inside the manipulator's reachable workspace, so the
    // least-squares optimum is zero (up to floating-point rounding).
    [[nodiscard]] Scalar optimal_value() const { return Scalar(0); }
};

}

#endif
