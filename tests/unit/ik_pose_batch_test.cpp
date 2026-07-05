// Regression tests for the SE(3) pose-batch IK test problem.
// Covers dimension, problem-class flags, joint bound symmetry,
// central-difference gradient consistency at the initial point, and a
// domain-verification suite that checks the Denavit-Hartenberg forward
// kinematics and the geometric (cross-product) position Jacobian
// against independent recomputations.
//
// Reference: Craig, Introduction to Robotics, 3rd ed., Section 3.6
//            (modified Denavit-Hartenberg convention) and Section 5.5
//            (cross-product form of the geometric Jacobian); Lynch and
//            Park, Modern Robotics, 2017, Chapter 5 (space Jacobian:
//            for a revolute joint with unit axis omega_i through point
//            q_i, the end-effector linear velocity contribution is
//            omega_i x (p_ee - q_i)); Siciliano et al., Foundations of
//            Robotics, 2025 (geometric Jacobian, DH transform). The
//            cross-product column J_p,i = z_i x (p_ee - p_i) asserted
//            below is exactly the linear part of that space-Jacobian
//            column.
//
// Consumer cross-check (IK): cartan drives full 6-DOF pose IK. It builds
// forward kinematics from a Product-of-Exponentials chain of screw axes
// (cartan/serial/chain/screw_axis.h: revolute S = (omega, v) with
// v = -omega x point), computes a 6-row space/body Jacobian
// (cartan/serial/fk/jacobian.h: J_s,i = Ad_{T_{i-1}} S_i), and measures
// the pose error as the full SE(3) body twist V_b = log(T_target^{-1} *
// FK(q)) (cartan/serial/ik/...). The argmin fixture is a deliberately
// reduced task: position-only error ||p_ee - p_target||^2, DH forward
// kinematics instead of PoE, and the 3-row position Jacobian. Crucially
// the axis/sign convention agrees: cartan's revolute screw yields an
// end-effector linear velocity omega x (p_ee - point) = z_i x (p_ee -
// p_i), the same cross-product form (and sign) the argmin position
// Jacobian uses. So the parameterizations differ in task dimension
// (3-DOF position vs 6-DOF pose) and in the FK formalism (DH vs PoE) but
// share the linear-velocity Jacobian convention with cartan and Lynch &
// Park. This divergence is documented, not a defect; the FK and Jacobian
// forms themselves are asserted against independent recomputations here.

#include "argmin/test_functions/ik_pose_batch.h"
#include "argmin/test_functions/problem_class.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <numbers>

using namespace argmin;

TEST_CASE("ik_pose_batch dimensions and class flags", "[ik_pose_batch]")
{
    ik_pose_batch<6, 5> p{};
    REQUIRE(p.dimension() == 30);
    REQUIRE(p.num_equality() == 0);
    REQUIRE(p.num_inequality() == 0);

    REQUIRE(has_class(p.pclass, problem_class::application));
    REQUIRE(has_class(p.pclass, problem_class::bound_constrained));
    REQUIRE(!has_class(p.pclass, problem_class::equality));
}

TEST_CASE("ik_pose_batch bounds are symmetric, finite, and within +/- pi",
          "[ik_pose_batch]")
{
    ik_pose_batch<6, 5> p{};
    const auto lb = p.lower_bounds();
    const auto ub = p.upper_bounds();
    constexpr double pi = std::numbers::pi_v<double>;
    REQUIRE(lb.size() == 30);
    REQUIRE(ub.size() == 30);
    for(int i = 0; i < lb.size(); ++i)
    {
        REQUIRE(std::isfinite(lb[i]));
        REQUIRE(std::isfinite(ub[i]));
        REQUIRE(lb[i] == -ub[i]);
        REQUIRE(std::abs(lb[i]) <= pi);
        REQUIRE(std::abs(ub[i]) <= pi);
    }
}

TEST_CASE("ik_pose_batch gradient matches central-difference at the initial point",
          "[ik_pose_batch]")
{
    using prob_t = ik_pose_batch<6, 5>;
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

TEST_CASE("ik_pose_batch value is finite and non-negative at the initial point",
          "[ik_pose_batch]")
{
    ik_pose_batch<6, 5> p{};
    const auto z0 = p.initial_point();
    const double f0 = p.value(z0);
    REQUIRE(std::isfinite(f0));
    REQUIRE(f0 >= 0.0);
}

// ---- Domain verification ------------------------------------------------
//
// Independent building blocks: elementary homogeneous transforms and a
// from-scratch DH chain walk. These do not call ik_pose_batch's
// dh_transform / forward_kinematics helpers; they reassemble the SE(3)
// product Rot_x(alpha) Trans_x(a) Rot_z(theta+offset) Trans_z(d) from
// primitives so the header's composition and chain walk are checked
// against an independent implementation, not against themselves.

namespace
{

using prob_t = ik_pose_batch<6, 5>;
using mat4 = Eigen::Matrix<double, 4, 4>;
using vec3 = Eigen::Matrix<double, 3, 1>;

mat4 rot_x(double a)
{
    mat4 T = mat4::Identity();
    T(1, 1) = std::cos(a);  T(1, 2) = -std::sin(a);
    T(2, 1) = std::sin(a);  T(2, 2) = std::cos(a);
    return T;
}

mat4 rot_z(double t)
{
    mat4 T = mat4::Identity();
    T(0, 0) = std::cos(t);  T(0, 1) = -std::sin(t);
    T(1, 0) = std::sin(t);  T(1, 1) = std::cos(t);
    return T;
}

mat4 trans_x(double a)
{
    mat4 T = mat4::Identity();
    T(0, 3) = a;
    return T;
}

mat4 trans_z(double d)
{
    mat4 T = mat4::Identity();
    T(2, 3) = d;
    return T;
}

// Independent single modified-DH transform from primitives.
mat4 ref_dh(double a, double alpha, double d, double offset, double theta)
{
    return rot_x(alpha) * trans_x(a) * rot_z(theta + offset) * trans_z(d);
}

// Independent chain walk: end-effector transform and the per-joint
// pre-rotation frame (the frame whose z-axis joint i rotates about), so
// the cross-product Jacobian can be reconstructed from the same walk.
struct ref_fk_result
{
    mat4 T_ee;
    std::array<vec3, prob_t::joint_count> z_axis;
    std::array<vec3, prob_t::joint_count> p_pre;
};

ref_fk_result ref_forward_kinematics(
    const Eigen::Matrix<double, prob_t::joint_count, 1>& theta)
{
    const auto table = prob_t::dh_table();
    ref_fk_result out;
    mat4 cum = mat4::Identity();
    for(int i = 0; i < prob_t::joint_count; ++i)
    {
        // Frame after the fixed (alpha, a) prefix; joint i's z-axis
        // rotation acts here.
        const mat4 pre = cum * rot_x(table[i].alpha) * trans_x(table[i].a);
        out.z_axis[i] = pre.block<3, 1>(0, 2);
        out.p_pre[i] = pre.block<3, 1>(0, 3);
        cum = pre * rot_z(theta[i] + table[i].theta_offset) * trans_z(table[i].d);
    }
    out.T_ee = cum;
    return out;
}

// A few nontrivial joint configurations for the domain assertions.
std::array<Eigen::Matrix<double, prob_t::joint_count, 1>, 3> test_configs()
{
    std::array<Eigen::Matrix<double, prob_t::joint_count, 1>, 3> cfgs;
    cfgs[0] = prob_t::theta_home();
    for(int j = 0; j < prob_t::joint_count; ++j)
    {
        cfgs[1][j] = 0.2 * (j + 1) - 0.3;
        cfgs[2][j] = -0.15 * j + 0.4 * std::sin(1.3 * j);
    }
    return cfgs;
}

}  // namespace

TEST_CASE("ik_pose_batch DH forward kinematics matches an independent transform walk",
          "[ik_pose_batch][domain]")
{
    // Confirm the independent single-transform primitive equals the
    // header's dh_transform for one row (composition sanity), then check
    // full-chain p_ee at several configurations.
    const auto table = prob_t::dh_table();
    const mat4 T_header0 = prob_t::dh_transform(table[0], 0.37);
    const mat4 T_ref0 =
        ref_dh(table[0].a, table[0].alpha, table[0].d, table[0].theta_offset, 0.37);
    REQUIRE((T_header0 - T_ref0).cwiseAbs().maxCoeff() < 1e-13);

    std::array<mat4, prob_t::joint_count + 1> cumulative;
    std::array<mat4, prob_t::joint_count> pre_joint;
    for(const auto& theta : test_configs())
    {
        const mat4 T_ee_header =
            prob_t::forward_kinematics(theta, cumulative, pre_joint);
        const auto ref = ref_forward_kinematics(theta);
        const vec3 p_header = T_ee_header.block<3, 1>(0, 3);
        const vec3 p_ref = ref.T_ee.block<3, 1>(0, 3);
        REQUIRE((p_header - p_ref).cwiseAbs().maxCoeff() < 1e-12);
        // Full pose (rotation + position) agreement, not just position.
        REQUIRE((T_ee_header - ref.T_ee).cwiseAbs().maxCoeff() < 1e-12);
    }
}

TEST_CASE("ik_pose_batch geometric Jacobian matches a finite-difference Jacobian",
          "[ik_pose_batch][domain]")
{
    constexpr int NJ = prob_t::joint_count;
    std::array<mat4, NJ + 1> cumulative;
    std::array<mat4, NJ> pre_joint;
    const double h = 1e-6;

    for(const auto& theta : test_configs())
    {
        prob_t::forward_kinematics(theta, cumulative, pre_joint);
        const Eigen::Matrix<double, 3, NJ> J_header =
            prob_t::position_jacobian(cumulative, pre_joint);

        // Independent cross-product Jacobian from the reference walk:
        // J_p,i = z_i x (p_ee - p_pre_i).
        const auto ref = ref_forward_kinematics(theta);
        const vec3 p_ee = ref.T_ee.block<3, 1>(0, 3);
        Eigen::Matrix<double, 3, NJ> J_cross;
        for(int i = 0; i < NJ; ++i)
            J_cross.col(i) = ref.z_axis[i].cross(p_ee - ref.p_pre[i]);
        REQUIRE((J_header - J_cross).cwiseAbs().maxCoeff() < 1e-11);

        // Independent finite-difference Jacobian of p_ee wrt theta.
        Eigen::Matrix<double, 3, NJ> J_fd;
        for(int i = 0; i < NJ; ++i)
        {
            auto tp = theta;
            auto tm = theta;
            tp[i] += h;
            tm[i] -= h;
            const vec3 pp = ref_forward_kinematics(tp).T_ee.block<3, 1>(0, 3);
            const vec3 pm = ref_forward_kinematics(tm).T_ee.block<3, 1>(0, 3);
            J_fd.col(i) = (pp - pm) / (2.0 * h);
        }
        REQUIRE((J_header - J_fd).cwiseAbs().maxCoeff() < 1e-6);
    }
}

TEST_CASE("ik_pose_batch domain check fires under a seeded DH / axis perturbation",
          "[ik_pose_batch][domain][sensitivity]")
{
    // Sensitivity proof: the domain check must FAIL when the kinematics
    // is perturbed. The header's dh_table is constexpr, so we perturb a
    // local copy and confirm the FK and Jacobian checks that pass on the
    // true model no longer hold on the perturbed one.
    constexpr int NJ = prob_t::joint_count;
    const auto theta = prob_t::theta_home();

    std::array<mat4, NJ + 1> cumulative;
    std::array<mat4, NJ> pre_joint;
    const mat4 T_ee_header = prob_t::forward_kinematics(theta, cumulative, pre_joint);
    const vec3 p_header = T_ee_header.block<3, 1>(0, 3);

    // (a) Perturb one DH link length and rebuild the chain independently:
    // p_ee must diverge, so a wrong DH entry cannot pass the FK check.
    auto table = prob_t::dh_table();
    table[1].a += 5e-3;  // seeded single-parameter error
    mat4 cum = mat4::Identity();
    for(int i = 0; i < NJ; ++i)
    {
        cum = cum * rot_x(table[i].alpha) * trans_x(table[i].a)
                  * rot_z(theta[i] + table[i].theta_offset) * trans_z(table[i].d);
    }
    const vec3 p_perturbed = cum.block<3, 1>(0, 3);
    REQUIRE((p_header - p_perturbed).cwiseAbs().maxCoeff() > 1e-4);

    // (b) Flip one Jacobian column's axis sign and confirm the resulting
    // column no longer matches the header's Jacobian, so a sign error in
    // the cross-product form would be caught.
    const Eigen::Matrix<double, 3, NJ> J_header =
        prob_t::position_jacobian(cumulative, pre_joint);
    const auto ref = ref_forward_kinematics(theta);
    const vec3 p_ee = ref.T_ee.block<3, 1>(0, 3);
    Eigen::Matrix<double, 3, NJ> J_bad;
    for(int i = 0; i < NJ; ++i)
        J_bad.col(i) = ref.z_axis[i].cross(p_ee - ref.p_pre[i]);
    // Flip the sign of the largest-norm column so the seeded error is
    // guaranteed observable (a near-singular column would otherwise mask
    // a sign flip).
    int worst = 0;
    for(int i = 1; i < NJ; ++i)
        if(J_bad.col(i).norm() > J_bad.col(worst).norm())
            worst = i;
    J_bad.col(worst) = -J_bad.col(worst);  // seeded axis-sign error
    REQUIRE((J_header - J_bad).cwiseAbs().maxCoeff() > 1e-3);
}
