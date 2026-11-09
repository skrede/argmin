#ifndef HPP_GUARD_ARGMIN_DETAIL_FILTER_RESTORATION_WORKSPACE_H
#define HPP_GUARD_ARGMIN_DETAIL_FILTER_RESTORATION_WORKSPACE_H

#include <Eigen/Core>

#include <vector>
#include <cstdint>

namespace argmin::detail
{

// Sized once for a problem instance, so a restoration at a fixed N allocates nothing. The constraint-axis buffers stay
// dynamic because the problem's constraint callbacks take dynamic vectors and matrices; a buffer of the wrong shape
// is resized on first use.
template <typename Scalar, int N>
struct filter_restoration_workspace
{
    Eigen::VectorX<Scalar> c_eq;
    Eigen::MatrixX<Scalar> J_eq;
    Eigen::VectorX<Scalar> b_eq;
    Eigen::VectorX<Scalar> c_all;
    Eigen::MatrixX<Scalar> J_all;
    Eigen::Vector<Scalar, N> zero;
    Eigen::VectorX<Scalar> c_ineq;
    Eigen::MatrixX<Scalar> J_ineq;
    Eigen::VectorX<Scalar> b_ineq;
    std::vector<Scalar> h_history;
    Eigen::VectorX<Scalar> sign_eq;
    Eigen::VectorX<Scalar> sign_ineq;
    Eigen::Vector<Scalar, N> descent;
    Eigen::Vector<Scalar, N> step_lower;
    Eigen::Vector<Scalar, N> step_upper;
    Eigen::Matrix<Scalar, N, N> identity;
    Eigen::Matrix<Scalar, Eigen::Dynamic, N> A_eq;
    Eigen::Matrix<Scalar, Eigen::Dynamic, N> A_ineq;

    // The feasibility-QP leg writes its step into the policy's own QP result, sized here with the rest.
    template <typename QpResult>
    void resize(Eigen::Index n, Eigen::Index n_eq, Eigen::Index n_ineq, std::uint16_t max_steps, QpResult& qp)
    {
        c_eq.resize(n_eq);
        J_eq.resize(n_eq, n);
        b_eq.resize(n_eq);
        c_all.resize(n_eq + n_ineq);
        J_all.resize(n_eq + n_ineq, n);
        zero.setZero(n);
        c_ineq.resize(n_ineq);
        J_ineq.resize(n_ineq, n);
        b_ineq.resize(n_ineq);
        h_history.reserve(max_steps);
        sign_eq.resize(n_eq);
        sign_ineq.resize(n_ineq);
        descent.resize(n);
        step_lower.resize(n);
        step_upper.resize(n);
        identity.setIdentity(n, n);
        A_eq.resize(n_eq, n);
        A_ineq.resize(n_ineq, n);
        qp.x.setZero(n);
        qp.lambda.setZero(n_eq + n_ineq);
    }
};

// Presents a problem's stacked constraint rows to the restoration leaves as separate equality and inequality rows.
template <typename Scalar, int N, typename Problem>
struct restoration_adapter
{
    Eigen::Index n_eq;
    Eigen::Index n_ineq;
    const Problem& problem;
    Eigen::VectorX<Scalar>& c_all;
    Eigen::MatrixX<Scalar>& J_all;

    void eval_constraints(const Eigen::Vector<Scalar, N>& x, Eigen::VectorX<Scalar>& c_eq, Eigen::VectorX<Scalar>& c_ineq) const
    {
        c_all.resize(n_eq + n_ineq);
        problem.constraints(x, c_all);
        c_eq = c_all.head(n_eq);
        c_ineq = c_all.tail(n_ineq);
    }

    void eval_constraint_jacobians(const Eigen::Vector<Scalar, N>& x, Eigen::MatrixX<Scalar>& J_eq, Eigen::MatrixX<Scalar>& J_ineq) const
    {
        J_all.resize(n_eq + n_ineq, x.size());
        problem.constraint_jacobian(x, J_all);
        J_eq = J_all.topRows(n_eq);
        J_ineq = J_all.bottomRows(n_ineq);
    }
};

}

#endif
