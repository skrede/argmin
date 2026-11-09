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
    Eigen::VectorX<Scalar> c_all;
    Eigen::MatrixX<Scalar> J_all;
    Eigen::VectorX<Scalar> c_ineq;
    Eigen::MatrixX<Scalar> J_ineq;
    std::vector<Scalar> h_history;
    Eigen::VectorX<Scalar> sign_eq;
    Eigen::Vector<Scalar, N> descent;
    Eigen::VectorX<Scalar> sign_ineq;

    void resize(Eigen::Index n, Eigen::Index n_eq, Eigen::Index n_ineq, std::uint16_t max_steps)
    {
        c_eq.resize(n_eq);
        J_eq.resize(n_eq, n);
        c_all.resize(n_eq + n_ineq);
        J_all.resize(n_eq + n_ineq, n);
        c_ineq.resize(n_ineq);
        J_ineq.resize(n_ineq, n);
        h_history.reserve(max_steps);
        sign_eq.resize(n_eq);
        descent.resize(n);
        sign_ineq.resize(n_ineq);
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
