#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_ORACLE_KKT_POINT_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_ORACLE_KKT_POINT_H

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <limits>

namespace argmin::oracle
{

template <typename Scalar>
struct kkt_point
{
    Eigen::Index n_eq;
    Eigen::VectorX<Scalar> c;
    Eigen::VectorX<Scalar> g;
    Eigen::VectorX<Scalar> x;
    Eigen::VectorX<Scalar> lower;
    Eigen::VectorX<Scalar> upper;
    Eigen::MatrixX<Scalar> jacobian;
};

template <typename Problem, typename Scalar>
using problem_vector = Eigen::Vector<Scalar, problem_dimension_v<Problem>>;

// A least-squares problem without gradient() takes f = 0.5 r^T r, the library's convention, so its
// gradient is J^T r.
template <typename Scalar, typename Problem>
Eigen::VectorX<Scalar> objective_gradient(const Problem& problem, const problem_vector<Problem, Scalar>& x)
{
    static_assert(differentiable<Problem, Scalar> || least_squares<Problem, Scalar>, "the oracle needs a gradient or a residual Jacobian");
    if constexpr(differentiable<Problem, Scalar>)
    {
        problem_vector<Problem, Scalar> g = problem_vector<Problem, Scalar>::Zero(x.size());
        problem.gradient(x, g);
        return g;
    }
    else
    {
        Eigen::VectorX<Scalar> r = Eigen::VectorX<Scalar>::Zero(problem.num_residuals());
        Eigen::MatrixX<Scalar> J = Eigen::MatrixX<Scalar>::Zero(problem.num_residuals(), x.size());
        problem.residuals(x, r);
        problem.jacobian(x, J);
        return J.transpose() * r;
    }
}

template <typename Scalar, typename Problem>
void evaluate_constraints(const Problem& problem, const problem_vector<Problem, Scalar>& x, kkt_point<Scalar>& pt)
{
    static_assert(constrained<Problem, Scalar> || !constrained_values<Problem, Scalar>, "constraint values without a Jacobian cannot be judged");
    if constexpr(constrained<Problem, Scalar>)
    {
        pt.n_eq = problem.num_equality();
        const Eigen::Index m = pt.n_eq + problem.num_inequality();
        pt.c = Eigen::VectorX<Scalar>::Zero(m);
        pt.jacobian = Eigen::MatrixX<Scalar>::Zero(m, x.size());
        problem.constraints(x, pt.c);
        problem.constraint_jacobian(x, pt.jacobian);
    }
    else
    {
        pt.n_eq = 0;
        pt.c.resize(0);
        pt.jacobian.resize(0, x.size());
    }
}

template <typename Scalar, typename Problem>
void evaluate_bounds(const Problem& problem, Eigen::Index n, kkt_point<Scalar>& pt)
{
    if constexpr(bound_constrained<Problem, Scalar>)
    {
        pt.lower = problem.lower_bounds();
        pt.upper = problem.upper_bounds();
    }
    else
    {
        pt.lower = Eigen::VectorX<Scalar>::Constant(n, -std::numeric_limits<Scalar>::infinity());
        pt.upper = Eigen::VectorX<Scalar>::Constant(n, std::numeric_limits<Scalar>::infinity());
    }
}

template <typename Problem, typename Scalar, int N>
kkt_point<Scalar> evaluate_point(const Problem& problem, const Eigen::Vector<Scalar, N>& x)
{
    const problem_vector<Problem, Scalar> xp = x;
    kkt_point<Scalar> pt{.n_eq = 0, .c = {}, .g = objective_gradient<Scalar>(problem, xp), .x = xp, .lower = {}, .upper = {}, .jacobian = {}};
    evaluate_constraints<Scalar>(problem, xp, pt);
    evaluate_bounds<Scalar>(problem, xp.size(), pt);
    return pt;
}

}

#endif
