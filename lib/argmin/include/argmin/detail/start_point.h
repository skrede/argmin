#ifndef HPP_GUARD_ARGMIN_DETAIL_START_POINT_H
#define HPP_GUARD_ARGMIN_DETAIL_START_POINT_H

#include "argmin/types.h"

#include <Eigen/Core>

namespace argmin::detail
{

template <int N, typename Derived>
inline constexpr bool x0_shape_ok = (Derived::ColsAtCompileTime == 1) && (N == dynamic_dimension || Derived::RowsAtCompileTime == Eigen::Dynamic || Derived::RowsAtCompileTime == N);

// Eigen checks a Ref binding's length only by eigen_assert, so a dynamic-rows start is measured before it binds.
template <int N, typename Derived>
bool start_length_ok(const Eigen::MatrixBase<Derived>& x0, Eigen::Index n)
{
    static_assert(x0_shape_ok<N, Derived>,
                  "x0 must be a column vector matching the solver dimension "
                  "N (an N x 1 Eigen vector for fixed N; any n x 1 vector "
                  "for dynamic_dimension).");
    if constexpr(N != dynamic_dimension && Derived::RowsAtCompileTime == N)
        return true;
    else
        return x0.size() == n;
}

// A problem that reports no dimension leaves a dynamic-N start's length unconstrained.
template <int N, typename Problem>
Eigen::Index problem_start_dimension(const Problem& problem, Eigen::Index start_length)
{
    if constexpr(N != dynamic_dimension)
        return N;
    else if constexpr(requires { problem.dimension(); })
        return problem.dimension();
    else
        return start_length;
}

// A dynamic-N state keeps the dimension it was built with, so once the problem reports another one no start matches.
template <int N, typename Problem, typename State>
Eigen::Index start_dimension(const Problem* problem, const State& s)
{
    if constexpr(N != dynamic_dimension)
        return N;
    else if constexpr(requires { problem->dimension(); })
        return problem->dimension() == s.x.size() ? s.x.size() : Eigen::Index{-1};
    else
        return s.x.size();
}

}

#endif
