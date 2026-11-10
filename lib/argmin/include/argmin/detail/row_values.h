#ifndef HPP_GUARD_ARGMIN_DETAIL_ROW_VALUES_H
#define HPP_GUARD_ARGMIN_DETAIL_ROW_VALUES_H

#include "argmin/detail/lagrangian.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

namespace argmin::detail
{

// The constraint values at the iterate of a method that searches without its problem's constraint rows, so the
// iterate's violation is measured instead of reported as zero; measure_constraint_values reads c_eq and c_ineq.
template <typename Scalar>
struct row_values
{
    Eigen::VectorX<Scalar> c_eq;
    Eigen::VectorX<Scalar> c_all;
    Eigen::VectorX<Scalar> c_ineq;

    template <typename Problem, typename Point>
    void measure_rows(const Problem& problem, const Point& x)
    {
        if constexpr(constrained_values<Problem, Scalar>)
        {
            c_all.resize(problem.num_equality() + problem.num_inequality());
            problem.constraints(x, c_all);
            c_eq = c_all.head(problem.num_equality());
            c_ineq = c_all.tail(problem.num_inequality());
        }
    }

    Scalar row_violation() const { return primal_feasibility_inf(c_eq, c_ineq); }
};

}

#endif
