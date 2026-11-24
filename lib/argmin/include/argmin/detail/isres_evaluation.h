#ifndef HPP_GUARD_ARGMIN_DETAIL_ISRES_EVALUATION_H
#define HPP_GUARD_ARGMIN_DETAIL_ISRES_EVALUATION_H

#include "argmin/detail/isres_operators.h"
#include "argmin/detail/evaluation_count.h"

#include <Eigen/Core>

#include <algorithm>

namespace argmin::detail
{

template <typename State>
void evaluate_population(State& s)
{
    using vector_type = decltype(State::x);
    using scalar_type = typename vector_type::Scalar;
    const int n_c = s.n_eq + s.n_ineq;
    for(int j = 0; j < s.lambda; ++j)
    {
        const vector_type xi = s.population.col(j);
        s.fitnesses[j] = counted_value<scalar_type>(s.evaluations, *s.problem, xi);
        if(n_c > 0)
        {
            Eigen::VectorX<scalar_type> c(n_c);
            s.problem->constraints(xi, c);
            s.all_constraints_buf.col(j) = c;
        }
    }
}

// L2-squared violation aggregation (NLopt isres.c:151,163): an equality row adds c^2, an inequality row, feasible
// at c >= 0, adds min(c, 0)^2.
template <typename State>
void squared_violations(State& s)
{
    using scalar_type = typename decltype(State::x)::Scalar;
    for(int j = 0; j < s.lambda; ++j)
    {
        scalar_type v = scalar_type(0.0);
        for(int i = 0; i < s.n_eq; ++i)
        {
            const scalar_type ce = s.all_constraints_buf(i, j);
            v += ce * ce;
        }
        for(int i = 0; i < s.n_ineq; ++i)
        {
            const scalar_type slack = inequality_slack(s.all_constraints_buf(s.n_eq + i, j));
            v += slack * slack;
        }
        s.violations[j] = v;
    }
}

}

#endif
