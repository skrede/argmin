#ifndef HPP_GUARD_ARGMIN_DETAIL_BOUND_MEASURE_H
#define HPP_GUARD_ARGMIN_DETAIL_BOUND_MEASURE_H

#include "argmin/detail/bound_projection.h"

#include "argmin/formulation/concepts.h"

#include "argmin/types.h"

#include <type_traits>

namespace argmin::detail
{

template <typename Problem, typename Scalar>
concept boxed_problem = !std::is_void_v<Problem> && bound_constrained<Problem, Scalar>;

template <typename State>
concept keeps_box = requires(const State& s) {
    s.lower;
    s.upper;
};

// The iterate's distance outside the box. A family that keeps the box holds the problem's bounds in its state, read
// here so a step calls no user code.
template <typename Scalar, int N, typename State, typename Problem>
class bound_measure
{
public:
    template <typename P>
    explicit bound_measure(const P&)
    {
    }

    template <typename P>
    void refresh(const P&)
    {
    }

    Scalar operator()(const State& s) const
    {
        if constexpr(boxed_problem<Problem, Scalar>)
            return box_violation_inf<Scalar, N>(s.x, s.lower, s.upper);
        else
            return Scalar(0);
    }
};

// A family that ignores the box is measured against a copy of the bounds taken when its state is built or reset: a
// problem returns its bounds by value, which allocates for a dynamic dimension or a dynamically typed box.
template <typename Scalar, int N, typename State, typename Problem>
    requires boxed_problem<Problem, Scalar> && (!keeps_box<State>)
class bound_measure<Scalar, N, State, Problem>
{
public:
    explicit bound_measure(const Problem& problem)
        : m_lower{problem.lower_bounds()}
        , m_upper{problem.upper_bounds()}
    {}

    void refresh(const Problem& problem)
    {
        m_lower = problem.lower_bounds();
        m_upper = problem.upper_bounds();
    }

    Scalar operator()(const State& s) const { return box_violation_inf<Scalar, N>(s.x, m_lower, m_upper); }

private:
    vector<Scalar, N> m_lower;
    vector<Scalar, N> m_upper;
};

}

#endif
