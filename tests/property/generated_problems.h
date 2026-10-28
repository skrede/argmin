#ifndef HPP_GUARD_ARGMIN_TESTS_PROPERTY_GENERATED_PROBLEMS_H
#define HPP_GUARD_ARGMIN_TESTS_PROPERTY_GENERATED_PROBLEMS_H

#include "generated_values.h"

#include "cases/linear_quadratic_problems.h"

#include <Eigen/Core>

#include <rapidcheck.h>

#include <cstdint>
#include <ostream>

namespace argmin::property
{

inline cases::convex_quadratic pick_convex_quadratic(Eigen::Index n)
{
    return cases::convex_quadratic{pick_spd(n), pick_vector(n), pick_vector(n)};
}

// The box surrounds a point that the rows are later built around, and the start lies inside the box.
inline cases::boxed_quadratic pick_boxed_quadratic(Eigen::Index n, const Eigen::VectorXd& feasible)
{
    cases::boxed_quadratic p{pick_convex_quadratic(n), Eigen::VectorXd(n), Eigen::VectorXd(n)};
    for(Eigen::Index i = 0; i < n; ++i)
    {
        p.lower[i] = feasible[i] - 0.1 - pick_value(0, 1);
        p.upper[i] = feasible[i] + 0.1 + pick_value(0, 1);
        p.start[i] = p.lower[i] + 0.5 * (pick_value(-1, 1) + 1.0) * (p.upper[i] - p.lower[i]);
    }
    return p;
}

enum class start_kind : std::uint8_t
{
    in_box,
    feasible
};

// Inequality rows hold at the feasible point with a slack in [0, 1], so some start active there.
inline cases::linearly_constrained_quadratic pick_linearly_constrained(Eigen::Index n, Eigen::Index equalities, Eigen::Index inequalities, start_kind start)
{
    const Eigen::VectorXd feasible = pick_vector(n);
    cases::linearly_constrained_quadratic p{pick_boxed_quadratic(n, feasible), pick_matrix(equalities, n), Eigen::VectorXd(), pick_matrix(inequalities, n), Eigen::VectorXd(inequalities)};
    p.e = p.E * feasible;
    for(Eigen::Index i = 0; i < inequalities; ++i)
        p.h[i] = p.G.row(i).dot(feasible) - pick_value(0, 1);
    if(start == start_kind::feasible)
        p.start = feasible;
    return p;
}

inline std::int32_t below(Eigen::Index n)
{
    return static_cast<std::int32_t>(n) - 1;
}

inline rc::Gen<cases::convex_quadratic> unconstrained_quadratic()
{
    return rc::gen::exec([] { return pick_convex_quadratic(pick_size(1, 6)); });
}

inline rc::Gen<cases::boxed_quadratic> bound_quadratic()
{
    return rc::gen::exec(
        []
        {
            const Eigen::Index n = pick_size(1, 6);
            return pick_boxed_quadratic(n, pick_vector(n));
        });
}

inline rc::Gen<cases::linearly_constrained_quadratic> rowless_bound_quadratic()
{
    return rc::gen::exec([] { return pick_linearly_constrained(pick_size(1, 6), 0, 0, start_kind::in_box); });
}

inline rc::Gen<cases::linearly_constrained_quadratic> inequality_quadratic(start_kind start = start_kind::in_box)
{
    return rc::gen::exec([start] { return pick_linearly_constrained(pick_size(1, 6), 0, pick_size(1, 4), start); });
}

// At most n - 1 equality rows leave the feasible set more than a point.
inline rc::Gen<cases::linearly_constrained_quadratic> equality_quadratic()
{
    return rc::gen::exec(
        []
        {
            const Eigen::Index n = pick_size(2, 6);
            return pick_linearly_constrained(n, pick_size(1, below(n)), 0, start_kind::in_box);
        });
}

inline rc::Gen<cases::linearly_constrained_quadratic> mixed_quadratic()
{
    return rc::gen::exec(
        []
        {
            const Eigen::Index n = pick_size(2, 6);
            return pick_linearly_constrained(n, pick_size(1, below(n)), pick_size(1, 4), start_kind::in_box);
        });
}

}

namespace argmin::cases
{

inline void showValue(const convex_quadratic& p, std::ostream& os)
{
    const Eigen::IOFormat full(Eigen::FullPrecision);
    os << "Q\n" << p.Q.format(full) << "\nc " << p.c.transpose().format(full) << "\nstart " << p.start.transpose().format(full);
}

inline void showValue(const boxed_quadratic& p, std::ostream& os)
{
    const Eigen::IOFormat full(Eigen::FullPrecision);
    showValue(static_cast<const convex_quadratic&>(p), os);
    os << "\nlower " << p.lower.transpose().format(full) << "\nupper " << p.upper.transpose().format(full);
}

inline void showValue(const linearly_constrained_quadratic& p, std::ostream& os)
{
    const Eigen::IOFormat full(Eigen::FullPrecision);
    showValue(static_cast<const boxed_quadratic&>(p), os);
    os << "\nE\n" << p.E.format(full) << "\ne " << p.e.transpose().format(full) << "\nG\n" << p.G.format(full) << "\nh " << p.h.transpose().format(full);
}

}

#endif
