#ifndef HPP_GUARD_ARGMIN_TESTS_PROPERTY_GENERATED_LEAST_SQUARES_H
#define HPP_GUARD_ARGMIN_TESTS_PROPERTY_GENERATED_LEAST_SQUARES_H

#include "generated_values.h"

#include "cases/linear_quadratic_problems.h"

#include <Eigen/QR>
#include <Eigen/Core>

#include <rapidcheck.h>

#include <limits>
#include <cstdint>
#include <ostream>

namespace argmin::property
{

enum class box_kind : std::uint8_t
{
    infinite,
    around_start,
    around_minimizer
};

// The 0.5 I block below the random rows gives M full column rank, so the minimizer is unique. A box
// around the minimizer never binds at the solution.
inline cases::linear_least_squares pick_least_squares(box_kind box)
{
    constexpr double inf = std::numeric_limits<double>::infinity();
    const Eigen::Index n = pick_size(1, 6);
    const Eigen::Index k = pick_size(0, 3);
    cases::linear_least_squares p{Eigen::MatrixXd(k + n, n), 2.0 * pick_vector(k + n), pick_vector(n), Eigen::VectorXd::Constant(n, -inf), Eigen::VectorXd::Constant(n, inf)};
    p.M << pick_matrix(k, n), 0.5 * Eigen::MatrixXd::Identity(n, n);
    if(box == box_kind::infinite)
        return p;
    const Eigen::VectorXd center = box == box_kind::around_start ? p.start : Eigen::VectorXd(p.M.colPivHouseholderQr().solve(p.b));
    for(Eigen::Index i = 0; i < n; ++i)
    {
        p.lower[i] = center[i] - 0.1 - pick_value(0, 1);
        p.upper[i] = center[i] + 0.1 + pick_value(0, 1);
        p.start[i] = box == box_kind::around_start ? p.start[i] : p.lower[i] + 0.5 * (pick_value(-1, 1) + 1.0) * (p.upper[i] - p.lower[i]);
    }
    return p;
}

inline rc::Gen<cases::unboxed_least_squares> unconstrained_least_squares()
{
    return rc::gen::exec([] { return cases::unboxed_least_squares{pick_least_squares(box_kind::infinite)}; });
}

inline rc::Gen<cases::linear_least_squares> boxed_least_squares(box_kind box)
{
    return rc::gen::exec([box] { return pick_least_squares(box); });
}

}

namespace argmin::cases
{

inline void showValue(const linear_least_squares& p, std::ostream& os)
{
    const Eigen::IOFormat full(Eigen::FullPrecision);
    os << "M\n" << p.M.format(full) << "\nb " << p.b.transpose().format(full) << "\nstart " << p.start.transpose().format(full);
    os << "\nlower " << p.lower.transpose().format(full) << "\nupper " << p.upper.transpose().format(full);
}

inline void showValue(const unboxed_least_squares& p, std::ostream& os)
{
    showValue(p.p, os);
}

}

#endif
