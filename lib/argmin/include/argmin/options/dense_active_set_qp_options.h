#ifndef HPP_GUARD_ARGMIN_OPTIONS_DENSE_ACTIVE_SET_QP_OPTIONS_H
#define HPP_GUARD_ARGMIN_OPTIONS_DENSE_ACTIVE_SET_QP_OPTIONS_H

#include "argmin/types.h"

#include <cstdint>

namespace argmin
{

template <scalar S>
struct active_set_qp_defaults;

template <>
struct active_set_qp_defaults<double>
{
    static constexpr double tolerance = 1e-6;
    static constexpr double equality_tolerance = 1e-9;
    static constexpr double feas_tol = 1e-7;
};

template <>
struct active_set_qp_defaults<float>
{
    // A precision-class value keeps the double row's fraction of the available precision:
    // t_f = eps_f ^ (ln t_d / ln eps_d), rounded to one significant figure.
    static constexpr float tolerance = 2e-3f;           // rule
    static constexpr float equality_tolerance = 1e-4f;  // rule
    static constexpr float feas_tol = 8e-4f;            // rule
};

// Tuning knobs for the dense active-set QP solver.
//
// The roster holds only what this solver honors. The operator-splitting
// solvers' penalty, relaxation, scaling, polish and warm-start knobs have no
// meaning for an active-set method, so they are absent rather than accepted
// and ignored.
//
// equality_tolerance is the relative width under which a two-sided bound row
// l <= a^T x <= u is treated as the equality a^T x = l; it is scaled by
// max(1, |l|, |u|), the same rule the operator-splitting solvers apply.
//
// The defaults are the values this solver received before it had its own
// options type: max_iterations and tolerance from a default-constructed
// dense_qp_options (its max_iterations and eps_abs), and the equality width it
// previously held as a fixed constant. A caller who sets nothing sees no change.
template <scalar Scalar = double>
struct dense_active_set_qp_options
{
    std::uint32_t max_iterations{4000};
    Scalar tolerance{active_set_qp_defaults<Scalar>::tolerance};
    Scalar equality_tolerance{active_set_qp_defaults<Scalar>::equality_tolerance};
};

}

#endif
