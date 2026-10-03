#ifndef HPP_GUARD_ARGMIN_OPTIONS_DENSE_ACTIVE_SET_QP_OPTIONS_H
#define HPP_GUARD_ARGMIN_OPTIONS_DENSE_ACTIVE_SET_QP_OPTIONS_H

#include <cstdint>

namespace argmin
{

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
struct dense_active_set_qp_options
{
    std::uint16_t max_iterations{4000};
    double tolerance{1e-6};
    double equality_tolerance{1e-9};
};

}

#endif
