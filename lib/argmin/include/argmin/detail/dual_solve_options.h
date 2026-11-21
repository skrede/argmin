#ifndef HPP_GUARD_ARGMIN_DETAIL_DUAL_SOLVE_OPTIONS_H
#define HPP_GUARD_ARGMIN_DETAIL_DUAL_SOLVE_OPTIONS_H

#include "argmin/solver/options.h"
#include "argmin/solver/convergence.h"

#include "argmin/options/mma_defaults.h"

namespace argmin
{
namespace detail
{

// The moving-asymptote dual loop applies its own stopping tests and these options only reach the dual policy's init,
// so they carry no stall criterion and none of its history buffer.
template <scalar S>
using dual_convergence = convergence_policy<gradient_tolerance_criterion<S>, step_tolerance_criterion<S>>;

template <scalar S>
solver_options<dual_convergence<S>> dual_solve_options()
{
    solver_options<dual_convergence<S>> opts;
    opts.max_iterations = 100;
    opts.set_gradient_threshold(mma_defaults<S>::dual_gradient_threshold);
    opts.set_step_threshold(mma_defaults<S>::dual_step_threshold);
    return opts;
}

}
}

#endif
