#ifndef HPP_GUARD_NABLAPP_SOLVER_OPTIONS_H
#define HPP_GUARD_NABLAPP_SOLVER_OPTIONS_H

namespace nablapp
{

// Base convergence tolerances for all solvers (per CORE-06).
//
// Individual solver policies may extend this or define their own options
// that include these fields. basic_solver uses these to decide when to
// stop iterating.

template <typename Scalar = double>
struct solver_options
{
    int max_iterations{1000};
    Scalar gradient_tolerance{Scalar(1e-8)};
    Scalar objective_tolerance{Scalar(1e-12)};
    Scalar step_tolerance{Scalar(1e-12)};
    Scalar constraint_tolerance{Scalar(1e-8)};
    int verbosity{0};
};

}

#endif
