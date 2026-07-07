#ifndef HPP_GUARD_ARGMIN_SOLVER_BASIC_SOLVER_H
#define HPP_GUARD_ARGMIN_SOLVER_BASIC_SOLVER_H

#include "argmin/types.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/step_budget_solver.h"

namespace argmin
{

// Transitional alias retained while the tree migrates to the named budget
// drivers. basic_solver is exactly the step-budget driver: iteration-only
// budgeting, wall-clock-free. Call sites that want a wall-clock deadline use
// time_budget_solver / step_and_time_budget_solver directly. This alias is
// intra-phase scaffolding: it keeps the many unmigrated call sites compiling
// through the split and is removed once the call sites adopt the driver names
// directly.
//
// Alias-template CTAD (C++20) forwards to step_budget_solver's deduction
// guides, so `basic_solver{policy, problem, x0, opts}` still deduces N /
// Problem / Convergence from the arguments.

template <typename Policy, int N = dynamic_dimension, typename Problem = void,
          typename Convergence = default_convergence>
using basic_solver = step_budget_solver<Policy, N, Problem, Convergence>;

}

#endif
