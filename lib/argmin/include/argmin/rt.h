#ifndef HPP_GUARD_ARGMIN_RT_H
#define HPP_GUARD_ARGMIN_RT_H

#include "argmin/expected.h"
#include "argmin/solver/stepper.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/formulation/concepts.h"

// Real-time-safe aggregator.
//
// rt.h re-exports the wall-clock-free subset of the solver surface for callers
// that want a single include naming exactly the pieces safe to drive on a hard
// real-time path: the iteration-only step_budget_solver driver, the passive
// caller-scheduled stepper primitive, and the throw-free argmin::expected
// result type. RT-safety is a property, not a location: this is a thin umbrella
// over the plain argmin/ headers, NOT a segregated header tree -- every type it
// names is the same type the ordinary includes provide, and nothing here is
// unavailable elsewhere. The one guarantee is negative: neither this header nor
// any header it pulls in includes <chrono>, so a translation unit that budgets
// purely by iterations (or hand-drives a stepper) never transitively acquires
// the wall clock. The clock-bearing time_budget_solver /
// step_and_time_budget_solver drivers are deliberately excluded.
//
// See docs/rt-safety-matrix.md for the per-facility RT-safety column this
// umbrella tracks (exceptions-off, RTTI-off, allocation, wall-clock).

namespace argmin
{

// The concept boundary this umbrella promises -- the step-budget driver owns a
// convergence loop and satisfies the loop-owning nlp_solver refinement, while
// the passive stepper satisfies only the core steppable single-step surface --
// is pinned by definition-site static_asserts inside step_budget_solver.h and
// stepper.h themselves. Those hold for every caller, so this umbrella need only
// re-export the surface; the guarantee travels with the types, not this header.

}

#endif
