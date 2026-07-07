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

// A representative instantiation pins the concept boundary this umbrella
// promises: the step-budget driver owns a convergence loop and satisfies the
// loop-owning nlp_solver refinement, while the passive stepper satisfies only
// the core steppable single-step surface. If a future edit accidentally gave
// the stepper a solve()/step_n() loop (or stripped one from the driver), one of
// these fires at the aggregation site rather than silently downstream.
namespace detail
{
struct rt_probe_policy
{
    using scalar_type = double;
    struct state_type { Eigen::VectorXd x; };

    template <typename Problem, typename Convergence = default_convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0,
                    const solver_options<Convergence>&)
    {
        return state_type{.x = x0};
    }

    step_result<double> step(state_type&) { return {}; }
    void reset(state_type& s, const Eigen::VectorXd& x0) { s.x = x0; }
    void reset_clear(state_type& s, const Eigen::VectorXd& x0) { s.x = x0; }
};

static_assert(steppable<step_budget_solver<rt_probe_policy>>,
              "step_budget_solver must satisfy steppable");
static_assert(nlp_solver<step_budget_solver<rt_probe_policy>>,
              "step_budget_solver owns a convergence loop and must satisfy "
              "nlp_solver");
static_assert(steppable<stepper<rt_probe_policy>>,
              "stepper must satisfy the passive steppable surface");
static_assert(!nlp_solver<stepper<rt_probe_policy>>,
              "stepper is a passive step primitive and must NOT satisfy the "
              "loop-owning nlp_solver refinement");
}

}

#endif
