#ifndef HPP_GUARD_ARGMIN_DETAIL_RUNNER_CONCEPT_PROBE_H
#define HPP_GUARD_ARGMIN_DETAIL_RUNNER_CONCEPT_PROBE_H

#include "argmin/result/step_result.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/options.h"

#include <Eigen/Core>

namespace argmin
{
namespace detail
{

// Minimal witness policy for pinning runner concept membership at each runner's
// definition site. It models solver_policy just enough to instantiate a runner
// facade over solver_core, so a runner header can static_assert its own concept
// conformance -- nlp_solver for the budget-owning drivers, steppable-but-not-
// nlp_solver for the passive stepper -- against a concrete type without forcing
// a dependency on any real policy into the policy-generic runner header.
//
// The method bodies exist only to satisfy the solver_policy surface at compile
// time; nothing ever calls them at runtime, so they are excluded from coverage
// as provably unreachable -- a concept witness, not a code path.
struct runner_concept_probe_policy
{
    using scalar_type = double;
    struct state_type { Eigen::VectorXd x; };

    // LCOV_EXCL_START
    template <typename Problem, typename Convergence = default_convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0,
                    const solver_options<Convergence>&)
    {
        return state_type{.x = x0};
    }

    step_result<double> step(state_type&) { return {}; }
    void reset(state_type& s, const Eigen::VectorXd& x0) { s.x = x0; }
    void reset_clear(state_type& s, const Eigen::VectorXd& x0) { s.x = x0; }
    // LCOV_EXCL_STOP
};

}
}

#endif
