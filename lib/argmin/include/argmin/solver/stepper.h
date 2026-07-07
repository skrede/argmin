#ifndef HPP_GUARD_ARGMIN_SOLVER_STEPPER_H
#define HPP_GUARD_ARGMIN_SOLVER_STEPPER_H

#include "argmin/types.h"
#include "argmin/detail/solver_core.h"
#include "argmin/result/step_result.h"
#include "argmin/result/status.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/options.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <utility>
#include <cstdint>
#include <type_traits>

namespace argmin
{

// Passive real-time step primitive: the caller owns all scheduling.
//
// stepper is the purest embeddable surface over detail::solver_core. It shares
// the driver constructor family, the templated MatrixBase x0 boundary, the CTAD
// deduction guides, and every accessor with step_budget_solver, but it exposes
// NO internal loop: it has neither a drive-to-convergence entry point nor a
// bounded-run one. A caller advances the optimization one iteration at a time
// via step() and decides -- entirely on its own schedule -- when to stop, most
// naturally by polling status() / converged() between ticks. This is the shape a
// hard-real-time control or IK loop wants: a single bounded-work call per tick
// with no hidden iteration and no wall clock.
//
// stepper satisfies the steppable concept (the passive single-step surface) but
// deliberately NOT nlp_solver (which refines steppable with the loop-owning
// drive/bounded-run surface). RT-safety here is a property, not a location: the
// type lives under the plain argmin/solver/ path alongside the drivers, and its
// translation unit pulls in NO <chrono> -- the loop drivers that budget on wall
// time are the only chrono carriers.
//
// step() keeps the exact public-step semantics of the drivers (solver_core::
// step()): the stored convergence is consulted exactly once per step, after the
// single policy iteration, and the verdict is published through status(). It is
// NOT the raw step_impl(): consulting convergence a second time per step would
// double-advance windowed criteria (stall / relative-tolerance windows). That
// hazard, and the split between the raw step_impl() and the convergence-aware
// step(), live in detail::solver_core -- stepper reuses step() verbatim rather
// than re-deriving a stepping surface.
//
// A hand-rolled caller loop
//     while(st.status() == solver_status::running && i < budget) { st.step(); ++i; }
// reproduces the step_budget_solver drive-to-convergence iterate/objective
// trajectories bit-for-bit on the same policy / problem / options, because both
// drive the identical solver_core over the identical policy.step(state)
// sequence.
//
// Reference: K&W Section 4.4 (convergence criteria), N&W Section 3.1.

template <typename Policy, int N = dynamic_dimension, typename Problem = void,
          typename Convergence = default_convergence>
class stepper
{
public:
    using core_type = detail::solver_core<Policy, N, Problem, Convergence>;
    using policy_type = Policy;
    // scalar_type is derived directly from Policy (not core_type) so the x0
    // constructor parameter mentions only Policy. Routing it through
    // core_type would drag N/Problem into the (non-deduced) x0 diagnostic and
    // perturb CTAD guide selection.
    using scalar_type = typename Policy::scalar_type;
    using state_type = typename core_type::state_type;

    // Diagnostic for the deduced x0 expression: a fixed-N caller must pass an
    // N-row column vector. A mismatch fails here with a one-line message
    // instead of deep inside Eigen's assignment. dynamic_dimension (N == -1)
    // accepts any column count.
    template <typename Derived>
    static constexpr bool x0_shape_ok =
        (Derived::ColsAtCompileTime == 1) &&
        (N == dynamic_dimension || Derived::RowsAtCompileTime == Eigen::Dynamic ||
         Derived::RowsAtCompileTime == N);

    template <typename Derived>
    static Eigen::Vector<scalar_type, N> to_x0(const Eigen::MatrixBase<Derived>& x0)
    {
        static_assert(x0_shape_ok<Derived>,
                      "x0 must be a column vector matching the solver dimension "
                      "N (an N x 1 Eigen vector for fixed N; any n x 1 vector "
                      "for dynamic_dimension).");
        return Eigen::Vector<scalar_type, N>(x0);
    }

    template <typename P, typename Derived>
    stepper(Policy policy, const P& problem,
            const Eigen::MatrixBase<Derived>& x0,
            const solver_options<Convergence>& opts = {})
        : core_{std::move(policy), problem, to_x0(x0), opts}
    {}

    // CTAD converting constructor: accepts an un-rebound policy and rebinds
    // it to the problem's compile-time dimension, preserving the caller's
    // configured options.
    template <typename OrigPolicy, typename P, typename Derived>
        requires (!std::same_as<std::remove_cvref_t<OrigPolicy>, Policy>)
              && requires { typename std::remove_cvref_t<OrigPolicy>::template rebind<N>; }
              && std::same_as<Policy, typename std::remove_cvref_t<OrigPolicy>::template rebind<N>>
    stepper(OrigPolicy&& orig, const P& problem,
            const Eigen::MatrixBase<Derived>& x0,
            const solver_options<Convergence>& opts = {})
        : core_{std::forward<OrigPolicy>(orig), problem, to_x0(x0), opts}
    {}

    template <typename P, typename Derived>
    stepper(const P& problem, const Eigen::MatrixBase<Derived>& x0,
            const solver_options<Convergence>& opts = {})
        : core_{problem, to_x0(x0), opts}
    {}

    template <typename P, typename Derived, typename PolicyOpts>
        requires has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    stepper(Policy policy, const P& problem,
            const Eigen::MatrixBase<Derived>& x0,
            const solver_options<Convergence>& opts,
            const PolicyOpts& policy_opts)
        : core_{std::move(policy), problem, to_x0(x0), opts, policy_opts}
    {}

    // CTAD converting constructor with policy options.
    template <typename OrigPolicy, typename P, typename Derived, typename PolicyOpts>
        requires (!std::same_as<std::remove_cvref_t<OrigPolicy>, Policy>)
              && requires { typename std::remove_cvref_t<OrigPolicy>::template rebind<N>; }
              && std::same_as<Policy, typename std::remove_cvref_t<OrigPolicy>::template rebind<N>>
              && has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    stepper(OrigPolicy&& orig, const P& problem,
            const Eigen::MatrixBase<Derived>& x0,
            const solver_options<Convergence>& opts,
            const PolicyOpts& policy_opts)
        : core_{std::forward<OrigPolicy>(orig), problem, to_x0(x0), opts, policy_opts}
    {}

    template <typename P, typename Derived, typename PolicyOpts>
        requires has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    stepper(const P& problem, const Eigen::MatrixBase<Derived>& x0,
            const solver_options<Convergence>& opts,
            const PolicyOpts& policy_opts)
        : core_{problem, to_x0(x0), opts, policy_opts}
    {}

    // Overload for policies without options_type: the fourth argument is
    // solver_options<> (the policy_options_t fallback) and is ignored.
    template <typename P, typename Derived>
        requires (!has_options_type<Policy>)
    stepper(const P& problem, const Eigen::MatrixBase<Derived>& x0,
            const solver_options<Convergence>& opts,
            const solver_options<>& policy_opts)
        : core_{problem, to_x0(x0), opts, policy_opts}
    {}

    stepper(const stepper&) = delete;
    stepper& operator=(const stepper&) = delete;
    stepper(stepper&&) noexcept = default;
    stepper& operator=(stepper&&) noexcept = default;

    // Single iteration with per-step convergence consultation. See
    // solver_core::step(). This is the ONLY advancement surface stepper
    // exposes: there is no internal loop.
    step_result<scalar_type> step() { return core_.step(); }

    const state_type& state() const { return core_.state(); }

    // Read the stored convergence policy (populated with per-criterion
    // last_check_results by the most recent step()).
    const auto& convergence() const { return core_.convergence(); }

    solver_status status() const { return core_.status(); }

    // Terminal-status query for the caller's tick loop. Returns true once the
    // solver has reached any terminal status -- i.e. status() != running --
    // whether by the convergence policy, a policy-reported failure, or an
    // abort() request. The specific terminal reason is read from status();
    // this is the one-liner a `while(!st.converged()) st.step();` loop wants
    // so it need not name solver_status::running itself.
    bool converged() const { return core_.status() != solver_status::running; }

    // The wrapped policy instance (options included).
    const Policy& policy() const { return core_.policy(); }

    // The gradient-like norm reported by the most recent step, or a NaN
    // sentinel when no step has run yet.
    scalar_type gradient_norm() const { return core_.gradient_norm(); }

    scalar_type constraint_violation() const { return core_.constraint_violation(); }

    void abort() { core_.abort(); }

    // Reset to a new starting point, preserving algorithm state (e.g. L-BFGS
    // curvature pairs).
    template <typename Derived>
    void reset(const Eigen::MatrixBase<Derived>& x0)
    {
        static_assert(x0_shape_ok<Derived>,
                      "x0 must be a column vector matching the solver dimension "
                      "N (an N x 1 Eigen vector for fixed N; any n x 1 vector "
                      "for dynamic_dimension).");
        core_.reset(x0);
    }

    // Reset to a new starting point, clearing all algorithm state.
    template <typename Derived>
    void reset_clear(const Eigen::MatrixBase<Derived>& x0)
    {
        static_assert(x0_shape_ok<Derived>,
                      "x0 must be a column vector matching the solver dimension "
                      "N (an N x 1 Eigen vector for fixed N; any n x 1 vector "
                      "for dynamic_dimension).");
        core_.reset_clear(x0);
    }

private:
    core_type core_;
};

// CTAD deduction guides mirroring step_budget_solver's: extract N from the
// problem type via problem_dimension_v<Problem> and rebind the policy to that
// dimension via Policy::template rebind<N>. Users never write N or Problem.

// Policy + Problem + x0 + opts
template <typename Policy, typename Problem, typename Derived, typename Convergence>
stepper(Policy, const Problem&,
        const Eigen::MatrixBase<Derived>&,
        const solver_options<Convergence>&)
    -> stepper<typename Policy::template rebind<problem_dimension_v<Problem>>,
               problem_dimension_v<Problem>,
               Problem,
               Convergence>;

// Policy + Problem + x0 (no opts): Convergence keeps the class default.
template <typename Policy, typename Problem, typename Derived>
stepper(Policy, const Problem&, const Eigen::MatrixBase<Derived>&)
    -> stepper<typename Policy::template rebind<problem_dimension_v<Problem>>,
               problem_dimension_v<Problem>,
               Problem>;

// Policy + Problem + x0 + opts + policy_opts
template <typename Policy, typename Problem, typename Derived, typename Convergence,
          typename PolicyOpts>
stepper(Policy, const Problem&,
        const Eigen::MatrixBase<Derived>&,
        const solver_options<Convergence>&, const PolicyOpts&)
    -> stepper<typename Policy::template rebind<problem_dimension_v<Problem>>,
               problem_dimension_v<Problem>,
               Problem,
               Convergence>;

}

#endif
