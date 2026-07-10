#ifndef HPP_GUARD_ARGMIN_DETAIL_SOLVER_CORE_H
#define HPP_GUARD_ARGMIN_DETAIL_SOLVER_CORE_H

#include "argmin/types.h"
#include "argmin/detail/lagrangian.h"
#include "argmin/detail/tuple_contains.h"
#include "argmin/result/step_result.h"
#include "argmin/result/solve_result.h"
#include "argmin/result/status.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/options.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <bit>
#include <tuple>
#include <atomic>
#include <limits>
#include <cstdint>
#include <utility>
#include <concepts>
#include <optional>
#include <algorithm>
#include <type_traits>

namespace argmin
{

template <typename P>
concept has_options_type = requires { typename P::options_type; };

namespace detail
{
template <typename Policy, typename Scalar, bool = has_options_type<Policy>>
struct policy_options_impl
{
    using type = typename Policy::options_type;
};

template <typename Policy, typename Scalar>
struct policy_options_impl<Policy, Scalar, false>
{
    using type = solver_options<>;
};
}

template <typename Policy, typename Scalar>
using policy_options_t = typename detail::policy_options_impl<Policy, Scalar>::type;

namespace detail
{
// Resolve state_type: if Policy has template state_type<P>, use it.
// Otherwise fall back to Policy::state_type (non-template).
// This enables incremental conversion of policies from std::function
// to direct problem pointer calls.
template <typename Policy, typename Problem>
consteval auto resolve_state_tag()
{
    if constexpr(requires { typename Policy::template state_type<Problem>; })
        return std::type_identity<typename Policy::template state_type<Problem>>{};
    else
        return std::type_identity<typename Policy::state_type>{};
}

template <typename Policy, typename Problem>
using resolve_state_t =
    typename decltype(resolve_state_tag<Policy, Problem>())::type;

// Shared step/binding/convergence engine for every driver facade.
//
// solver_core owns the wrapped policy instance, its resolved state, the
// problem binding, the iteration/evaluation counters, the stored convergence
// criteria, the abort flag, and the by-step convergence consultation. It
// carries the constructor family (including the CTAD rebind machinery), the
// public step()/reset()/reset_clear()/abort() semantics, and the accessors.
// The loop that drives step_n / solve lives in ONE helper
// (detail/solve_loop.h) parameterized by a budget predicate; every surface
// type (the basic step driver, the budget drivers, the passive stepper)
// composes this core rather than forking a copy of the loop.
//
// Convergence checking is performed here, not by the policy: the policy does
// NOT know about convergence. Reference: K&W Section 4.4 (convergence
// criteria), N&W Section 3.1.
template <typename Policy, int N = dynamic_dimension, typename Problem = void,
          typename Convergence = default_convergence>
class solver_core
{
public:
    using policy_type = Policy;
    using scalar_type = typename Policy::scalar_type;
    using state_type = resolve_state_t<Policy, Problem>;
    using convergence_type = Convergence;
    using vector_type = vector<scalar_type, N>;
    using solve_result_type = solve_result<scalar_type, N>;

    // Enforce the policy contract at the one scope where state_type resolves
    // (resolve_state_t depends on Problem). A policy missing step/reset/
    // reset_clear or a state without an x member fails here with a one-line
    // diagnostic at the construction site instead of a deep template error.
    static_assert(solver_policy<Policy, state_type, scalar_type>,
                  "Policy does not satisfy the solver_policy contract: it must "
                  "expose scalar_type and step(state) / reset(state, x0) / "
                  "reset_clear(state, x0), and its state must expose x. "
                  "Construct via CTAD from a problem + x0.");

    // Rebinds the incoming problem reference onto the class-template
    // Problem slot so the loop can call value()/constraints() directly
    // instead of probing policy state for a cached (f, cv) pair.
    // Returns nullptr when Problem==void (the legacy non-template
    // state_type policies, e.g. projected_gn, where no class-level
    // Problem type exists); returns &problem otherwise. Lifetime
    // follows the caller-owned problem object -- the same constraint
    // already applies to every policy state that caches &problem.
    template <typename P>
    static constexpr const Problem* bind_problem(const P& problem) noexcept
    {
        if constexpr(std::is_same_v<Problem, void>)
            return nullptr;
        else
        {
            static_assert(std::is_same_v<P, Problem>,
                          "solver_core: constructor problem type must "
                          "match class-template Problem (CTAD normally "
                          "guarantees this)");
            return &problem;
        }
    }

    // Copy a source policy's options into the rebound policy's options. The
    // CTAD rebind changes only the compile-time dimension N; options_type
    // carries no N-dependent members, so the source and rebound option
    // aggregates are structurally identical though distinct C++ types (each
    // is a nested type of a different Policy specialization). Their field
    // layout is therefore identical and a trivially-copyable relocation is
    // exact. When the types coincide (no dimension change) this degrades to
    // a plain assignment.
    template <typename Dst, typename Src>
    static void assign_rebound_options(Dst& dst, const Src& src)
    {
        if constexpr(std::is_same_v<Dst, Src>)
            dst = src;
        else
        {
            static_assert(std::is_trivially_copyable_v<Dst>
                          && std::is_trivially_copyable_v<Src>
                          && sizeof(Dst) == sizeof(Src),
                          "rebind must carry the configured policy options "
                          "across the dimension change: the source and "
                          "rebound option aggregates must be layout-identical "
                          "trivially-copyable types");
            dst = std::bit_cast<Dst>(src);
        }
    }

    // Build the rebound policy for the converting CTAD constructors,
    // preserving the caller's configured options instead of silently
    // default-constructing them away. Accepts both lvalue and rvalue source
    // policies.
    template <typename OrigPolicy>
    static Policy rebind_policy(OrigPolicy&& orig)
    {
        Policy p{};
        if constexpr(has_options_type<Policy>)
            assign_rebound_options(p.options, orig.options);
        return p;
    }

    template <typename P>
    solver_core(Policy policy, const P& problem,
                const Eigen::VectorX<scalar_type>& x0,
                const solver_options<Convergence>& opts = {})
        : policy_{std::move(policy)}
        , max_iterations_{opts.max_iterations}
        , constraint_tolerance_{opts.constraint_tolerance}
        , feasibility_tolerance_{opts.feasibility_tolerance}
        , problem_ptr_{bind_problem(problem)}
        , state_{policy_.init(problem, Eigen::Vector<scalar_type, N>(x0), opts)}
    {
        store_convergence(opts.convergence);
        forward_solver_hints(opts);
        if constexpr(requires { policy_.options; })
            forward_policy_hints(policy_.options);
    }

    // CTAD converting constructor: accepts an un-rebound policy and rebinds
    // it to the problem's compile-time dimension, preserving the caller's
    // configured options. The requires-clause strips cvref from OrigPolicy so
    // both lvalue and rvalue source policies are accepted (an lvalue deduces
    // OrigPolicy as a reference type, which has no nested rebind alias).
    template <typename OrigPolicy, typename P>
        requires (!std::same_as<std::remove_cvref_t<OrigPolicy>, Policy>)
              && requires { typename std::remove_cvref_t<OrigPolicy>::template rebind<N>; }
              && std::same_as<Policy, typename std::remove_cvref_t<OrigPolicy>::template rebind<N>>
    solver_core(OrigPolicy&& orig, const P& problem,
                const Eigen::VectorX<scalar_type>& x0,
                const solver_options<Convergence>& opts = {})
        : policy_{rebind_policy(std::forward<OrigPolicy>(orig))}
        , max_iterations_{opts.max_iterations}
        , constraint_tolerance_{opts.constraint_tolerance}
        , feasibility_tolerance_{opts.feasibility_tolerance}
        , problem_ptr_{bind_problem(problem)}
        , state_{policy_.init(problem, Eigen::Vector<scalar_type, N>(x0), opts)}
    {
        store_convergence(opts.convergence);
        forward_solver_hints(opts);
        if constexpr(requires { policy_.options; })
            forward_policy_hints(policy_.options);
    }

    template <typename P>
    solver_core(const P& problem, const Eigen::VectorX<scalar_type>& x0,
                const solver_options<Convergence>& opts = {})
        : max_iterations_{opts.max_iterations}
        , constraint_tolerance_{opts.constraint_tolerance}
        , feasibility_tolerance_{opts.feasibility_tolerance}
        , problem_ptr_{bind_problem(problem)}
        , state_{policy_.init(problem, Eigen::Vector<scalar_type, N>(x0), opts)}
    {
        store_convergence(opts.convergence);
        forward_solver_hints(opts);
        if constexpr(requires { policy_.options; })
            forward_policy_hints(policy_.options);
    }

    template <typename P, typename PolicyOpts>
        requires has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    solver_core(Policy policy, const P& problem,
                const Eigen::VectorX<scalar_type>& x0,
                const solver_options<Convergence>& opts,
                const PolicyOpts& policy_opts)
        : policy_{std::move(policy)}
        , max_iterations_{opts.max_iterations}
        , constraint_tolerance_{opts.constraint_tolerance}
        , feasibility_tolerance_{opts.feasibility_tolerance}
        , problem_ptr_{bind_problem(problem)}
        , state_{policy_.init(problem, Eigen::Vector<scalar_type, N>(x0), opts, policy_opts)}
    {
        store_convergence(opts.convergence);
        forward_solver_hints(opts);
        forward_policy_hints(policy_opts);
    }

    // CTAD converting constructor with policy options. The explicit
    // policy_opts (already rebound-typed) is installed by init(); the
    // requires-clause strips cvref from OrigPolicy so an lvalue source policy
    // is accepted.
    template <typename OrigPolicy, typename P, typename PolicyOpts>
        requires (!std::same_as<std::remove_cvref_t<OrigPolicy>, Policy>)
              && requires { typename std::remove_cvref_t<OrigPolicy>::template rebind<N>; }
              && std::same_as<Policy, typename std::remove_cvref_t<OrigPolicy>::template rebind<N>>
              && has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    solver_core(OrigPolicy&&, const P& problem,
                const Eigen::VectorX<scalar_type>& x0,
                const solver_options<Convergence>& opts,
                const PolicyOpts& policy_opts)
        : max_iterations_{opts.max_iterations}
        , constraint_tolerance_{opts.constraint_tolerance}
        , feasibility_tolerance_{opts.feasibility_tolerance}
        , problem_ptr_{bind_problem(problem)}
        , state_{policy_.init(problem, Eigen::Vector<scalar_type, N>(x0), opts, policy_opts)}
    {
        store_convergence(opts.convergence);
        forward_solver_hints(opts);
        forward_policy_hints(policy_opts);
    }

    template <typename P, typename PolicyOpts>
        requires has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    solver_core(const P& problem, const Eigen::VectorX<scalar_type>& x0,
                const solver_options<Convergence>& opts,
                const PolicyOpts& policy_opts)
        : max_iterations_{opts.max_iterations}
        , constraint_tolerance_{opts.constraint_tolerance}
        , feasibility_tolerance_{opts.feasibility_tolerance}
        , problem_ptr_{bind_problem(problem)}
        , state_{policy_.init(problem, Eigen::Vector<scalar_type, N>(x0), opts, policy_opts)}
    {
        store_convergence(opts.convergence);
        forward_solver_hints(opts);
        forward_policy_hints(policy_opts);
    }

    // Overload for policies without options_type: the fourth argument is
    // solver_options<> (the policy_options_t fallback) and is ignored.
    template <typename P>
        requires (!has_options_type<Policy>)
    solver_core(const P& problem, const Eigen::VectorX<scalar_type>& x0,
                const solver_options<Convergence>& opts,
                const solver_options<>&)
        : max_iterations_{opts.max_iterations}
        , constraint_tolerance_{opts.constraint_tolerance}
        , feasibility_tolerance_{opts.feasibility_tolerance}
        , problem_ptr_{bind_problem(problem)}
        , state_{policy_.init(problem, Eigen::Vector<scalar_type, N>(x0), opts)}
    {
        store_convergence(opts.convergence);
        forward_solver_hints(opts);
        // No policy hints for policies without options_type.
    }

    solver_core(const solver_core&) = delete;
    solver_core& operator=(const solver_core&) = delete;

    solver_core(solver_core&& other) noexcept
        : policy_{std::move(other.policy_)}
        , max_iterations_{other.max_iterations_}
        , constraint_tolerance_{other.constraint_tolerance_}
        , feasibility_tolerance_{other.feasibility_tolerance_}
        , stored_convergence_{other.stored_convergence_}
        , problem_ptr_{other.problem_ptr_}
        , state_{std::move(other.state_)}
        , iterations_{other.iterations_}
        , function_evaluations_{other.function_evaluations_}
        , last_gradient_norm_{other.last_gradient_norm_}
        , last_status_{other.last_status_}
        , abort_flag_{other.abort_flag_.load(std::memory_order_relaxed)}
    {}

    solver_core& operator=(solver_core&& other) noexcept
    {
        if(this != &other)
        {
            policy_ = std::move(other.policy_);
            max_iterations_ = other.max_iterations_;
            constraint_tolerance_ = other.constraint_tolerance_;
            feasibility_tolerance_ = other.feasibility_tolerance_;
            stored_convergence_ = other.stored_convergence_;
            problem_ptr_ = other.problem_ptr_;
            state_ = std::move(other.state_);
            iterations_ = other.iterations_;
            function_evaluations_ = other.function_evaluations_;
            last_gradient_norm_ = other.last_gradient_norm_;
            last_status_ = other.last_status_;
            abort_flag_.store(other.abort_flag_.load(std::memory_order_relaxed),
                              std::memory_order_relaxed);
        }
        return *this;
    }

    // Single iteration with per-step convergence consultation.
    //
    // Beyond executing the policy step and updating telemetry, step() checks
    // the stored convergence criteria against the produced result and updates
    // the solver's reported status(). A caller driving a tick loop via step()
    // alone (MPC/IK) therefore sees a by-criterion terminal status without
    // reimplementing the stopping tests or paying the step_n(1) cost of
    // rebuilding options and reseeding best-seen tracking. The stored
    // convergence and the running iteration counter are reused as-is; no
    // options are rebuilt. Policy-reported failure (policy_status) takes
    // precedence over the convergence verdict.
    step_result<scalar_type> step()
    {
        auto result = step_impl();

        if(result.policy_status)
            last_status_ = *result.policy_status;
        else if(auto conv = stored_convergence_.check(result, iterations_))
            last_status_ = *conv;
        else
            last_status_ = solver_status::running;

        return result;
    }

    // Raw single iteration shared by the public step() and the step_n()
    // convergence loop. Executes the policy step, fills x_norm centrally
    // (so step_tolerance_rel_criterion has a consistent denominator without
    // per-policy churn -- policies own constraint_violation, previously
    // overwritten here), advances the iteration and evaluation counters, and
    // records the reported gradient norm. Deliberately does NOT consult
    // convergence: the public step() adds that on top, while the solve loop
    // owns its own stopping decision.
    step_result<scalar_type> step_impl()
    {
        auto result = policy_.step(state_);
        result.x_norm = state_.x.norm();
        ++iterations_;
        function_evaluations_ += result.evaluations;
        last_gradient_norm_ = result.gradient_norm;
        return result;
    }

    const state_type& state() const { return state_; }

    // Read the stored convergence policy (populated with per-criterion
    // last_check_results by the most recent solve loop). Precedent:
    // line_search_calls on kraft_slsqp_policy::state_type.
    const Convergence& convergence() const { return stored_convergence_; }

    solver_status status() const { return last_status_; }

    // The wrapped policy instance (options included). Lets a caller confirm
    // the configuration that actually reached the policy -- notably that a
    // CTAD rebind preserved the configured options rather than resetting
    // them.
    const Policy& policy() const { return policy_; }

    // The gradient-like norm reported by the most recent step, or a NaN
    // sentinel when no step has run yet (genuinely unavailable -- never a
    // fabricated 0 that would read as a stationary point).
    scalar_type gradient_norm() const { return last_gradient_norm_; }

    scalar_type constraint_violation() const
    {
        if constexpr(requires { state_.c_eq; state_.c_ineq; })
            return detail::constraint_violation(state_.c_eq, state_.c_ineq);
        else
            return scalar_type(0);
    }

    // Configured stopping budget and best-seen feasibility floors, read back
    // by the driver facade when it rebuilds the local solver_options on the
    // defaulted solve() / step_n(budget) paths.
    std::uint32_t max_iterations() const { return max_iterations_; }
    std::optional<double> constraint_tolerance() const { return constraint_tolerance_; }
    double feasibility_tolerance() const { return feasibility_tolerance_; }

    void abort() { abort_flag_.store(true, std::memory_order_relaxed); }

    // Loop-support surface consumed by detail/solve_loop.h.
    bool abort_requested() const { return abort_flag_.load(std::memory_order_relaxed); }
    std::uint32_t iteration_count() const { return iterations_; }
    std::uint32_t evaluation_count() const { return function_evaluations_; }
    void set_status(solver_status s) { last_status_ = s; }

    // Back-copy per-criterion telemetry from a caller-owned solver_options
    // convergence instance into stored_convergence_ when the types match, so
    // convergence().last_check_results() reflects the most recent solve loop.
    // For an OptsConvergence differing from the solver's own stored
    // Convergence type, consumers read last_check_results directly from their
    // own opts.convergence instance.
    template <typename OptsConvergence>
    void back_copy_convergence(const OptsConvergence& oc)
    {
        if constexpr(std::is_same_v<OptsConvergence, Convergence>)
            stored_convergence_.last_check_results_ = oc.last_check_results_;
    }

    // Reset to a new starting point, preserving algorithm state (e.g. L-BFGS
    // curvature pairs). Delegates to policy.reset(), binding the caller buffer
    // straight into the policy's Eigen::Ref parameter -- a plain vector of the
    // matching scalar passes through with no conversion temporary (the warm
    // per-tick path is allocation-free); only a genuine expression argument
    // materializes, inside Ref's own storage.
    template <typename Derived>
    void reset(const Eigen::MatrixBase<Derived>& x0)
    {
        policy_.reset(state_, x0);
        iterations_ = 0;
        function_evaluations_ = 0;
        last_gradient_norm_ = std::numeric_limits<scalar_type>::quiet_NaN();
        last_status_ = solver_status::running;
        abort_flag_.store(false, std::memory_order_relaxed);
    }

    // Reset to a new starting point, clearing all algorithm state.
    // Delegates to policy.reset_clear() with the same pass-through binding.
    template <typename Derived>
    void reset_clear(const Eigen::MatrixBase<Derived>& x0)
    {
        policy_.reset_clear(state_, x0);
        iterations_ = 0;
        function_evaluations_ = 0;
        last_gradient_norm_ = std::numeric_limits<scalar_type>::quiet_NaN();
        last_status_ = solver_status::running;
        abort_flag_.store(false, std::memory_order_relaxed);
    }

    // Seed best-seen f at loop entry via a direct problem evaluation.
    // Requires Problem != void (i.e. the class-template Problem slot was
    // filled by CTAD or explicit instantiation). Legacy non-template
    // state_type policies (projected_gn, projected_gradient_gn -- which
    // target least_squares only) fall back to the +inf sentinel so the
    // first accepted step becomes the best-seen baseline; x_0 is still
    // returned via best_x if no step succeeds.
    //
    // Reference: argmin::objective concept (argmin/formulation/
    //            concepts.h) -- value(x) is total across supported
    //            Problem types.
    scalar_type seed_best_f() const
    {
        if constexpr(!std::is_same_v<Problem, void>)
            return static_cast<scalar_type>(problem_ptr_->value(state_.x));
        else
            return std::numeric_limits<scalar_type>::infinity();
    }

    // Seed best-seen cv at loop entry from the policy's cached
    // constraint values (populated by policy.init at x_0). L-infinity
    // composition matches step_result.constraint_violation reporting
    // downstream, so is_better compares like-against-like in the best-
    // seen branch. Unconstrained / bound-constrained-only Problems have
    // empty c_eq / c_ineq and yield cv = 0.
    //
    // Reference: detail::primal_feasibility_inf (argmin/detail/
    //            lagrangian.h); N&W 2e Def 12.1.
    scalar_type seed_best_cv() const
    {
        if constexpr(requires { state_.c_eq; state_.c_ineq; })
            return detail::primal_feasibility_inf(state_.c_eq, state_.c_ineq);
        else
            return scalar_type(0);
    }

    // Stores the constructor's convergence policy by value, with no
    // field-by-field remap: Convergence is the solver's own class-template
    // parameter (deduced from the ctor's solver_options<Convergence>
    // argument via CTAD), so stored_convergence_ always has the exact same
    // type as the argument here. A relative-tolerance Convergence is stored
    // -- and later checked -- as a relative-tolerance policy; there is no
    // absolute-criterion coercion.
    void store_convergence(const Convergence& conv)
    {
        stored_convergence_ = conv;
    }

    // Forwards solver-level options (currently constraint_tolerance) into
    // the stored convergence criteria. Called at construction time from
    // every constructor; once set here the tightened threshold persists
    // through subsequent solve() / step_n(budget) calls that reconstruct
    // local solver_options from stored state.
    //
    // constraint_tolerance is a solver_options field (not a policy_options
    // field), which is why this is a sibling to forward_policy_hints
    // instead of an extension of it -- the two forwarders honor different
    // scopes.
    //
    // Mapping rule (max-of-two): user-supplied constraint_tolerance now
    // gates the composite E-measure (N&W 2e Def 12.1) via
    // stationarity_threshold rather than a separate primal-feasibility
    // threshold, since kkt_residual now carries the primal-feasibility
    // legs directly. The max preserves the pre-existing unconstrained
    // default (1e-8) whenever constraint_tolerance is unset or looser.
    void forward_solver_hints(const solver_options<Convergence>& opts)
    {
        using criteria_tuple = decltype(stored_convergence_.criteria);
        if constexpr(detail::tuple_contains_v<objective_tolerance_criterion, criteria_tuple>)
        {
            auto& ftol = std::get<objective_tolerance_criterion>(stored_convergence_.criteria);
            if(opts.constraint_tolerance)
                ftol.stationarity_threshold = std::max(ftol.stationarity_threshold, *opts.constraint_tolerance);
        }
        if constexpr(detail::tuple_contains_v<objective_tolerance_rel_criterion, criteria_tuple>)
        {
            auto& ftol_rel = std::get<objective_tolerance_rel_criterion>(stored_convergence_.criteria);
            if(opts.constraint_tolerance)
                ftol_rel.stationarity_threshold = std::max(ftol_rel.stationarity_threshold, *opts.constraint_tolerance);
        }
    }

    template <typename PolicyOpts>
    void forward_policy_hints(const PolicyOpts& policy_opts)
    {
        if constexpr(requires { policy_opts.stall_window; })
        {
            // detail::tuple_contains_v (not a raw `requires { std::get<...> }`
            // probe) is required here: libstdc++'s std::get<T> on a tuple
            // lacking T fails via a static_assert inside the function body,
            // which is not a SFINAE-friendly substitution failure and would
            // hard-error for any Convergence lacking stall_tolerance_criterion
            // (e.g. a caller-supplied policy combining a stall_window-bearing
            // policy's options with a bespoke non-stall convergence type).
            if constexpr(detail::tuple_contains_v<stall_tolerance_criterion,
                                                   decltype(stored_convergence_.criteria)>)
            {
                auto& stall = std::get<stall_tolerance_criterion>(stored_convergence_.criteria);
                stall.window = policy_opts.stall_window;
            }
        }

        // MMA / GCMMA kkt_tolerance option -> stationarity_threshold via
        // the max-of-two rule. Preserves the user's explicit
        // stationarity_threshold while honoring the policy hint as a
        // floor. Mirrors the pattern in forward_solver_hints above for
        // constraint_tolerance -> stationarity_threshold.
        //
        // Reference: Nocedal and Wright, "Numerical Optimization" 2e,
        //            Definition 12.1 (KKT stationarity); canonical
        //            max-of-two mapping rule per forward_solver_hints.
        if constexpr(requires { policy_opts.kkt_tolerance; })
        {
            using criteria_tuple = decltype(stored_convergence_.criteria);
            if constexpr(detail::tuple_contains_v<objective_tolerance_criterion, criteria_tuple>)
            {
                auto& ftol = std::get<objective_tolerance_criterion>(stored_convergence_.criteria);
                if(policy_opts.kkt_tolerance)
                {
                    ftol.stationarity_threshold =
                        std::max(ftol.stationarity_threshold, *policy_opts.kkt_tolerance);
                }
            }
            if constexpr(detail::tuple_contains_v<objective_tolerance_rel_criterion, criteria_tuple>)
            {
                auto& ftol_rel = std::get<objective_tolerance_rel_criterion>(stored_convergence_.criteria);
                if(policy_opts.kkt_tolerance)
                {
                    ftol_rel.stationarity_threshold =
                        std::max(ftol_rel.stationarity_threshold, *policy_opts.kkt_tolerance);
                }
            }
        }

        // MMA / GCMMA stall_tolerance_threshold option -> stall_tolerance_criterion.
        // Unconditional set via value_or(default): the criterion's
        // `if(!threshold) return std::nullopt;` short-circuit means a nullopt
        // threshold disables the criterion silently.  stall_tolerance_threshold
        // has no baseline; this wire is what enables the criterion for the
        // MMA / GCMMA policy family.  Mirrors the canonical forward_solver_hints
        // constraint_tolerance -> stationarity_threshold precedent above but
        // uses value_or rather than max-of-two because stall detection has no
        // pre-existing default to preserve.
        //
        // The best_seen_feasible metric flag is auto-enabled when the policy
        // has the rho_init option (MMA) or the raa0 option (GCMMA), which
        // together identify the MMA policy family and the early-convergence-
        // then-destabilization mechanism documented in in-tree diagnosis
        // traces.  Non-MMA policies that happen to expose
        // stall_tolerance_threshold (none today) leave the flag in its
        // defaulted false state; the criterion then uses the combined
        // (objective + constraint_violation) metric byte-for-byte.
        if constexpr(requires { policy_opts.stall_tolerance_threshold; })
        {
            using criteria_tuple = decltype(stored_convergence_.criteria);
            if constexpr(detail::tuple_contains_v<stall_tolerance_criterion, criteria_tuple>)
            {
                auto& stall = std::get<stall_tolerance_criterion>(stored_convergence_.criteria);
                stall.threshold = policy_opts.stall_tolerance_threshold.value_or(1e-6);
                if constexpr(requires { policy_opts.rho_init; }
                          || requires { policy_opts.raa0; })
                {
                    stall.track_best_seen_feasible = true;
                }
            }
        }
    }

private:
    Policy policy_{};
    std::uint32_t max_iterations_{1000};
    std::optional<double> constraint_tolerance_{};
    // Mirrors solver_options::feasibility_tolerance (constructor-time
    // configured); read back into the locally-rebuilt solver_options on the
    // defaulted solve()/step_n(budget) paths so the best-seen feasibility
    // comparator honors the constructor's configured value instead of
    // reverting to solver_options's own 1e-6 default.
    double feasibility_tolerance_{1e-6};
    Convergence stored_convergence_{};
    const Problem* problem_ptr_{nullptr};
    state_type state_;
    std::uint32_t iterations_{0};
    // Accumulated genuine evaluation count (sum of per-step
    // step_result::evaluations); reported as solve_result::function_evaluations
    // instead of aliasing the iteration count.
    std::uint32_t function_evaluations_{0};
    // Gradient-like norm from the most recent step; NaN until the first step
    // (genuinely unavailable), never a fabricated 0.
    scalar_type last_gradient_norm_{std::numeric_limits<scalar_type>::quiet_NaN()};
    solver_status last_status_{solver_status::running};
    std::atomic<bool> abort_flag_{false};
};
}

}

#endif
