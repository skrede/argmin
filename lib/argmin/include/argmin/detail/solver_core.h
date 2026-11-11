#ifndef HPP_GUARD_ARGMIN_DETAIL_SOLVER_CORE_H
#define HPP_GUARD_ARGMIN_DETAIL_SOLVER_CORE_H

#include "argmin/types.h"
#include "argmin/detail/lagrangian.h"
#include "argmin/detail/start_point.h"
#include "argmin/detail/status_claim.h"
#include "argmin/detail/bound_measure.h"
#include "argmin/detail/problem_shape.h"
#include "argmin/detail/tuple_contains.h"
#include "argmin/result/step_result.h"
#include "argmin/result/solve_result.h"
#include "argmin/result/status.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/options.h"

#include "argmin/options/convergence_defaults.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <bit>
#include <new>
#include <tuple>
#include <atomic>
#include <limits>
#include <memory>
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
    using type = solver_options<default_convergence<Scalar>>;
};
}

template <typename Policy, typename Scalar>
using policy_options_t = typename detail::policy_options_impl<Policy, Scalar>::type;

namespace detail
{
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
// solver_core owns the wrapped policy instance, its resolved state, the
// problem binding, the iteration/evaluation counters, the stored convergence
// criteria, the abort flag, and the by-step convergence consultation. It
// carries the constructor family (including the CTAD rebind machinery), the
// public step()/reset()/reset_clear()/abort() semantics, and the accessors.
// The loop that drives step_n / solve lives in ONE helper
// (detail/solve_loop.h) parameterized by a budget predicate; every surface
// type (the basic step driver, the budget drivers, the passive stepper)
// composes this core rather than forking a copy of the loop.
// Convergence checking is performed here, not by the policy: the policy does
// NOT know about convergence. Reference: K&W Section 4.4 (convergence
// criteria), N&W Section 3.1.
template <typename Policy, int N = dynamic_dimension, typename Problem = void,
          typename Convergence = default_convergence<typename Policy::scalar_type>>
class solver_core
{
    struct build_tag
    {
    };

    // Rebuilds the placeholder when constructing a state throws, so state_ is never left destroyed. A placeholder that
    // threw here would terminate the program; a fixed-dimension placeholder allocates nothing, a dynamic-dimension one may.
    struct unbuilt_on_unwind
    {
        bool armed;
        solver_core& core;

        ~unbuilt_on_unwind()
        {
            if(!armed)
                return;
            core.built_ = false;
            ::new(static_cast<void*>(std::addressof(core.state_))) state_type(unbuilt_state());
        }
    };

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
                  "expose scalar_type, step(state), and reset(state, x0) / "
                  "reset_clear(state, x0) returning solver_status; its state must expose x. "
                  "Construct via CTAD from a problem + x0.");
    static_assert(std::same_as<typename Convergence::scalar_type, scalar_type>,
                  "Convergence criteria must use the same scalar type as the policy.");

    // Rebinds the incoming problem reference onto the class-template Problem slot so the loop can call value()/constraints()
    // directly instead of probing policy state for a cached (f, cv) pair. Returns nullptr when Problem==void (the legacy
    // non-template state_type policies, e.g. projected_gn, where no class-level Problem type exists); returns &problem
    // otherwise. Lifetime follows the caller-owned problem object, as for every policy state that caches &problem.
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

    template <typename P, typename Derived>
    solver_core(Policy policy, const P& problem, const Eigen::MatrixBase<Derived>& x0, const solver_options<Convergence>& opts = {})
        : solver_core{build_tag{}, std::move(policy), problem, x0, opts, std::nullopt}
    {}

    // CTAD converting constructor: rebinds an un-rebound policy to the problem's compile-time dimension, keeping its
    // options. cvref is stripped from OrigPolicy because an lvalue deduces a reference type, which has no nested rebind.
    template <typename OrigPolicy, typename P, typename Derived>
        requires (!std::same_as<std::remove_cvref_t<OrigPolicy>, Policy>)
              && requires { typename std::remove_cvref_t<OrigPolicy>::template rebind<N>; }
              && std::same_as<Policy, typename std::remove_cvref_t<OrigPolicy>::template rebind<N>>
    solver_core(OrigPolicy&& orig, const P& problem, const Eigen::MatrixBase<Derived>& x0, const solver_options<Convergence>& opts = {})
        : solver_core{build_tag{}, rebind_policy(std::forward<OrigPolicy>(orig)), problem, x0, opts, std::nullopt}
    {}

    template <typename P, typename Derived>
    solver_core(const P& problem, const Eigen::MatrixBase<Derived>& x0, const solver_options<Convergence>& opts = {})
        : solver_core{build_tag{}, Policy{}, problem, x0, opts, std::nullopt}
    {}

    template <typename P, typename Derived, typename PolicyOpts>
        requires has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    solver_core(Policy policy, const P& problem, const Eigen::MatrixBase<Derived>& x0, const solver_options<Convergence>& opts, const PolicyOpts& policy_opts)
        : solver_core{build_tag{}, std::move(policy), problem, x0, opts, policy_opts}
    {}

    // CTAD converting constructor with policy options: the explicit policy_opts, already of the rebound type, are
    // installed by init.
    template <typename OrigPolicy, typename P, typename Derived, typename PolicyOpts>
        requires (!std::same_as<std::remove_cvref_t<OrigPolicy>, Policy>)
              && requires { typename std::remove_cvref_t<OrigPolicy>::template rebind<N>; }
              && std::same_as<Policy, typename std::remove_cvref_t<OrigPolicy>::template rebind<N>>
              && has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    solver_core(OrigPolicy&&, const P& problem, const Eigen::MatrixBase<Derived>& x0, const solver_options<Convergence>& opts, const PolicyOpts& policy_opts)
        : solver_core{build_tag{}, Policy{}, problem, x0, opts, policy_opts}
    {}

    template <typename P, typename Derived, typename PolicyOpts>
        requires has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    solver_core(const P& problem, const Eigen::MatrixBase<Derived>& x0, const solver_options<Convergence>& opts, const PolicyOpts& policy_opts)
        : solver_core{build_tag{}, Policy{}, problem, x0, opts, policy_opts}
    {}

    // Overload for policies without options_type: the fourth argument is
    // the policy_options_t fallback solver_options and is ignored.
    template <typename P, typename Derived>
        requires (!has_options_type<Policy>)
    solver_core(const P& problem, const Eigen::MatrixBase<Derived>& x0, const solver_options<Convergence>& opts, const solver_options<default_convergence<scalar_type>>&)
        : solver_core{build_tag{}, Policy{}, problem, x0, opts, std::nullopt}
    {}

    solver_core(const solver_core&) = delete;
    solver_core& operator=(const solver_core&) = delete;

    solver_core(solver_core&& other) noexcept
        : policy_{std::move(other.policy_)}
        , max_iterations_{other.max_iterations_}
        , constraint_tolerance_{other.constraint_tolerance_}
        , feasibility_tolerance_{other.feasibility_tolerance_}
        , stored_convergence_{other.stored_convergence_}
        , problem_ptr_{other.problem_ptr_}
        , bound_measure_{std::move(other.bound_measure_)}
        , policy_opts_{std::move(other.policy_opts_)}
        , built_{other.built_}
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
            bound_measure_ = std::move(other.bound_measure_);
            policy_opts_ = std::move(other.policy_opts_);
            built_ = other.built_;
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

    // Consults the stored convergence, so a tick loop driving step() alone (MPC, IK) sees a by-criterion status
    // without the step_n(1) cost of rebuilding options. The claim is judged as the solve loop judges it.
    step_result<scalar_type> step()
    {
        auto result = step_impl();
        last_status_ = detail::resolve_step_status(result, stored_convergence_, iterations_, constraint_tolerance_.value_or(feasibility_tolerance_));
        return result;
    }

    // Does not consult convergence: the solve loop owns its stopping decision. x_norm and bound_violation are filled
    // here so step_tolerance_rel_criterion has one denominator and the claim gate one box measure for every policy.
    step_result<scalar_type> step_impl()
    {
        if(!built_)
            return detail::unbuilt_step<scalar_type>();
        if(last_status_ == solver_status::invalid_problem)
            return detail::rejected_step<scalar_type>(state_);
        auto result = policy_.step(state_);
        result.x_norm = state_.x.norm();
        result.bound_violation = bound_measure_(state_);
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

    solver_status status() const { return !built_ || detail::state_rejected(state_) ? solver_status::invalid_problem : last_status_; }

    // The wrapped policy instance (options included). Lets a caller confirm
    // the configuration that actually reached the policy -- notably that a
    // CTAD rebind preserved the configured options rather than resetting
    // them.
    const Policy& policy() const { return policy_; }

    // NaN before the first step, never a fabricated 0 that would read as a stationary point.
    scalar_type gradient_norm() const { return last_gradient_norm_; }

    // L-infinity, as step_result, solve_result and the feasibility tolerance measure the rows (N&W 2e Def 12.1).
    scalar_type constraint_violation() const
    {
        if(status() == solver_status::invalid_problem)
            return std::numeric_limits<scalar_type>::quiet_NaN();
        return detail::measure_constraint_values<scalar_type>(state_, [](const auto& c_eq, const auto& c_ineq) { return detail::primal_feasibility_inf(c_eq, c_ineq); });
    }

    scalar_type bound_violation() const { return status() == solver_status::invalid_problem ? std::numeric_limits<scalar_type>::quiet_NaN() : bound_measure_(state_); }

    // Configured stopping budget and best-seen feasibility floors, read back
    // by the driver facade when it rebuilds the local solver_options on the
    // defaulted solve() / step_n(budget) paths.
    std::uint32_t max_iterations() const { return max_iterations_; }
    std::optional<scalar_type> constraint_tolerance() const { return constraint_tolerance_; }
    scalar_type feasibility_tolerance() const { return feasibility_tolerance_; }

    void abort() { abort_flag_.store(true, std::memory_order_relaxed); }

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

    // Reset to a new starting point, preserving algorithm state (e.g. L-BFGS curvature pairs). A plain vector
    // of the matching scalar binds the policy's Eigen::Ref with no conversion temporary. A state never built
    // because its construction start was rejected is built here, as the constructor would have.
    template <typename Derived>
    solver_status reset(const Eigen::MatrixBase<Derived>& x0)
    {
        if(!detail::start_length_ok<N>(x0, expected_dimension(x0.size())))
            return last_status_ = solver_status::invalid_problem;
        return restart(built_ ? policy_.reset(state_, x0) : build(x0));
    }

    // Reset to a new starting point, clearing all algorithm state.
    template <typename Derived>
    solver_status reset_clear(const Eigen::MatrixBase<Derived>& x0)
    {
        if(!detail::start_length_ok<N>(x0, expected_dimension(x0.size())))
            return last_status_ = solver_status::invalid_problem;
        return restart(built_ ? policy_.reset_clear(state_, x0) : build(x0));
    }

    // Seed best-seen f at loop entry via a direct problem evaluation.
    // Requires Problem != void (i.e. the class-template Problem slot was
    // filled by CTAD or explicit instantiation). Legacy non-template
    // state_type policies (projected_gn, projected_gradient_gn -- which
    // target least_squares only) fall back to the +inf sentinel so the
    // first accepted step becomes the best-seen baseline; x_0 is still
    // returned via best_x if no step succeeds.
    // Reference: argmin::objective concept (argmin/formulation/
    //            concepts.h) -- value(x) is total across supported
    //            Problem types.
    scalar_type seed_best_f() const
    {
        // A rejected problem is never evaluated, not even at x_0.
        if(status() == solver_status::invalid_problem)
            return std::numeric_limits<scalar_type>::infinity();
        if constexpr(!std::is_same_v<Problem, void>)
            return static_cast<scalar_type>(problem_ptr_->value(state_.x));
        else
            return std::numeric_limits<scalar_type>::infinity();
    }

    // The constraint values at x_0 are cached by policy.init; a rejected problem's are never filled.
    scalar_type seed_best_cv() const { return status() == solver_status::invalid_problem ? std::numeric_limits<scalar_type>::infinity() : constraint_violation(); }

    // A rejected problem seeds the same +inf sentinel for the box as for its rows.
    scalar_type seed_best_bv() const { return status() == solver_status::invalid_problem ? std::numeric_limits<scalar_type>::infinity() : bound_measure_(state_); }

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
    // constraint_tolerance is a solver_options field (not a policy_options
    // field), which is why this is a sibling to forward_policy_hints
    // instead of an extension of it -- the two forwarders honor different
    // scopes.
    // Mapping rule (max-of-two): user-supplied constraint_tolerance now
    // gates the composite E-measure (N&W 2e Def 12.1) via
    // stationarity_threshold rather than a separate primal-feasibility
    // threshold, since kkt_residual now carries the primal-feasibility
    // legs directly. The max preserves the pre-existing unconstrained
    // default (1e-8) whenever constraint_tolerance is unset or looser.
    void forward_solver_hints(const solver_options<Convergence>& opts)
    {
        using criteria_tuple = decltype(stored_convergence_.criteria);
        if constexpr(detail::tuple_contains_v<objective_tolerance_criterion<scalar_type>, criteria_tuple>)
        {
            auto& ftol = std::get<objective_tolerance_criterion<scalar_type>>(stored_convergence_.criteria);
            if(opts.constraint_tolerance)
                ftol.stationarity_threshold = std::max(ftol.stationarity_threshold, *opts.constraint_tolerance);
        }
        if constexpr(detail::tuple_contains_v<objective_tolerance_rel_criterion<scalar_type>, criteria_tuple>)
        {
            auto& ftol_rel = std::get<objective_tolerance_rel_criterion<scalar_type>>(stored_convergence_.criteria);
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
            if constexpr(detail::tuple_contains_v<stall_tolerance_criterion<scalar_type>,
                                                   decltype(stored_convergence_.criteria)>)
            {
                auto& stall = std::get<stall_tolerance_criterion<scalar_type>>(stored_convergence_.criteria);
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
            if constexpr(detail::tuple_contains_v<objective_tolerance_criterion<scalar_type>, criteria_tuple>)
            {
                auto& ftol = std::get<objective_tolerance_criterion<scalar_type>>(stored_convergence_.criteria);
                if(policy_opts.kkt_tolerance)
                {
                    ftol.stationarity_threshold =
                        std::max(ftol.stationarity_threshold, *policy_opts.kkt_tolerance);
                }
            }
            if constexpr(detail::tuple_contains_v<objective_tolerance_rel_criterion<scalar_type>, criteria_tuple>)
            {
                auto& ftol_rel = std::get<objective_tolerance_rel_criterion<scalar_type>>(stored_convergence_.criteria);
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
            if constexpr(detail::tuple_contains_v<stall_tolerance_criterion<scalar_type>, criteria_tuple>)
            {
                auto& stall = std::get<stall_tolerance_criterion<scalar_type>>(stored_convergence_.criteria);
                stall.threshold = policy_opts.stall_tolerance_threshold.value_or(convergence_defaults<scalar_type>::stall_threshold);
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
    std::optional<scalar_type> constraint_tolerance_{};
    // Read back by the defaulted solve()/step_n(budget) paths, which rebuild solver_options and would otherwise
    // revert the best-seen feasibility comparator to the 1e-6 default.
    scalar_type feasibility_tolerance_{convergence_defaults<scalar_type>::feasibility_tolerance};
    Convergence stored_convergence_{};
    const Problem* problem_ptr_{nullptr};
    detail::bound_measure<scalar_type, N, state_type, Problem> bound_measure_;
    std::optional<policy_options_t<Policy, scalar_type>> policy_opts_;
    // False when the construction start was rejected: state_ then holds no value but a NaN x.
    bool built_;
    state_type state_;
    std::uint32_t iterations_{0};
    // Sum of the per-step step_result::evaluations, reported as solve_result::function_evaluations.
    std::uint32_t function_evaluations_{0};
    scalar_type last_gradient_norm_{std::numeric_limits<scalar_type>::quiet_NaN()};
    solver_status last_status_{solver_status::running};
    std::atomic<bool> abort_flag_{false};

    template <typename P, typename Derived>
    solver_core(build_tag, Policy policy, const P& problem, const Eigen::MatrixBase<Derived>& x0, const solver_options<Convergence>& opts, std::optional<policy_options_t<Policy, scalar_type>> policy_opts)
        : policy_{std::move(policy)}
        , max_iterations_{opts.max_iterations}
        , constraint_tolerance_{opts.constraint_tolerance}
        , feasibility_tolerance_{opts.feasibility_tolerance}
        , problem_ptr_{bind_problem(problem)}
        , bound_measure_{problem}
        , policy_opts_{std::move(policy_opts)}
        , built_{detail::start_length_ok<N>(x0, detail::problem_start_dimension<N>(problem, x0.size()))}
        , state_{make_state(problem, x0, opts)}
        , last_status_{built_ ? solver_status::running : solver_status::invalid_problem}
    {
        store_convergence(opts.convergence);
        forward_solver_hints(opts);
        if(policy_opts_)
            forward_policy_hints(*policy_opts_);
        else if constexpr(requires { policy_.options; })
            forward_policy_hints(policy_.options);
    }

    // Init gets the constructor's policy options again, so a policy copying them into its state is rebuilt with
    // them. No path puts a second state on the stack: each branch returns a prvalue and build constructs in place.
    template <typename P, typename Derived>
    state_type make_state(const P& problem, const Eigen::MatrixBase<Derived>& x0, const solver_options<Convergence>& opts)
    {
        if(!built_)
            return unbuilt_state();
        if constexpr(!std::is_same_v<Derived, vector_type>)
            return make_state(problem, vector_type(x0), opts);
        else if constexpr(has_options_type<Policy>)
        {
            if(policy_opts_)
                return policy_.init(problem, x0.derived(), opts, *policy_opts_);
            return policy_.init(problem, x0.derived(), opts);
        }
        else
            return policy_.init(problem, x0.derived(), opts);
    }

    static state_type unbuilt_state()
    {
        state_type s;
        if constexpr(N != dynamic_dimension)
            s.x = vector_type::Constant(std::numeric_limits<scalar_type>::quiet_NaN());
        return s;
    }

    template <typename Derived>
    solver_status build(const Eigen::MatrixBase<Derived>& x0)
    {
        if constexpr(std::is_same_v<Problem, void>)
            return solver_status::invalid_problem;
        else
        {
            // Copied before the state dies: a caller may pass a view into the placeholder state, such as state().x.
            const vector_type start(x0);
            built_ = true;
            std::destroy_at(std::addressof(state_));
            unbuilt_on_unwind guard{true, *this};
            ::new(static_cast<void*>(std::addressof(state_))) state_type(make_state(*problem_ptr_, start, {.max_iterations = max_iterations_, .constraint_tolerance = constraint_tolerance_, .feasibility_tolerance = feasibility_tolerance_, .convergence = stored_convergence_}));
            guard.armed = false;
            return detail::latch_status(state_);
        }
    }

    Eigen::Index expected_dimension(Eigen::Index start_length) const
    {
        if constexpr(!std::is_same_v<Problem, void>)
        {
            if(!built_)
                return detail::problem_start_dimension<N>(*problem_ptr_, start_length);
        }
        return detail::start_dimension<N>(problem_ptr_, state_);
    }

    solver_status restart(solver_status status)
    {
        if constexpr(!std::is_same_v<Problem, void>)
            bound_measure_.refresh(*problem_ptr_);
        last_status_ = status;
        iterations_ = 0;
        function_evaluations_ = 0;
        last_gradient_norm_ = std::numeric_limits<scalar_type>::quiet_NaN();
        abort_flag_.store(false, std::memory_order_relaxed);
        return status;
    }
};
}

}

#endif
