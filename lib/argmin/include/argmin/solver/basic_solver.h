#ifndef HPP_GUARD_ARGMIN_SOLVER_BASIC_SOLVER_H
#define HPP_GUARD_ARGMIN_SOLVER_BASIC_SOLVER_H

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
#include <cmath>
#include <tuple>
#include <atomic>
#include <chrono>
#include <limits>
#include <cstdint>
#include <concepts>
#include <optional>
#include <type_traits>

namespace argmin
{

template <typename P>
concept has_options_type = requires { typename P::options_type; };

namespace detail
{
// tuple_contains / tuple_contains_v are provided by
// "argmin/detail/tuple_contains.h" so detail-tier code (e.g.,
// detail/isres_operators.h) can reuse the same SFINAE gate without
// pulling solver/basic_solver.h into the detail tier.

// Feasibility-first best-seen comparator used by basic_solver::step_n
// to select the returned iterate.
//
// Tiered ordering:
//   1. Feasible beats infeasible unconditionally.
//   2. Both feasible: prefer lower objective.
//   3. Both infeasible: prefer lower constraint violation.
//
// An iterate is "feasible" when its constraint_violation is <= feas_tol;
// unconstrained problems (cv = 0) always take the feasible branch.
//
// Reference: NLopt nlopt_optimize convention (nlopt/src/api/nlopt.c) --
//            the caller receives the best point encountered, not the
//            terminal trial. This monotonically improves the reported
//            solve_result for oscillation-prone policies (MMA, GCMMA,
//            ISRES, CMA-ES) without regressing well-behaved policies
//            whose terminal iterate is already their best.
template <typename Scalar>
constexpr bool is_better(Scalar f_cand, Scalar cv_cand,
                         Scalar f_best, Scalar cv_best,
                         Scalar feas_tol) noexcept
{
    const bool cand_feasible = cv_cand <= feas_tol;
    const bool best_feasible = cv_best <= feas_tol;
    if(cand_feasible && !best_feasible) return true;
    if(!cand_feasible && best_feasible) return false;
    if(cand_feasible && best_feasible)
        return f_cand < f_best;
    return cv_cand < cv_best;
}
}

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
}

// Solver wrapper with convergence loop (per CORE-02, D-09, D-10, D-11, D-13, D-14).
//
// basic_solver wraps a Policy instance that defines the algorithm logic.
// The solver provides the iterative execution model: step(), solve(),
// step_n(budget). Convergence checking is performed by basic_solver --
// the policy does NOT know about convergence.
//
// basic_solver<Policy, N, Problem> has three template parameters:
// the policy, the compile-time dimension N, and the Problem type.
// N and Problem are deduced from the problem via CTAD deduction guides.
// Users never write N or Problem explicitly — CTAD handles both.
//
// For policies with non-template state_type (pre-conversion), Problem
// defaults to void and is ignored. For policies with template
// state_type<P>, Problem is used to instantiate the state.
//
// The policy contract is enforced by the solver_policy concept
// (argmin/formulation/concepts.h), static_assert-ed in the class body
// immediately after the state_type alias. It pins the harness-visible surface
// (scalar_type; step/reset/reset_clear; state.x); init() is intentionally left
// unconstrained since it is templated on Problem/Convergence and its shape is
// an implementation detail.
//
// Reference: K&W Section 4.4 (convergence criteria), N&W Section 3.1.

template <typename Policy, int N = dynamic_dimension, typename Problem = void,
          typename Convergence = default_convergence>
class basic_solver
{
public:
    using policy_type = Policy;
    using scalar_type = typename Policy::scalar_type;
    using state_type = detail::resolve_state_t<Policy, Problem>;

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
    // Problem slot so step_n can call value()/constraints() directly
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
                          "basic_solver: constructor problem type must "
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
    basic_solver(Policy policy, const P& problem,
                 const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts = {})
        : policy_{std::move(policy)}
        , max_iterations_{opts.max_iterations}
        , max_time_{opts.max_time}
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
    basic_solver(OrigPolicy&& orig, const P& problem,
                 const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts = {})
        : policy_{rebind_policy(std::forward<OrigPolicy>(orig))}
        , max_iterations_{opts.max_iterations}
        , max_time_{opts.max_time}
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
    basic_solver(const P& problem, const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts = {})
        : max_iterations_{opts.max_iterations}
        , max_time_{opts.max_time}
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
    basic_solver(Policy policy, const P& problem,
                 const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts,
                 const PolicyOpts& policy_opts)
        : policy_{std::move(policy)}
        , max_iterations_{opts.max_iterations}
        , max_time_{opts.max_time}
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
    basic_solver(OrigPolicy&&, const P& problem,
                 const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts,
                 const PolicyOpts& policy_opts)
        : max_iterations_{opts.max_iterations}
        , max_time_{opts.max_time}
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
    basic_solver(const P& problem, const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts,
                 const PolicyOpts& policy_opts)
        : max_iterations_{opts.max_iterations}
        , max_time_{opts.max_time}
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
    basic_solver(const P& problem, const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts,
                 const solver_options<>&)
        : max_iterations_{opts.max_iterations}
        , max_time_{opts.max_time}
        , constraint_tolerance_{opts.constraint_tolerance}
        , feasibility_tolerance_{opts.feasibility_tolerance}
        , problem_ptr_{bind_problem(problem)}
        , state_{policy_.init(problem, Eigen::Vector<scalar_type, N>(x0), opts)}
    {
        store_convergence(opts.convergence);
        forward_solver_hints(opts);
        // No policy hints for policies without options_type.
    }

    basic_solver(const basic_solver&) = delete;
    basic_solver& operator=(const basic_solver&) = delete;

    basic_solver(basic_solver&& other) noexcept
        : policy_{std::move(other.policy_)}
        , max_iterations_{other.max_iterations_}
        , max_time_{other.max_time_}
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

    basic_solver& operator=(basic_solver&& other) noexcept
    {
        if(this != &other)
        {
            policy_ = std::move(other.policy_);
            max_iterations_ = other.max_iterations_;
            max_time_ = other.max_time_;
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

    // Solve with default convergence (uses stored max_iterations).
    //
    // Builds solver_options<Convergence> (the class's own stored convergence
    // type, not solver_options<>) so a non-default Convergence is used
    // without coercion, and copies the constructor-configured
    // constraint_tolerance_/feasibility_tolerance_ into the local opts so the
    // best-seen feasibility comparator honors them instead of reverting to
    // solver_options's 1e-6 default.
    //
    // Re-entrancy: solve() and step_n(budget) are continuations, not
    // restarts. Each call executes up to its budget of ADDITIONAL iterations
    // from the current iterate, and the iteration counter, evaluation
    // counter, and any windowed convergence state accumulate across calls. A
    // second solve() therefore resumes where the first stopped (a warm
    // restart), which is the semantics a tick-loop caller wants. To restart
    // from a clean counter and a new starting point, call reset() (preserves
    // algorithm state) or reset_clear() (drops it) first.
    solve_result<scalar_type, N> solve()
    {
        solver_options<Convergence> opts;
        opts.max_iterations = max_iterations_;
        opts.max_time = max_time_;
        opts.constraint_tolerance = constraint_tolerance_;
        opts.feasibility_tolerance = feasibility_tolerance_;
        opts.convergence = stored_convergence_;
        auto result = step_n(max_iterations_, opts);
        // Propagate the per-criterion telemetry that step_n writes into
        // its local opts.convergence back to stored_convergence_ so
        // solver.convergence().last_check_results() reflects the last
        // solve's outcome.
        stored_convergence_.last_check_results_ =
            opts.convergence.last_check_results_;
        return result;
    }

    // Solve with explicit convergence policy via solver_options.
    template <typename ExplicitConvergence>
    solve_result<scalar_type, N> solve(const solver_options<ExplicitConvergence>& opts)
    {
        return step_n(opts.max_iterations, opts);
    }

    // Step with budget, default convergence. See solve() above for the
    // rationale on building solver_options<Convergence> and copying the
    // stored tolerances.
    solve_result<scalar_type, N> step_n(std::uint32_t budget)
    {
        solver_options<Convergence> opts;
        opts.max_iterations = max_iterations_;
        opts.max_time = max_time_;
        opts.constraint_tolerance = constraint_tolerance_;
        opts.feasibility_tolerance = feasibility_tolerance_;
        opts.convergence = stored_convergence_;
        auto result = step_n(budget, opts);
        stored_convergence_.last_check_results_ =
            opts.convergence.last_check_results_;
        return result;
    }

    // Step with budget and explicit convergence policy via solver_options.
    //
    // Convergence loop structure:
    //   1. Check abort flag
    //   2. Check time limit between steps (D-11: not inside policy, D-12: steady_clock)
    //   3. Execute policy step
    //   4. Check policy-reported failure (D-16: policy failure is final)
    //   5. Check convergence policy
    template <typename OptsConvergence>
    solve_result<scalar_type, N> step_n(std::uint32_t budget,
                                        const solver_options<OptsConvergence>& opts)
    {
        auto t0 = std::chrono::steady_clock::now();

        solver_status status = solver_status::running;
        step_result<scalar_type> last{};

        // Best-seen tracking (NLopt convention: the returned solve_result
        // reports the best point encountered across the entire loop, not
        // the terminal trial). Seed from the entry iterate x_0 via an
        // explicit pre-loop evaluation -- no state-field probing. The
        // problem interface guarantees value() on every supported
        // Problem (objective concept); cv is seeded from the policy's
        // cached c_eq/c_ineq (populated by policy.init at x_0) using the
        // L-infinity primal-feasibility measure so best_cv is
        // dimensionally consistent with step_result.constraint_violation
        // reported downstream. Policies without a problem binding
        // (projected_gn family, where Problem==void) fall back to +inf
        // sentinels; the first accepted step becomes the best-seen
        // baseline in that case, which still preserves the entry
        // invariant because state_.x is unchanged until step() runs.
        //
        // Reference: nlopt/src/api/nlopt.c nlopt_optimize (best-solution-
        //            returned semantics); N&W 2e Def 12.1 (primal
        //            feasibility, L-infinity composition).
        vector<scalar_type, N> best_x = state_.x;
        scalar_type best_f = seed_best_f();
        scalar_type best_cv = seed_best_cv();
        // User-supplied constraint_tolerance takes precedence over the
        // baked-in feasibility_tolerance default: a caller who bothered
        // to tighten the KKT residual gate expects the best-seen
        // comparator to honor the same floor.
        const scalar_type feas_tol = static_cast<scalar_type>(
            opts.constraint_tolerance.value_or(opts.feasibility_tolerance));

        for(std::uint32_t i = 0; i < budget; ++i)
        {
            if(abort_flag_.load(std::memory_order_relaxed))
            {
                status = solver_status::aborted;
                break;
            }

            // Time limit check (D-11: between steps, not inside policy)
            if(opts.max_time)
            {
                auto elapsed = std::chrono::steady_clock::now() - t0;
                if(elapsed >= *opts.max_time)
                {
                    status = solver_status::time_limit_reached;
                    break;
                }
            }

            // step_impl() (not the public step()) runs the raw iteration:
            // the convergence loop below owns the stopping decision through
            // opts.convergence, so a second per-step convergence consult
            // inside step() would double-advance windowed criteria.
            last = step_impl();

            // Best-seen update: compare after every step including the
            // one that triggers policy_status or convergence, so the
            // terminal iterate itself is eligible to be the winner.
            // state_.x has already been updated by policy.step(); the
            // policy-reported (f, cv) in last are the values at that x.
            if(detail::is_better(last.objective_value, last.constraint_violation,
                                 best_f, best_cv, feas_tol))
            {
                best_x = state_.x;
                best_f = last.objective_value;
                best_cv = last.constraint_violation;
            }

            // Policy-reported failure is final (D-16)
            if(last.policy_status)
            {
                status = *last.policy_status;
                break;
            }

            // Convergence policy check
            auto conv = opts.convergence.check(last, iterations_);
            if(conv)
            {
                status = *conv;
                break;
            }
        }

        if(status == solver_status::running)
        {
            status = (iterations_ >= opts.max_iterations)
                         ? solver_status::max_iterations
                         : solver_status::budget_exhausted;
        }

        last_status_ = status;

        // Back-copy per-criterion telemetry from the caller-owned opts into
        // stored_convergence_ when types match, so solver.convergence()
        // .last_check_results() reflects the most recent step_n(budget, opts)
        // call. For an OptsConvergence that differs from the solver's own
        // stored Convergence type (e.g. passing a solver_options<
        // slsqp_compatible_convergence> to a basic_solver whose Convergence
        // is default_convergence), consumers read last_check_results
        // directly from their own opts.convergence instance.
        if constexpr(std::is_same_v<OptsConvergence, decltype(stored_convergence_)>)
        {
            stored_convergence_.last_check_results_ =
                opts.convergence.last_check_results_;
        }

        auto t1 = std::chrono::steady_clock::now();

        return solve_result<scalar_type, N>{
            .status = status,
            .iterations = iterations_,
            // Genuine accumulated evaluation count (summed from each step's
            // reported step_result::evaluations), not an alias of the
            // iteration count. Policies that evaluate several times per
            // iteration (line searches) report > iterations here.
            .function_evaluations = function_evaluations_,
            .objective_value = best_f,
            .gradient_norm = last.gradient_norm,
            .constraint_violation = best_cv,
            .x = best_x,
            .wall_time = t1 - t0,
        };
    }

    const state_type& state() const { return state_; }

    // Read the stored convergence policy (populated with per-criterion
    // last_check_results by the most recent step_n() call). Precedent:
    // line_search_calls on kraft_slsqp_policy::state_type.
    const auto& convergence() const { return stored_convergence_; }

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

    void abort() { abort_flag_.store(true, std::memory_order_relaxed); }

    // Reset to a new starting point, preserving algorithm state (e.g. L-BFGS
    // curvature pairs). Delegates to policy.reset(). Per D-05.
    void reset(const Eigen::VectorX<scalar_type>& x0)
    {
        policy_.reset(state_, Eigen::Vector<scalar_type, N>(x0));
        iterations_ = 0;
        function_evaluations_ = 0;
        last_gradient_norm_ = std::numeric_limits<scalar_type>::quiet_NaN();
        last_status_ = solver_status::running;
        abort_flag_.store(false, std::memory_order_relaxed);
    }

    // Reset to a new starting point, clearing all algorithm state.
    // Delegates to policy.reset_clear(). Per D-05.
    void reset_clear(const Eigen::VectorX<scalar_type>& x0)
    {
        policy_.reset_clear(state_, Eigen::Vector<scalar_type, N>(x0));
        iterations_ = 0;
        function_evaluations_ = 0;
        last_gradient_norm_ = std::numeric_limits<scalar_type>::quiet_NaN();
        last_status_ = solver_status::running;
        abort_flag_.store(false, std::memory_order_relaxed);
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
    // every basic_solver ctor; once set here the tightened threshold
    // persists through subsequent solve() / step_n(budget) calls that
    // reconstruct local solver_options from stored state.
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
    // legs directly. The max preserves the pre-31.1 unconstrained default
    // (1e-8) whenever constraint_tolerance is unset or looser.
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
        //
        // Reference: Svanberg 1987, "The method of moving asymptotes",
        //            IJNME 24:359-373, Theorem 5.1 (Cauchy-sequence
        //            convergence precondition).
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
    // Raw single iteration shared by the public step() and the step_n()
    // convergence loop. Executes the policy step, fills x_norm centrally
    // (so step_tolerance_rel_criterion has a consistent denominator without
    // per-policy churn -- policies own constraint_violation, previously
    // overwritten here), advances the iteration and evaluation counters, and
    // records the reported gradient norm. Deliberately does NOT consult
    // convergence: the public step() adds that on top, while step_n() owns
    // its own stopping decision.
    step_result<scalar_type> step_impl()
    {
        auto result = policy_.step(state_);
        result.x_norm = state_.x.norm();
        ++iterations_;
        function_evaluations_ += result.evaluations;
        last_gradient_norm_ = result.gradient_norm;
        return result;
    }

    // Seed best-seen f at step_n entry via a direct problem evaluation.
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

    // Seed best-seen cv at step_n entry from the policy's cached
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

    Policy policy_{};
    std::uint32_t max_iterations_{1000};
    std::optional<std::chrono::nanoseconds> max_time_{};
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

// CTAD deduction guides.
//
// These guides extract N from the problem type via problem_dimension_v<Problem>
// and rebind the policy to that dimension via Policy::template rebind<N>.
// Problem is deduced from the constructor argument and stored as the third
// template parameter for policies with template state_type<P>.
//
// The x0 parameter uses Eigen::VectorX<typename Policy::scalar_type> explicitly
// (not a generic X) to match the constructor signature precisely. GCC 15 CTAD
// resolution requires this to correctly deduce Problem when state_type is a
// template -- a generic X parameter causes the guide to be ignored in favour
// of the implicit deduction guide which defaults Problem to void.

// Policy + Problem + x0 + opts
//
// Convergence is deduced here from the opts argument's solver_options<
// Convergence> type and flows into the class's own Convergence template
// parameter, so a non-default Convergence (e.g. a custom relative-tolerance
// policy) is stored without coercion to default_convergence.
template <typename Policy, typename Problem, typename Convergence>
basic_solver(Policy, const Problem&,
             const Eigen::VectorX<typename Policy::scalar_type>&,
             const solver_options<Convergence>&)
    -> basic_solver<typename Policy::template rebind<problem_dimension_v<Problem>>,
                    problem_dimension_v<Problem>,
                    Problem,
                    Convergence>;

// Policy + Problem + x0 (no opts): Convergence is not deducible here, so it
// keeps the class's own default_convergence default.
template <typename Policy, typename Problem>
basic_solver(Policy, const Problem&,
             const Eigen::VectorX<typename Policy::scalar_type>&)
    -> basic_solver<typename Policy::template rebind<problem_dimension_v<Problem>>,
                    problem_dimension_v<Problem>,
                    Problem>;

// Policy + Problem + x0 + opts + policy_opts
template <typename Policy, typename Problem, typename Convergence,
          typename PolicyOpts>
basic_solver(Policy, const Problem&,
             const Eigen::VectorX<typename Policy::scalar_type>&,
             const solver_options<Convergence>&, const PolicyOpts&)
    -> basic_solver<typename Policy::template rebind<problem_dimension_v<Problem>>,
                    problem_dimension_v<Problem>,
                    Problem,
                    Convergence>;

}

#endif
