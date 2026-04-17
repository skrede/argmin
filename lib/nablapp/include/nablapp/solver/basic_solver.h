#ifndef HPP_GUARD_NABLAPP_SOLVER_BASIC_SOLVER_H
#define HPP_GUARD_NABLAPP_SOLVER_BASIC_SOLVER_H

#include "nablapp/types.h"
#include "nablapp/detail/lagrangian.h"
#include "nablapp/result/step_result.h"
#include "nablapp/result/solve_result.h"
#include "nablapp/result/status.h"
#include "nablapp/solver/convergence.h"
#include "nablapp/solver/options.h"

#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <atomic>
#include <chrono>
#include <cmath>
#include <tuple>
#include <concepts>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace nablapp
{

template <typename P>
concept has_options_type = requires { typename P::options_type; };

namespace detail
{
// Tuple type membership trait.
//
// std::get<T>(tuple) is not SFINAE-friendly when T is not present: it
// triggers a static_assert ("the type T in std::get<T> must occur exactly
// once in the tuple") rather than substitution failure. if constexpr on
// `requires { std::get<T>(t); }` therefore silently passes for missing
// types and breaks at instantiation. Use this trait to gate the branches.
template <typename T, typename Tuple>
struct tuple_contains;

template <typename T, typename... Us>
struct tuple_contains<T, std::tuple<Us...>>
    : std::bool_constant<(std::is_same_v<T, Us> || ...)> {};

template <typename T, typename Tuple>
inline constexpr bool tuple_contains_v = tuple_contains<T, Tuple>::value;
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
// Policy contract (deducing this):
//   - Policy::scalar_type         -- scalar type (double, float)
//   - Policy::state_type          or Policy::template state_type<P>
//   - policy.init(problem, x0, opts) -> state_type or state_type<Problem>
//   - policy.step(state)          -> step_result<scalar_type>
//   - policy.reset(state, x0)     -> void
//   - policy.reset_clear(state, x0) -> void
//
// Reference: K&W Section 4.4 (convergence criteria), N&W Section 3.1.

template <typename Policy, int N = dynamic_dimension, typename Problem = void>
class basic_solver
{
public:
    using policy_type = Policy;
    using scalar_type = typename Policy::scalar_type;
    using state_type = detail::resolve_state_t<Policy, Problem>;

    template <typename P, typename Convergence = default_convergence>
    basic_solver(Policy policy, const P& problem,
                 const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts = {})
        : policy_{std::move(policy)}
        , max_iterations_{opts.max_iterations}
        , verbosity_{opts.verbosity}
        , max_time_{opts.max_time}
        , constraint_tolerance_{opts.constraint_tolerance}
        , state_{policy_.init(problem, Eigen::Vector<scalar_type, N>(x0), opts)}
    {
        store_convergence(opts.convergence);
        forward_solver_hints(opts);
        if constexpr(requires { policy_.options; })
            forward_policy_hints(policy_.options);
    }

    // CTAD converting constructor: accepts an un-rebound policy and discards
    // it, default-constructing the rebound policy. Enables deduction guides
    // that rebind Policy to the problem's compile-time dimension.
    template <typename OrigPolicy, typename P,
              typename Convergence = default_convergence>
        requires (!std::same_as<std::remove_cvref_t<OrigPolicy>, Policy>)
              && requires { typename OrigPolicy::template rebind<N>; }
              && std::same_as<Policy, typename OrigPolicy::template rebind<N>>
    basic_solver(OrigPolicy&&, const P& problem,
                 const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts = {})
        : max_iterations_{opts.max_iterations}
        , verbosity_{opts.verbosity}
        , max_time_{opts.max_time}
        , constraint_tolerance_{opts.constraint_tolerance}
        , state_{policy_.init(problem, Eigen::Vector<scalar_type, N>(x0), opts)}
    {
        store_convergence(opts.convergence);
        forward_solver_hints(opts);
        if constexpr(requires { policy_.options; })
            forward_policy_hints(policy_.options);
    }

    template <typename P, typename Convergence = default_convergence>
    basic_solver(const P& problem, const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts = {})
        : max_iterations_{opts.max_iterations}
        , verbosity_{opts.verbosity}
        , max_time_{opts.max_time}
        , constraint_tolerance_{opts.constraint_tolerance}
        , state_{policy_.init(problem, Eigen::Vector<scalar_type, N>(x0), opts)}
    {
        store_convergence(opts.convergence);
        forward_solver_hints(opts);
        if constexpr(requires { policy_.options; })
            forward_policy_hints(policy_.options);
    }

    template <typename P, typename PolicyOpts, typename Convergence = default_convergence>
        requires has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    basic_solver(Policy policy, const P& problem,
                 const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts,
                 const PolicyOpts& policy_opts)
        : policy_{std::move(policy)}
        , max_iterations_{opts.max_iterations}
        , verbosity_{opts.verbosity}
        , max_time_{opts.max_time}
        , constraint_tolerance_{opts.constraint_tolerance}
        , state_{policy_.init(problem, Eigen::Vector<scalar_type, N>(x0), opts, policy_opts)}
    {
        store_convergence(opts.convergence);
        forward_solver_hints(opts);
        forward_policy_hints(policy_opts);
    }

    // CTAD converting constructor with policy options.
    template <typename OrigPolicy, typename P, typename PolicyOpts,
              typename Convergence = default_convergence>
        requires (!std::same_as<std::remove_cvref_t<OrigPolicy>, Policy>)
              && requires { typename OrigPolicy::template rebind<N>; }
              && std::same_as<Policy, typename OrigPolicy::template rebind<N>>
              && has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    basic_solver(OrigPolicy&&, const P& problem,
                 const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts,
                 const PolicyOpts& policy_opts)
        : max_iterations_{opts.max_iterations}
        , verbosity_{opts.verbosity}
        , max_time_{opts.max_time}
        , constraint_tolerance_{opts.constraint_tolerance}
        , state_{policy_.init(problem, Eigen::Vector<scalar_type, N>(x0), opts, policy_opts)}
    {
        store_convergence(opts.convergence);
        forward_solver_hints(opts);
        forward_policy_hints(policy_opts);
    }

    template <typename P, typename PolicyOpts, typename Convergence = default_convergence>
        requires has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    basic_solver(const P& problem, const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts,
                 const PolicyOpts& policy_opts)
        : max_iterations_{opts.max_iterations}
        , verbosity_{opts.verbosity}
        , max_time_{opts.max_time}
        , constraint_tolerance_{opts.constraint_tolerance}
        , state_{policy_.init(problem, Eigen::Vector<scalar_type, N>(x0), opts, policy_opts)}
    {
        store_convergence(opts.convergence);
        forward_solver_hints(opts);
        forward_policy_hints(policy_opts);
    }

    // Overload for policies without options_type: the fourth argument is
    // solver_options<> (the policy_options_t fallback) and is ignored.
    template <typename P, typename Convergence = default_convergence>
        requires (!has_options_type<Policy>)
    basic_solver(const P& problem, const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts,
                 const solver_options<>&)
        : max_iterations_{opts.max_iterations}
        , verbosity_{opts.verbosity}
        , max_time_{opts.max_time}
        , constraint_tolerance_{opts.constraint_tolerance}
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
        , verbosity_{other.verbosity_}
        , max_time_{other.max_time_}
        , constraint_tolerance_{other.constraint_tolerance_}
        , stored_convergence_{other.stored_convergence_}
        , state_{std::move(other.state_)}
        , iterations_{other.iterations_}
        , last_status_{other.last_status_}
        , abort_flag_{other.abort_flag_.load(std::memory_order_relaxed)}
    {}

    basic_solver& operator=(basic_solver&& other) noexcept
    {
        if(this != &other)
        {
            policy_ = std::move(other.policy_);
            max_iterations_ = other.max_iterations_;
            verbosity_ = other.verbosity_;
            max_time_ = other.max_time_;
            constraint_tolerance_ = other.constraint_tolerance_;
            stored_convergence_ = other.stored_convergence_;
            state_ = std::move(other.state_);
            iterations_ = other.iterations_;
            last_status_ = other.last_status_;
            abort_flag_.store(other.abort_flag_.load(std::memory_order_relaxed),
                              std::memory_order_relaxed);
        }
        return *this;
    }

    step_result<scalar_type> step()
    {
        auto result = policy_.step(state_);
        // Central x_norm fill so step_tolerance_rel_criterion can use a
        // consistent denominator without per-policy churn. Policies own
        // constraint_violation (previously overwritten here, which destroyed
        // filter-policy semantics where the reported violation corresponds
        // to the returned trial/restoration step).
        result.x_norm = state_.x.norm();
        ++iterations_;
        return result;
    }

    // Solve with default convergence (uses stored max_iterations).
    solve_result<scalar_type, N> solve()
    {
        solver_options<> opts;
        opts.max_iterations = max_iterations_;
        opts.max_time = max_time_;
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
    template <typename Convergence>
    solve_result<scalar_type, N> solve(const solver_options<Convergence>& opts)
    {
        return step_n(opts.max_iterations, opts);
    }

    // Step with budget, default convergence.
    solve_result<scalar_type, N> step_n(std::uint32_t budget)
    {
        solver_options<> opts;
        opts.max_iterations = max_iterations_;
        opts.max_time = max_time_;
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
    template <typename Convergence>
    solve_result<scalar_type, N> step_n(std::uint32_t budget,
                                        const solver_options<Convergence>& opts)
    {
        auto t0 = std::chrono::steady_clock::now();

        solver_status status = solver_status::running;
        step_result<scalar_type> last{};

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

            last = step();

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
        // call. For Convergence != default_convergence (e.g. the
        // slsqp_compatible_convergence alias), consumers read last_check_results
        // directly from their own opts.convergence instance.
        if constexpr(std::is_same_v<Convergence, default_convergence>)
        {
            stored_convergence_.last_check_results_ =
                opts.convergence.last_check_results_;
        }

        auto t1 = std::chrono::steady_clock::now();

        return solve_result<scalar_type, N>{
            .status = status,
            .iterations = iterations_,
            .function_evaluations = iterations_,
            .objective_value = last.objective_value,
            .gradient_norm = last.gradient_norm,
            .constraint_violation = constraint_violation(),
            .x = state_.x,
            .wall_time = t1 - t0,
        };
    }

    const state_type& state() const { return state_; }

    // Read the stored convergence policy (populated with per-criterion
    // last_check_results by the most recent step_n() call). Precedent:
    // line_search_calls on kraft_slsqp_policy::state_type.
    const auto& convergence() const { return stored_convergence_; }

    solver_status status() const { return last_status_; }

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
        last_status_ = solver_status::running;
        abort_flag_.store(false, std::memory_order_relaxed);
    }

    // Reset to a new starting point, clearing all algorithm state.
    // Delegates to policy.reset_clear(). Per D-05.
    void reset_clear(const Eigen::VectorX<scalar_type>& x0)
    {
        policy_.reset_clear(state_, Eigen::Vector<scalar_type, N>(x0));
        iterations_ = 0;
        last_status_ = solver_status::running;
        abort_flag_.store(false, std::memory_order_relaxed);
    }

    template <typename Criterion>
    void store_criterion(const Criterion& c)
    {
        if constexpr(std::same_as<Criterion, gradient_tolerance_criterion>)
        {
            if constexpr(requires { std::get<gradient_tolerance_criterion>(stored_convergence_.criteria); })
                std::get<gradient_tolerance_criterion>(stored_convergence_.criteria).threshold = c.threshold;
        }
        else if constexpr(std::same_as<Criterion, objective_tolerance_criterion>)
        {
            if constexpr(requires { std::get<objective_tolerance_criterion>(stored_convergence_.criteria); })
            {
                auto& dst = std::get<objective_tolerance_criterion>(stored_convergence_.criteria);
                dst.threshold = c.threshold;
                dst.stationarity_threshold = c.stationarity_threshold;
            }
        }
        else if constexpr(std::same_as<Criterion, step_tolerance_criterion>)
        {
            if constexpr(requires { std::get<step_tolerance_criterion>(stored_convergence_.criteria); })
                std::get<step_tolerance_criterion>(stored_convergence_.criteria).threshold = c.threshold;
        }
        else if constexpr(std::same_as<Criterion, objective_tolerance_rel_criterion>)
        {
            using criteria_tuple = decltype(stored_convergence_.criteria);
            if constexpr(detail::tuple_contains_v<objective_tolerance_rel_criterion, criteria_tuple>)
            {
                auto& dst = std::get<objective_tolerance_rel_criterion>(stored_convergence_.criteria);
                dst.threshold = c.threshold;
                dst.stationarity_threshold = c.stationarity_threshold;
            }
        }
        else if constexpr(std::same_as<Criterion, step_tolerance_rel_criterion>)
        {
            using criteria_tuple = decltype(stored_convergence_.criteria);
            if constexpr(detail::tuple_contains_v<step_tolerance_rel_criterion, criteria_tuple>)
                std::get<step_tolerance_rel_criterion>(stored_convergence_.criteria).threshold = c.threshold;
        }
        else if constexpr(std::same_as<Criterion, stall_tolerance_criterion>)
        {
            if constexpr(requires { std::get<stall_tolerance_criterion>(stored_convergence_.criteria); })
            {
                auto& dst = std::get<stall_tolerance_criterion>(stored_convergence_.criteria);
                dst.threshold = c.threshold;
                dst.window = c.window;
            }
        }
    }

    template <typename Convergence>
    void store_convergence(const Convergence& conv)
    {
        if constexpr(requires { conv.criteria; })
        {
            std::apply([this](const auto&... cs) {
                (store_criterion(cs), ...);
            }, conv.criteria);
        }
        if constexpr(requires { conv.inner; })
        {
            store_convergence(conv.inner);
        }
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
    template <typename Convergence>
    void forward_solver_hints(const solver_options<Convergence>& opts)
    {
        using criteria_tuple = decltype(stored_convergence_.criteria);
        if constexpr(detail::tuple_contains_v<objective_tolerance_criterion, criteria_tuple>)
        {
            auto& ftol = std::get<objective_tolerance_criterion>(stored_convergence_.criteria);
            if(opts.constraint_tolerance)
            {
                const double current = ftol.stationarity_threshold.value_or(1e-8);
                ftol.stationarity_threshold = std::max(current, *opts.constraint_tolerance);
            }
        }
        if constexpr(detail::tuple_contains_v<objective_tolerance_rel_criterion, criteria_tuple>)
        {
            auto& ftol_rel = std::get<objective_tolerance_rel_criterion>(stored_convergence_.criteria);
            if(opts.constraint_tolerance)
            {
                const double current = ftol_rel.stationarity_threshold.value_or(1e-8);
                ftol_rel.stationarity_threshold = std::max(current, *opts.constraint_tolerance);
            }
        }
    }

    template <typename PolicyOpts>
    void forward_policy_hints(const PolicyOpts& policy_opts)
    {
        if constexpr(requires { policy_opts.stall_window; })
        {
            if constexpr(requires { std::get<stall_tolerance_criterion>(stored_convergence_.criteria); })
            {
                auto& stall = std::get<stall_tolerance_criterion>(stored_convergence_.criteria);
                stall.window = policy_opts.stall_window;
            }
        }
    }

private:
    Policy policy_{};
    std::uint32_t max_iterations_{1000};
    std::uint8_t verbosity_{0};
    std::optional<std::chrono::nanoseconds> max_time_{};
    std::optional<double> constraint_tolerance_{};
    default_convergence stored_convergence_{};
    state_type state_;
    std::uint32_t iterations_{0};
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
template <typename Policy, typename Problem, typename Convergence>
basic_solver(Policy, const Problem&,
             const Eigen::VectorX<typename Policy::scalar_type>&,
             const solver_options<Convergence>&)
    -> basic_solver<typename Policy::template rebind<problem_dimension_v<Problem>>,
                    problem_dimension_v<Problem>,
                    Problem>;

// Policy + Problem + x0 (no opts)
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
                    Problem>;

// D-04: convenience type aliases for common configurations.

template <typename Policy, int N = dynamic_dimension, typename Problem = void>
using realtime_solver = basic_solver<Policy, N, Problem>;

template <typename Policy, int N = dynamic_dimension, typename Problem = void>
using gradient_solver = basic_solver<Policy, N, Problem>;

}

#endif
