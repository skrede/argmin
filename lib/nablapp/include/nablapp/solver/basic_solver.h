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

// Solver wrapper with convergence loop (per CORE-02, D-09, D-10, D-11, D-13, D-14).
//
// basic_solver wraps a Policy instance that defines the algorithm logic.
// The solver provides the iterative execution model: step(), solve(),
// step_n(budget). Convergence checking is performed by basic_solver --
// the policy does NOT know about convergence.
//
// basic_solver<Policy, N> has two template parameters: the policy and
// the compile-time dimension N. N is deduced from the problem via CTAD
// deduction guides and is used to rebind the policy to the correct
// dimension. Users never write N explicitly.
//
// Policy contract (deducing this):
//   - Policy::scalar_type         -- scalar type (double, float)
//   - Policy::state_type          -- owns problem ref, current iterate, algorithm state
//   - policy.init(problem, x0, opts) -> state_type
//   - policy.step(state)          -> step_result<scalar_type>
//   - policy.reset(state, x0)     -> void
//   - policy.reset_clear(state, x0) -> void
//
// Reference: K&W Section 4.4 (convergence criteria), N&W Section 3.1.

template <typename Policy, int N = dynamic_dimension>
class basic_solver
{
public:
    using policy_type = Policy;
    using scalar_type = typename Policy::scalar_type;
    using state_type = typename Policy::state_type;

    template <typename Problem, typename Convergence = default_convergence>
    basic_solver(Policy policy, const Problem& problem,
                 const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts = {})
        : policy_{std::move(policy)}
        , max_iterations_{opts.max_iterations}
        , verbosity_{opts.verbosity}
        , max_time_{opts.max_time}
        , constraint_tolerance_{opts.constraint_tolerance}
        , state_{policy_.init(problem, Eigen::Vector<scalar_type, N>(x0), opts)}
    {}

    // CTAD converting constructor: accepts an un-rebound policy and discards
    // it, default-constructing the rebound policy. Enables deduction guides
    // that rebind Policy to the problem's compile-time dimension.
    template <typename OrigPolicy, typename Problem,
              typename Convergence = default_convergence>
        requires (!std::same_as<std::remove_cvref_t<OrigPolicy>, Policy>)
              && requires { typename OrigPolicy::template rebind<N>; }
              && std::same_as<Policy, typename OrigPolicy::template rebind<N>>
    basic_solver(OrigPolicy&&, const Problem& problem,
                 const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts = {})
        : max_iterations_{opts.max_iterations}
        , verbosity_{opts.verbosity}
        , max_time_{opts.max_time}
        , constraint_tolerance_{opts.constraint_tolerance}
        , state_{policy_.init(problem, Eigen::Vector<scalar_type, N>(x0), opts)}
    {}

    template <typename Problem, typename Convergence = default_convergence>
    basic_solver(const Problem& problem, const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts = {})
        : max_iterations_{opts.max_iterations}
        , verbosity_{opts.verbosity}
        , max_time_{opts.max_time}
        , constraint_tolerance_{opts.constraint_tolerance}
        , state_{policy_.init(problem, Eigen::Vector<scalar_type, N>(x0), opts)}
    {}

    template <typename Problem, typename PolicyOpts, typename Convergence = default_convergence>
        requires has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    basic_solver(Policy policy, const Problem& problem,
                 const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts,
                 const PolicyOpts& policy_opts)
        : policy_{std::move(policy)}
        , max_iterations_{opts.max_iterations}
        , verbosity_{opts.verbosity}
        , max_time_{opts.max_time}
        , constraint_tolerance_{opts.constraint_tolerance}
        , state_{policy_.init(problem, Eigen::Vector<scalar_type, N>(x0), opts, policy_opts)}
    {}

    // CTAD converting constructor with policy options.
    template <typename OrigPolicy, typename Problem, typename PolicyOpts,
              typename Convergence = default_convergence>
        requires (!std::same_as<std::remove_cvref_t<OrigPolicy>, Policy>)
              && requires { typename OrigPolicy::template rebind<N>; }
              && std::same_as<Policy, typename OrigPolicy::template rebind<N>>
              && has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    basic_solver(OrigPolicy&&, const Problem& problem,
                 const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts,
                 const PolicyOpts& policy_opts)
        : max_iterations_{opts.max_iterations}
        , verbosity_{opts.verbosity}
        , max_time_{opts.max_time}
        , constraint_tolerance_{opts.constraint_tolerance}
        , state_{policy_.init(problem, Eigen::Vector<scalar_type, N>(x0), opts, policy_opts)}
    {}

    template <typename Problem, typename PolicyOpts, typename Convergence = default_convergence>
        requires has_options_type<Policy>
              && std::same_as<std::remove_cvref_t<PolicyOpts>, typename Policy::options_type>
    basic_solver(const Problem& problem, const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts,
                 const PolicyOpts& policy_opts)
        : max_iterations_{opts.max_iterations}
        , verbosity_{opts.verbosity}
        , max_time_{opts.max_time}
        , constraint_tolerance_{opts.constraint_tolerance}
        , state_{policy_.init(problem, Eigen::Vector<scalar_type, N>(x0), opts, policy_opts)}
    {}

    // Overload for policies without options_type: the fourth argument is
    // solver_options<> (the policy_options_t fallback) and is ignored.
    template <typename Problem, typename Convergence = default_convergence>
        requires (!has_options_type<Policy>)
    basic_solver(const Problem& problem, const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<Convergence>& opts,
                 const solver_options<>&)
        : max_iterations_{opts.max_iterations}
        , verbosity_{opts.verbosity}
        , max_time_{opts.max_time}
        , constraint_tolerance_{opts.constraint_tolerance}
        , state_{policy_.init(problem, Eigen::Vector<scalar_type, N>(x0), opts)}
    {}

    basic_solver(const basic_solver&) = delete;
    basic_solver& operator=(const basic_solver&) = delete;

    basic_solver(basic_solver&& other) noexcept
        : policy_{std::move(other.policy_)}
        , max_iterations_{other.max_iterations_}
        , verbosity_{other.verbosity_}
        , max_time_{other.max_time_}
        , constraint_tolerance_{other.constraint_tolerance_}
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
        result.constraint_violation = constraint_violation();
        ++iterations_;
        return result;
    }

    // Solve with default convergence (uses stored max_iterations).
    solve_result<scalar_type, N> solve()
    {
        solver_options<> opts;
        opts.max_iterations = max_iterations_;
        opts.max_time = max_time_;
        return step_n(max_iterations_, opts);
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
        return step_n(budget, opts);
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

private:
    Policy policy_{};
    std::uint32_t max_iterations_{1000};
    std::uint8_t verbosity_{0};
    std::optional<std::chrono::nanoseconds> max_time_{};
    std::optional<double> constraint_tolerance_{};
    state_type state_;
    std::uint32_t iterations_{0};
    solver_status last_status_{solver_status::running};
    std::atomic<bool> abort_flag_{false};
};

// CTAD deduction guides.
//
// These guides extract N from the problem type via problem_dimension_v<Problem>
// and rebind the policy to that dimension via Policy::template rebind<N>.
// Users write basic_solver solver(lbfgsb_policy{}, problem, x0, opts)
// and get basic_solver<lbfgsb_policy<problem_dimension_v<Problem>>,
//                       problem_dimension_v<Problem>>.

// Policy + Problem + x0 + opts
template <typename Policy, typename Problem, typename X, typename Convergence>
basic_solver(Policy, const Problem&, const X&,
             const solver_options<Convergence>&)
    -> basic_solver<typename Policy::template rebind<problem_dimension_v<Problem>>,
                    problem_dimension_v<Problem>>;

// Policy + Problem + x0 (no opts)
template <typename Policy, typename Problem, typename X>
basic_solver(Policy, const Problem&, const X&)
    -> basic_solver<typename Policy::template rebind<problem_dimension_v<Problem>>,
                    problem_dimension_v<Problem>>;

// Policy + Problem + x0 + opts + policy_opts
template <typename Policy, typename Problem, typename X, typename Convergence,
          typename PolicyOpts>
basic_solver(Policy, const Problem&, const X&,
             const solver_options<Convergence>&, const PolicyOpts&)
    -> basic_solver<typename Policy::template rebind<problem_dimension_v<Problem>>,
                    problem_dimension_v<Problem>>;

// D-04: convenience type aliases for common configurations.
// These are identity aliases -- basic_solver has two template parameters,
// and the convergence configuration flows through solver_options at call time.

template <typename Policy, int N = dynamic_dimension>
using realtime_solver = basic_solver<Policy, N>;

template <typename Policy, int N = dynamic_dimension>
using gradient_solver = basic_solver<Policy, N>;

}

#endif
