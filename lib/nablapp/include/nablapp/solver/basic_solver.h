#ifndef HPP_GUARD_NABLAPP_SOLVER_BASIC_SOLVER_H
#define HPP_GUARD_NABLAPP_SOLVER_BASIC_SOLVER_H

#include "nablapp/result/step_result.h"
#include "nablapp/result/solve_result.h"
#include "nablapp/result/status.h"
#include "nablapp/solver/options.h"

#include <Eigen/Core>

#include <atomic>
#include <chrono>
#include <cmath>

namespace nablapp
{

// Solver wrapper with convergence loop (per CORE-02, D-09, D-10, D-11, D-13, D-14).
//
// basic_solver wraps a Policy instance that defines the algorithm logic.
// The solver provides the iterative execution model: step(), solve(),
// step_n(budget). Convergence checking is performed by basic_solver --
// the policy does NOT know about convergence.
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

template <typename Policy>
class basic_solver
{
public:
    using policy_type = Policy;
    using scalar_type = typename Policy::scalar_type;
    using state_type = typename Policy::state_type;

    template <typename Problem>
    basic_solver(Policy policy, const Problem& problem,
                 const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<scalar_type>& opts = {})
        : policy_{std::move(policy)}
        , options_{opts}
        , state_{policy_.init(problem, x0, opts)}
    {}

    template <typename Problem>
    basic_solver(const Problem& problem, const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<scalar_type>& opts = {})
        : options_{opts}
        , state_{policy_.init(problem, x0, opts)}
    {}

    basic_solver(const basic_solver&) = delete;
    basic_solver& operator=(const basic_solver&) = delete;

    basic_solver(basic_solver&& other) noexcept
        : policy_{std::move(other.policy_)}
        , options_{std::move(other.options_)}
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
            options_ = std::move(other.options_);
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
        ++iterations_;
        return result;
    }

    solve_result<scalar_type> solve()
    {
        return step_n(options_.max_iterations);
    }

    solve_result<scalar_type> step_n(int budget)
    {
        auto t0 = std::chrono::steady_clock::now();

        solver_status status = solver_status::running;
        step_result<scalar_type> last{};

        for(int i = 0; i < budget; ++i)
        {
            if(abort_flag_.load(std::memory_order_relaxed))
            {
                status = solver_status::aborted;
                break;
            }

            last = step();

            if(last.gradient_norm < options_.gradient_tolerance)
            {
                status = solver_status::converged;
                break;
            }

            if(iterations_ > 1
               && std::abs(last.objective_change) < options_.objective_tolerance)
            {
                status = solver_status::ftol_reached;
                break;
            }

            if(last.step_size < options_.step_tolerance && iterations_ > 1)
            {
                status = solver_status::stalled;
                break;
            }
        }

        if(status == solver_status::running)
        {
            status = (iterations_ >= options_.max_iterations)
                         ? solver_status::max_iterations
                         : solver_status::budget_exhausted;
        }

        last_status_ = status;

        auto t1 = std::chrono::steady_clock::now();

        return solve_result<scalar_type>{
            .status = status,
            .iterations = iterations_,
            .function_evaluations = iterations_,
            .objective_value = last.objective_value,
            .gradient_norm = last.gradient_norm,
            .x = state_.x,
            .wall_time = t1 - t0,
        };
    }

    const state_type& state() const { return state_; }

    solver_status status() const { return last_status_; }

    void abort() { abort_flag_.store(true, std::memory_order_relaxed); }

    // Reset to a new starting point, preserving algorithm state (e.g. L-BFGS
    // curvature pairs). Delegates to policy.reset(). Per D-05.
    void reset(const Eigen::VectorX<scalar_type>& x0)
    {
        policy_.reset(state_, x0);
        iterations_ = 0;
        last_status_ = solver_status::running;
        abort_flag_.store(false, std::memory_order_relaxed);
    }

    // Reset to a new starting point, clearing all algorithm state.
    // Delegates to policy.reset_clear(). Per D-05.
    void reset_clear(const Eigen::VectorX<scalar_type>& x0)
    {
        policy_.reset_clear(state_, x0);
        iterations_ = 0;
        last_status_ = solver_status::running;
        abort_flag_.store(false, std::memory_order_relaxed);
    }

private:
    Policy policy_{};
    solver_options<scalar_type> options_;
    state_type state_;
    int iterations_{0};
    solver_status last_status_{solver_status::running};
    std::atomic<bool> abort_flag_{false};
};

// CTAD deduction guides (per D-13).

template <typename Policy, typename Problem, typename Scalar>
basic_solver(Policy, const Problem&, const Eigen::VectorX<Scalar>&,
             const solver_options<Scalar>&) -> basic_solver<Policy>;

template <typename Policy, typename Problem, typename Scalar>
basic_solver(Policy, const Problem&, const Eigen::VectorX<Scalar>&)
    -> basic_solver<Policy>;

}

#endif
