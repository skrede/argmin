#ifndef HPP_GUARD_NABLAPP_SOLVER_BASIC_SOLVER_H
#define HPP_GUARD_NABLAPP_SOLVER_BASIC_SOLVER_H

#include "nablapp/result/step_result.h"
#include "nablapp/result/solve_result.h"
#include "nablapp/result/status.h"
#include "nablapp/solver/options.h"

#include <Eigen/Core>

#include <chrono>
#include <cmath>

namespace nablapp
{

// Solver wrapper with convergence loop (per CORE-02, D-09, D-10, D-11).
//
// basic_solver wraps a stateless Policy that defines the algorithm logic.
// The solver provides the iterative execution model: step(), solve(),
// step_n(budget). Convergence checking is performed by basic_solver --
// the policy does NOT know about convergence.
//
// Policy contract:
//   - Policy::scalar_type         -- scalar type (double, float)
//   - Policy::state_type          -- owns problem ref, current iterate, algorithm state
//   - Policy::init(problem, x0, opts) -> state_type
//   - Policy::step(state)         -> step_result<scalar_type>
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
    basic_solver(const Problem& problem, const Eigen::VectorX<scalar_type>& x0,
                 const solver_options<scalar_type>& opts = {})
        : options_{opts}
        , state_{Policy::init(problem, x0, opts)}
    {}

    step_result<scalar_type> step()
    {
        auto result = Policy::step(state_);
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
            last = step();

            if(last.gradient_norm < options_.gradient_tolerance)
            {
                status = solver_status::converged;
                break;
            }

            if(iterations_ > 1
               && std::abs(last.objective_change) < options_.objective_tolerance)
            {
                status = solver_status::converged;
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

    // Reset to a new starting point, preserving algorithm state (e.g. L-BFGS
    // curvature pairs). Delegates to Policy::reset(). Per D-05.
    void reset(const Eigen::VectorX<scalar_type>& x0)
    {
        Policy::reset(state_, x0);
        iterations_ = 0;
    }

    // Reset to a new starting point, clearing all algorithm state.
    // Delegates to Policy::reset_clear(). Per D-05.
    void reset_clear(const Eigen::VectorX<scalar_type>& x0)
    {
        Policy::reset_clear(state_, x0);
        iterations_ = 0;
    }

private:
    solver_options<scalar_type> options_;
    state_type state_;
    int iterations_{0};
};

}

#endif
