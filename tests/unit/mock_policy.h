#ifndef HPP_GUARD_NABLAPP_TESTS_MOCK_POLICY_H
#define HPP_GUARD_NABLAPP_TESTS_MOCK_POLICY_H

#include "nablapp/result/step_result.h"
#include "nablapp/solver/options.h"

#include <Eigen/Core>

#include <cmath>

namespace nablapp::test
{

// Mock policy for testing basic_solver.
//
// Implements gradient descent on f(x) = 0.5 * x^T * x (quadratic bowl).
// gradient = x, so each step: x <- x - alpha * x = (1 - alpha) * x.
// With alpha = 0.5, converges geometrically: ||x_k|| = ||x_0|| * 0.5^k.

struct mock_policy
{
    using scalar_type = double;

    struct state_type
    {
        Eigen::VectorXd x;
        double objective_value{};
        double step_size{0.5};
    };

    template <typename Problem>
    static state_type init(const Problem&, const Eigen::VectorXd& x0,
                           const nablapp::solver_options<double>&)
    {
        return state_type{
            .x = x0,
            .objective_value = 0.5 * x0.squaredNorm(),
        };
    }

    static nablapp::step_result<double> step(state_type& s)
    {
        double old_f = s.objective_value;
        s.x *= (1.0 - s.step_size);
        s.objective_value = 0.5 * s.x.squaredNorm();

        return nablapp::step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = s.x.norm(),
            .step_size = s.step_size,
            .objective_change = s.objective_value - old_f,
            .improved = s.objective_value < old_f,
        };
    }
};

// Mock policy that never converges (gradient_norm stays above tolerance).

struct non_converging_policy
{
    using scalar_type = double;

    struct state_type
    {
        Eigen::VectorXd x;
        double objective_value{};
        int step_count{0};
    };

    template <typename Problem>
    static state_type init(const Problem&, const Eigen::VectorXd& x0,
                           const nablapp::solver_options<double>&)
    {
        return state_type{
            .x = x0,
            .objective_value = 100.0,
        };
    }

    static nablapp::step_result<double> step(state_type& s)
    {
        ++s.step_count;
        return nablapp::step_result<double>{
            .objective_value = 100.0,
            .gradient_norm = 10.0,
            .step_size = 1.0,
            .objective_change = -1.0,
            .improved = true,
        };
    }
};

}

#endif
