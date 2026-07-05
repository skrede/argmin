#ifndef HPP_GUARD_ARGMIN_TESTS_MOCK_POLICY_H
#define HPP_GUARD_ARGMIN_TESTS_MOCK_POLICY_H

#include "argmin/result/step_result.h"
#include "argmin/result/status.h"
#include "argmin/solver/options.h"

#include <Eigen/Core>

#include <cmath>
#include <cstdint>

namespace argmin::test
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

    template <typename Problem, typename Convergence = argmin::default_convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0,
                    const argmin::solver_options<Convergence>&)
    {
        return state_type{
            .x = x0,
            .objective_value = 0.5 * x0.squaredNorm(),
        };
    }

    argmin::step_result<double> step(state_type& s)
    {
        double old_f = s.objective_value;
        s.x *= (1.0 - s.step_size);
        s.objective_value = 0.5 * s.x.squaredNorm();

        return argmin::step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = s.x.norm(),
            .step_size = s.step_size,
            .objective_change = s.objective_value - old_f,
            .improved = s.objective_value < old_f,
            .x_norm = s.x.norm(),
        };
    }

    void reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
        s.objective_value = 0.5 * x0.squaredNorm();
    }

    void reset_clear(state_type& s, const Eigen::VectorXd& x0)
    {
        reset(s, x0);
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

    template <typename Problem, typename Convergence = argmin::default_convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0,
                    const argmin::solver_options<Convergence>&)
    {
        return state_type{
            .x = x0,
            .objective_value = 100.0,
        };
    }

    argmin::step_result<double> step(state_type& s)
    {
        ++s.step_count;
        return argmin::step_result<double>{
            .objective_value = 100.0,
            .gradient_norm = 10.0,
            .step_size = 1.0,
            .objective_change = -1.0,
            .improved = true,
        };
    }

    void reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
        s.objective_value = 100.0;
        s.step_count = 0;
    }

    void reset_clear(state_type& s, const Eigen::VectorXd& x0)
    {
        reset(s, x0);
    }
};

// Constrained mock policy for testing constraint propagation.
//
// Solves f(x) = 0.5 * x^T * x subject to x(0) >= 1.
// Inequality constraint: c_ineq(0) = x(0) - 1.0 (negative means violated).

struct constrained_mock_policy
{
    using scalar_type = double;

    struct state_type
    {
        Eigen::VectorXd x;
        double objective_value{};
        double step_size{0.5};
        Eigen::VectorXd c_eq;
        Eigen::VectorXd c_ineq;
    };

    template <typename Problem, typename Convergence = argmin::default_convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0,
                    const argmin::solver_options<Convergence>&)
    {
        Eigen::VectorXd c_ineq(1);
        c_ineq(0) = x0(0) - 1.0;

        return state_type{
            .x = x0,
            .objective_value = 0.5 * x0.squaredNorm(),
            .c_eq = Eigen::VectorXd(0),
            .c_ineq = c_ineq,
        };
    }

    argmin::step_result<double> step(state_type& s)
    {
        double old_f = s.objective_value;
        s.x *= (1.0 - s.step_size);
        s.objective_value = 0.5 * s.x.squaredNorm();
        s.c_ineq(0) = s.x(0) - 1.0;

        const double violation = std::max(0.0, -s.c_ineq(0));

        return argmin::step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = s.x.norm(),
            .step_size = s.step_size,
            .objective_change = s.objective_value - old_f,
            .improved = s.objective_value < old_f,
            .constraint_violation = violation,
        };
    }

    void reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
        s.objective_value = 0.5 * x0.squaredNorm();
        s.c_ineq(0) = x0(0) - 1.0;
    }

    void reset_clear(state_type& s, const Eigen::VectorXd& x0)
    {
        reset(s, x0);
    }
};

// Mock policy that is always feasible with high objective.
// Used for feasibility-first ranking tests.

struct feasible_mock_policy
{
    using scalar_type = double;

    struct state_type
    {
        Eigen::VectorXd x;
        double objective_value{};
        Eigen::VectorXd c_eq;
        Eigen::VectorXd c_ineq;
    };

    template <typename Problem, typename Convergence = argmin::default_convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0,
                    const argmin::solver_options<Convergence>&)
    {
        return state_type{
            .x = x0,
            .objective_value = 10.0,
            .c_eq = Eigen::VectorXd(0),
            .c_ineq = Eigen::VectorXd{{1.0}},
        };
    }

    argmin::step_result<double> step(state_type& s)
    {
        return argmin::step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = s.x.norm(),
            .step_size = 0.1,
            .objective_change = 0.0,
            .improved = false,
        };
    }

    void reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
    }

    void reset_clear(state_type& s, const Eigen::VectorXd& x0)
    {
        reset(s, x0);
    }
};

// Mock policy that is always infeasible with low objective.
// Used for feasibility-first ranking tests.

struct infeasible_mock_policy
{
    using scalar_type = double;

    struct state_type
    {
        Eigen::VectorXd x;
        double objective_value{};
        Eigen::VectorXd c_eq;
        Eigen::VectorXd c_ineq;
    };

    template <typename Problem, typename Convergence = argmin::default_convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0,
                    const argmin::solver_options<Convergence>&)
    {
        return state_type{
            .x = x0,
            .objective_value = 1.0,
            .c_eq = Eigen::VectorXd(0),
            .c_ineq = Eigen::VectorXd{{-2.0}},
        };
    }

    argmin::step_result<double> step(state_type& s)
    {
        return argmin::step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = s.x.norm(),
            .step_size = 0.1,
            .objective_change = 0.0,
            .improved = false,
        };
    }

    void reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
    }

    void reset_clear(state_type& s, const Eigen::VectorXd& x0)
    {
        reset(s, x0);
    }
};

// Mock policy that is infeasible with high violation.
// Used for testing "lowest violation wins among infeasible" ranking.

struct high_violation_mock_policy
{
    using scalar_type = double;

    struct state_type
    {
        Eigen::VectorXd x;
        double objective_value{};
        Eigen::VectorXd c_eq;
        Eigen::VectorXd c_ineq;
    };

    template <typename Problem, typename Convergence = argmin::default_convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0,
                    const argmin::solver_options<Convergence>&)
    {
        return state_type{
            .x = x0,
            .objective_value = 0.5,
            .c_eq = Eigen::VectorXd(0),
            .c_ineq = Eigen::VectorXd{{-5.0}},
        };
    }

    argmin::step_result<double> step(state_type& s)
    {
        return argmin::step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = s.x.norm(),
            .step_size = 0.1,
            .objective_change = 0.0,
            .improved = false,
        };
    }

    void reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
    }

    void reset_clear(state_type& s, const Eigen::VectorXd& x0)
    {
        reset(s, x0);
    }
};

// Mock policy that reports converged after exactly 2 steps.
// Used for testing that a converged solver is retired by the group (never
// re-stepped) rather than kept in rotation forever.

struct converging_mock_policy
{
    using scalar_type = double;

    struct state_type
    {
        Eigen::VectorXd x;
        double objective_value{};
        std::uint32_t step_count{};
    };

    template <typename Problem, typename Convergence = argmin::default_convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0,
                    const argmin::solver_options<Convergence>&)
    {
        return {.x = x0, .objective_value = 0.5 * x0.squaredNorm()};
    }

    argmin::step_result<double> step(state_type& s)
    {
        ++s.step_count;
        s.x *= 0.5;
        s.objective_value = 0.5 * s.x.squaredNorm();
        argmin::step_result<double> r{
            .objective_value = s.objective_value,
            .gradient_norm = s.x.norm(),
            .step_size = 0.5,
            .objective_change = -s.objective_value,
            .improved = true,
            .x_norm = s.x.norm(),
        };
        if(s.step_count >= 2)
            r.policy_status = argmin::solver_status::converged;
        return r;
    }

    void reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
        s.objective_value = 0.5 * x0.squaredNorm();
        s.step_count = 0;
    }

    void reset_clear(state_type& s, const Eigen::VectorXd& x0)
    {
        reset(s, x0);
    }
};

// Mock policy at the feasibility_tolerance boundary: constraint_violation
// is 5e-7, strictly inside basic_solver's default feasibility_tolerance
// (1e-6, options.h) but strictly outside a bare-zero tolerance. Used to
// check that basic_solver_group's feasibility default agrees with
// basic_solver's, rather than judging this borderline iterate infeasible.

struct borderline_feasible_mock_policy
{
    using scalar_type = double;

    struct state_type
    {
        Eigen::VectorXd x;
        double objective_value{};
        Eigen::VectorXd c_eq;
        Eigen::VectorXd c_ineq;
    };

    template <typename Problem, typename Convergence = argmin::default_convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0,
                    const argmin::solver_options<Convergence>&)
    {
        return state_type{
            .x = x0,
            .objective_value = 10.0,
            .c_eq = Eigen::VectorXd(0),
            .c_ineq = Eigen::VectorXd{{-5e-7}},
        };
    }

    argmin::step_result<double> step(state_type& s)
    {
        return argmin::step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = s.x.norm(),
            .step_size = 0.1,
            .objective_change = 0.0,
            .improved = false,
            .constraint_violation = 5e-7,
        };
    }

    void reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
    }

    void reset_clear(state_type& s, const Eigen::VectorXd& x0)
    {
        reset(s, x0);
    }
};

// Mock policy that is exactly feasible (cv = 0) with a worse objective than
// borderline_feasible_mock_policy. Paired with it to detect a feasibility
// default mismatch: under an incorrect value_or(0) tolerance this policy
// alone would be judged feasible and would win on the feasible-beats-
// infeasible rule despite its worse objective.

struct feasible_worse_objective_mock_policy
{
    using scalar_type = double;

    struct state_type
    {
        Eigen::VectorXd x;
        double objective_value{};
        Eigen::VectorXd c_eq;
        Eigen::VectorXd c_ineq;
    };

    template <typename Problem, typename Convergence = argmin::default_convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0,
                    const argmin::solver_options<Convergence>&)
    {
        return state_type{
            .x = x0,
            .objective_value = 20.0,
            .c_eq = Eigen::VectorXd(0),
            .c_ineq = Eigen::VectorXd{{0.0}},
        };
    }

    argmin::step_result<double> step(state_type& s)
    {
        return argmin::step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = s.x.norm(),
            .step_size = 0.1,
            .objective_change = 0.0,
            .improved = false,
            .constraint_violation = 0.0,
        };
    }

    void reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
    }

    void reset_clear(state_type& s, const Eigen::VectorXd& x0)
    {
        reset(s, x0);
    }
};

// Mock policy for testing basic_solver's own best-seen feasibility
// tolerance in isolation: step 1 is borderline-feasible (cv = 5e-7,
// obj = 10), step 2 is unambiguously feasible but worse (cv = 0, obj = 20).
// Also seeds the same borderline c_ineq at init() so the pre-loop best-seen
// seed (basic_solver.h seed_best_cv) starts from the same borderline value.

struct borderline_then_worse_mock_policy
{
    using scalar_type = double;

    struct state_type
    {
        Eigen::VectorXd x;
        double objective_value{};
        Eigen::VectorXd c_eq;
        Eigen::VectorXd c_ineq;
        std::uint32_t step_count{};
    };

    template <typename Problem, typename Convergence = argmin::default_convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0,
                    const argmin::solver_options<Convergence>&)
    {
        return state_type{
            .x = x0,
            .objective_value = 10.0,
            .c_eq = Eigen::VectorXd(0),
            .c_ineq = Eigen::VectorXd{{-5e-7}},
        };
    }

    argmin::step_result<double> step(state_type& s)
    {
        ++s.step_count;
        if(s.step_count == 1)
        {
            s.objective_value = 10.0;
            s.c_ineq(0) = -5e-7;
        }
        else
        {
            s.objective_value = 20.0;
            s.c_ineq(0) = 0.0;
        }

        return argmin::step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = s.x.norm(),
            .step_size = 0.1,
            .objective_change = 0.0,
            .improved = false,
            .constraint_violation = (s.step_count == 1 ? 5e-7 : 0.0),
        };
    }

    void reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
        s.objective_value = 10.0;
        s.c_ineq(0) = -5e-7;
        s.step_count = 0;
    }

    void reset_clear(state_type& s, const Eigen::VectorXd& x0)
    {
        reset(s, x0);
    }
};

// Mock policy that reports roundoff_limited after exactly 3 steps.
// Used for testing policy_status propagation and solver group retirement.

struct roundoff_mock_policy
{
    using scalar_type = double;

    struct state_type
    {
        Eigen::VectorXd x;
        double objective_value{};
        std::uint32_t step_count{};
    };

    template <typename Problem, typename Convergence = argmin::default_convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0,
                    const argmin::solver_options<Convergence>&)
    {
        return {.x = x0, .objective_value = 0.5 * x0.squaredNorm()};
    }

    argmin::step_result<double> step(state_type& s)
    {
        ++s.step_count;
        s.x *= 0.5;
        s.objective_value = 0.5 * s.x.squaredNorm();
        argmin::step_result<double> r{
            .objective_value = s.objective_value,
            .gradient_norm = s.x.norm(),
            .step_size = 0.5,
            .objective_change = -s.objective_value,
            .improved = true,
            .x_norm = s.x.norm(),
        };
        if(s.step_count >= 3)
            r.policy_status = argmin::solver_status::roundoff_limited;
        return r;
    }

    void reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
        s.objective_value = 0.5 * x0.squaredNorm();
        s.step_count = 0;
    }

    void reset_clear(state_type& s, const Eigen::VectorXd& x0)
    {
        reset(s, x0);
    }
};

// Mock policy that reports diverged after 5 steps.
// Used for testing policy_status propagation and solver group retirement.

struct diverging_mock_policy
{
    using scalar_type = double;

    struct state_type
    {
        Eigen::VectorXd x;
        double objective_value{};
        std::uint32_t step_count{};
    };

    template <typename Problem, typename Convergence = argmin::default_convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0,
                    const argmin::solver_options<Convergence>&)
    {
        return {.x = x0, .objective_value = 1.0};
    }

    argmin::step_result<double> step(state_type& s)
    {
        ++s.step_count;
        s.objective_value *= 10.0;
        s.x *= 2.0;
        argmin::step_result<double> r{
            .objective_value = s.objective_value,
            .gradient_norm = 100.0,
            .step_size = 1.0,
            .objective_change = s.objective_value,
            .improved = false,
            .x_norm = s.x.norm(),
        };
        if(s.step_count >= 5)
            r.policy_status = argmin::solver_status::diverged;
        return r;
    }

    void reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
        s.objective_value = 1.0;
        s.step_count = 0;
    }

    void reset_clear(state_type& s, const Eigen::VectorXd& x0)
    {
        reset(s, x0);
    }
};

// Mock policy with options_type for testing per-policy options forwarding.
//
// Like mock_policy but step_size is configurable via options_type.

struct mock_policy_with_opts
{
    using scalar_type = double;

    struct options_type
    {
        double custom_step_size{0.5};
    };

    struct state_type
    {
        Eigen::VectorXd x;
        double objective_value{};
        double step_size{0.5};
    };

    template <typename Problem, typename Convergence = argmin::default_convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0,
                    const argmin::solver_options<Convergence>&)
    {
        return {.x = x0, .objective_value = 0.5 * x0.squaredNorm()};
    }

    template <typename Problem, typename Convergence = argmin::default_convergence>
    state_type init(const Problem&, const Eigen::VectorXd& x0,
                    const argmin::solver_options<Convergence>&,
                    const options_type& policy_opts)
    {
        return {.x = x0, .objective_value = 0.5 * x0.squaredNorm(),
                .step_size = policy_opts.custom_step_size};
    }

    argmin::step_result<double> step(state_type& s)
    {
        double old_f = s.objective_value;
        s.x *= (1.0 - s.step_size);
        s.objective_value = 0.5 * s.x.squaredNorm();

        return argmin::step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = s.x.norm(),
            .step_size = s.step_size,
            .objective_change = s.objective_value - old_f,
            .improved = s.objective_value < old_f,
        };
    }

    void reset(state_type& s, const Eigen::VectorXd& x0)
    {
        s.x = x0;
        s.objective_value = 0.5 * x0.squaredNorm();
    }

    void reset_clear(state_type& s, const Eigen::VectorXd& x0)
    {
        reset(s, x0);
    }
};

}

#endif
