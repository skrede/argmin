#ifndef HPP_GUARD_ARGMIN_TESTS_UNIT_SCRIPTED_CLAIM_POLICY_H
#define HPP_GUARD_ARGMIN_TESTS_UNIT_SCRIPTED_CLAIM_POLICY_H

#include "argmin/solver/options.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/step_budget_solver.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"

#include "argmin/options/convergence_defaults.h"

#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <algorithm>

namespace argmin::test
{

template <typename S>
struct line
{
    static constexpr std::int32_t problem_dimension = dynamic_dimension;

    std::int32_t dimension() const { return 1; }
    S value(const Eigen::VectorX<S>& x) const { return x(0); }
};

template <typename S>
struct scripted_step
{
    S constraint_violation;
    std::optional<S> kkt_residual;
    std::optional<solver_status> policy_status;
    S step_size{1};
};

// Replays a fixed step script and repeats its last entry, so a test controls the residual and the violation each
// stopping test sees.
template <typename S>
struct scripted_policy
{
    using scalar_type = S;

    struct state_type
    {
        Eigen::VectorX<S> x;
        std::size_t step_count;
    };

    std::array<scripted_step<S>, 2> script;

    template <typename Problem, typename Convergence>
    state_type init(const Problem&, const Eigen::VectorX<S>& x0, const solver_options<Convergence>&)
    {
        return state_type{x0, 0};
    }

    step_result<S> step(state_type& s)
    {
        const scripted_step<S>& next = script[std::min(s.step_count, script.size() - 1)];
        ++s.step_count;
        s.x(0) = S(s.step_count);
        return step_result<S>{.objective_value = S(1),
                              .gradient_norm = S(1),
                              .step_size = next.step_size,
                              .constraint_violation = next.constraint_violation,
                              .x_norm = s.x(0),
                              .kkt_residual = next.kkt_residual,
                              .policy_status = next.policy_status};
    }

    solver_status reset(state_type& s, const Eigen::VectorX<S>& x0)
    {
        s = state_type{x0, 0};
        return solver_status::running;
    }

    solver_status reset_clear(state_type& s, const Eigen::VectorX<S>& x0) { return reset(s, x0); }
};

template <typename S>
constexpr S feas_tol = convergence_defaults<S>::feasibility_tolerance;

// A residual that passes the default gradient threshold while the violation it may fold in exceeds the tolerance.
template <typename S>
constexpr S passing_kkt = convergence_defaults<S>::gradient_threshold / S(2);

template <typename S>
constexpr scripted_step<S> infeasible{S(2) * feas_tol<S>, passing_kkt<S>, std::nullopt};

template <typename S>
constexpr scripted_step<S> feasible{feas_tol<S> / S(2), passing_kkt<S>, std::nullopt};

template <typename S, typename Convergence = default_convergence<S>>
step_budget_solver<scripted_policy<S>, dynamic_dimension, line<S>, Convergence> make_solver(const std::array<scripted_step<S>, 2>& script, solver_options<Convergence> opts = {})
{
    static const line<S> problem{};
    opts.max_iterations = 3;
    return {scripted_policy<S>{script}, problem, Eigen::VectorX<S>::Constant(1, S(3)), opts};
}

}

#endif
