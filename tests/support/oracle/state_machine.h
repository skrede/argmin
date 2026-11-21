#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_ORACLE_STATE_MACHINE_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_ORACLE_STATE_MACHINE_H

#include "oracle/result_oracle.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <bit>
#include <cmath>
#include <string>
#include <cstdint>
#include <optional>

namespace argmin::oracle
{

struct snapshot
{
    solver_status status;
    Eigen::VectorXd x;
    double gradient_norm;
    double constraint_violation;
    double bound_violation;
    std::optional<bool> converged;
    std::optional<bool> finished;
    std::optional<std::uint32_t> iterations;
    std::optional<std::uint32_t> evaluations;
};

// The oracle's own transition table: only the two statuses meaning the call's own iteration or time budget ran out
// let step_n and solve continue without a reset.
inline bool resumes(solver_status status)
{
    return status == solver_status::budget_exhausted || status == solver_status::time_limit_reached;
}

inline bool is_claim(solver_status status)
{
    const claim claimed = claim_of(status);
    return claimed == claim::optimality || claimed == claim::ftol_optimality || claimed == claim::feasibility;
}

template <typename Solver>
snapshot take_snapshot(const Solver& solver)
{
    snapshot s{
        .status = solver.status(),
        .x = solver.state().x.template cast<double>(),
        .gradient_norm = static_cast<double>(solver.gradient_norm()),
        .constraint_violation = static_cast<double>(solver.constraint_violation()),
        .bound_violation = static_cast<double>(solver.bound_violation()),
        .converged = std::nullopt,
        .finished = std::nullopt,
        .iterations = std::nullopt,
        .evaluations = std::nullopt,
    };
    if constexpr(requires { solver.converged(); })
        s.converged = solver.converged();
    if constexpr(requires { solver.finished(); })
        s.finished = solver.finished();
    return s;
}

template <typename Solver, typename Scalar, int N>
snapshot take_snapshot(const Solver& solver, const solve_result<Scalar, N>& last)
{
    snapshot s = take_snapshot(solver);
    s.iterations = last.iterations;
    s.evaluations = last.function_evaluations;
    return s;
}

inline bool same_bits(double a, double b)
{
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

inline bool same_bits(const Eigen::VectorXd& a, const Eigen::VectorXd& b)
{
    if(a.size() != b.size())
        return false;
    for(Eigen::Index i = 0; i < a.size(); ++i)
    {
        if(!same_bits(a[i], b[i]))
            return false;
    }
    return true;
}

inline std::string first_difference(const snapshot& a, const snapshot& b)
{
    if(a.status != b.status)
        return "status";
    if(!same_bits(a.x, b.x))
        return "x";
    if(!same_bits(a.gradient_norm, b.gradient_norm))
        return "gradient_norm";
    if(!same_bits(a.constraint_violation, b.constraint_violation))
        return "constraint_violation";
    if(!same_bits(a.bound_violation, b.bound_violation))
        return "bound_violation";
    if(a.converged != b.converged || a.finished != b.finished)
        return "converged/finished";
    if(a.iterations != b.iterations)
        return "iterations";
    if(a.evaluations != b.evaluations)
        return "evaluations";
    return {};
}

template <typename Scalar>
std::string unevaluated_failure(const step_result<Scalar>& r)
{
    if(r.evaluations != 0 || !r.is_null_step || r.improved)
        return "the step evaluated or moved";
    if(!std::isnan(r.objective_value) || !std::isnan(r.gradient_norm) || !std::isnan(r.constraint_violation) || !std::isnan(r.bound_violation))
        return "the step reported a value";
    return {};
}

template <typename Scalar>
std::string sticky_failure(const snapshot& before, const snapshot& after, const step_result<Scalar>& r)
{
    if(before.status == solver_status::running)
        return "the snapshot before the step is running";
    if(r.policy_status != before.status)
        return "policy_status";
    if(std::string field = first_difference(before, after); !field.empty())
        return field;
    return unevaluated_failure(r);
}

template <typename Scalar>
std::string abort_failure(const snapshot& before, const snapshot& after, const step_result<Scalar>& r)
{
    if(after.status != solver_status::aborted || r.policy_status != solver_status::aborted)
        return "status";
    snapshot expected = before;
    expected.status = after.status;
    expected.converged = after.converged;
    expected.finished = after.finished;
    if(std::string field = first_difference(expected, after); !field.empty())
        return field;
    return unevaluated_failure(r);
}

inline std::string claim_query_failure(const snapshot& s)
{
    if(s.converged && *s.converged != is_claim(s.status))
        return "converged";
    if(s.finished && *s.finished != (s.status != solver_status::running))
        return "finished";
    return {};
}

inline std::string reset_failure(const snapshot& s, const Eigen::VectorXd& start)
{
    if(s.status != solver_status::running && s.status != solver_status::invalid_problem)
        return "status";
    if(s.iterations && *s.iterations != 0)
        return "iterations";
    if(!std::isnan(s.gradient_norm))
        return "gradient_norm";
    if(!same_bits(s.x, start))
        return "x";
    return {};
}

// For a step_n or solve call with a nonzero budget, judged on the snapshots around it.
inline std::string continuation_failure(const snapshot& before, const snapshot& after)
{
    if(!before.iterations || !after.iterations)
        return "iterations unknown";
    if(before.status == solver_status::running || resumes(before.status))
        return *after.iterations > *before.iterations ? std::string{} : std::string{"iterations"};
    return first_difference(before, after);
}

}

#endif
