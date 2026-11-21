#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_ORACLE_TRAJECTORY_MEMBERSHIP_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_ORACLE_TRAJECTORY_MEMBERSHIP_H

#include "oracle/state_machine.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"
#include "argmin/result/solve_result.h"

#include <Eigen/Core>

#include <limits>
#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>

namespace argmin::oracle
{

struct trajectory_row
{
    Eigen::VectorXd x;
    double objective_value;
    double constraint_violation;
    double gradient_norm;
    solver_status status;
};

// Row 0 is the start point, where no step has evaluated a gradient; each later row is one step() of the solver.
template <typename Solver, typename Problem>
std::vector<trajectory_row> record_trajectory(Solver& solver, const Problem& problem, std::uint32_t max_steps)
{
    constexpr double no_gradient = std::numeric_limits<double>::quiet_NaN();
    std::vector<trajectory_row> rows{{solver.state().x.template cast<double>(), static_cast<double>(problem.value(solver.state().x)), static_cast<double>(solver.constraint_violation()), no_gradient, solver.status()}};
    for(std::uint32_t i = 0; i < max_steps && solver.status() == solver_status::running; ++i)
    {
        const auto r = solver.step();
        rows.push_back({solver.state().x.template cast<double>(), static_cast<double>(r.objective_value), static_cast<double>(r.constraint_violation), static_cast<double>(r.gradient_norm), solver.status()});
    }
    return rows;
}

inline bool ends_on_budget(solver_status status)
{
    return status == solver_status::max_iterations || status == solver_status::budget_exhausted || status == solver_status::time_limit_reached;
}

template <typename Scalar, int N>
bool same_iterate(const solve_result<Scalar, N>& result, const trajectory_row& row)
{
    return same_bits(result.x.template cast<double>(), row.x) && same_bits(static_cast<double>(result.objective_value), row.objective_value)
        && same_bits(static_cast<double>(result.constraint_violation), row.constraint_violation);
}

// The result's x, f, cv and gradient norm must be one recorded row, bit for bit, and its status the last row's or a
// budget end.
template <typename Scalar, int N>
std::string membership_failure(const solve_result<Scalar, N>& result, const std::vector<trajectory_row>& rows)
{
    const auto same_row = [&](const trajectory_row& row) { return same_iterate(result, row) && same_bits(static_cast<double>(result.gradient_norm), row.gradient_norm); };
    if(std::ranges::none_of(rows, [&](const trajectory_row& row) { return same_iterate(result, row); }))
        return "iterate";
    if(std::ranges::none_of(rows, same_row))
        return "gradient_norm";
    if(result.status != rows.back().status && !ends_on_budget(result.status))
        return "status";
    return {};
}

}

#endif
