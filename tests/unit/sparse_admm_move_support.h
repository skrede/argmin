#ifndef HPP_GUARD_ARGMIN_TESTS_UNIT_SPARSE_ADMM_MOVE_SUPPORT_H
#define HPP_GUARD_ARGMIN_TESTS_UNIT_SPARSE_ADMM_MOVE_SUPPORT_H

#include "sparse_control_qp_family.h"

#include "argmin/qp/qp_types.h"
#include "argmin/qp/sparse_admm_qp.h"
#include "argmin/options/sparse_qp_options.h"

#include <Eigen/Core>

#include <bit>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <optional>

namespace argmin::sparse_admm_move
{

using result_list = std::vector<qp_result<double>>;

struct step_problem
{
    Eigen::VectorXd q;
    Eigen::VectorXd l;
    Eigen::VectorXd u;
};

inline std::vector<step_problem> receding_schedule(const control_qp_data::control_qp& cell, std::int32_t steps)
{
    std::vector<step_problem> schedule;
    schedule.reserve(static_cast<std::size_t>(steps));
    for(std::int32_t k = 0; k < steps; ++k)
    {
        step_problem s{cell.gradient_for(cell.reference_at(k)), cell.l, cell.u};
        const double nudge = 1e-3 * static_cast<double>((k % 5) - 2);
        for(std::int32_t j = 0; j < cell.n_state; ++j)
        {
            s.l[cell.initial_row(j)] = cell.x0[j] + nudge;
            s.u[cell.initial_row(j)] = cell.x0[j] + nudge;
        }
        schedule.push_back(std::move(s));
    }
    return schedule;
}

inline std::vector<step_problem> slice(const std::vector<step_problem>& schedule, std::size_t first, std::size_t last)
{
    return {schedule.begin() + static_cast<std::ptrdiff_t>(first), schedule.begin() + static_cast<std::ptrdiff_t>(last)};
}

inline sparse_qp_options<> options_for(bool polish, bool warm)
{
    sparse_qp_options<> opts;
    opts.polish = polish;
    opts.warm_start = warm;
    return opts;
}

// A step whose call returned an error is recorded with iterations = -1, which no solve reports, so
// a failed step can never pass as a twin of a successful one.
inline result_list drive(sparse_admm_qp_solver<double>& solver, const control_qp_data::control_qp& cell,
                         const std::vector<step_problem>& schedule, const sparse_qp_options<>& opts, bool pose_first)
{
    result_list results;
    for(std::size_t k = 0; k < schedule.size(); ++k)
    {
        const step_problem& s = schedule[k];
        qp_result<double> out;
        const bool pose = pose_first && k == 0;
        const std::optional<qp_error> err = pose ? solver.solve_into(cell.P, s.q, cell.A, s.l, s.u, out, opts)
                                                 : solver.resolve_into(s.q, s.l, s.u, out, opts);
        if(err)
            out.iterations = -1;
        results.push_back(std::move(out));
    }
    return results;
}

inline void append(result_list& run, result_list tail)
{
    for(qp_result<double>& r : tail)
        run.push_back(std::move(r));
}

inline bool same_bits(const Eigen::VectorXd& a, const Eigen::VectorXd& b)
{
    if(a.size() != b.size())
        return false;
    return a.size() == 0 || std::memcmp(a.data(), b.data(), static_cast<std::size_t>(a.size()) * sizeof(double)) == 0;
}

inline bool same_bits(double a, double b)
{
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

inline bool bit_identical(const qp_result<double>& a, const qp_result<double>& b)
{
    return same_bits(a.x, b.x) && same_bits(a.y, b.y) && a.status == b.status && a.iterations == b.iterations
           && a.polished == b.polished && a.warm_started == b.warm_started && same_bits(a.objective_value, b.objective_value)
           && same_bits(a.primal_residual, b.primal_residual) && same_bits(a.dual_residual, b.dual_residual);
}

inline bool all_steps_solved(const result_list& run)
{
    for(const qp_result<double>& r : run)
        if(r.iterations < 0)
            return false;
    return !run.empty();
}

inline bool any_polished(const result_list& run)
{
    for(const qp_result<double>& r : run)
        if(r.polished)
            return true;
    return false;
}

inline bool warm_after_first(const result_list& run)
{
    for(std::size_t k = 1; k < run.size(); ++k)
        if(run[k].warm_started)
            return true;
    return false;
}

}

#endif
