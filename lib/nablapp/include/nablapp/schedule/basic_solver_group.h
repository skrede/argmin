#ifndef HPP_GUARD_NABLAPP_SCHEDULE_BASIC_SOLVER_GROUP_H
#define HPP_GUARD_NABLAPP_SCHEDULE_BASIC_SOLVER_GROUP_H

#include "nablapp/solver/basic_solver.h"
#include "nablapp/result/step_result.h"
#include "nablapp/result/solve_result.h"
#include "nablapp/result/status.h"
#include "nablapp/solver/options.h"

#include <Eigen/Core>

#include <chrono>
#include <concepts>
#include <cstddef>
#include <tuple>
#include <utility>

namespace nablapp
{

// Multi-solver group with scheduling (per CORE-03, CORE-04, D-12).
//
// Holds a tuple of basic_solver<Policies>... and uses a Schedule to decide
// which solver to step next. The compile-time fold-expression requires clause
// ensures the Problem type satisfies the union of all policy requirements.

template <typename Schedule, typename... Policies>
class basic_solver_group
{
public:
    using scalar_type = typename std::tuple_element_t<0, std::tuple<Policies...>>::scalar_type;

    template <typename Problem>
        requires (std::constructible_from<basic_solver<Policies>, const Problem&,
                  const Eigen::VectorX<scalar_type>&, const solver_options<scalar_type>&> && ...)
    basic_solver_group(const Problem& problem,
                       const Eigen::VectorX<scalar_type>& x0,
                       const solver_options<scalar_type>& opts = {},
                       Schedule schedule = {})
        : solvers_{basic_solver<Policies>{problem, x0, opts}...}
        , schedule_{std::move(schedule)}
        , opts_{opts}
    {
        init_schedule();
    }

    // Per-policy options constructor.
    // Each element of policy_opts is forwarded to the corresponding
    // policy's basic_solver constructor.
    template <typename Problem>
        requires (std::constructible_from<basic_solver<Policies>, const Problem&,
                  const Eigen::VectorX<scalar_type>&, const solver_options<scalar_type>&,
                  const policy_options_t<Policies, scalar_type>&> && ...)
    basic_solver_group(const Problem& problem,
                       const Eigen::VectorX<scalar_type>& x0,
                       const solver_options<scalar_type>& opts,
                       std::tuple<policy_options_t<Policies, scalar_type>...> policy_opts,
                       Schedule schedule = {})
        : solvers_{make_solvers_with_opts(problem, x0, opts, policy_opts,
                                         std::index_sequence_for<Policies...>{})}
        , schedule_{std::move(schedule)}
        , opts_{opts}
    {
        init_schedule();
    }

    step_result<scalar_type> step()
    {
        std::size_t idx = schedule_.select(sizeof...(Policies));
        auto result = step_at(idx, std::index_sequence_for<Policies...>{});
        schedule_.notify(result);
        return result;
    }

    solve_result<scalar_type> solve()
    {
        return step_n(max_iterations());
    }

    solve_result<scalar_type> step_n(int budget)
    {
        auto t0 = std::chrono::steady_clock::now();

        step_result<scalar_type> last{};
        solver_status status = solver_status::running;

        for(int i = 0; i < budget; ++i)
        {
            last = step();

            if(last.gradient_norm < gradient_tolerance())
            {
                status = solver_status::converged;
                break;
            }
        }

        if(status == solver_status::running)
        {
            status = solver_status::budget_exhausted;
        }

        auto t1 = std::chrono::steady_clock::now();

        auto best_idx = best_solver_index(std::index_sequence_for<Policies...>{});
        auto seq = std::index_sequence_for<Policies...>{};

        return solve_result<scalar_type>{
            .status = status,
            .iterations = budget,
            .function_evaluations = budget,
            .objective_value = objective_at(best_idx, seq),
            .gradient_norm = last.gradient_norm,
            .constraint_violation = constraint_violation_at(best_idx, seq),
            .x = best_x(seq),
            .wall_time = t1 - t0,
        };
    }

private:
    void init_schedule()
    {
        if constexpr(requires { schedule_.set_num_solvers(std::size_t{}); })
        {
            schedule_.set_num_solvers(sizeof...(Policies));
        }
    }

    template <typename Problem, std::size_t... Is>
    static auto make_solvers_with_opts(
        const Problem& problem,
        const Eigen::VectorX<scalar_type>& x0,
        const solver_options<scalar_type>& opts,
        const std::tuple<policy_options_t<Policies, scalar_type>...>& policy_opts,
        std::index_sequence<Is...>)
    {
        return std::tuple{basic_solver<Policies>{
            problem, x0, opts, std::get<Is>(policy_opts)}...};
    }

    int max_iterations() const
    {
        return opts_.max_iterations;
    }

    scalar_type gradient_tolerance() const
    {
        return opts_.gradient_tolerance;
    }

    template <std::size_t... Is>
    step_result<scalar_type> step_at(std::size_t idx, std::index_sequence<Is...>)
    {
        step_result<scalar_type> result{};
        ((Is == idx ? (result = std::get<Is>(solvers_).step(), true) : false), ...);
        return result;
    }

    // Feasibility-first ranking.
    // Reference: N&W Ch. 15.
    template <std::size_t... Is>
    std::size_t best_solver_index(std::index_sequence<Is...>) const
    {
        scalar_type best_obj = std::numeric_limits<scalar_type>::max();
        scalar_type best_cv = std::numeric_limits<scalar_type>::max();
        std::size_t best_idx = 0;
        scalar_type tol = opts_.constraint_tolerance;

        auto check = [&](std::size_t idx, const auto& solver)
        {
            scalar_type obj = solver.state().objective_value;
            scalar_type cv = solver.constraint_violation();
            bool feas = cv <= tol;
            bool best_feas = best_cv <= tol;

            bool better = false;
            if(feas && !best_feas) better = true;
            else if(!feas && best_feas) better = false;
            else if(feas && best_feas) better = obj < best_obj;
            else better = cv < best_cv;

            if(better)
            {
                best_obj = obj;
                best_cv = cv;
                best_idx = idx;
            }
        };

        (check(Is, std::get<Is>(solvers_)), ...);
        return best_idx;
    }

    template <std::size_t... Is>
    Eigen::VectorX<scalar_type> best_x(std::index_sequence<Is...>) const
    {
        auto idx = best_solver_index(std::index_sequence<Is...>{});
        const Eigen::VectorX<scalar_type>* ptr = nullptr;
        ((Is == idx ? (ptr = &std::get<Is>(solvers_).state().x, true) : false), ...);
        return *ptr;
    }

    template <std::size_t... Is>
    scalar_type objective_at(std::size_t idx, std::index_sequence<Is...>) const
    {
        scalar_type val{};
        ((Is == idx ? (val = std::get<Is>(solvers_).state().objective_value, true) : false), ...);
        return val;
    }

    template <std::size_t... Is>
    scalar_type constraint_violation_at(std::size_t idx, std::index_sequence<Is...>) const
    {
        scalar_type cv{};
        ((Is == idx ? (cv = std::get<Is>(solvers_).constraint_violation(), true) : false), ...);
        return cv;
    }

    std::tuple<basic_solver<Policies>...> solvers_;
    Schedule schedule_;
    solver_options<scalar_type> opts_;
};

}

#endif
