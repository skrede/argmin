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

        return solve_result<scalar_type>{
            .status = status,
            .iterations = budget,
            .function_evaluations = budget,
            .objective_value = last.objective_value,
            .gradient_norm = last.gradient_norm,
            .constraint_violation = constraint_violation_at(best_idx, std::index_sequence_for<Policies...>{}),
            .x = best_x(std::index_sequence_for<Policies...>{}),
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

    template <std::size_t... Is>
    std::size_t best_solver_index(std::index_sequence<Is...>) const
    {
        scalar_type best_val = std::numeric_limits<scalar_type>::max();
        std::size_t best_idx = 0;

        auto check = [&](std::size_t idx, const auto& solver)
        {
            if(solver.state().objective_value < best_val)
            {
                best_val = solver.state().objective_value;
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
