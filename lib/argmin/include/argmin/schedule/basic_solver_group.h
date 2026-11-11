#ifndef HPP_GUARD_ARGMIN_SCHEDULE_BASIC_SOLVER_GROUP_H
#define HPP_GUARD_ARGMIN_SCHEDULE_BASIC_SOLVER_GROUP_H

#include "argmin/solver/step_budget_solver.h"
#include "argmin/detail/solve_loop.h"
#include "argmin/detail/status_claim.h"
#include "argmin/formulation/concepts.h"
#include "argmin/result/step_result.h"
#include "argmin/result/solve_result.h"
#include "argmin/result/status.h"
#include "argmin/solver/convergence.h"
#include "argmin/solver/options.h"
#include "argmin/types.h"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>

namespace argmin
{

// Multi-solver group with scheduling.
//
// Holds a tuple of step_budget_solver<Policies, N>... and uses a Schedule to decide
// which solver to step next. The compile-time fold-expression requires clause
// ensures the Problem type satisfies the union of all policy requirements.

template <typename Schedule, int N = dynamic_dimension, typename Problem = void, typename... Policies>
class basic_solver_group
{
public:
    using scalar_type = typename std::tuple_element_t<0, std::tuple<Policies...>>::scalar_type;

    // Enforce the schedule contract: an ill-formed schedule (missing select /
    // notify / reset) fails here with a one-line diagnostic instead of a deep
    // template error at the select()/notify() call sites.
    static_assert(schedule<Schedule>,
                  "Schedule does not satisfy the schedule contract: it must "
                  "expose select(n), reset(), and notify(step_result).");

    template <typename P, typename Convergence = default_convergence<scalar_type>>
        requires (std::constructible_from<step_budget_solver<Policies, N, Problem>, const P&,
                  const Eigen::VectorX<scalar_type>&, const solver_options<Convergence>&> && ...)
    basic_solver_group(const P& problem,
                       const Eigen::VectorX<scalar_type>& x0,
                       const solver_options<Convergence>& opts = {},
                       Schedule schedule = {})
        : solvers_{step_budget_solver<Policies, N, Problem>{problem, x0, opts}...}
        , schedule_{std::move(schedule)}
        , max_iterations_{opts.max_iterations}
        , constraint_tolerance_{opts.constraint_tolerance}
        , feasibility_tolerance_{opts.feasibility_tolerance}
    {
        init_schedule();
    }

    // Per-policy options constructor.
    template <typename P, typename Convergence = default_convergence<scalar_type>>
        requires (std::constructible_from<step_budget_solver<Policies, N, Problem>, const P&,
                  const Eigen::VectorX<scalar_type>&, const solver_options<Convergence>&,
                  const policy_options_t<Policies, scalar_type>&> && ...)
    basic_solver_group(const P& problem,
                       const Eigen::VectorX<scalar_type>& x0,
                       const solver_options<Convergence>& opts,
                       std::tuple<policy_options_t<Policies, scalar_type>...> policy_opts,
                       Schedule schedule = {})
        : solvers_{make_solvers_with_opts(problem, x0, opts, policy_opts,
                                         std::index_sequence_for<Policies...>{})}
        , schedule_{std::move(schedule)}
        , max_iterations_{opts.max_iterations}
        , constraint_tolerance_{opts.constraint_tolerance}
        , feasibility_tolerance_{opts.feasibility_tolerance}
    {
        init_schedule();
    }

    step_result<scalar_type> step()
    {
        constexpr std::size_t n = sizeof...(Policies);

        if(all_retired())
        {
            // No live solver remains. Returning a value-initialized
            // step_result here would masquerade as a real step (objective
            // 0, gradient 0) and could falsely satisfy a caller's own
            // convergence check; report a truthful terminal outcome
            // instead.
            return terminal_result();
        }

        // Skip-scan for the next live solver. fallback_schedule and
        // time_boxed_schedule only advance their internal index on stall
        // / time-slice expiry, not on retirement, so a repeated
        // schedule_.select(n) call can keep returning an already-retired
        // index forever. Query the schedule once for its proposed index,
        // then scan forward over the live solvers from there -- since
        // all_retired() is false above, at least one live index exists
        // within the next n slots.
        std::size_t idx = schedule_.select(n);
        for(std::size_t attempt = 0; attempt < n; ++attempt)
        {
            if(!retired_[idx])
            {
                auto result = step_at(idx, std::index_sequence_for<Policies...>{});
                schedule_.notify(result);

                // Retire on failure or (per-policy) convergence status so a
                // converged solver stops being re-stepped and cannot
                // livelock the group.
                const auto policy_status = detail::admit_claim(result.policy_status, result, feasibility_tolerance_in_force());
                if(policy_status == solver_status::roundoff_limited || policy_status == solver_status::diverged || policy_status == solver_status::converged)
                {
                    retired_[idx] = true;
                    populate_result(idx, result, std::index_sequence_for<Policies...>{});
                }

                return result;
            }
            idx = (idx + 1) % n;
        }

        // Unreachable given the all_retired() guard above: every index was
        // found retired mid-scan even though all_retired() reported false.
        // Fall back to the same truthful terminal outcome rather than a
        // fabricated result.
        return terminal_result();
    }

    const std::array<solve_result<scalar_type, N>, sizeof...(Policies)>& results() const
    {
        return results_;
    }

    // Every member is reset even after one rejects the start; the first member status that is not running is
    // reported.
    solver_status reset(const Eigen::VectorX<scalar_type>& x0)
    {
        const auto statuses = reset_solvers(x0, std::index_sequence_for<Policies...>{});
        retired_ = {};
        init_schedule();
        const auto rejected = std::ranges::find_if(statuses, [](solver_status member) { return member != solver_status::running; });
        return rejected == statuses.end() ? solver_status::running : *rejected;
    }

    solve_result<scalar_type, N> solve()
    {
        solver_options<default_convergence<scalar_type>> opts;
        opts.max_iterations = max_iterations_;
        return step_n(max_iterations_, opts);
    }

    template <typename Convergence = default_convergence<scalar_type>>
    solve_result<scalar_type, N> step_n(std::uint32_t budget,
                                        const solver_options<Convergence>& opts = {})
    {
        step_result<scalar_type> last{};
        solver_status status = solver_status::running;

        // Real executed counts, so an early break (all solvers retired or a
        // converged criterion) reports what actually ran rather than the full
        // budget.
        std::uint32_t executed = 0;
        std::uint32_t evaluations = 0;

        for(std::uint32_t i = 0; i < budget; ++i)
        {
            // All policies retired -- stop iteration
            if(all_retired())
            {
                status = solver_status::budget_exhausted;
                break;
            }

            last = step();
            ++executed;
            evaluations += last.evaluations;

            // For groups, policy failure retires the individual policy (handled in step()).
            // Only stop the entire group if ALL policies are retired.
            if(all_retired())
            {
                status = solver_status::budget_exhausted;
                break;
            }

            // Convergence policy check
            auto conv = detail::admit_claim(opts.convergence.check(last, i + 1), last, feasibility_tolerance_in_force());
            if(conv)
            {
                status = *conv;
                break;
            }
        }

        if(status == solver_status::running)
        {
            status = solver_status::budget_exhausted;
        }

        // Populate results for non-retired policies
        populate_active_results(status, std::index_sequence_for<Policies...>{});

        auto best_idx = best_solver_index(std::index_sequence_for<Policies...>{});
        auto seq = std::index_sequence_for<Policies...>{};

        return solve_result<scalar_type, N>{
            .status = status,
            // Real executed iteration count and accumulated evaluation count,
            // not the requested budget -- an early break reports what ran.
            .iterations = executed,
            .function_evaluations = evaluations,
            .objective_value = objective_at(best_idx, seq),
            .gradient_norm = last.gradient_norm,
            .constraint_violation = constraint_violation_at(best_idx, seq),
            .x = best_x(seq),
        };
    }

private:
    void init_schedule()
    {
        // Restart the schedule's selection state so a reused group (or a
        // caller-supplied, pre-advanced schedule) resumes deterministically
        // from the first solver. set_num_solvers stays opt-in: only stateful
        // schedules that cache the solver count provide it.
        schedule_.reset();
        if constexpr(requires { schedule_.set_num_solvers(std::size_t{}); })
        {
            schedule_.set_num_solvers(sizeof...(Policies));
        }
    }

    template <std::size_t... Is>
    std::array<solver_status, sizeof...(Policies)> reset_solvers(const Eigen::VectorX<scalar_type>& x0, std::index_sequence<Is...>)
    {
        return {std::get<Is>(solvers_).reset(x0)...};
    }

    template <typename P, typename Convergence, std::size_t... Is>
    static auto make_solvers_with_opts(
        const P& problem,
        const Eigen::VectorX<scalar_type>& x0,
        const solver_options<Convergence>& opts,
        const std::tuple<policy_options_t<Policies, scalar_type>...>& policy_opts,
        std::index_sequence<Is...>)
    {
        return std::tuple{step_budget_solver<Policies, N, Problem>{
            problem, x0, opts, std::get<Is>(policy_opts)}...};
    }

    template <std::size_t... Is>
    step_result<scalar_type> step_at(std::size_t idx, std::index_sequence<Is...>)
    {
        step_result<scalar_type> result{};
        ((Is == idx ? void(result = std::get<Is>(solvers_).step()) : void()), ...);
        return result;
    }

    // The precedence step_budget_solver applies, so a group and a standalone solver judge an iterate's feasibility
    // and its claims identically.
    scalar_type feasibility_tolerance_in_force() const
    {
        return constraint_tolerance_.value_or(static_cast<scalar_type>(feasibility_tolerance_));
    }

    // Members rank as one solve's iterates do, each with its own status.
    template <std::size_t... Is>
    std::size_t best_solver_index(std::index_sequence<Is...>) const
    {
        constexpr scalar_type unset = std::numeric_limits<scalar_type>::max();
        const scalar_type tol = feasibility_tolerance_in_force();
        detail::iterate_rank<scalar_type> best{solver_status::running, unset, unset};
        std::size_t best_idx = 0;

        auto rank = [&](std::size_t idx, const auto& solver)
        {
            const detail::iterate_rank<scalar_type> cand{solver.status(), solver.state().objective_value, solver.constraint_violation()};
            if(detail::is_better(cand, best, tol))
            {
                best = cand;
                best_idx = idx;
            }
        };

        (rank(Is, std::get<Is>(solvers_)), ...);
        return best_idx;
    }

    template <std::size_t... Is>
    Eigen::Vector<scalar_type, N> best_x(std::index_sequence<Is...>) const
    {
        auto idx = best_solver_index(std::index_sequence<Is...>{});
        const Eigen::Vector<scalar_type, N>* ptr = nullptr;
        ((Is == idx ? void(ptr = &std::get<Is>(solvers_).state().x) : void()), ...);
        return *ptr;
    }

    template <std::size_t... Is>
    scalar_type objective_at(std::size_t idx, std::index_sequence<Is...>) const
    {
        scalar_type val{};
        ((Is == idx ? void(val = std::get<Is>(solvers_).state().objective_value) : void()), ...);
        return val;
    }

    template <std::size_t... Is>
    scalar_type constraint_violation_at(std::size_t idx, std::index_sequence<Is...>) const
    {
        scalar_type cv{};
        ((Is == idx ? void(cv = std::get<Is>(solvers_).constraint_violation()) : void()), ...);
        return cv;
    }

    bool all_retired() const
    {
        for(std::size_t i = 0; i < sizeof...(Policies); ++i)
        {
            if(!retired_[i]) return false;
        }
        return true;
    }

    // Truthful "no live solver" outcome for step(). Distinguishable from a
    // real step by construction: policy_status is explicitly set (never
    // nullopt, unlike an in-progress real step) and the metrics are NaN
    // rather than the misleading value-initialized 0, so a NaN-oblivious
    // comparison (e.g. `< threshold`) can never read this as convergence.
    static step_result<scalar_type> terminal_result()
    {
        step_result<scalar_type> r{};
        r.objective_value = std::numeric_limits<scalar_type>::quiet_NaN();
        r.gradient_norm = std::numeric_limits<scalar_type>::quiet_NaN();
        r.policy_status = solver_status::budget_exhausted;
        return r;
    }

    template <std::size_t... Is>
    void populate_result(std::size_t idx, const step_result<scalar_type>& sr,
                         std::index_sequence<Is...>)
    {
        auto fill = [&](std::size_t i, const auto& solver)
        {
            if(i == idx)
            {
                results_[i] = solve_result<scalar_type, N>{
                    .status = sr.policy_status.value_or(solver_status::running),
                    .objective_value = sr.objective_value,
                    .gradient_norm = sr.gradient_norm,
                    .constraint_violation = solver.constraint_violation(),
                    .x = solver.state().x,
                };
            }
        };
        (fill(Is, std::get<Is>(solvers_)), ...);
    }

    template <std::size_t... Is>
    void populate_active_results(solver_status group_status,
                                 std::index_sequence<Is...>)
    {
        auto fill = [&](std::size_t i, const auto& solver)
        {
            if(!retired_[i])
            {
                results_[i] = solve_result<scalar_type, N>{
                    .status = group_status,
                    .objective_value = solver.state().objective_value,
                    // Real last-reported gradient norm for this active solver
                    // (NaN sentinel if it never stepped) -- never a fabricated
                    // 0 that would read as a stationary point.
                    .gradient_norm = solver.gradient_norm(),
                    .constraint_violation = solver.constraint_violation(),
                    .x = solver.state().x,
                };
            }
        };
        (fill(Is, std::get<Is>(solvers_)), ...);
    }

    std::tuple<step_budget_solver<Policies, N, Problem>...> solvers_;
    Schedule schedule_;
    std::uint32_t max_iterations_{1000};
    std::optional<scalar_type> constraint_tolerance_{};
    scalar_type feasibility_tolerance_{convergence_defaults<scalar_type>::feasibility_tolerance};
    std::array<bool, sizeof...(Policies)> retired_{};
    std::array<solve_result<scalar_type, N>, sizeof...(Policies)> results_{};
};

}

#endif
