#ifndef HPP_GUARD_ARGMIN_TESTS_UNIT_PRECEDENCE_VISITOR_H
#define HPP_GUARD_ARGMIN_TESTS_UNIT_PRECEDENCE_VISITOR_H

#include "cases/case_options.h"

#include "argmin/solver/options.h"
#include "argmin/solver/convergence.h"

#include "argmin/detail/solver_core.h"

#include "argmin/options/convergence_defaults.h"

#include <catch2/catch_test_macros.hpp>

#include <set>
#include <tuple>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace argmin::precedence
{

template <typename Convergence>
const stall_tolerance_criterion<double>& stall_of(const Convergence& convergence)
{
    return std::get<stall_tolerance_criterion<double>>(convergence.criteria);
}

template <typename Convergence>
stall_tolerance_criterion<double>& stall_of(Convergence& convergence)
{
    return std::get<stall_tolerance_criterion<double>>(convergence.criteria);
}

template <typename Policy>
std::uint16_t hinted_window(const Policy& policy)
{
    if constexpr(requires { policy.options.stall_window; })
        return policy.options.stall_window;
    else
        return convergence_defaults<double>::stall_window;
}

inline solver_options<> one_iteration(const solver_options<>& options)
{
    solver_options<> opts = options;
    opts.max_iterations = 1;
    return opts;
}

// One cell per case key: the stored path is read after construction, the explicit path after a one-iteration solve
// on the caller's own options. Each cell runs once on the policy's own options and, where the policy takes options at
// construction, once more on options carrying a window no default uses, so the caller, hint and default all differ.
class visitor
{
public:
    visitor() : keys_{}, asymptote_cells_{0} {}

    template <typename Problem, typename MakeSolver>
    void operator()(const cases::case_id& id, const Problem& problem, const solver_options<>& options, const MakeSolver& make_solver)
    {
        if(!keys_.insert(id.key).second)
            return;
        INFO(id.key);
        using solver_type = decltype(make_solver());
        const auto policy = make_solver().policy();
        check_window([&](const solver_options<>& opts) { return solver_type{policy, problem, problem.initial_point(), opts}; }, options, hinted_window(policy));
        if constexpr(has_options_type<std::remove_cvref_t<decltype(policy)>>)
            check_configured<solver_type>(policy, problem, options);
    }

    std::size_t asymptote_cells() const { return asymptote_cells_; }

private:
    std::set<std::string_view> keys_;
    std::size_t asymptote_cells_;

    template <typename Solver, typename Policy, typename Problem>
    void check_configured(const Policy& policy, const Problem& problem, const solver_options<>& options)
    {
        typename Policy::options_type configured{};
        constexpr bool hints_window = requires { configured.stall_window; };
        if constexpr(hints_window)
            configured.stall_window = 81;
        const auto build = [&](const solver_options<>& opts) { return Solver{policy, problem, problem.initial_point(), opts, configured}; };
        check_window(build, options, hints_window ? std::uint16_t{81} : convergence_defaults<double>::stall_window);
        if constexpr(requires { configured.stall_tolerance_threshold; })
        {
            check_asymptote(build, options, requires { configured.rho_init; } || requires { configured.raa0; });
            ++asymptote_cells_;
        }
    }

    template <typename Build>
    static void check_window(const Build& build, const solver_options<>& options, std::uint16_t hint)
    {
        const auto caller_window = static_cast<std::uint16_t>(hint + 7);
        solver_options<> caller = one_iteration(options);
        caller.set_stall_window(caller_window);
        CHECK(stall_of(build(caller).convergence()).window_in_force() == caller_window);
        auto probe = build(options);
        CHECK_FALSE(stall_of(probe.convergence()).window.has_value());
        CHECK(stall_of(probe.convergence()).window_in_force() == hint);
        solver_options<> unset = one_iteration(options);
        probe.solve(unset);
        CHECK_FALSE(stall_of(unset.convergence).window.has_value());
        CHECK(stall_of(unset.convergence).window_in_force() == hint);
        build(options).solve(caller);
        CHECK(stall_of(caller.convergence).window == caller_window);
        CHECK(stall_of(caller.convergence).window_in_force() == caller_window);
    }

    template <typename Build>
    static void check_asymptote(const Build& build, const solver_options<>& options, bool tracks)
    {
        CHECK(stall_of(build(options).convergence()).threshold_in_force() == convergence_defaults<double>::stall_threshold);
        CHECK(stall_of(build(options).convergence()).best_seen_feasible_in_force() == tracks);
        solver_options<> caller = one_iteration(options);
        caller.set_stall_threshold(1e-3);
        stall_of(caller.convergence).track_best_seen_feasible = !tracks;
        CHECK(stall_of(build(caller).convergence()).threshold_in_force() == 1e-3);
        CHECK(stall_of(build(caller).convergence()).best_seen_feasible_in_force() != tracks);
        build(options).solve(caller);
        CHECK(stall_of(caller.convergence).threshold_in_force() == 1e-3);
        CHECK(stall_of(caller.convergence).best_seen_feasible_in_force() != tracks);
        solver_options<> unset = one_iteration(options);
        build(options).solve(unset);
        CHECK_FALSE(stall_of(unset.convergence).threshold.has_value());
        CHECK_FALSE(stall_of(unset.convergence).track_best_seen_feasible.has_value());
        CHECK(stall_of(unset.convergence).threshold_in_force() == convergence_defaults<double>::stall_threshold);
        CHECK(stall_of(unset.convergence).best_seen_feasible_in_force() == tracks);
    }
};

}

#endif
