#ifndef HPP_GUARD_ARGMIN_DETAIL_HINT_PRECEDENCE_H
#define HPP_GUARD_ARGMIN_DETAIL_HINT_PRECEDENCE_H

#include "argmin/options/convergence_defaults.h"

#include <tuple>
#include <cstdint>
#include <optional>

namespace argmin
{
namespace detail
{

template <typename S>
struct stall_hints
{
    std::optional<S> threshold;
    std::optional<std::uint16_t> window;
    std::optional<bool> track_best_seen_feasible;
};

struct no_policy_hints
{
};

template <typename T>
std::optional<T> in_force(const std::optional<T>& caller, const std::optional<T>& hint)
{
    return caller ? caller : hint;
}

template <typename T>
T in_force(const std::optional<T>& caller, const std::optional<T>& hint, T fallback)
{
    return caller.value_or(hint.value_or(fallback));
}

template <typename S, typename PolicyOpts>
stall_hints<S> stall_hints_of(const PolicyOpts& policy_opts)
{
    stall_hints<S> hints;
    if constexpr(requires { policy_opts.stall_window; })
        hints.window = policy_opts.stall_window;
    // A moving-asymptote policy arms the stall test; one with a conservativeness rho (rho_init or raa0) judges it on
    // the best feasible objective seen, which an objective oscillating about a reached optimum cannot raise.
    if constexpr(requires { policy_opts.stall_tolerance_threshold; })
    {
        hints.threshold = policy_opts.stall_tolerance_threshold.value_or(convergence_defaults<S>::stall_threshold);
        if constexpr(requires { policy_opts.rho_init; } || requires { policy_opts.raa0; })
            hints.track_best_seen_feasible = true;
    }
    return hints;
}

template <typename Criterion, typename PolicyOpts>
void write_criterion_hints(const Criterion& criterion, const PolicyOpts& policy_opts)
{
    if constexpr(requires { criterion.policy_hints; })
        criterion.policy_hints = stall_hints_of<typename Criterion::scalar_type>(policy_opts);
}

// Every slot is rewritten, empty where the policy gives no hint, so an options object reused with another policy
// keeps nothing from the previous one.
template <typename Convergence, typename PolicyOpts>
void write_hints(const Convergence& convergence, const PolicyOpts& policy_opts)
{
    if constexpr(requires { convergence.criteria; })
        std::apply([&](const auto&... each) { (write_criterion_hints(each, policy_opts), ...); }, convergence.criteria);
}

}
}

#endif
