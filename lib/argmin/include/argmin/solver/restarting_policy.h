#ifndef HPP_GUARD_ARGMIN_SOLVER_RESTARTING_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_RESTARTING_POLICY_H

// IPOP restart decorator for population-based solver policies.
//
// Transparent GoF decorator that wraps any policy and adds restart-on-
// stagnation with increasing population size (IPOP). The decorator
// forwards init/step/reset/reset_clear to the inner policy and monitors
// for stagnation. When triggered, restart consumes one step() call.
//
// rebind<M> forwards through the decorator so basic_solver CTAD works
// transparently.
//
// Reference: Auger & Hansen (2005), "A Restart CMA Evolution Strategy
//            with Increasing Population Size", CEC 2005.

#include "argmin/result/step_result.h"
#include "argmin/solver/options.h"
#include "argmin/types.h"

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace argmin
{

namespace detail
{
// Resolve state_type: use Inner::state_type<P> when available, else Inner::state_type.
// Mirrors basic_solver's resolve_state_t for use in policy decorators.
template <typename InnerPolicy, typename P>
consteval auto resolve_inner_state_tag()
{
    if constexpr(requires { typename InnerPolicy::template state_type<P>; })
        return std::type_identity<typename InnerPolicy::template state_type<P>>{};
    else
        return std::type_identity<typename InnerPolicy::state_type>{};
}

template <typename InnerPolicy, typename P>
using resolve_inner_state_t =
    typename decltype(resolve_inner_state_tag<InnerPolicy, P>())::type;
}

template <typename Inner>
struct restarting_policy
{
    using scalar_type = typename Inner::scalar_type;

    template <int M>
    using rebind = restarting_policy<typename Inner::template rebind<M>>;

    struct options_type
    {
        typename Inner::options_type inner{};
        std::optional<std::uint32_t> stagnation_limit{};
        std::optional<double> population_multiplier{};
        // stall_window / feasibility_gate intentionally not declared on the decorator.
        // restarting_policy emits synthetic (obj_change=1.0, step_size=1.0) step_results on
        // restart and maintains its own internal stall logic (see the restart-trigger block
        // below). Exposing an external stall_window would double-fire against this internal
        // logic; feasibility_gate is inner-policy territory. Silent no-op made explicit here —
        // see forward_policy_hints in basic_solver.h.
    };

    template <typename P = void>
    struct state_type
    {
        detail::resolve_inner_state_t<Inner, P> inner;
        decltype(std::declval<detail::resolve_inner_state_t<Inner, P>>().x) x{};
        const P* problem{nullptr};
        decltype(x) x0_saved{};
        std::uint32_t stagnation_count{0};
        double best_ever_value{std::numeric_limits<double>::infinity()};
        bool restart_pending{false};
        std::uint32_t restart_count{0};
        std::uint32_t dimension{0};
        std::uint32_t initial_lambda{0};

        Eigen::VectorXd c_eq{};
        Eigen::VectorXd c_ineq{};
    };

    Inner inner_policy_{};
    options_type options{};

    template <typename Problem, typename Convergence = default_convergence>
    state_type<Problem> init(const Problem& problem,
                    const auto& x0,
                    const solver_options<Convergence>& opts)
    {
        inner_policy_.options = options.inner;

        // Reproducibility: when no inner seed has been specified, plumb a
        // deterministic default into the inner policy's local options copy.
        // Without this, population-based inner policies (cmaes, isres, ...)
        // fall through to a std::random_device fallback at init() time,
        // which gives every rebuild a different RNG seed and breaks
        // bit-identity of the final objective across rebuilds at the same
        // SHA / same caller-visible configuration. The default-constructed
        // decorator is the relevant case: bench harnesses and concept
        // probes instantiate `restarting_policy<cmaes_policy<>>{}` without
        // supplying a seed, and a stochastic inner default propagates into
        // user-visible numeric output. Callers that want stochastic
        // seeding set `options.inner.seed` explicitly before constructing
        // the decorator (e.g. per-seed publish_bench sweeps); that path is
        // unaffected because we only fill the field when it is empty.
        //
        // Reference: Hansen (2023), "The CMA Evolution Strategy: A
        //            Tutorial", arXiv:1604.00772 §B.5 (reproducibility).
        if constexpr(requires { inner_policy_.options.seed; })
        {
            if(!inner_policy_.options.seed.has_value())
                inner_policy_.options.seed = std::uint64_t{1};
        }

        state_type<Problem> s;
        s.problem = &problem;
        s.x0_saved = x0;
        s.inner = inner_policy_.init(problem, x0, opts);
        s.best_ever_value = s.inner.objective_value;
        s.stagnation_count = 0;
        s.restart_count = 0;
        s.restart_pending = false;
        s.dimension = static_cast<std::uint32_t>(x0.size());

        // Extract initial lambda from inner state
        if constexpr(requires { s.inner.lambda; })
            s.initial_lambda = static_cast<std::uint32_t>(s.inner.lambda);
        else if constexpr(requires { s.inner.params.lambda; })
            s.initial_lambda = static_cast<std::uint32_t>(s.inner.params.lambda);
        else
            s.initial_lambda = static_cast<std::uint32_t>(
                4 + 3 * std::log(static_cast<double>(s.dimension)));

        sync_from_inner(s);
        return s;
    }

    template <typename Problem, typename Convergence>
    state_type<Problem> init(const Problem& problem,
                    const auto& x0,
                    const solver_options<Convergence>& opts,
                    const options_type& policy_opts)
    {
        options = policy_opts;
        return init(problem, x0, opts);
    }

    template <typename P>
    step_result<scalar_type> step(state_type<P>& s)
    {
        if(s.restart_pending)
        {
            reinit(s);
            s.restart_pending = false;
            s.stagnation_count = 0;

            return step_result<scalar_type>{
                .objective_value = s.inner.objective_value,
                .gradient_norm = 1.0,
                .step_size = 1.0,
                .objective_change = 1.0,
                .improved = false,
                .x_norm = s.inner.x.norm(),
            };
        }

        auto result = inner_policy_.step(s.inner);

        // Stall detection
        if(result.objective_value < s.best_ever_value)
        {
            s.best_ever_value = result.objective_value;
            s.stagnation_count = 0;
        }
        else
        {
            ++s.stagnation_count;
        }

        std::uint32_t limit = options.stagnation_limit.value_or(
            static_cast<std::uint32_t>(
                10 + std::ceil(30.0 * s.dimension / s.initial_lambda)));

        if(s.stagnation_count >= limit)
            s.restart_pending = true;

        sync_from_inner(s);
        return result;
    }

    template <typename P>
    void reset(state_type<P>& s, const auto& x0)
    {
        inner_policy_.reset(s.inner, x0);
        s.stagnation_count = 0;
        s.best_ever_value = s.inner.objective_value;
        s.restart_pending = false;
        sync_from_inner(s);
    }

    template <typename P>
    void reset_clear(state_type<P>& s, const auto& x0)
    {
        inner_policy_.reset_clear(s.inner, x0);
        s.stagnation_count = 0;
        s.restart_count = 0;
        s.best_ever_value = s.inner.objective_value;
        s.restart_pending = false;
        sync_from_inner(s);
    }

    // Accessor for the wrapped inner policy. Lets a caller mutate the
    // inner policy's options (e.g. options.lambda for the
    // F-R-01-style "bump lambda then reset" contract test in
    // tests/unit/restarting_policy_test.cpp). The decorator otherwise
    // owns the inner via the reinit() codepath; callers should not
    // rely on this accessor for non-test mutation.
    //
    // Reference: Auger & Hansen (2005), CEC 2005 §III (IPOP-CMA-ES) --
    // the decorator's reinit() is the canonical lambda-bump path; this
    // accessor enables locking the inner-policy reset() contract that
    // reinit() depends on.
    Inner& inner_policy() { return inner_policy_; }
    const Inner& inner_policy() const { return inner_policy_; }

private:

    template <typename P>
    void reinit(state_type<P>& s)
    {
        ++s.restart_count;

        // IPOP: population = initial_lambda * 2^restart_count
        double multiplier = options.population_multiplier.value_or(2.0);
        auto new_pop = static_cast<std::uint32_t>(
            s.initial_lambda * std::pow(multiplier, s.restart_count));

        // Update inner population size via the appropriate field
        if constexpr(requires { inner_policy_.options.population_size; })
            inner_policy_.options.population_size = new_pop;
        else if constexpr(requires { inner_policy_.options.lambda; })
            inner_policy_.options.lambda = new_pop;

        inner_policy_.reset_clear(s.inner, s.x0_saved);
        sync_from_inner(s);
    }

    template <typename P>
    static void sync_from_inner(state_type<P>& s)
    {
        s.x = s.inner.x;
        if constexpr(requires { s.inner.c_eq; s.inner.c_ineq; })
        {
            s.c_eq = s.inner.c_eq;
            s.c_ineq = s.inner.c_ineq;
        }
    }
};

}

#endif
