#ifndef HPP_GUARD_ARGMIN_SOLVER_RESTARTING_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_RESTARTING_POLICY_H

// IPOP restart decorator: wraps a population-based policy and restarts it with an increasing population. A run ends on
// its own verdict (the inner policy's status, else the decorator's copy of the convergence settings) or when it stops
// improving; a converged, ftol_reached or xtol_reached claim or invalid_problem ends the solve, every other end
// restarts, and at max_restarts the solve ends with the best run's result and status. rebind<M> forwards through the
// decorator so step_budget_solver CTAD works.
//
// Reference: Auger & Hansen (2005), "A Restart CMA Evolution Strategy
//            with Increasing Population Size", CEC 2005.

#include "argmin/detail/xoshiro256.h"
#include "argmin/detail/cmaes_sampling.h"
#include "argmin/detail/stochastic_seed.h"
#include "argmin/detail/evaluation_count.h"
#include "argmin/detail/decorator_verdict.h"
#include "argmin/result/step_result.h"
#include "argmin/solver/options.h"
#include "argmin/types.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace argmin
{

namespace detail
{
// Resolve state_type: use Inner::state_type<P> when available, else Inner::state_type.
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
        std::optional<std::uint32_t> stagnation_limit{};  // truly optional: unset = auto formula 10 + ceil(30 n / lambda)
        std::optional<std::uint32_t> max_restarts{};  // unset: no cap, the solve's iteration budget ends the restarts
        scalar_type population_multiplier{scalar_type(2.0)};  // IPOP doubling (Auger & Hansen 2005)
        // No stall_window or feasibility_gate: each run is judged by the decorator's own convergence copy under the
        // inner policy's options, and the solve's criteria never see a run's steps.
    };

    template <typename P = void>
    struct state_type : detail::wrapped_evaluations<detail::resolve_inner_state_t<Inner, P>>
    {
        detail::resolve_inner_state_t<Inner, P> inner;
        decltype(std::declval<detail::resolve_inner_state_t<Inner, P>>().x) x{};
        decltype(x) x0_saved{};
        decltype(x) lower{};
        decltype(x) upper{};
        detail::xoshiro256 rng{1};
        detail::member_judge<scalar_type> judge{};
        detail::best_member<decltype(x), scalar_type> best{};
        std::uint64_t base_seed{1};
        bool restart_pending{false};
        std::uint32_t restart_count{0};
        std::uint32_t dimension{0};
        std::uint32_t initial_lambda{0};

        Eigen::VectorX<scalar_type> c_eq{};
        Eigen::VectorX<scalar_type> c_ineq{};
    };

    Inner inner_policy_{};
    options_type options{};

    template <typename Problem, typename Convergence = default_convergence<scalar_type>>
    state_type<Problem> init(const Problem& problem, const auto& x0, const solver_options<Convergence>& opts)
    {
        inner_policy_.options = options.inner;
        state_type<Problem> s;
        s.x0_saved = x0;
        s.base_seed = pin_inner_seed();
        s.rng = detail::xoshiro256{detail::restart_seed(s.base_seed, 0)};
        s.inner = inner_policy_.init(problem, x0, opts);
        s.dimension = static_cast<std::uint32_t>(x0.size());
        s.initial_lambda = lambda_of(s);
        s.judge = detail::member_judge<scalar_type>{opts, inner_policy_, stagnation_limit(s)};
        if constexpr(detail::boxed_problem<Problem, scalar_type>)
        {
            s.lower = problem.lower_bounds();
            s.upper = problem.upper_bounds();
        }
        sync_from_inner(s);
        return s;
    }

    template <typename Problem, typename Convergence>
    state_type<Problem> init(const Problem& problem, const auto& x0, const solver_options<Convergence>& opts, const options_type& policy_opts)
    {
        options = policy_opts;
        return init(problem, x0, opts);
    }

    template <typename P>
    step_result<scalar_type> step(state_type<P>& s)
    {
        if(s.restart_pending)
            reinit(s);
        const step_result<scalar_type> result = inner_policy_.step(s.inner);
        sync_from_inner(s);
        return detail::member_step<P>(s, result, [&](solver_status verdict) { return finishes(verdict) || restarts_used(s); });
    }

    template <typename P>
    solver_status reset(state_type<P>& s, const auto& x0)
    {
        rewind_seeds(s);
        return settle(s, inner_policy_.reset(s.inner, x0));
    }

    template <typename P>
    solver_status reset_clear(state_type<P>& s, const auto& x0)
    {
        rewind_seeds(s);
        s.restart_count = 0;
        return settle(s, inner_policy_.reset_clear(s.inner, x0));
    }

    // Lets a test mutate the inner options, such as options.lambda, to lock the inner reset() contract that a restart
    // depends on; the decorator otherwise owns the inner policy.
    Inner& inner_policy() { return inner_policy_; }
    const Inner& inner_policy() const { return inner_policy_; }

private:

    template <typename P>
    void reinit(state_type<P>& s)
    {
        ++s.restart_count;

        // IPOP: population = initial_lambda * multiplier^restart_count
        const auto new_pop = static_cast<std::uint32_t>(static_cast<scalar_type>(s.initial_lambda) * std::pow(options.population_multiplier, static_cast<scalar_type>(s.restart_count)));

        if constexpr(requires { inner_policy_.options.population_size; })
            inner_policy_.options.population_size = new_pop;
        else if constexpr(requires { inner_policy_.options.lambda; })
            inner_policy_.options.lambda = new_pop;

        if constexpr(requires { inner_policy_.options.seed; })
            inner_policy_.options.seed = detail::restart_seed(s.base_seed, s.restart_count);
        inner_policy_.reset_clear(s.inner, restart_mean(s));
        s.restart_pending = false;
        s.judge.restart();
        sync_from_inner(s);
    }

    static constexpr bool finishes(solver_status verdict)
    {
        return verdict == solver_status::converged || verdict == solver_status::ftol_reached || verdict == solver_status::xtol_reached
            || verdict == solver_status::invalid_problem;
    }

    template <typename P>
    bool restarts_used(const state_type<P>& s) const
    {
        return options.max_restarts && s.restart_count >= *options.max_restarts;
    }

    // Auger and Hansen (2005): a run stagnates when its best objective has not improved for 10 + ceil(30 n / lambda)
    // generations.
    template <typename P>
    std::uint32_t stagnation_limit(const state_type<P>& s) const
    {
        return options.stagnation_limit.value_or(static_cast<std::uint32_t>(10 + std::ceil(30.0 * s.dimension / s.initial_lambda)));
    }

    template <typename P>
    static std::uint32_t lambda_of(const state_type<P>& s)
    {
        if constexpr(requires { s.inner.lambda; })
            return static_cast<std::uint32_t>(s.inner.lambda);
        else if constexpr(requires { s.inner.params.lambda; })
            return static_cast<std::uint32_t>(s.inner.params.lambda);
        else
            return static_cast<std::uint32_t>(4 + 3 * std::log(static_cast<scalar_type>(s.dimension)));
    }

    // Auger and Hansen (2005) section III: a restart draws its mean in the search box to probe a new basin. The draw
    // uses the decorator's own generator, so the inner policy's reset stays equal to its construction.
    template <typename P>
    decltype(state_type<P>::x0_saved) restart_mean(state_type<P>& s)
    {
        decltype(state_type<P>::x0_saved) mean = s.x0_saved;
        if constexpr(detail::boxed_problem<P, scalar_type>)
            detail::uniform_in_finite_box<scalar_type>(mean, s.lower, s.upper, s.rng);
        return mean;
    }

    // An unset inner seed is pinned to 1, so a default-constructed decorator, as bench harnesses and concept probes
    // build it, reproduces its solve; a caller who wants a random stream sets options.inner.seed.
    std::uint64_t pin_inner_seed()
    {
        if constexpr(requires { inner_policy_.options.seed; })
        {
            if(!inner_policy_.options.seed.has_value())
                inner_policy_.options.seed = std::uint64_t{1};
            return *inner_policy_.options.seed;
        }
        return std::uint64_t{1};
    }

    template <typename P>
    static solver_status settle(state_type<P>& s, solver_status status)
    {
        s.judge.restart();
        s.best.clear();
        s.restart_pending = false;
        sync_from_inner(s);
        return status;
    }

    template <typename P>
    void rewind_seeds(state_type<P>& s)
    {
        if constexpr(requires { inner_policy_.options.seed; })
            inner_policy_.options.seed = s.base_seed;
        s.rng = detail::xoshiro256{detail::restart_seed(s.base_seed, 0)};
    }

    template <typename P>
    static void sync_from_inner(state_type<P>& s)
    {
        s.x = s.inner.x;
        detail::mirror_evaluations(s, s.inner);
        if constexpr(requires { s.inner.c_eq; s.inner.c_ineq; })
        {
            s.c_eq = s.inner.c_eq;
            s.c_ineq = s.inner.c_ineq;
        }
    }
};

}

#endif
