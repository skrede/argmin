#ifndef HPP_GUARD_ARGMIN_DETAIL_CMAES_SEARCH_START_H
#define HPP_GUARD_ARGMIN_DETAIL_CMAES_SEARCH_START_H

#include "argmin/detail/problem_shape.h"
#include "argmin/detail/cmaes_constants.h"
#include "argmin/detail/stochastic_seed.h"
#include "argmin/detail/evaluation_count.h"
#include "argmin/detail/cmaes_initial_shape.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <algorithm>

namespace argmin::detail
{

// Hansen (2023) arXiv:1604.00772 section B.2: K = max(1, floor(1 / (10 n (c_1 + c_mu)))). The floor term is 0 at low
// n, where K = 1 is kept: a small eigendecomposition costs under 1% of a generation there, and libcmaes's
// counter-based skip (cmastrategy.cc optimize_) gives a similar frequency.
template <typename Scalar, typename Params>
std::uint32_t decomposition_skip_period(const std::optional<std::uint32_t>& requested, int n, const Params& params)
{
    if(requested.has_value())
        return *requested;
    const Scalar skip_denom = Scalar(10.0) * static_cast<Scalar>(n) * (params.c_1 + params.c_mu);
    return std::max(std::uint32_t{1}, static_cast<std::uint32_t>(std::floor(Scalar(1.0) / skip_denom)));
}

// Hansen (2023) arXiv:1604.00772 section B.3, "Stagnation": a history of at least 120 + ceil(30 n / lambda).
template <typename Scalar>
std::uint32_t stagnation_window_floor(const std::optional<std::uint32_t>& requested, int n, int lambda)
{
    if(requested.has_value())
        return *requested;
    return std::uint32_t{120} + static_cast<std::uint32_t>(std::ceil(Scalar(30.0) * static_cast<Scalar>(n) / static_cast<Scalar>(lambda)));
}

template <typename State>
void size_generation_buffers(State& s, int n)
{
    s.fitnesses_buf.resize(s.params.lambda);
    s.unpenalized_buf.resize(s.params.lambda);
    s.idx_buf.resize(static_cast<std::size_t>(s.params.lambda));
    s.deltas_buf.resize(n, s.params.lambda);
    s.gen_fitnesses_buf.resize(static_cast<std::size_t>(s.params.lambda));
}

struct cmaes_start_rules
{
    bool bounded_population_floor;
    bool box_scaled_covariance;
};

// Everything a search derives from the options and its start point, so a reset rebuilds what construction builds. An
// over-cap population leaves the buffers unsized, since they have a static maximum.
template <typename Scalar, typename State, typename Options, typename Start>
void start_cmaes_search(State& s, const Options& options, const Start& x0, cmaes_start_rules rules)
{
    const int n = static_cast<int>(x0.size());
    s.mean = x0;
    s.x0 = x0;
    s.sigma = initial_step_size<Scalar>(options.initial_sigma, s.has_bounds, s.lower, s.upper);
    s.initial_sigma = s.sigma;
    s.params = compute_constants<Scalar>(n, population_size<Scalar>(options.lambda, State::MaxPop, n, rules.bounded_population_floor && s.has_bounds));
    s.invalid_problem = (s.params.lambda > State::MaxPop);
    if(!s.invalid_problem)
        size_generation_buffers(s, n);
    initial_shape<Scalar>(s, n, rules.box_scaled_covariance && s.has_bounds && !options.initial_sigma.has_value());
    s.decomposition_skip_k = decomposition_skip_period<Scalar>(options.eigendecomposition_skip_generations, n, s.params);
    s.rng = seeded_generator(options.seed);
    s.stagnation_window_min = stagnation_window_floor<Scalar>(options.stagnation_window, n, s.params.lambda);
    s.generation = 0;
    s.best_fitness_history.clear();
    s.median_fitness_history.clear();
    s.x = x0;
    if(latch_rejection<Scalar>(s, s.invalid_problem || declares_constraint_rows(*s.problem)))
        return;
    s.objective_value = counted_value<Scalar>(s.evaluations, *s.problem, x0);
    s.best_ever_value = s.objective_value;
}

}

#endif
