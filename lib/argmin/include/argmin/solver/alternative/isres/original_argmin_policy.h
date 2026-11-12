#ifndef HPP_GUARD_ARGMIN_SOLVER_ALTERNATIVE_ISRES_ORIGINAL_ARGMIN_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_ALTERNATIVE_ISRES_ORIGINAL_ARGMIN_POLICY_H

// FROZEN SEARCH — DO NOT MODIFY THE OPERATORS.
//
// This variant's search is the byte-for-byte pre-rewrite ISRES implementation.
// It is preserved as a research artifact for the empirical comparison
// reported in the published paper. New project rules (no_dynamic_eigen,
// no_pre-existing_disclaimers, allocation hygiene, sigma upper clamp,
// alpha-smoothing remap, dead-code removal) are CONSCIOUSLY NOT
// APPLIED here. Modifications are accepted only if they preserve
// the search trajectory byte-identical to the pre-rewrite version; the
// point it returns and what it claims follow the other ISRES variants.
//
// The mismapping of `alpha` to pull-to-best (rather than sigma-smoothing)
// is the key algorithmic divergence vs Runarsson-Yao 2005 / NLopt
// isres.c and is intentionally PRESERVED here.

// ISRES (Improved Stochastic Ranking Evolution Strategy) solver policy.
//
// Global constrained optimizer using stochastic ranking for selection
// and log-normal self-adaptive step sizes with differential variation.
// Requires bound constraints for population initialization and
// constraint values (no Jacobian needed).
//
// step() = one generation: mutate, evaluate, rank, select.
//
// Reference: Runarsson & Yao (2005), "Search Biases in Constrained
//            Evolutionary Optimization", IEEE Trans. SMC-C.
//            K&W Section 8.6 (evolution strategies).

#include "argmin/detail/lagrangian.h"
#include "argmin/detail/isres_record.h"
#include "argmin/detail/isres_operators.h"
#include "argmin/detail/problem_shape.h"
#include "argmin/detail/stochastic_ranking.h"
#include "argmin/detail/xoshiro256.h"
#include "argmin/options/isres_defaults.h"
#include "argmin/result/step_result.h"
#include "argmin/solver/options.h"
#include "argmin/types.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <vector>

namespace argmin::alternative::isres
{

template <scalar Scalar = double, int N = dynamic_dimension>
struct original_argmin_policy
{
    using scalar_type = Scalar;

    template <int M>
    using rebind = original_argmin_policy<Scalar, M>;

    struct options_type
    {
        std::optional<std::uint32_t> population_size{};
        std::optional<Scalar> parent_fraction{};
        std::optional<Scalar> ranking_probability{};
        // pull_to_best_weight controls the alpha-pull-to-best operator
        // in the frozen-baseline mutation path (NOT differential
        // variation -- the field name reflects the actual semantic).
        std::optional<Scalar> pull_to_best_weight{};
        std::optional<std::uint64_t> seed{};
        std::uint16_t stall_window{200};
    };

    template <typename P = void>
    struct state_type
    {
        static constexpr int M = [] {
            if constexpr(has_constraint_count<P>) return constraint_count_v<P>;
            else return dynamic_dimension;
        }();

        const P* problem{nullptr};
        Eigen::Vector<Scalar, N> x;
        Scalar objective_value{};

        Eigen::Matrix<Scalar, N, Eigen::Dynamic> population;
        Eigen::MatrixX<Scalar> sigmas;
        Eigen::VectorX<Scalar> fitnesses;
        Eigen::VectorX<Scalar> violations;

        Eigen::Vector<Scalar, N> lower;
        Eigen::Vector<Scalar, N> upper;

        std::optional<detail::xoshiro256> rng;
        std::uint32_t generation{0};
        detail::best_evaluated<Scalar> record;
        // The pull-to-best operator's target, kept by the baseline's own rule, which the search depends on; the
        // least-infeasibility witness lets the pre-feasible update compare against the rolling best instead of the
        // just-overwritten s.violations[0].
        Eigen::Vector<Scalar, N> pull_target;
        Scalar pull_target_value{};
        Scalar best_ever_value{std::numeric_limits<Scalar>::infinity()};
        Scalar best_ever_violation{std::numeric_limits<Scalar>::infinity()};

        Eigen::VectorX<Scalar> c_eq;
        Eigen::VectorX<Scalar> c_ineq;
        int n_eq{0};
        int n_ineq{0};
        bool invalid_problem{false};

        int lambda{0};
        int mu{0};
        Scalar pf{Scalar(0.45)};
        Scalar alpha{Scalar(0.2)};
        detail::es_learning_rates<Scalar> rates{};

        std::optional<detail::isres_operator_workspace<Scalar, N>> mutation_workspace;

        // Per-step buffers, sized to lambda in init() so step() never heap-resizes.
        Eigen::Matrix<Scalar, N, Eigen::Dynamic> offspring_buf;
        Eigen::MatrixX<Scalar> new_sigmas_buf;
        Eigen::VectorX<Scalar> offspring_fitnesses_buf;
        Eigen::VectorX<Scalar> offspring_violations_buf;
        Eigen::MatrixX<Scalar> all_constraints_buf;
        std::vector<std::uint32_t> indices_buf;
    };

    options_type options{};

    template <typename Problem, typename Convergence>
    state_type<Problem> init(const Problem& problem,
                    const Eigen::Vector<Scalar, N>& x0,
                    const solver_options<Convergence>& opts, const options_type& policy_opts)
    {
        options = policy_opts;
        return init(problem, x0, opts);
    }

    template <typename Problem, typename Convergence = default_convergence<scalar_type>>
        requires objective<Problem, Scalar> && constrained_values<Problem, Scalar> && bound_constrained<Problem, Scalar>
    state_type<Problem> init(const Problem& problem,
                    const Eigen::Vector<Scalar, N>& x0,
                    const solver_options<Convergence>& opts)
    {
        const int n = problem.dimension();
        state_type<Problem> s;
        s.problem = &problem;

        s.lower = problem.lower_bounds();
        s.upper = problem.upper_bounds();

        s.n_eq = problem.num_equality();
        s.n_ineq = problem.num_inequality();
        const int n_c = s.n_eq + s.n_ineq;

        s.lambda = static_cast<int>(options.population_size.value_or(static_cast<std::uint32_t>(20 * (n + 1))));
        s.mu = parent_count(s.lambda, options.parent_fraction.value_or(Scalar(1.0) / Scalar(7.0)));
        s.pf = options.ranking_probability.value_or(Scalar(0.45));
        s.alpha = options.pull_to_best_weight.value_or(Scalar(0.2));
        s.rates = detail::compute_es_rates<Scalar>(n);

        // Seed RNG (xoshiro256+; consistent with cmaes_policy).
        const std::uint64_t seed = options.seed.value_or(
            static_cast<std::uint64_t>(std::random_device{}()));
        s.rng.emplace(seed);

        s.invalid_problem = rejects_population(s);
        if(s.invalid_problem)
        {
            s.x = x0;
            s.objective_value = std::numeric_limits<Scalar>::infinity();
            return s;
        }

        s.mutation_workspace.emplace(n);

        // Pre-allocate per-step buffers.
        s.offspring_buf.resize(n, s.lambda);
        s.new_sigmas_buf.resize(n, s.lambda);
        s.offspring_fitnesses_buf.resize(s.lambda);
        s.offspring_violations_buf.resize(s.lambda);
        s.all_constraints_buf.resize(n_c, s.lambda);
        s.indices_buf.resize(static_cast<std::size_t>(s.lambda));

        s.population = detail::initialize_population<N>(
            n, s.lambda, s.lower, s.upper, *s.rng);

        s.sigmas = detail::initialize_sigmas(n, s.lambda, s.lower, s.upper);

        // Evaluate all individuals (use the pre-allocated all_constraints_buf)
        s.fitnesses.resize(s.lambda);

        for(int j = 0; j < s.lambda; ++j)
        {
            Eigen::Vector<Scalar, N> xi = s.population.col(j);
            s.fitnesses[j] = s.problem->value(xi);

            if(n_c > 0)
            {
                Eigen::VectorX<Scalar> c(n_c);
                s.problem->constraints(xi, c);
                s.all_constraints_buf.col(j) = c;
            }
        }

        if(n_c > 0)
            s.violations = detail::compute_violations(s.all_constraints_buf, s.n_eq, s.n_ineq);
        else
            s.violations = Eigen::VectorX<Scalar>::Zero(s.lambda);

        // The pull-to-best target starts at the least-violating individual, then the lowest objective.
        int best_idx = 0;
        for(int j = 1; j < s.lambda; ++j)
        {
            bool feasible_j = s.violations[j] <= Scalar(0.0);
            bool feasible_best = s.violations[best_idx] <= Scalar(0.0);

            if(feasible_j && !feasible_best)
                best_idx = j;
            else if(feasible_j == feasible_best)
            {
                if(feasible_j && s.fitnesses[j] < s.fitnesses[best_idx])
                    best_idx = j;
                else if(!feasible_j && s.violations[j] < s.violations[best_idx])
                    best_idx = j;
            }
        }

        s.pull_target = s.population.col(best_idx);
        s.pull_target_value = s.fitnesses[best_idx];
        s.best_ever_value = s.pull_target_value;
        s.best_ever_violation = s.violations[best_idx];

        detail::start_best_evaluated(s, s.population, s.fitnesses, static_cast<Scalar>(opts.feasibility_tolerance_in_force()));
        return s;
    }

    template <typename P>
    step_result<Scalar> step(state_type<P>& s)
    {
        if(s.invalid_problem)
            return detail::rejected_step<Scalar>(s);
        const int n = static_cast<int>(s.x.size());
        const int n_c = s.n_eq + s.n_ineq;
        Scalar old_best = s.objective_value;

        // 1. Generate offspring via mutation + differential variation.
        for(int j = 0; j < s.lambda; ++j)
        {
            int parent_idx = j % s.mu;
            auto [child, sig] = s.mutation_workspace->mutate(
                Eigen::Vector<Scalar, N>(s.population.col(parent_idx)),
                s.sigmas.col(parent_idx),
                s.pull_target,
                s.alpha, s.rates.tau, s.rates.tau_prime,
                s.lower, s.upper, *s.rng);

            s.offspring_buf.col(j) = child;
            // Element-wise: gcc reports the packet copy of a short float vector into a dynamic
            // column as an out-of-bounds read (-Warray-bounds, -Wstringop-overread).
            for(int i = 0; i < n; ++i)
                s.new_sigmas_buf(i, j) = sig[i];
        }

        // 2. Evaluate offspring (state-owned buffers).
        for(int j = 0; j < s.lambda; ++j)
        {
            Eigen::Vector<Scalar, N> xi = s.offspring_buf.col(j);
            s.offspring_fitnesses_buf[j] = s.problem->value(xi);

            if(n_c > 0)
            {
                Eigen::VectorX<Scalar> c(n_c);
                s.problem->constraints(xi, c);
                s.all_constraints_buf.col(j) = c;
            }
        }

        // 3. Compute violations (write into the persistent buffer).
        if(n_c > 0)
            s.offspring_violations_buf = detail::compute_violations(
                s.all_constraints_buf, s.n_eq, s.n_ineq);
        else
            s.offspring_violations_buf.setZero();

        // 4. Stochastic ranking (state-owned indices buffer).
        for(int j = 0; j < s.lambda; ++j)
            s.indices_buf[static_cast<std::size_t>(j)]
                = static_cast<std::uint32_t>(j);

        detail::stochastic_rank(s.indices_buf,
                                s.offspring_fitnesses_buf,
                                s.offspring_violations_buf,
                                s.pf, *s.rng);

        // 5. Select mu best-ranked as new population
        for(int j = 0; j < s.mu; ++j)
        {
            auto idx = s.indices_buf[static_cast<std::size_t>(j)];
            s.population.col(j) = s.offspring_buf.col(idx);
            s.sigmas.col(j) = s.new_sigmas_buf.col(idx);
            s.fitnesses[j] = s.offspring_fitnesses_buf[idx];
            s.violations[j] = s.offspring_violations_buf[idx];
        }

        // Dead fill: mutation reads parents via j % mu, so slots [mu, lambda) are never read between steps. Kept
        // for observable-state hygiene.
        for(int j = s.mu; j < s.lambda; ++j)
        {
            int src = j % s.mu;
            s.population.col(j) = s.population.col(src);
            s.sigmas.col(j) = s.sigmas.col(src);
            s.fitnesses[j] = s.fitnesses[src];
            s.violations[j] = s.violations[src];
        }

        // 6. Move the pull-to-best target if improved.
        auto best_ranked = s.indices_buf[0];
        const Scalar best_ranked_f = s.offspring_fitnesses_buf[best_ranked];
        const Scalar best_ranked_v = s.offspring_violations_buf[best_ranked];

        if(best_ranked_f < s.best_ever_value && best_ranked_v <= Scalar(0.0))
        {
            s.best_ever_value = best_ranked_f;
            s.best_ever_violation = Scalar(0.0);
            s.pull_target_value = best_ranked_f;
            s.pull_target = s.offspring_buf.col(best_ranked);
        }
        else if(s.best_ever_value == std::numeric_limits<Scalar>::infinity())
        {
            // No feasible point yet: update only on a Pareto improvement.
            if(best_ranked_v < s.best_ever_violation
               && best_ranked_f <= s.pull_target_value)
            {
                s.best_ever_violation = best_ranked_v;
                s.pull_target_value = best_ranked_f;
                s.pull_target = s.offspring_buf.col(best_ranked);
            }
        }

        // 7. Every offspring is offered to the record, judged by the solve's own comparator and tolerance.
        detail::keep_best_evaluated(s, s.offspring_buf, s.offspring_fitnesses_buf);
        ++s.generation;

        Scalar mean_sigma = s.sigmas.leftCols(s.mu).mean();
        bool improved = s.objective_value < old_best;
        Scalar obj_change = improved ? (s.objective_value - old_best) : mean_sigma;

        return step_result<Scalar>{
            .objective_value = s.objective_value,
            .gradient_norm = std::numeric_limits<Scalar>::quiet_NaN(),
            .step_size = mean_sigma,
            .objective_change = obj_change,
            .improved = improved,
            .constraint_violation = detail::primal_feasibility_inf(s.c_eq, s.c_ineq),
            .x_norm = s.x.norm(),
        };
    }

    template <typename P>
    solver_status reset(state_type<P>& s, Eigen::Ref<const Eigen::Vector<Scalar, N>> x0)
    {
        s.invalid_problem = rejects_population(s) || detail::constraint_rows_changed(*s.problem, s.n_eq, s.n_ineq);
        if(s.invalid_problem)
            return solver_status::invalid_problem;
        const int n = static_cast<int>(x0.size());

        s.population = detail::initialize_population<N>(
            n, s.lambda, s.lower, s.upper, *s.rng);
        s.sigmas = detail::initialize_sigmas(n, s.lambda, s.lower, s.upper);

        detail::restart_best_evaluated(s, x0);
        s.pull_target = x0;
        s.pull_target_value = s.objective_value;
        s.best_ever_value = s.objective_value;
        s.best_ever_violation = std::numeric_limits<Scalar>::infinity();
        s.generation = 0;
        return detail::latch_status(s);
    }

    template <typename P>
    solver_status reset_clear(state_type<P>& s, Eigen::Ref<const Eigen::Vector<Scalar, N>> x0)
    {
        return reset(s, x0);
    }

private:
    // A population_size above INT32_MAX arrives here negative, since the unsigned-to-int conversion is modular.
    template <typename P>
    static bool rejects_population(const state_type<P>& s)
    {
        return s.lambda < 1 || s.mu < 1;
    }

    // Zero flags a NaN fraction or a count above lambda; the int64 step holds a lambda a float Scalar rounds to 2^31.
    static int parent_count(int lambda, Scalar fraction)
    {
        const Scalar rounded = std::trunc(Scalar(lambda) * fraction);
        if(!(rounded <= Scalar(lambda)))
            return 0;
        return static_cast<int>(std::min<std::int64_t>(static_cast<std::int64_t>(std::max(Scalar(1), rounded)), lambda));
    }
};

}

#endif
