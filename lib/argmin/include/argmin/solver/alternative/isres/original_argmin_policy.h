#ifndef HPP_GUARD_ARGMIN_SOLVER_ALTERNATIVE_ISRES_ORIGINAL_ARGMIN_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_ALTERNATIVE_ISRES_ORIGINAL_ARGMIN_POLICY_H

// FROZEN BASELINE — DO NOT MODIFY.
//
// This variant is the byte-for-byte pre-rewrite ISRES implementation.
// It is preserved as a research artifact for the empirical comparison
// reported in the published paper. New project rules (no_dynamic_eigen,
// no_pre-existing_disclaimers, allocation hygiene, sigma upper clamp,
// alpha-smoothing remap, dead-code removal) are CONSCIOUSLY NOT
// APPLIED here. Modifications are accepted only if they preserve
// observable behavior byte-identical to the pre-rewrite version.
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

#include "argmin/detail/isres_operators.h"
#include "argmin/detail/stochastic_ranking.h"
#include "argmin/detail/xoshiro256.h"
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

template <int N = dynamic_dimension>
struct original_argmin_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = original_argmin_policy<M>;

    struct options_type
    {
        std::optional<std::uint32_t> population_size{};
        std::optional<double> parent_fraction{};
        std::optional<double> ranking_probability{};
        // pull_to_best_weight controls the alpha-pull-to-best operator
        // in the frozen-baseline mutation path (NOT differential
        // variation -- the field name reflects the actual semantic).
        std::optional<double> pull_to_best_weight{};
        std::optional<std::uint64_t> seed{};
        std::uint16_t stall_window{200};
        double feasibility_gate{1e-4};
    };

    template <typename P = void>
    struct state_type
    {
        static constexpr int M = [] {
            if constexpr(has_constraint_count<P>) return constraint_count_v<P>;
            else return dynamic_dimension;
        }();

        const P* problem{nullptr};
        Eigen::Vector<double, N> x;
        double objective_value{};

        Eigen::Matrix<double, N, Eigen::Dynamic> population;
        Eigen::MatrixXd sigmas;
        Eigen::VectorXd fitnesses;
        Eigen::VectorXd violations;

        Eigen::Vector<double, N> lower;
        Eigen::Vector<double, N> upper;

        // RNG: xoshiro256+ matches cmaes (consistent across the global
        // / stochastic policies). std::mt19937 was 2.5 KB of state;
        // xoshiro256 is 32 B.
        std::optional<detail::xoshiro256> rng;
        std::uint32_t generation{0};
        double best_ever_value{std::numeric_limits<double>::infinity()};
        // Track the least-infeasibility witness across the run so the
        // pre-feasible best update can compare against the rolling
        // best instead of the just-overwritten s.violations[0].
        double best_ever_violation{std::numeric_limits<double>::infinity()};

        Eigen::VectorXd c_eq;
        Eigen::VectorXd c_ineq;
        int n_eq{0};
        int n_ineq{0};

        int lambda{0};
        int mu{0};
        double pf{0.45};
        double alpha{0.2};
        detail::es_learning_rates rates{};

        std::optional<detail::isres_operator_workspace<double, N>> mutation_workspace;

        // Per-step buffers hoisted to state so step() does not
        // heap-resize them every generation. Sized to
        // current lambda in init(); persistent across steps.
        Eigen::Matrix<double, N, Eigen::Dynamic> offspring_buf;
        Eigen::MatrixXd new_sigmas_buf;
        Eigen::VectorXd offspring_fitnesses_buf;
        Eigen::VectorXd offspring_violations_buf;
        Eigen::MatrixXd all_constraints_buf;
        std::vector<std::uint32_t> indices_buf;
    };

    options_type options{};

    template <typename Problem, typename Convergence>
    state_type<Problem> init(const Problem& problem,
                    const Eigen::Vector<double, N>& x0,
                    const solver_options<Convergence>& opts, const options_type& policy_opts)
    {
        options = policy_opts;
        return init(problem, x0, opts);
    }

    template <typename Problem, typename Convergence = default_convergence>
        requires objective<Problem> && constrained_values<Problem> && bound_constrained<Problem>
    state_type<Problem> init(const Problem& problem,
                    const Eigen::Vector<double, N>& x0,
                    const solver_options<Convergence>& opts)
    {
        constexpr int MC = state_type<Problem>::M;

        const int n = problem.dimension();
        state_type<Problem> s;
        s.problem = &problem;

        // Store bounds
        s.lower = problem.lower_bounds();
        s.upper = problem.upper_bounds();

        // Constraint counts
        s.n_eq = problem.num_equality();
        s.n_ineq = problem.num_inequality();
        const int n_c = s.n_eq + s.n_ineq;

        // Population parameters
        s.lambda = static_cast<int>(
            options.population_size.value_or(
                static_cast<std::uint32_t>(20 * (n + 1))));
        double frac = options.parent_fraction.value_or(1.0 / 7.0);
        s.mu = std::max(1, static_cast<int>(s.lambda * frac));
        s.pf = options.ranking_probability.value_or(0.45);
        s.alpha = options.pull_to_best_weight.value_or(0.2);
        s.rates = detail::compute_es_rates(n);

        // Seed RNG (xoshiro256+; consistent with cmaes_policy).
        const std::uint64_t seed = options.seed.value_or(
            static_cast<std::uint64_t>(std::random_device{}()));
        s.rng.emplace(seed);

        // Pre-allocate mutation workspace
        s.mutation_workspace.emplace(n);

        // Pre-allocate per-step buffers.
        s.offspring_buf.resize(n, s.lambda);
        s.new_sigmas_buf.resize(n, s.lambda);
        s.offspring_fitnesses_buf.resize(s.lambda);
        s.offspring_violations_buf.resize(s.lambda);
        s.all_constraints_buf.resize(n_c, s.lambda);
        s.indices_buf.resize(static_cast<std::size_t>(s.lambda));

        // Initialize population uniformly in bounds
        s.population = detail::initialize_population<N>(
            n, s.lambda, s.lower, s.upper, *s.rng);

        // Initialize step sizes
        s.sigmas = detail::initialize_sigmas(n, s.lambda, s.lower, s.upper);

        // Evaluate all individuals (use the pre-allocated all_constraints_buf)
        s.fitnesses.resize(s.lambda);

        for(int j = 0; j < s.lambda; ++j)
        {
            Eigen::Vector<double, N> xi = s.population.col(j);
            s.fitnesses[j] = s.problem->value(xi);

            if(n_c > 0)
            {
                Eigen::VectorXd c(n_c);
                s.problem->constraints(xi, c);
                s.all_constraints_buf.col(j) = c;
            }
        }

        // Compute violations
        if(n_c > 0)
            s.violations = detail::compute_violations(s.all_constraints_buf, s.n_eq, s.n_ineq);
        else
            s.violations = Eigen::VectorXd::Zero(s.lambda);

        // Find best individual (least violation first, then best objective)
        int best_idx = 0;
        for(int j = 1; j < s.lambda; ++j)
        {
            bool feasible_j = s.violations[j] <= 0.0;
            bool feasible_best = s.violations[best_idx] <= 0.0;

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

        s.x = s.population.col(best_idx);
        s.objective_value = s.fitnesses[best_idx];
        s.best_ever_value = s.objective_value;
        // Initial best-ever-violation is the violation at the chosen
        // best init point. Used by the pre-feasible best-update path
        // in step().
        s.best_ever_violation = s.violations[best_idx];

        // Store constraint values at best point
        s.c_eq.resize(s.n_eq);
        s.c_ineq.resize(s.n_ineq);
        if(n_c > 0)
        {
            Eigen::VectorXd c(n_c);
            s.problem->constraints(s.x, c);
            s.c_eq = c.head(s.n_eq);
            s.c_ineq = c.tail(s.n_ineq);
        }

        return s;
    }

    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        constexpr int MC = state_type<P>::M;

        const int n = s.x.size();
        const int n_c = s.n_eq + s.n_ineq;
        double old_best = s.objective_value;

        // 1. Generate offspring via mutation + differential variation.
        // Buffers live on s; mutation operator is unchanged in this
        // commit (the operator rewrite is a follow-up).
        for(int j = 0; j < s.lambda; ++j)
        {
            // Select parent from current population (top mu by rank)
            int parent_idx = j % s.mu;
            auto [child, sig] = s.mutation_workspace->mutate(
                Eigen::Vector<double, N>(s.population.col(parent_idx)),
                s.sigmas.col(parent_idx),
                s.x,
                s.alpha, s.rates.tau, s.rates.tau_prime,
                s.lower, s.upper, *s.rng);

            s.offspring_buf.col(j) = child;
            s.new_sigmas_buf.col(j) = sig;
        }

        // 2. Evaluate offspring (state-owned buffers).
        for(int j = 0; j < s.lambda; ++j)
        {
            Eigen::Vector<double, N> xi = s.offspring_buf.col(j);
            s.offspring_fitnesses_buf[j] = s.problem->value(xi);

            if(n_c > 0)
            {
                Eigen::VectorXd c(n_c);
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

        // Fill remaining slots by copying parents cyclically.
        // (Note: this fill is dead code -- the
        // mutation step at line 1 above reads via `parent_idx = j % mu`,
        // which only touches columns [0, mu). Slots [mu, lambda) are
        // structurally unread between step() invocations. Kept for
        // observable-state hygiene; eliminating it is a separate
        // micro-cleanup.)
        for(int j = s.mu; j < s.lambda; ++j)
        {
            int src = j % s.mu;
            s.population.col(j) = s.population.col(src);
            s.sigmas.col(j) = s.sigmas.col(src);
            s.fitnesses[j] = s.fitnesses[src];
            s.violations[j] = s.violations[src];
        }

        // 6. Update best if improved.
        auto best_ranked = s.indices_buf[0];
        const double best_ranked_f = s.offspring_fitnesses_buf[best_ranked];
        const double best_ranked_v = s.offspring_violations_buf[best_ranked];

        if(best_ranked_f < s.best_ever_value && best_ranked_v <= 0.0)
        {
            s.best_ever_value = best_ranked_f;
            s.best_ever_violation = 0.0;
            s.objective_value = best_ranked_f;
            s.x = s.offspring_buf.col(best_ranked);

            if(n_c > 0)
            {
                Eigen::VectorXd c(n_c);
                s.problem->constraints(s.x, c);
                s.c_eq = c.head(s.n_eq);
                s.c_ineq = c.tail(s.n_ineq);
            }
        }
        else if(s.best_ever_value == std::numeric_limits<double>::infinity())
        {
            // No feasible point found yet -- track least-violating
            // witness. Pre-fix this branch compared `s.violations[0]`,
            // but s.violations[0] was overwritten by the just-selected
            // top-ranked offspring at the population-update loop above
            // (comparing best-ranked to itself never updates).
            // Now compare against the rolling
            // s.best_ever_violation which is preserved across steps.
            //
            // Pre-fix the predicate also used `||` between violation-
            // decrease and fitness-decrease, which
            // admitted offspring with worse feasibility but better
            // fitness, masking the least-violating witness. Use `&&`
            // so we only update on a Pareto improvement (both lower
            // violation AND not-worse fitness).
            if(best_ranked_v < s.best_ever_violation
               && best_ranked_f <= s.objective_value)
            {
                s.best_ever_violation = best_ranked_v;
                s.objective_value = best_ranked_f;
                s.x = s.offspring_buf.col(best_ranked);

                if(n_c > 0)
                {
                    Eigen::VectorXd c(n_c);
                    s.problem->constraints(s.x, c);
                    s.c_eq = c.head(s.n_eq);
                    s.c_ineq = c.tail(s.n_ineq);
                }
            }
        }

        ++s.generation;

        // Convergence signalling
        double mean_sigma = s.sigmas.leftCols(s.mu).mean();
        double grad_proxy = mean_sigma * std::sqrt(static_cast<double>(n));
        bool improved = s.objective_value < old_best;
        double obj_change = improved ? (s.objective_value - old_best) : mean_sigma;

        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = grad_proxy,
            .step_size = mean_sigma,
            .objective_change = obj_change,
            .improved = improved,
            .x_norm = s.x.norm(),
        };
    }

    template <typename P>
    void reset(state_type<P>& s, Eigen::Ref<const Eigen::Vector<double, N>> x0)
    {
        constexpr int MC = state_type<P>::M;

        const int n = x0.size();

        // Re-initialize population uniformly
        s.population = detail::initialize_population<N>(
            n, s.lambda, s.lower, s.upper, *s.rng);
        s.sigmas = detail::initialize_sigmas(n, s.lambda, s.lower, s.upper);

        s.x = x0;
        s.objective_value = s.problem->value(x0);
        s.best_ever_value = s.objective_value;
        s.best_ever_violation = std::numeric_limits<double>::infinity();
        s.generation = 0;

        if(s.n_eq + s.n_ineq > 0)
        {
            const int n_c = s.n_eq + s.n_ineq;
            Eigen::VectorXd c(n_c);
            s.problem->constraints(s.x, c);
            s.c_eq = c.head(s.n_eq);
            s.c_ineq = c.tail(s.n_ineq);
        }
    }

    template <typename P>
    void reset_clear(state_type<P>& s, Eigen::Ref<const Eigen::Vector<double, N>> x0)
    {
        reset(s, x0);
    }
};

}

#endif
