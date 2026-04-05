#ifndef HPP_GUARD_NABLAPP_SOLVER_ISRES_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_ISRES_POLICY_H

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

#include "nablapp/detail/isres_operators.h"
#include "nablapp/detail/stochastic_ranking.h"
#include "nablapp/result/step_result.h"
#include "nablapp/solver/options.h"
#include "nablapp/types.h"

#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <vector>

namespace nablapp
{

template <int N = dynamic_dimension>
struct isres_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = isres_policy<M>;

    struct options_type
    {
        std::optional<std::uint32_t> population_size{};
        std::optional<double> parent_fraction{};
        std::optional<double> ranking_probability{};
        std::optional<double> differential_weight{};
        std::optional<std::uint64_t> seed{};
    };

    template <typename P = void>
    struct state_type
    {
        const P* problem{nullptr};
        Eigen::Vector<double, N> x;
        double objective_value{};

        Eigen::Matrix<double, N, Eigen::Dynamic> population;
        Eigen::MatrixXd sigmas;
        Eigen::VectorXd fitnesses;
        Eigen::VectorXd violations;

        Eigen::Vector<double, N> lower;
        Eigen::Vector<double, N> upper;

        std::mt19937 rng;
        std::uint32_t generation{0};
        double best_ever_value{std::numeric_limits<double>::infinity()};

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
        s.alpha = options.differential_weight.value_or(0.2);
        s.rates = detail::compute_es_rates(n);

        // Seed RNG
        if(options.seed.has_value())
            s.rng.seed(static_cast<unsigned>(options.seed.value()));
        else
            s.rng.seed(std::random_device{}());

        // Pre-allocate mutation workspace
        s.mutation_workspace.emplace(n);

        // Initialize population uniformly in bounds
        s.population = detail::initialize_population<N>(
            n, s.lambda, s.lower, s.upper, s.rng);

        // Initialize step sizes
        s.sigmas = detail::initialize_sigmas(n, s.lambda, s.lower, s.upper);

        // Evaluate all individuals
        s.fitnesses.resize(s.lambda);
        Eigen::MatrixXd all_constraints(n_c, s.lambda);

        for(int j = 0; j < s.lambda; ++j)
        {
            Eigen::Vector<double, N> xi = s.population.col(j);
            s.fitnesses[j] = s.problem->value(xi);

            if(n_c > 0)
            {
                Eigen::VectorXd c(n_c);
                s.problem->constraints(xi, c);
                all_constraints.col(j) = c;
            }
        }

        // Compute violations
        if(n_c > 0)
            s.violations = detail::compute_violations(all_constraints, s.n_eq, s.n_ineq);
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
        const int n = s.x.size();
        const int n_c = s.n_eq + s.n_ineq;
        double old_best = s.objective_value;

        // 1. Generate offspring via mutation + differential variation
        Eigen::Matrix<double, N, Eigen::Dynamic> offspring(n, s.lambda);
        Eigen::MatrixXd new_sigmas(n, s.lambda);

        for(int j = 0; j < s.lambda; ++j)
        {
            // Select parent from current population (top mu by rank)
            int parent_idx = j % s.mu;
            auto [child, sig] = s.mutation_workspace->mutate(
                Eigen::Vector<double, N>(s.population.col(parent_idx)),
                s.sigmas.col(parent_idx),
                s.x,
                s.alpha, s.rates.tau, s.rates.tau_prime,
                s.lower, s.upper, s.rng);

            offspring.col(j) = child;
            new_sigmas.col(j) = sig;
        }

        // 2. Evaluate offspring
        Eigen::VectorXd fitnesses(s.lambda);
        Eigen::MatrixXd all_constraints(n_c, s.lambda);

        for(int j = 0; j < s.lambda; ++j)
        {
            Eigen::Vector<double, N> xi = offspring.col(j);
            fitnesses[j] = s.problem->value(xi);

            if(n_c > 0)
            {
                Eigen::VectorXd c(n_c);
                s.problem->constraints(xi, c);
                all_constraints.col(j) = c;
            }
        }

        // 3. Compute violations
        Eigen::VectorXd violations = (n_c > 0)
            ? detail::compute_violations(all_constraints, s.n_eq, s.n_ineq)
            : Eigen::VectorXd::Zero(s.lambda);

        // 4. Stochastic ranking
        std::vector<std::uint32_t> indices(static_cast<std::size_t>(s.lambda));
        for(int j = 0; j < s.lambda; ++j)
            indices[static_cast<std::size_t>(j)] = static_cast<std::uint32_t>(j);

        detail::stochastic_rank(indices, fitnesses, violations, s.pf, s.rng);

        // 5. Select mu best-ranked as new population
        for(int j = 0; j < s.mu; ++j)
        {
            auto idx = indices[static_cast<std::size_t>(j)];
            s.population.col(j) = offspring.col(idx);
            s.sigmas.col(j) = new_sigmas.col(idx);
            s.fitnesses[j] = fitnesses[idx];
            s.violations[j] = violations[idx];
        }

        // Fill remaining slots by copying parents cyclically
        for(int j = s.mu; j < s.lambda; ++j)
        {
            int src = j % s.mu;
            s.population.col(j) = s.population.col(src);
            s.sigmas.col(j) = s.sigmas.col(src);
            s.fitnesses[j] = s.fitnesses[src];
            s.violations[j] = s.violations[src];
        }

        // 6. Update best if improved
        auto best_ranked = indices[0];
        if(fitnesses[best_ranked] < s.best_ever_value
           && violations[best_ranked] <= 0.0)
        {
            s.best_ever_value = fitnesses[best_ranked];
            s.objective_value = fitnesses[best_ranked];
            s.x = offspring.col(best_ranked);

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
            if(violations[best_ranked] < s.violations[0]
               || fitnesses[best_ranked] < s.objective_value)
            {
                s.objective_value = fitnesses[best_ranked];
                s.x = offspring.col(best_ranked);

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
    void reset(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        const int n = x0.size();

        // Re-initialize population uniformly
        s.population = detail::initialize_population<N>(
            n, s.lambda, s.lower, s.upper, s.rng);
        s.sigmas = detail::initialize_sigmas(n, s.lambda, s.lower, s.upper);

        s.x = x0;
        s.objective_value = s.problem->value(x0);
        s.best_ever_value = s.objective_value;
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
    void reset_clear(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        reset(s, x0);
    }
};

}

#endif
