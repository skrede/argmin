#ifndef HPP_GUARD_NABLAPP_SOLVER_CMAES_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_CMAES_POLICY_H

// CMA-ES (Covariance Matrix Adaptation Evolution Strategy) solver policy.
//
// Implements the CMA Evolution Strategy where step() = one generation:
// sample lambda offspring, evaluate, rank, update mean/covariance/sigma.
// Supports optional IPOP restart on stagnation detection (D-04) and
// box constraint handling via repair+penalty (D-03).
//
// Requires: objective<P,S> (no gradient needed).
// When bound_constrained<P,S>, applies repair+penalty boundary handling.
//
// Reference: Hansen & Ostermeier (2001), "Completely Derandomized
//            Self-Adaptation in Evolution Strategies."
//            Hansen (2023) "The CMA Evolution Strategy: A Tutorial",
//            arXiv:1604.00772.
//            K&W Section 8.7, Algorithm 8.10.

#include "nablapp/detail/cmaes_constants.h"
#include "nablapp/detail/cmaes_covariance.h"
#include "nablapp/detail/cmaes_sampling.h"
#include "nablapp/result/step_result.h"
#include "nablapp/solver/options.h"

#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

namespace nablapp
{

struct cmaes_policy
{
    using scalar_type = double;

    enum class restart_strategy { none, ipop };

    struct options_type
    {
        int lambda{0};
        double initial_sigma{0.3};
        restart_strategy restart{restart_strategy::none};
        std::optional<unsigned> seed{std::nullopt};
    };

    struct state_type
    {
        Eigen::VectorXd x;
        double objective_value{};

        Eigen::VectorXd mean;
        Eigen::MatrixXd C;
        Eigen::MatrixXd B;
        Eigen::VectorXd D;
        Eigen::VectorXd p_sigma;
        Eigen::VectorXd p_c;
        double sigma{};
        int generation{0};
        detail::cmaes_params params;
        std::mt19937 rng;

        std::function<double(const Eigen::VectorXd&)> eval_value;
        Eigen::VectorXd lower;
        Eigen::VectorXd upper;
        bool has_bounds{false};

        double initial_sigma{};
        int stagnation_count{0};
        double best_ever_value{std::numeric_limits<double>::infinity()};
    };

    options_type options{};

    template <typename Problem, typename Convergence>
    state_type init(this auto&& self, const Problem& problem, const Eigen::VectorXd& x0,
                    const solver_options<Convergence>& opts, const options_type& policy_opts)
    {
        self.options = policy_opts;
        return self.init(problem, x0, opts);
    }

    template <typename Problem, typename Convergence = default_convergence>
    state_type init(this auto&& self, const Problem& problem, const Eigen::VectorXd& x0,
                    const solver_options<Convergence>& opts)
    {
        const int n = problem.dimension();
        state_type s;

        s.mean = x0;
        s.sigma = self.options.initial_sigma;
        s.initial_sigma = s.sigma;

        s.eval_value = [&problem](const Eigen::VectorXd& v) {
            return problem.value(v);
        };

        if constexpr(bound_constrained<Problem>)
        {
            s.lower = problem.lower_bounds();
            s.upper = problem.upper_bounds();
            s.has_bounds = true;
        }

        s.params = detail::compute_constants(n, self.options.lambda);

        s.C = Eigen::MatrixXd::Identity(n, n);
        s.p_sigma = Eigen::VectorXd::Zero(n);
        s.p_c = Eigen::VectorXd::Zero(n);
        s.B = Eigen::MatrixXd::Identity(n, n);
        s.D = Eigen::VectorXd::Ones(n);

        if(self.options.seed.has_value())
            s.rng.seed(self.options.seed.value());
        else
            s.rng.seed(std::random_device{}());

        s.objective_value = s.eval_value(x0);
        s.x = x0;
        s.best_ever_value = s.objective_value;

        return s;
    }

    step_result<double> step(this auto&& self, state_type& s)
    {
        const int n = s.mean.size();
        const int lambda = s.params.lambda;
        const int mu = s.params.mu;
        double old_best = s.objective_value;

        // 1. Sample offspring
        auto offspring = detail::sample_offspring(
            s.mean, s.sigma, s.B, s.D, lambda, s.rng);

        // 2. Evaluate with optional repair+penalty
        Eigen::VectorXd fitnesses(lambda);
        for(int i = 0; i < lambda; ++i)
        {
            Eigen::VectorXd xi = offspring.col(i);
            double penalty = 0.0;
            if(s.has_bounds)
                penalty = detail::repair_and_penalize(xi, s.lower, s.upper);
            offspring.col(i) = xi;
            fitnesses[i] = s.eval_value(xi) + penalty;
        }

        // 3. Rank offspring (ascending -- minimization)
        std::vector<int> idx(lambda);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return fitnesses[a] < fitnesses[b];
        });

        // 4. Update mean: weighted recombination of mu best
        Eigen::VectorXd mean_old = s.mean;
        Eigen::VectorXd mean_new = Eigen::VectorXd::Zero(n);
        for(int i = 0; i < mu; ++i)
            mean_new += s.params.weights[i] * offspring.col(idx[i]);

        // 5. delta_w = (mean_new - mean_old) / sigma
        Eigen::VectorXd delta_w = ((mean_new - mean_old) / s.sigma).eval();

        // 6. Update evolution path p_sigma (CSA)
        // C^{-1/2} * v = B * D^{-1} * B^T * v
        Eigen::VectorXd C_inv_sqrt_dw =
            (s.B * s.D.cwiseInverse().asDiagonal() * (s.B.transpose() * delta_w)).eval();

        s.p_sigma = (1.0 - s.params.c_sigma) * s.p_sigma
                  + std::sqrt(s.params.c_sigma * (2.0 - s.params.c_sigma) * s.params.mu_eff)
                    * C_inv_sqrt_dw;

        // 7. Update sigma
        s.sigma *= std::exp(s.params.c_sigma / s.params.d_sigma
                          * (s.p_sigma.norm() / s.params.chi_n - 1.0));

        // 8. h_sigma indicator
        double gen_factor = 1.0 - std::pow(1.0 - s.params.c_sigma,
                                           2.0 * static_cast<double>(s.generation + 1));
        double h_sigma = (s.p_sigma.norm() / std::sqrt(gen_factor)
                         < (1.4 + 2.0 / (static_cast<double>(n) + 1.0)) * s.params.chi_n)
                         ? 1.0 : 0.0;

        // 9. Update evolution path p_c
        s.p_c = (1.0 - s.params.c_c) * s.p_c
              + h_sigma * std::sqrt(s.params.c_c * (2.0 - s.params.c_c) * s.params.mu_eff)
                * delta_w;

        // 10. Compute deltas for covariance update
        Eigen::MatrixXd deltas(n, lambda);
        for(int i = 0; i < lambda; ++i)
            deltas.col(i) = (offspring.col(idx[i]) - mean_old) / s.sigma;

        // 11. Update covariance
        detail::update_covariance(s.C, s.p_c, h_sigma,
                                  s.params.c_1, s.params.c_mu, s.params.c_c,
                                  s.params.weights, deltas, s.B, s.D, mu);

        // 12. Set new mean
        s.mean = mean_new;

        // 13. Eigendecompose updated C
        detail::eigendecompose(s.C, s.B, s.D);

        // 14. Track best
        double gen_best_f = fitnesses[idx[0]];
        if(gen_best_f < s.objective_value)
        {
            s.objective_value = gen_best_f;
            s.x = offspring.col(idx[0]);
        }
        double mean_f = s.eval_value(s.mean);
        if(mean_f < s.objective_value)
        {
            s.objective_value = mean_f;
            s.x = s.mean;
        }
        if(s.objective_value < s.best_ever_value)
            s.best_ever_value = s.objective_value;

        // 15. IPOP stagnation check (D-04)
        if(self.options.restart == restart_strategy::ipop)
        {
            bool stagnated = false;

            // Collapsed: sigma * max(D) too small
            if(s.sigma * s.D.maxCoeff() < 1e-12 * s.initial_sigma)
                stagnated = true;

            // No improvement for many generations
            int stag_limit = 10 + static_cast<int>(std::ceil(30.0 * n / s.params.lambda));
            if(gen_best_f >= old_best)
                ++s.stagnation_count;
            else
                s.stagnation_count = 0;

            if(s.stagnation_count >= stag_limit)
                stagnated = true;

            // Condition number too large
            double cond = s.D.maxCoeff() / std::max(s.D.minCoeff(), 1e-30);
            if(cond * cond > 1e14)
                stagnated = true;

            if(stagnated)
            {
                // IPOP: double lambda, reset state, keep best
                int new_lambda = s.params.lambda * 2;
                s.params = detail::compute_constants(n, new_lambda);
                s.C = Eigen::MatrixXd::Identity(n, n);
                s.B = Eigen::MatrixXd::Identity(n, n);
                s.D = Eigen::VectorXd::Ones(n);
                s.p_sigma.setZero();
                s.p_c.setZero();
                s.sigma = s.initial_sigma;
                s.generation = 0;
                s.stagnation_count = 0;
                // Keep s.x, s.objective_value, s.best_ever_value
            }
        }

        // 16. Increment generation
        ++s.generation;

        // 17. Return step_result
        // Derivative-free convergence signalling (same pattern as BOBYQA):
        // - gradient_norm proxy: sigma * max(D) -- collapses when converged
        // - When no improvement, use sigma as proxy for objective_change and
        //   step_size to prevent basic_solver's ftol/stall checks firing early
        double grad_proxy = s.sigma * s.D.maxCoeff();
        bool improved = s.objective_value < old_best;
        double obj_change = improved ? (s.objective_value - old_best) : s.sigma;
        double effective_step = improved ? (delta_w.norm() * s.sigma)
                                        : s.sigma;

        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = grad_proxy,
            .step_size = effective_step,
            .objective_change = obj_change,
            .improved = improved,
        };
    }

    void reset(this auto&&, state_type& s, const Eigen::VectorXd& x0)
    {
        const int n = x0.size();
        s.mean = x0;
        s.x = x0;
        s.objective_value = s.eval_value(x0);
        s.best_ever_value = s.objective_value;
        s.p_sigma.setZero();
        s.p_c.setZero();
        s.C = Eigen::MatrixXd::Identity(n, n);
        s.B = Eigen::MatrixXd::Identity(n, n);
        s.D = Eigen::VectorXd::Ones(n);
        s.sigma = s.initial_sigma;
        s.generation = 0;
        s.stagnation_count = 0;
    }

    void reset_clear(this auto&& self, state_type& s, const Eigen::VectorXd& x0)
    {
        self.reset(s, x0);
    }
};

}

#endif
