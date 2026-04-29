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

#include "nablapp/detail/xoshiro256.h"
#include "nablapp/detail/cmaes_constants.h"
#include "nablapp/detail/cmaes_covariance.h"
#include "nablapp/detail/cmaes_sampling.h"
#include "nablapp/options/cmaes_options.h"
#include "nablapp/result/step_result.h"
#include "nablapp/solver/options.h"
#include "nablapp/types.h"

#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

namespace nablapp
{

template <int N = dynamic_dimension, int MaxPopulation = dynamic_dimension>
struct cmaes_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = cmaes_policy<M, MaxPopulation>;

    enum class restart_strategy { none, ipop };

    struct options_type
    {
        std::optional<std::uint32_t> lambda{};           // population size (default: auto, K&W 8.19)
        std::optional<double> initial_sigma{};           // initial step size (default: bound-scaled, Hansen tutorial)
        restart_strategy restart{restart_strategy::none};
        std::optional<std::uint64_t> seed{};             // RNG seed
        cmaes_options cmaes{};                           // Detection params (Hansen tutorial)
        std::optional<std::uint32_t> eigendecomposition_skip_generations{}; // Override Hansen formula
        std::optional<std::uint32_t> stagnation_window{};  // Override Hansen stagnation window formula
        std::uint16_t stall_window{100};
        double feasibility_gate{std::numeric_limits<double>::infinity()};
    };

    template <typename P = void>
    struct state_type
    {
        const P* problem{nullptr};
        Eigen::Vector<double, N> x;
        double objective_value{};

        Eigen::Vector<double, N> mean;
        Eigen::Matrix<double, N, N> C;
        Eigen::Matrix<double, N, N> B;
        Eigen::Vector<double, N> D;
        Eigen::Vector<double, N> p_sigma;
        Eigen::Vector<double, N> p_c;
        double sigma{};
        std::uint32_t generation{0};
        detail::cmaes_params<> params;
        detail::xoshiro256 rng{1};

        Eigen::Vector<double, N> lower;
        Eigen::Vector<double, N> upper;
        bool has_bounds{false};

        double initial_sigma{};
        double best_ever_value{std::numeric_limits<double>::infinity()};

        // Hansen stagnation detection history.
        // Reference: Hansen (2023) arXiv:1604.00772, Section B.3 item 5.
        std::vector<double> best_fitness_history;
        std::vector<double> median_fitness_history;
        std::uint32_t stagnation_window_min{120};

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> eigen_solver;

        bool covariance_dirty{false};
        std::uint32_t decomposition_skip_k{1};
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
    state_type<Problem> init(const Problem& problem,
                    const Eigen::Vector<double, N>& x0,
                    const solver_options<Convergence>& opts)
    {
        const int n = problem.dimension();
        state_type<Problem> s;
        s.problem = &problem;

        s.mean = x0;

        if constexpr(bound_constrained<Problem>)
        {
            s.lower = problem.lower_bounds();
            s.upper = problem.upper_bounds();
            s.has_bounds = true;
        }

        // Sigma initialization: explicit > bound-scaled > default.
        // Hansen (2023) arXiv:1604.00772: sigma = 1/4 to 1/3 of search space.
        if(options.initial_sigma.has_value())
        {
            s.sigma = options.initial_sigma.value();
        }
        else if(s.has_bounds)
        {
            double max_range = 0.0;
            for(int i = 0; i < n; ++i)
            {
                double range = s.upper[i] - s.lower[i];
                if(std::isfinite(range))
                    max_range = std::max(max_range, range);
            }
            s.sigma = max_range > 0.0 ? max_range / 3.0 : 0.3;
        }
        else
        {
            s.sigma = 0.3;
        }
        s.initial_sigma = s.sigma;

        // Lambda: user override or auto-compute, with bounded-problem minimum.
        // Hansen (2023): lambda >= 4*N recommended for multimodal landscape coverage.
        int pop_lambda = options.lambda.has_value()
            ? static_cast<int>(options.lambda.value())
            : 0;
        if(pop_lambda == 0 && s.has_bounds)
            pop_lambda = std::max(4 * n,
                static_cast<int>(4 + std::floor(3.0 * std::log(n))));
        s.params = detail::compute_constants(n, pop_lambda);

        s.C = Eigen::Matrix<double, N, N>::Identity(n, n);
        s.p_sigma = Eigen::Vector<double, N>::Zero(n);
        s.p_c = Eigen::Vector<double, N>::Zero(n);
        s.B = Eigen::Matrix<double, N, N>::Identity(n, n);
        s.D = Eigen::Vector<double, N>::Ones(n);
        s.eigen_solver.compute(s.C);
        s.covariance_dirty = false;

        // Hansen (2023) arXiv:1604.00772: eigendecomposition skip period.
        // K = max(1, floor(1 / (10 * n * (c_1 + c_mu)))).
        if(options.eigendecomposition_skip_generations.has_value())
        {
            s.decomposition_skip_k = options.eigendecomposition_skip_generations.value();
        }
        else
        {
            double skip_denom = 10.0 * n * (s.params.c_1 + s.params.c_mu);
            s.decomposition_skip_k = std::max(
                std::uint32_t{1},
                static_cast<std::uint32_t>(std::floor(1.0 / skip_denom)));
        }

        // xoshiro256+ RNG: 32 bytes state vs 2.5KB for mt19937.
        // Reference: Blackman & Vigna (2021), ACM TOMS 47(4).
        if(options.seed.has_value())
            s.rng = detail::xoshiro256{options.seed.value()};
        else
            s.rng = detail::xoshiro256{static_cast<std::uint64_t>(std::random_device{}())};

        // Hansen stagnation window: 120 + ceil(30*n/lambda) (additive,
        // not max). Pre-fix the `max(120, 30*n/lambda)` form picked 120
        // for any (n, lambda) where 30*n/lambda <= 120 -- which is
        // basically every typical low-n CMA-ES run (n=2, lambda=6:
        // 30*2/6 = 10, max picks 120; n=10, lambda=10: 30*10/10 = 30,
        // max picks 120). Stagnation then fired at gen=120 regardless
        // of elapsed history. The Hansen 2023 §B.3 formula is additive
        // -- 120 + ceil(30*n/lambda) -- so the term scales with the
        // dimension/popsize ratio rather than being clamped out.
        s.stagnation_window_min = std::uint32_t{120}
            + static_cast<std::uint32_t>(std::ceil(30.0 * n / s.params.lambda));
        if(options.stagnation_window.has_value())
            s.stagnation_window_min = *options.stagnation_window;

        s.objective_value = s.problem->value(x0);
        s.x = x0;
        s.best_ever_value = s.objective_value;

        return s;
    }

    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        const int n = s.mean.size();
        const int lambda = s.params.lambda;
        const int mu = s.params.mu;
        double old_best = s.objective_value;

        // MaxPop covers IPOP doubling for n=16 with K=3 restarts:
        // lambda_max = 4*16*2^3 = 512.
        constexpr int MaxPop = 512;

        // 1. Sample offspring
        auto offspring = detail::sample_offspring<double, N, MaxPop>(
            s.mean, s.sigma, s.B, s.D, lambda, s.rng);

        // 2. Evaluate with optional repair+penalty.
        //
        // Two parallel arrays: `fitnesses` carries the penalty-inflated
        // value used for ranking and Hansen selection; `unpenalized`
        // carries `problem.value(repaired_xi)`, the feasible objective
        // at the repaired point. The unpenalized value is what the
        // caller should see as `s.objective_value` -- the penalized
        // value is meaningless for cross-library benchmark comparison.
        // Static-audit G12.
        Eigen::Matrix<double, Eigen::Dynamic, 1, 0, MaxPop, 1> fitnesses(lambda);
        Eigen::Matrix<double, Eigen::Dynamic, 1, 0, MaxPop, 1> unpenalized(lambda);
        for(int i = 0; i < lambda; ++i)
        {
            Eigen::Vector<double, N> xi = offspring.col(i);
            double penalty = 0.0;
            if(s.has_bounds)
                penalty = detail::repair_and_penalize(xi, s.lower, s.upper);
            offspring.col(i) = xi;
            double f_raw = s.problem->value(xi);
            unpenalized[i] = f_raw;
            fitnesses[i] = f_raw + penalty;
        }

        // 3. Rank offspring (ascending -- minimization)
        std::vector<int> idx(lambda);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return fitnesses[a] < fitnesses[b];
        });

        // 4. Update mean: weighted recombination of mu best
        Eigen::Vector<double, N> mean_old = s.mean;
        Eigen::Vector<double, N> mean_new = Eigen::Vector<double, N>::Zero(n);
        for(int i = 0; i < mu; ++i)
            mean_new += s.params.weights[i] * offspring.col(idx[i]);

        // 5. delta_w = (mean_new - mean_old) / sigma
        Eigen::Vector<double, N> delta_w = ((mean_new - mean_old) / s.sigma).eval();

        // 6. Update evolution path p_sigma (CSA)
        // C^{-1/2} * v = B * D^{-1} * B^T * v
        Eigen::Vector<double, N> C_inv_sqrt_dw =
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
        Eigen::Matrix<double, N, Eigen::Dynamic, 0,
            N == Eigen::Dynamic ? Eigen::Dynamic : N, MaxPop> deltas(n, lambda);
        for(int i = 0; i < lambda; ++i)
            deltas.col(i) = (offspring.col(idx[i]) - mean_old) / s.sigma;

        // 11. Update covariance
        detail::update_covariance(s.C, s.p_c, h_sigma,
                                  s.params.c_1, s.params.c_mu, s.params.c_c,
                                  s.params.weights, deltas, s.B, s.D, mu,
                                  s.covariance_dirty);

        // 12. Set new mean
        s.mean = mean_new;

        // 13. Eigendecompose updated C (reuses pre-allocated solver workspace).
        // Hansen (2023) arXiv:1604.00772: eigendecomposition skip.
        // Dirty flag (D-09): skip when covariance unchanged.
        // Periodic (D-10): decompose every K generations.
        if(s.covariance_dirty || s.generation % s.decomposition_skip_k == 0)
        {
            detail::eigendecompose(s.C, s.B, s.D, s.eigen_solver);
            s.covariance_dirty = false;
        }

        // 14. Track best across the offspring. Hansen Algorithm 11 line 11:
        // best-ranked offspring (selected by penalized fitness) is the
        // candidate. We rank by `fitnesses` but record the unpenalized
        // `problem.value(repaired_xi)` in `s.objective_value` so a
        // cross-library benchmark reads a comparable feasible-objective
        // value rather than a repair-penalty-inflated number. Static-
        // audit G12.
        //
        // Pre-fix step() also evaluated `s.problem->value(s.mean)` every
        // generation and replaced s.x / objective_value if the mean came
        // in lower; that added ~1 eval per gen (~16% inflation at popsize
        // 6) for no algorithmic reason. Static-audit G6 (removed in the
        // commit immediately preceding this one).
        const int best_offspring = idx[0];
        double gen_best_f = unpenalized[best_offspring];
        if(gen_best_f < s.objective_value)
        {
            s.objective_value = gen_best_f;
            s.x = offspring.col(best_offspring);
        }
        if(s.objective_value < s.best_ever_value)
            s.best_ever_value = s.objective_value;

        // 15. Status detection: roundoff_limited and diverged (Hansen tutorial)
        std::optional<solver_status> policy_status{};

        // NaN guard: numerical overflow in sigma or eigendecomposition
        // produces NaN which defeats all subsequent comparisons.
        if(!std::isfinite(s.sigma) || !std::isfinite(s.D.maxCoeff()))
            policy_status = solver_status::diverged;

        // Sigma collapse detection (roundoff_limited)
        double sigma_collapse_thr = options.cmaes.sigma_collapse_threshold.value_or(1e-12);
        if(!policy_status.has_value()
           && s.sigma * s.D.maxCoeff() < sigma_collapse_thr * s.initial_sigma)
            policy_status = solver_status::roundoff_limited;

        // Condition number explosion detection (diverged)
        double cond_limit = options.cmaes.condition_number_limit.value_or(1e14);
        double cond = s.D.maxCoeff() / std::max(s.D.minCoeff(), 1e-30);
        if(!policy_status.has_value() && cond * cond > cond_limit)
            policy_status = solver_status::diverged;

        // 16. IPOP stagnation check using Hansen's criteria.
        // Reference: Hansen (2023) arXiv:1604.00772, Section B.3.
        if(options.restart == restart_strategy::ipop)
        {
            // Track fitness history for stagnation detection. Use the
            // PENALIZED fitness here (what the search actually optimizes)
            // -- the unpenalized value is what we expose to callers as
            // s.objective_value, but the search's progress signal is the
            // penalized one. Static-audit G12.
            s.best_fitness_history.push_back(fitnesses[best_offspring]);
            {
                // Compute median fitness of this generation.
                std::vector<double> gen_fitnesses(lambda);
                for(int i = 0; i < lambda; ++i)
                    gen_fitnesses[i] = fitnesses[i];
                auto mid = gen_fitnesses.begin() + lambda / 2;
                std::nth_element(gen_fitnesses.begin(), mid, gen_fitnesses.end());
                s.median_fitness_history.push_back(*mid);
            }
            // Cap history size.
            if(s.best_fitness_history.size() > 20000)
            {
                s.best_fitness_history.erase(s.best_fitness_history.begin());
                s.median_fitness_history.erase(s.median_fitness_history.begin());
            }

            bool stagnated = false;

            // Collapsed: sigma * max(D) too small
            if(policy_status == solver_status::roundoff_limited)
                stagnated = true;

            // Hansen B.3 item 5: Stagnation criterion with median history window.
            // When both histories have >= window entries, check if the median
            // of the last 30% is not improving vs the median of the first 30%.
            //
            // Dynamic cap: as the search progresses, expand the comparison
            // window so stagnation becomes harder to fire late in the run
            // (Hansen 2023 §B.3 "0.2 * generation" cap). Effective window
            // grows from `stagnation_window_min` (early) to
            // `ceil(0.2 * generation)` (late). This keeps the stagnation
            // detector usable across the full IPOP-style restart sweep
            // without needing a per-restart hand-tune.
            auto window = std::max(
                static_cast<std::size_t>(s.stagnation_window_min),
                static_cast<std::size_t>(std::ceil(0.2 * static_cast<double>(s.generation))));
            if(!stagnated && s.best_fitness_history.size() >= window)
            {
                auto check_stagnation = [](const std::vector<double>& history,
                                           std::size_t win) {
                    std::size_t seg = win * 3 / 10;
                    if(seg == 0)
                        return false;
                    std::size_t h = history.size();
                    // Median of first 30% of window.
                    std::vector<double> old_seg(history.begin() + static_cast<std::ptrdiff_t>(h - win),
                                                history.begin() + static_cast<std::ptrdiff_t>(h - win + seg));
                    auto om = old_seg.begin() + static_cast<std::ptrdiff_t>(seg / 2);
                    std::nth_element(old_seg.begin(), om, old_seg.end());
                    double old_median = *om;
                    // Median of last 30% of window.
                    std::vector<double> new_seg(history.end() - static_cast<std::ptrdiff_t>(seg),
                                                history.end());
                    auto nm = new_seg.begin() + static_cast<std::ptrdiff_t>(seg / 2);
                    std::nth_element(new_seg.begin(), nm, new_seg.end());
                    double new_median = *nm;
                    return new_median >= old_median;
                };

                if(check_stagnation(s.best_fitness_history, window)
                   && check_stagnation(s.median_fitness_history, window))
                    stagnated = true;
            }

            // Hansen B.3 item 4: EqualFunValues.
            // Stop if range of best values over last K generations is zero.
            if(!stagnated)
            {
                auto efv_window = static_cast<std::size_t>(
                    10 + std::ceil(30.0 * n / s.params.lambda));
                if(s.best_fitness_history.size() >= efv_window)
                {
                    auto it = s.best_fitness_history.end()
                              - static_cast<std::ptrdiff_t>(efv_window);
                    auto [mn, mx] = std::minmax_element(it, s.best_fitness_history.end());
                    if(*mx - *mn < 1e-15 * (std::abs(*mx) + std::abs(*mn) + 1e-30))
                        stagnated = true;
                }
            }

            // Condition number too large
            if(policy_status == solver_status::diverged)
                stagnated = true;

            if(stagnated)
            {
                // IPOP: double lambda, reset state, keep best.
                int new_lambda = s.params.lambda * 2;

                // Guard: the doubled population must not exceed the
                // stack-allocated ceiling (MaxPop) or the policy template
                // ceiling (MaxPopulation).
                bool pop_overflow = (new_lambda > MaxPop);
                if constexpr(MaxPopulation != dynamic_dimension)
                    pop_overflow = pop_overflow || (new_lambda > MaxPopulation);

                if(pop_overflow)
                {
                    policy_status = solver_status::stalled;
                    ++s.generation;
                    return step_result<double>{
                        .objective_value = s.objective_value,
                        .gradient_norm = s.sigma * s.D.maxCoeff(),
                        .step_size = 0.0,
                        .objective_change = 0.0,
                        .improved = false,
                        .x_norm = s.x.norm(),
                        .policy_status = policy_status,
                    };
                }

                s.params = detail::compute_constants(n, new_lambda);
                s.C = Eigen::Matrix<double, N, N>::Identity(n, n);
                s.B = Eigen::Matrix<double, N, N>::Identity(n, n);
                s.D = Eigen::Vector<double, N>::Ones(n);
                s.covariance_dirty = false;
                s.p_sigma.setZero();
                s.p_c.setZero();
                s.sigma = s.initial_sigma;
                s.generation = 0;
                s.best_fitness_history.clear();
                s.median_fitness_history.clear();
                policy_status = std::nullopt;
            }
        }

        // 17. Increment generation
        ++s.generation;

        // 18. Return step_result
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
            .x_norm = s.x.norm(),
            .policy_status = policy_status,
        };
    }

    template <typename P>
    void reset(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        const int n = x0.size();
        s.mean = x0;
        s.x = x0;
        s.objective_value = s.problem->value(x0);
        s.best_ever_value = s.objective_value;
        s.p_sigma.setZero();
        s.p_c.setZero();
        s.C = Eigen::Matrix<double, N, N>::Identity(n, n);
        s.B = Eigen::Matrix<double, N, N>::Identity(n, n);
        s.D = Eigen::Vector<double, N>::Ones(n);
        s.sigma = s.initial_sigma;
        s.generation = 0;
        s.best_fitness_history.clear();
        s.median_fitness_history.clear();
    }

    template <typename P>
    void reset_clear(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        reset(s, x0);
    }
};

}

#endif
