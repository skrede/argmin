#ifndef HPP_GUARD_NABLAPP_SOLVER_ALTERNATIVE_CMAES_REPAIR_L2_PENALTY_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_ALTERNATIVE_CMAES_REPAIR_L2_PENALTY_POLICY_H

// CMA-ES boundary-handling variant: clip x_i to [lower, upper] and add
// a sum-of-squared-distances penalty to the objective; CMA-ES sees the
// repaired x_i.
//
// This was the prior production cmaes_policy<> behavior, preserved
// verbatim under solver/alternative/ to enable an empirical comparison
// against pwq_reparameterization_policy and no_repair_adaptive_penalty_policy
// on bounded global cells (rastrigin_2/10, griewank_2, schwefel_2,
// ackley_2/10).
//
// Trade-off vs pwq_reparameterization_policy:
//   + Simple (one helper call); no per-coord buffer machinery.
//   - The L2 penalty introduces non-physical gradient signal at the
//     box boundary; biases the search away from boundary-active
//     optima on multimodal cells.
//
// CMA-ES algorithmic core: step() = one generation: sample lambda
// offspring, evaluate, rank, update mean/covariance/sigma. Supports
// optional IPOP restart on stagnation detection.
//
// Requires: objective<P,S> (no gradient needed).
// When bound_constrained<P,S>, applies repair+penalty boundary handling.
//
// References:
//   Hansen & Ostermeier (2001), "Completely Derandomized
//     Self-Adaptation in Evolution Strategies."
//   Hansen (2023) "The CMA Evolution Strategy: A Tutorial",
//     arXiv:1604.00772.
//   Hansen (2009) "A Method for Handling Uncertainty in Evolutionary
//     Optimization", §2 (boundary handling tutorial).
//   K&W (2025) Algorithms for Optimization 2e, §8.7, Algorithm 8.10.

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

namespace nablapp::alternative::cmaes
{

template <int N = dynamic_dimension, int MaxPopulation = dynamic_dimension>
struct repair_l2_penalty_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = repair_l2_penalty_policy<M, MaxPopulation>;

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

        // Auger & Hansen 2005 ("A Restart CMA Evolution Strategy with
        // Increasing Population Size", CEC 2005) §III: on each IPOP restart
        // the distribution mean is re-anchored to the user-provided initial
        // point x0. We capture x0 at init() and preserve it across restarts
        // so the IPOP branch can write `s.mean = s.x0` (libcmaes parity:
        // see cmasolutions.cc:49 _xmean = p._x0min within
        // CMASolutions(Parameters&), invoked from
        // ipopcmastrategy.cc:reset_search_state per restart).
        Eigen::Vector<double, N> x0;

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

        // Hansen 2023 (arXiv:1604.00772) section B.3 item 1 NoEffectAxis,
        // footnote 31: the axis index cycles 0..n-1 across calls per the
        // formula `i = (g mod n) + 1`. We carry the cycle position in
        // state so each step() advances to the next axis. Preserved across
        // IPOP restarts: although s.generation resets to 0 in the IPOP
        // branch, the cycle position is independent of generation count.
        int axis_cycle_index{0};

        // Hansen 2023 (arXiv:1604.00772) section B.3 item 8 TolXUp baseline:
        // sigma * max(diag(D)) at init time. The criterion fires when the
        // current value exceeds 1e4 * (initial sigma * initial max(D)).
        // D defaults to all-ones at init, so initial_d_max == 1.0 in the
        // canonical case; we record the actual init-time value to be
        // robust to any future init() change. Preserved across IPOP
        // restart (same baseline, same divergence cap).
        double initial_d_max{1.0};

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> eigen_solver;

        bool covariance_dirty{false};
        std::uint32_t decomposition_skip_k{1};

        // Per-step buffers (static-audit G10). Pre-allocated to MaxPop
        // so step() does not heap-resize them on the hot path. lambda
        // (current popsize) may be smaller than MaxPop on early
        // generations / pre-IPOP-doubling runs; the working size is
        // tracked by the per-step `lambda` argument and segments are
        // sized via .head(lambda) / .leftCols(lambda).
        static constexpr int MaxPop = 512;
        Eigen::Matrix<double, Eigen::Dynamic, 1, 0, MaxPop, 1> fitnesses_buf;
        Eigen::Matrix<double, Eigen::Dynamic, 1, 0, MaxPop, 1> unpenalized_buf;
        std::vector<int> idx_buf;
        Eigen::Matrix<double, N, Eigen::Dynamic, 0,
                      N == Eigen::Dynamic ? Eigen::Dynamic : N, MaxPop> deltas_buf;
        std::vector<double> gen_fitnesses_buf;
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
        // Captured for the IPOP restart re-anchor per Auger & Hansen 2005
        // §III: each restart sets s.mean back to the user's initial point.
        s.x0 = x0;

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

        // Pre-allocate per-step buffers (static-audit G10). Sized to
        // current lambda; step() uses .head(lambda) / .leftCols(lambda)
        // to track the working size, and re-resizes only on IPOP
        // doublings. The MaxPop=512 cap on the matrix template still
        // protects against pathological growth.
        s.fitnesses_buf.resize(s.params.lambda);
        s.unpenalized_buf.resize(s.params.lambda);
        s.idx_buf.resize(static_cast<std::size_t>(s.params.lambda));
        s.deltas_buf.resize(n, s.params.lambda);
        s.gen_fitnesses_buf.resize(static_cast<std::size_t>(s.params.lambda));

        s.C = Eigen::Matrix<double, N, N>::Identity(n, n);
        s.p_sigma = Eigen::Vector<double, N>::Zero(n);
        s.p_c = Eigen::Vector<double, N>::Zero(n);
        s.B = Eigen::Matrix<double, N, N>::Identity(n, n);
        s.D = Eigen::Vector<double, N>::Ones(n);
        s.eigen_solver.compute(s.C);
        s.covariance_dirty = false;

        // Hansen 2023 (arXiv:1604.00772) section B.3 item 8 (TolXUp) baseline.
        // Recorded at init so the divergence cap survives even if the D
        // initialization is changed in the future. With the canonical D = 1
        // initialization, initial_d_max == 1.0.
        s.initial_d_max = s.D.maxCoeff();
        s.axis_cycle_index = 0;

        // Hansen 2023 (arXiv:1604.00772) §B.2 (Strategy internal numerical
        // effort): the skip factor floor(1 / (10*n*(c_1 + c_mu))) degenerates
        // to 0 for low n with default weights (e.g. n=2, default lambda=6:
        // c_1 ~= 0.18, c_mu ~= 0.04, denom = 10*2*0.22 = 4.4, floor(1/4.4)
        // = 0), so the max(1, ...) guard pins K to 1 and a SelfAdjointEigen-
        // Solver<2> compute() runs every iter. K=1 is accepted at low n:
        // a 2x2 self-adjoint eigendecomposition is sub-microsecond on
        // commodity hardware, while the publish_bench wall_us per iter on
        // n=2 cells is ~25 us (5-seed snapshot at
        // 2026-04-30-libcmaes-comparison: ackley_2 1322 iters / 35.4 ms),
        // putting per-iter eigendecomp cost below 1% of total per-iter wall
        // and below the bench noise floor. libcmaes uses a counter-based
        // equivalent (`eigeneval - counteval > lambda / (10*(c_1 + c_mu))`,
        // src/cmastrategy.cc::optimize_), which gives a numerically-similar
        // skip frequency in this regime. K = max(1, floor(1 / (10 * n *
        // (c_1 + c_mu)))).
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
        //
        // Buffers live on s (state-owned, static-audit G10). On the
        // first IPOP doubling lambda may exceed the buffers' current
        // size; resize on demand (one-time alloc per lambda growth).
        if(static_cast<int>(s.fitnesses_buf.size()) < lambda)
        {
            s.fitnesses_buf.resize(lambda);
            s.unpenalized_buf.resize(lambda);
            s.idx_buf.resize(static_cast<std::size_t>(lambda));
            s.deltas_buf.resize(n, lambda);
            s.gen_fitnesses_buf.resize(static_cast<std::size_t>(lambda));
        }
        auto fitnesses = s.fitnesses_buf.head(lambda);
        auto unpenalized = s.unpenalized_buf.head(lambda);

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
        auto& idx = s.idx_buf;
        std::iota(idx.begin(), idx.begin() + lambda, 0);
        std::sort(idx.begin(), idx.begin() + lambda, [&](int a, int b) {
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

        // 10. Compute deltas for covariance update (state-owned buffer
        // per static-audit G10). Working width is `lambda`; deltas_buf
        // was sized to current lambda in init() / on IPOP growth above.
        auto deltas = s.deltas_buf.leftCols(lambda);
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

        // Hansen 2023 (arXiv:1604.00772) §B.3 item 7 (TolX): legacy
        // single-axis sigma collapse check. The §B.3 paper criterion checks
        // `sigma * sqrt(C(i,i))` in ALL coordinates AND `|sigma * p_c(i)|`
        // in ALL components against `1e-12 * initial_sigma`; that
        // strictly-stricter all-coord-AND-p_c form is implemented by the
        // §B.3 EXIT-only TolX block below (item 7) and is the canonical
        // disposition for the criterion. This site is RETAINED as a
        // user-override hook for legacy callers that explicitly set
        // `cmaes_options::sigma_collapse_threshold`. The
        // `value_or(value_or(default))` chain documents precedence:
        //   1. explicit `sigma_collapse_threshold`,
        //   2. else explicit `step_size_tolerance` (so a user setting the
        //      §B.3 TolX option transparently lifts the legacy single-axis
        //      check to the same threshold; the two checks then share a
        //      numerical baseline and the §B.3 per-coord TolX strictly
        //      subsumes this max-axis legacy check),
        //   3. else the §B.3 default 1e-12 * initial_sigma.
        // The legacy probe also remains an IPOP restart trigger (recycled
        // by the {Stagnation, sigma_collapse, ConditionCov} restart-set
        // logic below), while the §B.3 EXIT-only criteria always exit and
        // are never recycled.
        const double legacy_tol_factor =
            options.cmaes.sigma_collapse_threshold.value_or(
                options.cmaes.step_size_tolerance.value_or(1e-12));
        if(!policy_status.has_value()
           && s.sigma * s.D.maxCoeff() < legacy_tol_factor * s.initial_sigma)
            policy_status = solver_status::roundoff_limited;

        // Condition number explosion detection (diverged)
        double cond_limit = options.cmaes.condition_number_limit.value_or(1e14);
        double cond = s.D.maxCoeff() / std::max(s.D.minCoeff(), 1e-30);
        if(!policy_status.has_value() && cond * cond > cond_limit)
            policy_status = solver_status::diverged;

        // Snapshot the pre-EXIT-criteria status so the IPOP block (below)
        // can distinguish (sigma_collapse / cond_explosion -> recycle as
        // restart trigger) from (§B.3 EXIT criterion -> always exit, never
        // recycle). The new EXIT criteria below are gated on
        // `!policy_status.has_value()` so they cannot fire if either of
        // sigma_collapse / cond_explosion already set policy_status.
        const bool legacy_status_pre_exit = policy_status.has_value();

        // 16. Hansen 2023 (arXiv:1604.00772) section B.3 termination criteria.
        //
        // History tracking runs unconditionally so the §B.3 EXIT criteria
        // (TolFun, EqualFunValues) can read it regardless of restart_strategy.
        // The penalized fitness is the search's progress signal (the
        // unpenalized value is what we expose to callers as s.objective_value;
        // see static-audit G12). median_fitness_history is only consumed by
        // the IPOP-mode Stagnation detector but is cheap to maintain and
        // simplifies the gating below.
        s.best_fitness_history.push_back(fitnesses[best_offspring]);
        {
            // Compute median fitness of this generation. Reuses the
            // state-owned buffer (static-audit G10); nth_element mutates it
            // in place, but the values are read-only after the median
            // extraction so the next-generation overwrite is fine.
            auto& gen_fitnesses = s.gen_fitnesses_buf;
            for(int i = 0; i < lambda; ++i)
                gen_fitnesses[static_cast<std::size_t>(i)] = fitnesses[i];
            auto mid = gen_fitnesses.begin() + lambda / 2;
            std::nth_element(gen_fitnesses.begin(),
                             mid,
                             gen_fitnesses.begin() + lambda);
            s.median_fitness_history.push_back(*mid);
        }
        if(s.best_fitness_history.size() > 20000)
        {
            s.best_fitness_history.erase(s.best_fitness_history.begin());
            s.median_fitness_history.erase(s.median_fitness_history.begin());
        }

        // EXIT-only §B.3 criteria. These run regardless of restart_strategy
        // and return immediately on fire, mapping the §B.3 paper criterion
        // to the appropriate solver_status. The `policy_status` flow is
        // single-shot: once one of these fires we set it and return; the
        // IPOP-restart block below cannot subsequently restart on the same
        // step. The IPOP path stays gated on Stagnation (and the existing
        // sigma_collapse / condition-number probes set above) -- not on
        // these new exit criteria, per the EXIT-only contract.

        // Hansen 2023 §B.3 item 6 (TolFun): "stop if the range of the best
        // objective function values of the last 10 + ceil(30n/lambda)
        // generations and all function values of the recent generation is
        // below TolFun. Choosing TolFun depends on the problem, while
        // 10^-12 is a conservative first guess." User-tunable via
        // cmaes_options::objective_value_tolerance.
        // Status mapping: ftol_reached.
        if(!policy_status.has_value())
        {
            const double tol_fun = options.cmaes.objective_value_tolerance.value_or(1e-12);
            const auto tol_fun_window = static_cast<std::size_t>(
                10 + std::ceil(30.0 * static_cast<double>(n)
                               / static_cast<double>(s.params.lambda)));
            if(s.best_fitness_history.size() >= tol_fun_window)
            {
                auto win_begin = s.best_fitness_history.end()
                                 - static_cast<std::ptrdiff_t>(tol_fun_window);
                auto [hist_min, hist_max] =
                    std::minmax_element(win_begin, s.best_fitness_history.end());
                auto [gen_min, gen_max] =
                    std::minmax_element(fitnesses.data(), fitnesses.data() + lambda);
                const double full_min = std::min(*hist_min, *gen_min);
                const double full_max = std::max(*hist_max, *gen_max);
                if(full_max - full_min < tol_fun)
                    policy_status = solver_status::ftol_reached;
            }
        }

        // Hansen 2023 §B.3 item 4 (EqualFunValues): "stop if the range of
        // the best objective function values of the last 10 + ceil(30n/lambda)
        // generations is zero." The paper text is the literal "is zero" --
        // not a relaxed 1e-15 epsilon. The relaxed form widens the trigger
        // into TolHistFun-style behaviour, which is a different §B.3
        // criterion (TolFun, above). On a true plateau (offspring resampling
        // the same x bits, or a flat objective) the bit-equal range IS
        // achievable.
        // Status mapping: ftol_reached, matching libcmaes's
        // bench_libcmaes.cpp:122-138 EQUALFUNVALS -> ftol_reached convention.
        // EqualFunValues exits the solve even under restart_strategy::ipop --
        // it does NOT trigger a restart -- so the IPOP-restart trigger set
        // is {Stagnation, sigma_collapse, ConditionCov}, with EqualFunValues
        // and the other §B.3 EXIT criteria deciding "stop" not "restart".
        if(!policy_status.has_value())
        {
            const auto efv_window = static_cast<std::size_t>(
                10 + std::ceil(30.0 * static_cast<double>(n)
                               / static_cast<double>(s.params.lambda)));
            if(s.best_fitness_history.size() >= efv_window)
            {
                auto it = s.best_fitness_history.end()
                          - static_cast<std::ptrdiff_t>(efv_window);
                auto [mn, mx] = std::minmax_element(it, s.best_fitness_history.end());
                if(*mx == *mn)
                    policy_status = solver_status::ftol_reached;
            }
        }

        // Hansen 2023 §B.3 item 7 (TolX): "stop if the standard deviation
        // of the normal distribution is smaller than TolX in all coordinates
        // and sigma*p_c is smaller than TolX in all components. By default
        // we set TolX to 10^-12 times the initial sigma." User-tunable via
        // cmaes_options::step_size_tolerance (interpreted as a multiplicative
        // factor against initial_sigma, matching Hansen's "10^-12 times the
        // initial sigma" formulation).
        // Status mapping: roundoff_limited (numerical floor reached).
        if(!policy_status.has_value())
        {
            const double tol_x = options.cmaes.step_size_tolerance.value_or(1e-12)
                                 * s.initial_sigma;
            bool tol_x_all_coords = true;
            for(int i = 0; i < n; ++i)
            {
                const double std_i = s.sigma * std::sqrt(s.C(i, i));
                if(std_i >= tol_x)
                {
                    tol_x_all_coords = false;
                    break;
                }
            }
            bool tol_x_all_pc = true;
            if(tol_x_all_coords)
            {
                for(int i = 0; i < n; ++i)
                {
                    if(std::abs(s.sigma * s.p_c(i)) >= tol_x)
                    {
                        tol_x_all_pc = false;
                        break;
                    }
                }
            }
            if(tol_x_all_coords && tol_x_all_pc)
                policy_status = solver_status::roundoff_limited;
        }

        // Hansen 2023 §B.3 item 8 (TolXUp): "stop if sigma * max(diag(D))
        // increased by more than 10^4. This usually indicates a far too
        // small initial sigma, or divergent behavior."
        // Status mapping: diverged (matches the paper's "divergent
        // behavior" attribution).
        if(!policy_status.has_value())
        {
            constexpr double tol_x_up_factor = 1.0e4;
            const double current_max_d = s.D.maxCoeff();
            if(s.sigma * current_max_d
               > tol_x_up_factor * s.initial_sigma * s.initial_d_max)
                policy_status = solver_status::diverged;
        }

        // Hansen 2023 §B.3 item 1 (NoEffectAxis), footnote 31: "terminate
        // if m equals m + 0.1*sigma*d_ii*b_i, where i = (g mod n) + 1, and
        // d_ii^2 and b_i are respectively the i-th eigenvalue and
        // eigenvector of C." The (g mod n) cycle ensures every axis is
        // checked over n consecutive generations. nablapp implements the
        // cycle via state_type::axis_cycle_index advanced once per step()
        // call so the index does not depend on s.generation (which resets
        // on IPOP restart).
        // Status mapping: roundoff_limited (the mean is bit-stable along
        // that axis, which is a numerical-floor signal).
        if(!policy_status.has_value())
        {
            const int axis_i = s.axis_cycle_index;
            s.axis_cycle_index = (s.axis_cycle_index + 1) % n;
            const double d_ii = s.D(axis_i);
            const double scale = 0.1 * s.sigma * d_ii;
            bool no_effect_axis = true;
            for(int j = 0; j < n; ++j)
            {
                const double m_j = s.mean(j);
                const double m_j_perturbed = m_j + scale * s.B(j, axis_i);
                if(m_j != m_j_perturbed)
                {
                    no_effect_axis = false;
                    break;
                }
            }
            if(no_effect_axis)
                policy_status = solver_status::roundoff_limited;
        }

        // Hansen 2023 §B.3 item 2 (NoEffectCoord): "stop if adding 0.2-
        // standard deviations in any single coordinate does not change m
        // (i.e. m_i equals m_i + 0.2*sigma*c_{i,i} for any i)."
        // Note the §B.3 text uses c_{i,i} for the i-th covariance diagonal;
        // the paper's typesetting is ambiguous about whether that means the
        // variance entry C(i,i) or the standard-deviation entry sqrt(C(i,i)).
        // libcmaes uses sqrt(C(i,i)) (cmaes/src/cmastopcriteria.cc), and
        // this matches the §B.3 footnote phrasing "0.2-standard deviations";
        // we follow libcmaes here.
        // Status mapping: roundoff_limited.
        if(!policy_status.has_value())
        {
            for(int i = 0; i < n; ++i)
            {
                const double m_i = s.mean(i);
                const double m_i_perturbed = m_i + 0.2 * s.sigma * std::sqrt(s.C(i, i));
                if(m_i == m_i_perturbed)
                {
                    policy_status = solver_status::roundoff_limited;
                    break;
                }
            }
        }

        // 17. IPOP restart machinery. Trigger set is
        // {Stagnation, ConditionCov, sigma_collapse}. The §B.3 EXIT-only
        // criteria (TolFun, TolX, TolXUp, NoEffectAxis, NoEffectCoord,
        // EqualFunValues) above set policy_status and bypass restart --
        // they exit the solve regardless of restart_strategy. The legacy
        // sigma_collapse (`policy_status == roundoff_limited` from line
        // ~420) and ConditionCov (`policy_status == diverged` from line
        // ~426) probes set above are RECYCLED here in IPOP mode: instead
        // of exiting we clear them to fire a restart.
        // Reference: Hansen (2023) arXiv:1604.00772, Section B.3 item 5.
        // EXIT-criterion-fired test: a §B.3 EXIT criterion fired iff
        // policy_status is currently set AND the pre-EXIT snapshot was
        // empty. In that case the IPOP restart machinery must be skipped
        // entirely: the EXIT contract is "always exit, never recycle".
        const bool exit_criterion_fired =
            policy_status.has_value() && !legacy_status_pre_exit;

        if(options.restart == restart_strategy::ipop && !exit_criterion_fired)
        {
            bool stagnated = false;

            // Recycle the legacy sigma_collapse / cond_explosion probes
            // (§B.3 ConditionCov-aligned) as IPOP triggers. Gated by
            // legacy_status_pre_exit so that a §B.3 EXIT criterion
            // setting roundoff_limited / diverged is NOT mistaken for
            // a legacy probe signal -- the EXIT contract is preserved
            // by the outer `!exit_criterion_fired` guard, and this
            // inner check then narrows to just the legacy probes.
            if(legacy_status_pre_exit
               && (policy_status == solver_status::roundoff_limited
                   || policy_status == solver_status::diverged))
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
            if(s.best_fitness_history.size() >= window)
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

                // Hansen 2023 (arXiv:1604.00772) section B.3 paragraph
                // "Stagnation": the minimum stagnation history window is
                // 120 + ceil(30 * n / lambda) and depends on the CURRENT
                // lambda. libcmaes recomputes this implicitly per-iter via
                // _max_hist (cmasolutions.cc:111). nablapp computes it once
                // in init(); on an IPOP restart lambda has just doubled, so
                // the floor must be recomputed for the new lambda. Without
                // this recompute the dynamic ceil(0.2 * generation) cap
                // regrows from zero each restart against the obsolete
                // init-time floor and stagnation refires too early, driving
                // the lambda-doubling staircase reported in the
                // 2026-04-30 libcmaes head-to-head.
                s.stagnation_window_min = std::uint32_t{120}
                    + static_cast<std::uint32_t>(
                        std::ceil(30.0 * static_cast<double>(n)
                                  / static_cast<double>(s.params.lambda)));

                // Auger & Hansen 2005, "A Restart CMA Evolution Strategy
                // with Increasing Population Size", CEC 2005 §III: each
                // IPOP restart re-anchors the distribution mean to the
                // user-provided initial point x0. Without this re-anchor
                // the lambda-doubling staircase only widens the sampling
                // distribution around the prior run's terminal mean and
                // re-explores the SAME basin. libcmaes-aligned per
                // ipopcmastrategy.cc reset_search_state ->
                // CMASolutions(Parameters&) at cmasolutions.cc:49
                // (_xmean = p._x0min for the fixed-x0 case).
                s.mean = s.x0;

                // Hansen 2023 (arXiv:1604.00772) §B.2 (Strategy internal
                // numerical effort): K = max(1, floor(1 / (10 * n *
                // (c_1 + c_mu)))). Both c_1 and c_mu were just refreshed
                // by detail::compute_constants(n, new_lambda) above, so
                // the eigendecomposition skip period must be recomputed
                // against the new params. Honors any user override on
                // options.eigendecomposition_skip_generations exactly
                // like init() does.
                if(options.eigendecomposition_skip_generations.has_value())
                {
                    s.decomposition_skip_k =
                        options.eigendecomposition_skip_generations.value();
                }
                else
                {
                    const double skip_denom =
                        10.0 * static_cast<double>(n)
                        * (s.params.c_1 + s.params.c_mu);
                    s.decomposition_skip_k = std::max(
                        std::uint32_t{1},
                        static_cast<std::uint32_t>(
                            std::floor(1.0 / skip_denom)));
                }

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
        // Refresh strategy parameters for the current options.lambda value
        // so any caller that updates options.lambda and then calls reset()
        // gets a refreshed adaptation parameter set (mu, mueff, c_sigma,
        // d_sigma, c_c, c_1, c_mu, weights, chi_n). Mirrors the in-policy
        // IPOP branch's recompute (in step()) and the libcmaes contract
        // (cmaparameters.cc::initialize_parameters re-invoked from
        // ipopcmastrategy.cc on each lambda bump).
        //
        // References:
        //   Auger & Hansen (2005), CEC 2005 §III (IPOP-CMA-ES).
        //   libcmaes ipopcmastrategy.cc::reset_search_state.
        s.params = detail::compute_constants(n,
            static_cast<int>(options.lambda.value_or(0)));
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
