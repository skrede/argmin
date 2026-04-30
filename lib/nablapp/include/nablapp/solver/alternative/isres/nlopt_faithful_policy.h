#ifndef HPP_GUARD_NABLAPP_SOLVER_ALTERNATIVE_ISRES_NLOPT_FAITHFUL_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_ALTERNATIVE_ISRES_NLOPT_FAITHFUL_POLICY_H

// ISRES (Improved Stochastic Ranking Evolution Strategy) variant:
// NLopt 2.10.0 isres.c-faithful reproduction.
//
// Reproduces the NLopt isres.c operator semantics line-for-line:
// physical-slot top-mu x0 snapshot at start-of-generation, irank[k]
// survivor pairing, DE-style differential variation with
// gamma * (x0[0] - x0[k+1]) (NOT pull-to-best), per-mutation sigma
// upper clamp (ub-lb)/sqrt(n), log-normal sigma update with
// alpha-smoothing sigma_out = sigma_parent + alpha * (sigma_new -
// sigma_parent), resample-on-bound with bounded retry budget,
// L2-squared violation aggregation. The [mu, lambda) cyclic fill
// present in the pre-rewrite implementation is removed because the
// irank[k] indexing makes those slots unread.
//
// Engineering defaults gamma=0.85 and alpha=0.2 are sourced from
// NLopt isres.c lines 67-68; the original Runarsson-Yao 2005 paper
// does not pin numeric values for either constant. The defaults are
// kept here for cross-implementation parity with NLopt and pymoo.
//
// References:
//   Runarsson, T. P., and Yao, X. (2005), "Search Biases in
//     Constrained Evolutionary Optimization," IEEE Trans. Systems,
//     Man, and Cybernetics, Part C: Applications and Reviews,
//     35(2):233-243.
//   Kochenderfer, M. J., and Wheeler, T. A., "Algorithms for
//     Optimization", 2e, MIT Press 2019, §8.6 (Evolution Strategies).
//   NLopt 2.10.0 isres.c -- Steven G. Johnson 2009, line cites:
//     67  (ALPHA = 0.2)
//     68  (GAMMA = 0.85)
//     151,163  (L2-squared violation aggregation)
//     233-242  (per-mutation sigma upper clamp placement)
//     243-244  ((ub-lb)/sqrt(n) clamp value)
//     251-252  (alpha-smoothing applied per mutation)
//     254-260  (DE-style differential variation operator body)
//     270-271  (bottom [mu, lambda) standard mutation loop).
//   Hansen, N. (2023), "The CMA Evolution Strategy: A Tutorial,"
//     arXiv:1604.00772, §B.5 (TolX termination convention; design
//     rationale for sigma-collapse predicate, not a literal port).

#include "nablapp/detail/isres_operators.h"
#include "nablapp/detail/stochastic_ranking.h"
#include "nablapp/detail/tuple_contains.h"
#include "nablapp/detail/xoshiro256.h"
#include "nablapp/result/step_result.h"
#include "nablapp/result/status.h"
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

namespace nablapp::alternative::isres
{

template <int N = dynamic_dimension>
struct nlopt_faithful_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = nlopt_faithful_policy<M>;

    // Runtime selectors (D-10, D-14). Both ship as data-only fields in
    // this plan; the conditional logic that interprets them is plumbed
    // by Plan 05.
    enum class sigma_clamp_placement_type { per_mutation, end_of_step };
    enum class sigma_collapse_form_type { bound_relative_configurable, xtol_coupled };

    struct options_type
    {
        // Population sizing (inherited from production isres_policy).
        std::optional<std::uint32_t> population_size{};      // default lambda = 20*(n+1)
        std::optional<double> parent_fraction{};             // default mu = ceil(lambda/7)
        std::optional<double> ranking_probability{};         // default pf = 0.45

        // RNG seed.
        std::optional<std::uint64_t> seed{};

        // Stall + feasibility (inherited from production isres_policy).
        std::uint16_t stall_window{200};
        double feasibility_gate{1e-4};                       // L2-squared in this variant

        // D-07 + D-11 + D-12: NLopt-faithful operator constants.
        // gamma = 0.85, alpha = 0.2; engineering defaults from NLopt
        // isres.c:67-68 (NOT pinned numerically by Runarsson-Yao 2005).
        // gamma is the differential-variation strength
        // (Runarsson-Yao 2005 §III); name preserves the paper's symbol
        // per project citation policy.
        std::optional<double> differential_variation_gamma{};  // default 0.85
        std::optional<double> sigma_smoothing_weight{};        // default 0.2

        // D-14: sigma-collapse predicate threshold (bound-relative form).
        // 1e-9 default per RESEARCH Q3 (1000x looser than CMA-ES's
        // 1e-12 because per-individual ISRES sigma is noisier than
        // CMA-ES covariance scaling).
        std::optional<double> sigma_collapse_ratio{};          // default 1e-9

        // D-18 + RESEARCH Q5: bounded retry on resample-on-bound to
        // avoid infinite loop on degenerate-geometry problems
        // (NLopt isres.c:245-248 unbounded loop). On budget exhaustion,
        // fall back to std::clamp clip semantics.
        std::optional<std::uint16_t> bound_resample_budget{};  // default 100

        // D-10 axis: clamp placement runtime selector.
        sigma_clamp_placement_type sigma_clamp_placement{
            sigma_clamp_placement_type::per_mutation};

        // D-14 axis: sigma-collapse form runtime selector. Plumbed in
        // step() during Plan 05; value here is data-only at this point
        // in the phase.
        sigma_collapse_form_type sigma_collapse_form{
            sigma_collapse_form_type::bound_relative_configurable};
    };

    template <typename P = void>
    struct state_type
    {
        static constexpr int M = [] {
            if constexpr(has_constraint_count<P>) return constraint_count_v<P>;
            else return dynamic_dimension;
        }();

        // RESEARCH Q7 sizing rationale:
        //   MaxN  = 64    -- 2x headroom over current Phase 26 ISRES
        //                    test-set max (n=20).
        //   MaxMu = 256   -- ceil((20*(MaxN+1))/7) rounded to power of 2.
        //   MaxLambda = 7 * MaxMu -- 1792, used for full-population
        //                    buffers (population, sigmas, offspring).
        static constexpr int MaxN = 64;
        static constexpr int MaxMu = 256;
        static constexpr int MaxLambda = 1792;

        const P* problem{nullptr};
        Eigen::Vector<double, N> x;
        double objective_value{};

        // Bounded-storage population matrices (RESEARCH Q7;
        // feedback_no_dynamic_eigen). Heap is used only when N is
        // dynamic_dimension; with a concrete N these are stack-resident
        // up to their compile-time max bounds.
        Eigen::Matrix<double,
                      N == Eigen::Dynamic ? Eigen::Dynamic : N,
                      Eigen::Dynamic, 0,
                      N == Eigen::Dynamic ? Eigen::Dynamic : N,
                      MaxLambda> population;

        Eigen::Matrix<double,
                      N == Eigen::Dynamic ? Eigen::Dynamic : N,
                      Eigen::Dynamic, 0,
                      N == Eigen::Dynamic ? Eigen::Dynamic : N,
                      MaxLambda> sigmas;

        // fitnesses / violations stay as Eigen::VectorXd (the
        // canonical Eigen::Vector<double, Dynamic> typedef) so
        // detail::stochastic_rank's Eigen::Vector<Scalar, Lambda> ref
        // signature deduces cleanly. The dominant cost in this variant
        // is the n x lambda matrices, which DO use bounded storage.
        Eigen::VectorXd fitnesses;
        Eigen::VectorXd violations;

        Eigen::Vector<double, N> lower;
        Eigen::Vector<double, N> upper;

        // RNG: xoshiro256+ shared with cmaes_policy + production
        // isres_policy (commit 66c0fc0). 32 B vs std::mt19937's 2.5 KB.
        std::optional<detail::xoshiro256> rng;
        std::uint32_t generation{0};
        double best_ever_value{std::numeric_limits<double>::infinity()};
        double best_ever_violation{std::numeric_limits<double>::infinity()};

        // Constraint evaluation buffers. n_eq / n_ineq are problem
        // properties only known at init() time, so these stay dynamic.
        Eigen::Matrix<double, Eigen::Dynamic, 1, 0,
                      Eigen::Dynamic, 1> c_eq;
        Eigen::Matrix<double, Eigen::Dynamic, 1, 0,
                      Eigen::Dynamic, 1> c_ineq;
        int n_eq{0};
        int n_ineq{0};

        int lambda{0};
        int mu{0};
        double pf{0.45};
        // alpha here is the sigma-smoothing weight (D-11/D-12),
        // re-mapped from its prior pull-to-best meaning.
        double alpha{0.2};
        detail::es_learning_rates rates{};

        // D-08: x0 snapshot of the top-mu physical slots, captured at
        // start-of-step before mutation. Read by the differential
        // variation operator in Plan 03 Task 2.
        Eigen::Matrix<double,
                      N == Eigen::Dynamic ? Eigen::Dynamic : N,
                      Eigen::Dynamic, 0,
                      N == Eigen::Dynamic ? Eigen::Dynamic : N,
                      MaxMu> x0_snapshot_buf;

        // Per-step buffers (hoisted per RESEARCH Q7;
        // feedback_no_dynamic_eigen).
        Eigen::Matrix<double,
                      N == Eigen::Dynamic ? Eigen::Dynamic : N,
                      Eigen::Dynamic, 0,
                      N == Eigen::Dynamic ? Eigen::Dynamic : N,
                      MaxLambda> offspring_buf;
        Eigen::Matrix<double,
                      N == Eigen::Dynamic ? Eigen::Dynamic : N,
                      Eigen::Dynamic, 0,
                      N == Eigen::Dynamic ? Eigen::Dynamic : N,
                      MaxLambda> new_sigmas_buf;
        Eigen::Matrix<double, Eigen::Dynamic, 1, 0, MaxLambda, 1>
            offspring_fitnesses_buf;
        Eigen::Matrix<double, Eigen::Dynamic, 1, 0, MaxLambda, 1>
            offspring_violations_buf;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0,
                      Eigen::Dynamic, MaxLambda> all_constraints_buf;
        std::vector<std::uint32_t> indices_buf;

        // User-set step_tolerance threshold captured at init() time when
        // the convergence policy carries a step_tolerance_criterion. step()
        // cannot thread the Convergence template through state_type<P>, so
        // the threshold value is captured here. Stays std::nullopt for
        // convergence policies that lack step_tolerance_criterion (e.g.
        // slsqp_compatible_convergence carries step_tolerance_rel_criterion
        // only); xtol_coupled sigma-collapse falls back to bound-relative
        // semantics in that case.
        std::optional<double> convergence_xtol_threshold;
    };

    options_type options{};

    template <typename Problem, typename Convergence>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<double, N>& x0,
                             const solver_options<Convergence>& opts,
                             const options_type& policy_opts)
    {
        options = policy_opts;
        return init(problem, x0, opts);
    }

    template <typename Problem, typename Convergence = default_convergence>
        requires objective<Problem>
              && constrained_values<Problem>
              && bound_constrained<Problem>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<double, N>& x0,
                             const solver_options<Convergence>& opts)
    {
        const int n = problem.dimension();
        state_type<Problem> s;
        s.problem = &problem;

        // Bounds.
        s.lower = problem.lower_bounds();
        s.upper = problem.upper_bounds();

        // Constraint counts.
        s.n_eq = problem.num_equality();
        s.n_ineq = problem.num_inequality();
        const int n_c = s.n_eq + s.n_ineq;

        // Population parameters.
        s.lambda = static_cast<int>(
            options.population_size.value_or(
                static_cast<std::uint32_t>(20 * (n + 1))));
        const double frac = options.parent_fraction.value_or(1.0 / 7.0);
        s.mu = std::max(1, static_cast<int>(s.lambda * frac));
        s.pf = options.ranking_probability.value_or(0.45);
        // D-12 remap: the alpha state field now carries the
        // sigma-smoothing weight, NOT pull-to-best (which this variant
        // does not implement). Default 0.2 from NLopt isres.c:67.
        s.alpha = options.sigma_smoothing_weight.value_or(0.2);
        s.rates = detail::compute_es_rates(n);

        // Seed RNG (xoshiro256+ per static-audit I8).
        const std::uint64_t seed = options.seed.value_or(
            static_cast<std::uint64_t>(std::random_device{}()));
        s.rng.emplace(seed);

        // Pre-allocate per-step buffers. The bounded-storage Eigen
        // matrices keep their max-storage allocation; resize() only
        // adjusts the active dim metadata.
        s.population.resize(n, s.lambda);
        s.sigmas.resize(n, s.lambda);
        s.fitnesses.resize(s.lambda);
        s.violations.resize(s.lambda);
        s.offspring_buf.resize(n, s.lambda);
        s.new_sigmas_buf.resize(n, s.lambda);
        s.offspring_fitnesses_buf.resize(s.lambda);
        s.offspring_violations_buf.resize(s.lambda);
        s.all_constraints_buf.resize(std::max(n_c, 0), s.lambda);
        s.indices_buf.resize(static_cast<std::size_t>(s.lambda));

        // D-08: x0 snapshot buffer (n x mu). Required by step() Task 2;
        // sizing here so step() does not index a zero-sized buffer.
        s.x0_snapshot_buf.resize(n, s.mu);

        // Initialize population uniformly in bounds. The free-function
        // returns a heap-backed Eigen::Matrix<double, N, Dynamic>; we
        // copy into the bounded-storage member matrix. The block copy
        // is a fixed n x lambda traversal -- no per-step concern.
        const auto initial_population =
            detail::initialize_population<N>(
                n, s.lambda, s.lower, s.upper, *s.rng);
        s.population.leftCols(s.lambda) = initial_population;

        const auto initial_sigmas =
            detail::initialize_sigmas<N>(n, s.lambda, s.lower, s.upper);
        s.sigmas.leftCols(s.lambda) = initial_sigmas;

        // Evaluate all individuals into the persistent buffers.
        for(int j = 0; j < s.lambda; ++j)
        {
            const Eigen::Vector<double, N> xi = s.population.col(j);
            s.fitnesses[j] = s.problem->value(xi);

            if(n_c > 0)
            {
                Eigen::VectorXd c(n_c);
                s.problem->constraints(xi, c);
                s.all_constraints_buf.col(j) = c;
            }
        }

        // D-19: L2-squared violation aggregation (NLopt isres.c:151,163).
        // Caller convention: c_ineq[j] >= 0 feasible, equality term is
        // c_eq[j]^2.
        if(n_c > 0)
        {
            for(int j = 0; j < s.lambda; ++j)
            {
                double v = 0.0;
                for(int i = 0; i < s.n_eq; ++i)
                {
                    const double ce = s.all_constraints_buf(i, j);
                    v += ce * ce;
                }
                for(int i = 0; i < s.n_ineq; ++i)
                {
                    const double slack =
                        std::max(0.0, -s.all_constraints_buf(s.n_eq + i, j));
                    v += slack * slack;
                }
                s.violations[j] = v;
            }
        }
        else
        {
            for(int j = 0; j < s.lambda; ++j)
                s.violations[j] = 0.0;
        }

        // Pick best init individual (least violation first, then best
        // objective). Mirrors production isres_policy::init logic.
        int best_idx = 0;
        for(int j = 1; j < s.lambda; ++j)
        {
            const bool feasible_j = s.violations[j] <= 0.0;
            const bool feasible_best = s.violations[best_idx] <= 0.0;

            if(feasible_j && !feasible_best)
                best_idx = j;
            else if(feasible_j == feasible_best)
            {
                if(feasible_j && s.fitnesses[j] < s.fitnesses[best_idx])
                    best_idx = j;
                else if(!feasible_j
                        && s.violations[j] < s.violations[best_idx])
                    best_idx = j;
            }
        }

        s.x = s.population.col(best_idx);
        s.objective_value = s.fitnesses[best_idx];
        s.best_ever_value = s.objective_value;
        s.best_ever_violation = s.violations[best_idx];

        // Cache constraint values at best init point.
        s.c_eq.resize(s.n_eq);
        s.c_ineq.resize(s.n_ineq);
        if(n_c > 0)
        {
            Eigen::VectorXd c(n_c);
            s.problem->constraints(s.x, c);
            s.c_eq = c.head(s.n_eq);
            s.c_ineq = c.tail(s.n_ineq);
        }

        // Capture the user-set step_tolerance_criterion threshold (if the
        // convergence policy carries one) so step()'s xtol_coupled
        // sigma-collapse predicate can read it without threading the
        // Convergence type through state_type. The tuple_contains gate
        // keeps init() compilable against convergence policies that lack
        // step_tolerance_criterion (e.g. slsqp_compatible_convergence,
        // which carries step_tolerance_rel_criterion only); the field
        // stays std::nullopt and step() silently falls back to the
        // bound-relative form.
        if constexpr(detail::tuple_contains_v<
                         step_tolerance_criterion,
                         decltype(opts.convergence.criteria)>)
        {
            const auto& crit = std::get<step_tolerance_criterion>(
                opts.convergence.criteria);
            s.convergence_xtol_threshold = crit.threshold;
        }

        return s;
    }

    // step() : one ISRES generation, NLopt isres.c-faithful body.
    //
    // Order of operations (matches NLopt isres.c lines 233-280):
    //   1. Stochastic ranking populates s.indices_buf so that
    //      s.indices_buf[0] is the best-ranked individual; ties broken
    //      probabilistically per Runarsson-Yao 2005 eq. 2.
    //   2. Snapshot the top-mu PHYSICAL slots of the population into
    //      s.x0_snapshot_buf (D-08; NLopt isres.c:253). The snapshot
    //      is NOT rank-permuted; it captures slots [0, mu) as-is, so
    //      that the differential-variation difference at line 260 of
    //      NLopt isres.c reads physical-slot[0] - physical-slot[k+1].
    //      This is the operator semantics the Plan 01 free function
    //      detail::differential_variation() encodes; the body inlines
    //      the per-component form to interleave the boundary-fallback
    //      check that the per-column free function cannot express.
    //   3. Top-mu DE-style differential variation. The k+1 == mu
    //      boundary case and OOB components fall back to the standard
    //      log-normal mutation + bounded resample.
    //   4. Bottom [mu, lambda) standard mutation only -- no DE here.
    //      The dead [mu, lambda) cyclic fill present in the production
    //      isres_policy.h:316-323 is removed (D-17): the irank[k]
    //      pairing makes those slots overwritten by mutation, so the
    //      copy was unread.
    //   5. Evaluate offspring and aggregate violations as
    //         v_k = sum_i c_eq[i]^2 + sum_j max(0, -c_ineq[j])^2
    //      (D-19; NLopt isres.c:151,163). Differs from production
    //      isres_policy which uses L1 via detail::compute_violations.
    //   6. Best-ever bookkeeping mirrors the post-66c0fc0 production
    //      pattern (rolling best_ever_violation; tightened
    //      Pareto-improvement predicate).
    //   7. Return step_result with policy_status defaulted to
    //      std::nullopt; Plan 05 plumbs the sigma-collapse + feasibility
    //      predicate-and-emission into this slot.
    //
    // Reference: NLopt 2.10.0 isres.c lines 233-280;
    //            Runarsson & Yao (2005), IEEE Trans. SMC-C
    //            35(2):233-243; K&W 2e §8.6.
    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        const int n = static_cast<int>(s.x.size());
        const int mu = s.mu;
        const int lambda = s.lambda;
        const int n_c = s.n_eq + s.n_ineq;
        auto& rng = *s.rng;
        std::normal_distribution<double> normal(0.0, 1.0);

        // Capture pre-step best for objective_change / improved
        // bookkeeping. Plan 05 references this name when plumbing the
        // status emission, so it must exist verbatim (W7).
        const double prev_best_ever_value = s.best_ever_value;

        const double gamma =
            options.differential_variation_gamma.value_or(0.85);
        const double alpha =
            options.sigma_smoothing_weight.value_or(0.2);
        const std::uint16_t resample_budget =
            options.bound_resample_budget.value_or(
                static_cast<std::uint16_t>(100));

        // (1) Stochastic ranking. The buffer must start as the identity
        // permutation [0, lambda); stochastic_rank sorts in place.
        for(int j = 0; j < lambda; ++j)
            s.indices_buf[static_cast<std::size_t>(j)]
                = static_cast<std::uint32_t>(j);

        detail::stochastic_rank(s.indices_buf,
                                s.fitnesses,
                                s.violations,
                                s.pf,
                                rng);

        // (2) x0 snapshot of the top-mu PHYSICAL slots
        // (D-08; NLopt isres.c:253). NOT rank-permuted.
        for(int j = 0; j < mu; ++j)
            s.x0_snapshot_buf.col(j) = s.population.col(j);

        // (3) Top-mu DE-style differential variation (D-06; NLopt
        // isres.c:254-260). The per-component fallback check at the
        // boundary makes the body inline rather than calling the
        // column-level free function detail::differential_variation();
        // the operator is, however, the same: x[rk] += gamma *
        // (x0_snapshot[0] - x0_snapshot[k+1]).
        for(int k = 0; k < mu; ++k)
        {
            const std::uint32_t rk =
                s.indices_buf[static_cast<std::size_t>(k)];
            const double taup_rand = s.rates.tau_prime * normal(rng);

            for(int j = 0; j < n; ++j)
            {
                const double xi = s.population(j, rk);
                if(k + 1 < mu)
                    s.population(j, rk) += gamma
                        * (s.x0_snapshot_buf(j, 0)
                           - s.x0_snapshot_buf(j, k + 1));

                const bool need_fallback = (k + 1 == mu)
                    || s.population(j, rk) < s.lower(j)
                    || s.population(j, rk) > s.upper(j);
                if(need_fallback)
                {
                    // D-09: per-mutation sigma upper clamp
                    // sigma_max = (ub - lb) / sqrt(n).
                    const double sigmamax = (s.upper(j) - s.lower(j))
                        / std::sqrt(static_cast<double>(n));
                    const double sigi = s.sigmas(j, rk);

                    // D-10 per_mutation form: log-normal sigma update
                    // with the upper clamp folded inside the operator.
                    s.sigmas(j, rk) = detail::log_normal_mutate(
                        sigi, s.rates.tau, taup_rand, sigmamax, rng);

                    // D-18 + RESEARCH Q5: bounded resample on bound;
                    // budget=100 default. On exhaustion, std::clamp
                    // fallback (vs NLopt's unbounded do/while).
                    for(std::uint16_t try_k = 0;
                        try_k < resample_budget;
                        ++try_k)
                    {
                        s.population(j, rk) =
                            xi + s.sigmas(j, rk) * normal(rng);
                        if(s.population(j, rk) >= s.lower(j)
                           && s.population(j, rk) <= s.upper(j))
                            break;
                        if(try_k + 1 == resample_budget)
                            s.population(j, rk) = std::clamp(
                                s.population(j, rk),
                                s.lower(j), s.upper(j));
                    }

                    // D-11: alpha-smoothing on sigma.
                    //   sigma_out = sigma_parent
                    //             + alpha * (sigma_new - sigma_parent)
                    s.sigmas(j, rk) =
                        sigi + alpha * (s.sigmas(j, rk) - sigi);
                }
            }
        }

        // (4) Bottom [mu, lambda) standard mutation only
        // (NLopt isres.c:266-277). Parent for slot k is the rank-(k%mu)
        // individual (NLopt convention); no differential variation
        // because there is no second anchor.
        for(int k = mu; k < lambda; ++k)
        {
            const std::uint32_t rk =
                s.indices_buf[static_cast<std::size_t>(k)];
            const std::uint32_t ri =
                s.indices_buf[static_cast<std::size_t>(k % mu)];
            const double taup_rand = s.rates.tau_prime * normal(rng);

            for(int j = 0; j < n; ++j)
            {
                const double sigmamax = (s.upper(j) - s.lower(j))
                    / std::sqrt(static_cast<double>(n));
                const double sigi = s.sigmas(j, ri);

                s.sigmas(j, rk) = detail::log_normal_mutate(
                    sigi, s.rates.tau, taup_rand, sigmamax, rng);

                for(std::uint16_t try_k = 0;
                    try_k < resample_budget;
                    ++try_k)
                {
                    s.population(j, rk) = s.population(j, ri)
                        + s.sigmas(j, rk) * normal(rng);
                    if(s.population(j, rk) >= s.lower(j)
                       && s.population(j, rk) <= s.upper(j))
                        break;
                    if(try_k + 1 == resample_budget)
                        s.population(j, rk) = std::clamp(
                            s.population(j, rk),
                            s.lower(j), s.upper(j));
                }
                s.sigmas(j, rk) =
                    sigi + alpha * (s.sigmas(j, rk) - sigi);
            }
        }

        // (5) Evaluate offspring and aggregate violations
        // (D-19; NLopt isres.c:151,163). The mutation step writes
        // directly into s.population, so we evaluate s.population
        // (no offspring_buf needed in this variant -- the irank[k]
        // pairing makes mutation idempotent on the active slot).
        for(int k = 0; k < lambda; ++k)
        {
            const Eigen::Vector<double, N> xk = s.population.col(k);
            s.fitnesses[k] = s.problem->value(xk);

            if(n_c > 0)
            {
                Eigen::VectorXd c(n_c);
                s.problem->constraints(xk, c);
                s.all_constraints_buf.col(k) = c;
            }

            // Inline L2-squared aggregation. NLopt's c_ineq >= 0
            // feasible convention: ineq slack = max(0, -c_ineq[j]).
            double violation_sum = 0.0;
            for(int j_eq = 0; j_eq < s.n_eq; ++j_eq)
            {
                const double ce = s.all_constraints_buf(j_eq, k);
                violation_sum += ce * ce;
            }
            for(int j_in = 0; j_in < s.n_ineq; ++j_in)
            {
                const double slack = std::max(
                    0.0,
                    -s.all_constraints_buf(s.n_eq + j_in, k));
                violation_sum += slack * slack;
            }
            s.violations[k] = violation_sum;
        }

        // (6) Best-ever bookkeeping. Mirrors production isres_policy
        // post-66c0fc0: rolling best_ever_violation, Pareto-tightened
        // pre-feasible update predicate.
        const std::uint32_t best_ranked = s.indices_buf[0];
        const double best_ranked_f = s.fitnesses[best_ranked];
        const double best_ranked_v = s.violations[best_ranked];

        if(best_ranked_f < s.best_ever_value && best_ranked_v <= 0.0)
        {
            s.best_ever_value = best_ranked_f;
            s.best_ever_violation = 0.0;
            s.objective_value = best_ranked_f;
            s.x = s.population.col(best_ranked);

            if(n_c > 0)
            {
                Eigen::VectorXd c(n_c);
                s.problem->constraints(s.x, c);
                s.c_eq = c.head(s.n_eq);
                s.c_ineq = c.tail(s.n_ineq);
            }
        }
        else if(s.best_ever_value
                == std::numeric_limits<double>::infinity())
        {
            // Pre-feasible witness: only update on Pareto improvement
            // (lower violation AND not-worse fitness).
            if(best_ranked_v < s.best_ever_violation
               && best_ranked_f <= s.objective_value)
            {
                s.best_ever_violation = best_ranked_v;
                s.objective_value = best_ranked_f;
                s.x = s.population.col(best_ranked);

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

        // (7) Step report.
        //
        // Status emission per Runarsson-Yao 2005 Section V (termination):
        // emit solver_status::ftol_reached when the population's mean
        // step size has collapsed AND the rank-0 individual's violation
        // sits within the feasibility gate. Two sigma-collapse forms are
        // available via the runtime selector options.sigma_collapse_form:
        //
        //   - bound_relative_configurable: compares mean_sigma against
        //     `ratio * mean(ub - lb) / sqrt(n)` (default ratio 1e-9).
        //   - xtol_coupled: compares mean_sigma against the user-set
        //     step_tolerance threshold captured in
        //     s.convergence_xtol_threshold at init() time. If the
        //     convergence policy lacked step_tolerance_criterion or its
        //     threshold is std::nullopt, the predicate silently falls
        //     back to the bound-relative form using `ratio` (fail-safe).
        const double mean_sigma = s.sigmas.leftCols(mu).mean();
        const double grad_proxy =
            mean_sigma * std::sqrt(static_cast<double>(n));
        const bool improved = (s.best_ever_value < prev_best_ever_value);
        const double objective_change = improved
            ? (s.best_ever_value - prev_best_ever_value)
            : mean_sigma;

        const std::uint32_t rank0 = s.indices_buf[0];
        const double v_best = s.violations[rank0];
        const bool feasible = (v_best <= options.feasibility_gate);

        const double collapse_ratio =
            options.sigma_collapse_ratio.value_or(1e-9);

        bool collapsed;
        if(options.sigma_collapse_form
           == sigma_collapse_form_type::xtol_coupled)
        {
            const double threshold_value =
                s.convergence_xtol_threshold.value_or(
                    std::numeric_limits<double>::quiet_NaN());
            collapsed = detail::sigma_collapsed_xtol_coupled(
                s, threshold_value, collapse_ratio, n,
                s.lower, s.upper);
        }
        else
        {
            collapsed = detail::sigma_collapsed_bound_relative(
                s, collapse_ratio, n, s.lower, s.upper);
        }

        std::optional<solver_status> emitted_status;
        if(collapsed && feasible)
            emitted_status = solver_status::ftol_reached;

        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = grad_proxy,
            .step_size = mean_sigma,
            .objective_change = objective_change,
            .improved = improved,
            .x_norm = s.x.norm(),
            .policy_status = emitted_status,
        };
    }

    template <typename P>
    void reset(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        const int n = static_cast<int>(x0.size());
        const int n_c = s.n_eq + s.n_ineq;

        // Re-initialize population uniformly. Same bounded-storage
        // copy idiom as init().
        const auto initial_population =
            detail::initialize_population<N>(
                n, s.lambda, s.lower, s.upper, *s.rng);
        s.population.leftCols(s.lambda) = initial_population;

        const auto initial_sigmas =
            detail::initialize_sigmas<N>(n, s.lambda, s.lower, s.upper);
        s.sigmas.leftCols(s.lambda) = initial_sigmas;

        s.x = x0;
        s.objective_value = s.problem->value(x0);
        s.best_ever_value = s.objective_value;
        s.best_ever_violation = std::numeric_limits<double>::infinity();
        s.generation = 0;

        if(n_c > 0)
        {
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
