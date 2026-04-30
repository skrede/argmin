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
#include "nablapp/detail/xoshiro256.h"
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

        Eigen::Matrix<double, Eigen::Dynamic, 1, 0, MaxLambda, 1> fitnesses;
        Eigen::Matrix<double, Eigen::Dynamic, 1, 0, MaxLambda, 1> violations;

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

        return s;
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
