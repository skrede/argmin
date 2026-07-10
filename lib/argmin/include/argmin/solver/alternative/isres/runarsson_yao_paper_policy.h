#ifndef HPP_GUARD_ARGMIN_SOLVER_ALTERNATIVE_ISRES_RUNARSSON_YAO_PAPER_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_ALTERNATIVE_ISRES_RUNARSSON_YAO_PAPER_POLICY_H

// ISRES (Improved Stochastic Ranking Evolution Strategy) variant:
// Runarsson-Yao 2005 paper-faithful reproduction.
//
// Implements the operator forms specified in Runarsson & Yao (2005)
// Section III without committing to NLopt's engineering shortcuts. Two
// algorithmic legs diverge from `nlopt_faithful_policy`:
//
//   1. x0 snapshot is the rank-0 (best) individual + rank-permuted
//      offspring, captured at start-of-generation BEFORE mutation.
//      NLopt instead snapshots the physical-slot population (which
//      under NLopt's separate `irank[]` index is NOT rank-permuted).
//
//   2. Differential variation is BEST-anchored:
//          x_new = x_1 + paper_gamma * (x_i - x_1)
//      moving the best individual along an offspring difference.
//      NLopt instead uses the array-anchored form
//          x[rk] += gamma * (x_first[j] - x_(k+1)[j])
//      which moves rank-k along a difference between two arbitrary
//      array slots.
//
// All other operator legs match `nlopt_faithful_policy` because the
// paper text does not diverge from NLopt on those axes:
//   - sigma upper clamp (ub-lb)/sqrt(n) per dim, per-mutation;
//   - alpha-smoothing sigma_out = sigma_parent + alpha *
//     (sigma_new - sigma_parent);
//   - resample-on-bound with bounded retry budget;
//   - L2-squared violation aggregation;
//   - removal of the dead [mu, lambda) cyclic fill.
//
// Citation-gap on numeric defaults:
//
//   Paper specifies the operator FORMS but does not pin gamma or
//   alpha numerically; defaults match NLopt (gamma=0.85, alpha=0.2)
//   for cross-implementation comparability with NLopt and pymoo.
//   See Runarsson & Yao (2005) Section III; ISRES+ Liu et al. (2023)
//   Bioinformatics 39(7) btad403, Table 2 (which is the source for
//   the bibliographic claim that the original paper does not pin
//   either constant).
//
// References:
//   Runarsson, T. P., and Yao, X. (2005), "Search Biases in
//     Constrained Evolutionary Optimization," IEEE Trans. Systems,
//     Man, and Cybernetics, Part C: Applications and Reviews,
//     35(2):233-243.
//   Kochenderfer, M. J., and Wheeler, T. A., "Algorithms for
//     Optimization", 2e, MIT Press 2019, Section 8.6 (Evolution
//     Strategies).
//   Liu, Z., et al. (2023), "ISRES+: an improved evolutionary
//     strategy for function minimization," Bioinformatics 39(7):
//     btad403, Table 2 (numeric default attribution gap).

#include "argmin/detail/isres_operators.h"
#include "argmin/detail/stochastic_ranking.h"
#include "argmin/detail/tuple_contains.h"
#include "argmin/detail/xoshiro256.h"
#include "argmin/result/step_result.h"
#include "argmin/result/status.h"
#include "argmin/solver/options.h"
#include "argmin/types.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <cmath>
#include <random>
#include <vector>
#include <cstdint>
#include <optional>
#include <algorithm>

namespace argmin::alternative::isres
{

template <int N = dynamic_dimension>
struct runarsson_yao_paper_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = runarsson_yao_paper_policy<M>;

    enum class sigma_clamp_placement_type { per_mutation, end_of_step };
    enum class sigma_collapse_form_type { bound_relative_configurable, xtol_coupled };

    struct options_type
    {
        // Population sizing.
        std::optional<std::uint32_t> population_size{};
        std::optional<double> parent_fraction{};
        std::optional<double> ranking_probability{};

        // RNG seed.
        std::optional<std::uint64_t> seed{};

        // Stall + feasibility.
        std::uint16_t stall_window{200};
        double feasibility_gate{1e-4};

        // Numeric values are NOT paper-pinned (see file-header citation gap).
        // Defaults match NLopt for cross-implementation comparability.
        // Field names use the paper_ prefix to mark the citation gap.
        std::optional<double> paper_gamma{};   // default 0.85 (citation gap; not in paper text)
        std::optional<double> paper_alpha{};   // default 0.2  (citation gap; not in paper text)

        // Sigma-collapse predicate threshold.
        std::optional<double> sigma_collapse_ratio{};  // default 1e-9

        // Bounded resample budget — caps NLopt's unbounded retry loop.
        std::optional<std::uint16_t> bound_resample_budget{};  // default 100

        // Operator-axis runtime selectors.
        sigma_clamp_placement_type sigma_clamp_placement{
            sigma_clamp_placement_type::per_mutation};
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

        // Bounded-storage scaffolding shared with nlopt_faithful_policy
        // (cmaes_policy MaxPop=512 idiom). For fixed-N specializations the
        // row count is exact (N); for dynamic_dimension the row count is
        // heap-sized (Eigen::Dynamic). Column count is capped at MaxLambda
        // / MaxMu so the column axis does not heap-resize on the hot path.
        // fitnesses / violations remain ordinary VectorXd because
        // detail::stochastic_rank takes an Eigen::Vector<Scalar, N>
        // (no Max-cap) signature.
        static constexpr int MaxN = 64;
        static constexpr int MaxMu = 256;
        static constexpr int MaxLambda = 1792;
        static constexpr int RowsCap =
            (N == dynamic_dimension) ? Eigen::Dynamic : N;

        const P* problem{nullptr};
        Eigen::Vector<double, N> x;
        double objective_value{};

        Eigen::Matrix<double, N, Eigen::Dynamic, 0, RowsCap, MaxLambda> population;
        Eigen::Matrix<double, N, Eigen::Dynamic, 0, RowsCap, MaxLambda> sigmas;
        Eigen::VectorXd fitnesses;
        Eigen::VectorXd violations;

        Eigen::Vector<double, N> lower;
        Eigen::Vector<double, N> upper;

        // RNG matches cmaes_policy + production isres_policy (xoshiro256+).
        std::optional<detail::xoshiro256> rng;
        std::uint32_t generation{0};
        double best_ever_value{std::numeric_limits<double>::infinity()};
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

        // Per-step buffers hoisted to state to keep step() alloc-free.
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0,
                      Eigen::Dynamic, MaxLambda> all_constraints_buf;
        std::vector<std::uint32_t> indices_buf;

        // PAPER DELTA #1 storage: rank-permuted x0 snapshot at
        // start-of-generation. The mu rank-0..rank-(mu-1) survivors are
        // copied here BEFORE the differential-variation update; col(0)
        // is rank-0 (best) and is the BEST anchor for the DE update.
        Eigen::Matrix<double, N, Eigen::Dynamic, 0, RowsCap, MaxMu> x0_snapshot_buf;

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
        requires objective<Problem> && constrained_values<Problem> && bound_constrained<Problem>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<double, N>& /*x0*/,
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

        s.lambda = static_cast<int>(
            options.population_size.value_or(
                static_cast<std::uint32_t>(20 * (n + 1))));
        const double frac = options.parent_fraction.value_or(1.0 / 7.0);
        s.mu = std::max(1, static_cast<int>(s.lambda * frac));
        s.pf = options.ranking_probability.value_or(0.45);
        s.alpha = options.paper_alpha.value_or(0.2);
        s.rates = detail::compute_es_rates(n);

        const std::uint64_t seed = options.seed.value_or(
            static_cast<std::uint64_t>(std::random_device{}()));
        s.rng.emplace(seed);

        s.population.resize(n, s.lambda);
        s.sigmas.resize(n, s.lambda);
        s.fitnesses.resize(s.lambda);
        s.violations.resize(s.lambda);
        s.x0_snapshot_buf.resize(n, s.mu);
        s.all_constraints_buf.resize(n_c > 0 ? n_c : 0, s.lambda);
        s.indices_buf.resize(static_cast<std::size_t>(s.lambda));

        s.population = detail::initialize_population<N>(
            n, s.lambda, s.lower, s.upper, *s.rng);
        s.sigmas = detail::initialize_sigmas<N>(
            n, s.lambda, s.lower, s.upper);

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

        // L2-squared violation aggregation: paper and NLopt agree on this
        // axis (NLopt isres.c:151,163).
        if(n_c > 0)
        {
            for(int j = 0; j < s.lambda; ++j)
            {
                double violation_sum = 0.0;
                for(int i_eq = 0; i_eq < s.n_eq; ++i_eq)
                {
                    const double c_val = s.all_constraints_buf(i_eq, j);
                    violation_sum += c_val * c_val;
                }
                for(int i_in = 0; i_in < s.n_ineq; ++i_in)
                {
                    const double c_val = s.all_constraints_buf(s.n_eq + i_in, j);
                    const double v = std::max(0.0, -c_val);
                    violation_sum += v * v;
                }
                s.violations[j] = violation_sum;
            }
        }
        else
        {
            s.violations.setZero();
        }

        // Find best individual (least violation first, then best objective).
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
                else if(!feasible_j && s.violations[j] < s.violations[best_idx])
                    best_idx = j;
            }
        }

        s.x = s.population.col(best_idx);
        s.objective_value = s.fitnesses[best_idx];
        s.best_ever_value = s.objective_value;
        s.best_ever_violation = s.violations[best_idx];

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

    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        // Capture previous best-ever for objective_change / improved
        // bookkeeping; also referenced by the sigma-collapse status
        // emission below.
        const double prev_best_ever_value = s.best_ever_value;

        const int n = static_cast<int>(s.x.size());
        const int mu = s.mu;
        const int lambda = s.lambda;
        const int n_c = s.n_eq + s.n_ineq;
        auto& rng = *s.rng;
        std::normal_distribution<double> normal(0.0, 1.0);

        const double gamma = options.paper_gamma.value_or(0.85);
        const double alpha = options.paper_alpha.value_or(0.2);
        const std::uint16_t resample_budget =
            options.bound_resample_budget.value_or(100);

        // (1) Stochastic ranking populates s.indices_buf with the
        //     rank-permutation over [0, lambda).
        for(int j = 0; j < lambda; ++j)
            s.indices_buf[static_cast<std::size_t>(j)]
                = static_cast<std::uint32_t>(j);
        detail::stochastic_rank(s.indices_buf, s.fitnesses, s.violations, s.pf, rng);

        // (2) PAPER DELTA #1: x0 snapshot is the rank-permuted population
        //     (rank-0 = best, rank-1 = second-best, ...) at start-of-
        //     generation. NLopt instead snapshots physical slots [0, mu).
        for(int j = 0; j < mu; ++j)
            s.x0_snapshot_buf.col(j) =
                s.population.col(s.indices_buf[static_cast<std::size_t>(j)]);

        // (3) PAPER DELTA #2: BEST-anchored DE recombination.
        //         x_new = x_1 + paper_gamma * (x_i - x_1)
        //     where x_1 = rank-0 (best) survivor, x_i = rank-i (i-th best).
        //     Skip k=0 since that IS x_1 itself; k=0 is mutated via the
        //     standard log-normal ES path. OOB / last-survivor fallback
        //     uses standard mutation; treating "differential variation
        //     for survivors" as covering OOB recovery would conflate the
        //     two operator regimes.
        for(int k = 0; k < mu; ++k)
        {
            const std::uint32_t rk = s.indices_buf[static_cast<std::size_t>(k)];
            const double taup_rand = s.rates.tau_prime * normal(rng);

            for(int j = 0; j < n; ++j)
            {
                const double xi = s.population(j, rk);

                if(k > 0)
                {
                    s.population(j, rk) =
                        s.x0_snapshot_buf(j, 0)
                        + gamma * (s.x0_snapshot_buf(j, k)
                                   - s.x0_snapshot_buf(j, 0));
                }

                const bool need_fallback = (k == 0)
                    || s.population(j, rk) < s.lower(j)
                    || s.population(j, rk) > s.upper(j);
                if(need_fallback)
                {
                    // Inherited NLopt-form leg (paper agrees on these axes):
                    //   sigma upper clamp + log-normal mutate + bounded
                    //   resample + alpha-smoothing.
                    const double sigmamax = (s.upper(j) - s.lower(j))
                                              / std::sqrt(static_cast<double>(n));
                    const double sigi = s.sigmas(j, rk);

                    s.sigmas(j, rk) = detail::log_normal_mutate(
                        sigi, s.rates.tau, taup_rand, sigmamax, rng);

                    for(std::uint16_t try_k = 0; try_k < resample_budget; ++try_k)
                    {
                        s.population(j, rk) = xi
                            + s.sigmas(j, rk) * normal(rng);
                        if(s.population(j, rk) >= s.lower(j)
                           && s.population(j, rk) <= s.upper(j))
                            break;
                        if(try_k + 1 == resample_budget)
                            s.population(j, rk) = std::clamp(
                                s.population(j, rk), s.lower(j), s.upper(j));
                    }

                    s.sigmas(j, rk) = sigi + alpha * (s.sigmas(j, rk) - sigi);
                }
            }
        }

        // (4) Bottom [mu, lambda) standard mutation. The paper does NOT
        //     specify this leg; NLopt convention (parent = rank-(k % mu))
        //     is the dominant practice across reference implementations.
        for(int k = mu; k < lambda; ++k)
        {
            const std::uint32_t rk = s.indices_buf[static_cast<std::size_t>(k)];
            const std::uint32_t ri = s.indices_buf[static_cast<std::size_t>(k % mu)];
            const double taup_rand = s.rates.tau_prime * normal(rng);

            for(int j = 0; j < n; ++j)
            {
                const double sigmamax = (s.upper(j) - s.lower(j))
                                          / std::sqrt(static_cast<double>(n));
                const double sigi = s.sigmas(j, ri);

                s.sigmas(j, rk) = detail::log_normal_mutate(
                    sigi, s.rates.tau, taup_rand, sigmamax, rng);

                for(std::uint16_t try_k = 0; try_k < resample_budget; ++try_k)
                {
                    s.population(j, rk) = s.population(j, ri)
                                           + s.sigmas(j, rk) * normal(rng);
                    if(s.population(j, rk) >= s.lower(j)
                       && s.population(j, rk) <= s.upper(j))
                        break;
                    if(try_k + 1 == resample_budget)
                        s.population(j, rk) = std::clamp(
                            s.population(j, rk), s.lower(j), s.upper(j));
                }
                s.sigmas(j, rk) = sigi + alpha * (s.sigmas(j, rk) - sigi);
            }
        }

        // (5) Evaluation + L2-squared violation aggregation. Paper and
        //     NLopt agree on this axis (NLopt isres.c:151,163).
        for(int k = 0; k < lambda; ++k)
        {
            Eigen::Vector<double, N> xk = s.population.col(k);
            s.fitnesses[k] = s.problem->value(xk);

            if(n_c > 0)
            {
                Eigen::VectorXd c(n_c);
                s.problem->constraints(xk, c);
                s.all_constraints_buf.col(k) = c;
            }
        }

        if(n_c > 0)
        {
            for(int k = 0; k < lambda; ++k)
            {
                double violation_sum = 0.0;
                for(int i_eq = 0; i_eq < s.n_eq; ++i_eq)
                {
                    const double c_val = s.all_constraints_buf(i_eq, k);
                    violation_sum += c_val * c_val;
                }
                for(int i_in = 0; i_in < s.n_ineq; ++i_in)
                {
                    const double c_val = s.all_constraints_buf(s.n_eq + i_in, k);
                    const double v = std::max(0.0, -c_val);
                    violation_sum += v * v;
                }
                s.violations[k] = violation_sum;
            }
        }
        else
        {
            s.violations.setZero();
        }

        // (6) Re-rank the post-mutation population so the rank-0 column
        //     reflects the new generation's best.
        for(int j = 0; j < lambda; ++j)
            s.indices_buf[static_cast<std::size_t>(j)]
                = static_cast<std::uint32_t>(j);
        detail::stochastic_rank(s.indices_buf, s.fitnesses, s.violations, s.pf, rng);

        // (7) Best-update + best_ever bookkeeping: feasible improvement
        //     first, then a Pareto-tightened pre-feasible least-infeasibility
        //     witness (rolling best_ever_violation).
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
        else if(s.best_ever_value == std::numeric_limits<double>::infinity())
        {
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

        // (8) Return step_result with sigma-collapse + feasibility status
        //     emission per Runarsson-Yao 2005 Section V (termination).
        //     Emit solver_status::ftol_reached when the population's mean
        //     step size has collapsed AND the rank-0 individual's violation
        //     sits within the feasibility gate. Two sigma-collapse forms
        //     are available via the runtime selector
        //     options.sigma_collapse_form:
        //
        //       - bound_relative_configurable: compares mean_sigma against
        //         `ratio * mean(ub - lb) / sqrt(n)` (default ratio 1e-9).
        //       - xtol_coupled: compares mean_sigma against the user-set
        //         step_tolerance threshold captured in
        //         s.convergence_xtol_threshold at init() time. If the
        //         convergence policy lacked step_tolerance_criterion or
        //         its threshold is std::nullopt, the predicate silently
        //         falls back to the bound-relative form using `ratio`
        //         (fail-safe).
        const double mean_sigma = s.sigmas.leftCols(mu).mean();

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
            .gradient_norm = std::numeric_limits<double>::infinity(),
            .step_size = mean_sigma,
            .objective_change = std::abs(s.best_ever_value - prev_best_ever_value),
            .improved = (s.best_ever_value < prev_best_ever_value),
            .x_norm = s.x.norm(),
            .policy_status = emitted_status,
        };
    }

    template <typename P>
    void reset(state_type<P>& s, Eigen::Ref<const Eigen::Vector<double, N>> x0)
    {
        const int n = static_cast<int>(x0.size());

        s.population = detail::initialize_population<N>(
            n, s.lambda, s.lower, s.upper, *s.rng);
        s.sigmas = detail::initialize_sigmas<N>(
            n, s.lambda, s.lower, s.upper);

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
