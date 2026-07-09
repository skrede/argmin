// dlib comparison benchmarks for argmin benchmark suite.
//
// Each dlib solver is benchmarked on applicable problems using dlib's native
// API (per D-01: no common adapter interface). Results are collected as
// benchmark_result structs with library = "dlib".
//
// Per Research pitfall 3: dlib BOBYQA and L-BFGS have different tuning than
// NLopt equivalents.
//
// Per Research open question 2: dlib does NOT expose f-eval counts, so we
// wrap objectives with a counting functor.
//
// Solver mapping (per D-04):
//   Bound-constrained:       find_min_box_constrained (LBFGS) -> "dlib_lbfgs_box"
//   Bound-constrained (DFO): find_min_bobyqa          -> "dlib_bobyqa"
//   Global:                  find_min_global            -> "dlib_global"
//
// dlib has no general constraint support (no equality/inequality), so
// only bound-constrained and global problems are benchmarked.
//
// NOTE: This file is only compiled when dlib is found via find_package
// (controlled by ARGMIN_HAS_DLIB compile definition from CMake).

#include "bench_dlib.h"
#include "counting_problem.h"
#include "problem_registry.h"

#include "argmin/formulation/concepts.h"

#include <dlib/global_optimization.h>
#include <dlib/matrix.h>
#include <dlib/optimization.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace argmin::bench
{

namespace detail
{

template <typename Problem, typename Vec>
[[nodiscard]] auto constraint_violation_at(const Problem& prob,
                                           const Vec& x) -> double
{
    double violation = 0.0;
    if constexpr(bound_constrained<Problem>)
    {
        const auto lb = prob.lower_bounds();
        const auto ub = prob.upper_bounds();
        for(int i = 0; i < prob.dimension(); ++i)
        {
            if(std::isfinite(lb[i]))
                violation = std::max(violation, static_cast<double>(lb[i] - x[i]));
            if(std::isfinite(ub[i]))
                violation = std::max(violation, static_cast<double>(x[i] - ub[i]));
        }
    }
    return violation;
}

[[nodiscard]] inline auto cap_status_string(std::string_view status,
                                            const eval_counts& counts,
                                            const bench_config& config) -> std::string
{
    if(status == "maxtime_reached")
        return "wall";
    if(status == "maxeval_reached")
        return "f_eval";
    if(config.max_f_evals > 0 && counts.f >= config.max_f_evals)
        return "f_eval";
    return std::string{counts.cap_status()};
}

// Forwarding objective functor -- routes dlib callbacks through a
// counting_problem<P> wrapper so the {f,g}_evals counters in the bench
// summary are populated independently of dlib's (absent) eval-count
// reporting.
template <typename Problem>
struct dlib_objective
{
    counting_problem<Problem>* wrapped;

    double operator()(const dlib::matrix<double, 0, 1>& x) const
    {
        Eigen::Map<const Eigen::VectorXd> xv(x.begin(), x.size());
        return wrapped->value(xv);
    }
};

// Forwarding gradient functor for dlib's find_min_box_constrained.
template <typename Problem>
struct dlib_gradient
{
    counting_problem<Problem>* wrapped;

    dlib::matrix<double, 0, 1> operator()(
        const dlib::matrix<double, 0, 1>& x) const
    {
        Eigen::Map<const Eigen::VectorXd> xmap(x.begin(), x.size());
        Eigen::Vector<double, Problem::problem_dimension> xv(xmap);
        Eigen::Vector<double, Problem::problem_dimension> g;
        wrapped->gradient(xv, g);

        dlib::matrix<double, 0, 1> result(x.size());
        for(long i = 0; i < x.size(); ++i)
            result(i) = g[i];
        return result;
    }
};

// Convert Eigen vector to dlib column vector.
[[nodiscard]] inline auto to_dlib(const Eigen::VectorXd& v)
    -> dlib::matrix<double, 0, 1>
{
    dlib::matrix<double, 0, 1> m(v.size());
    for(int i = 0; i < v.size(); ++i)
        m(i) = v[i];
    return m;
}

// Clamp x into [lb, ub] element-wise. dlib's find_min_bobyqa aborts if the
// start point violates box bounds (DLIB_ASSERT is non-recoverable).
inline void clamp_to_box(dlib::matrix<double, 0, 1>& x,
                         const dlib::matrix<double, 0, 1>& lb,
                         const dlib::matrix<double, 0, 1>& ub)
{
    for(long i = 0; i < x.size(); ++i)
        x(i) = std::clamp(x(i), lb(i), ub(i));
}

// Run dlib L-BFGS with box constraints on a problem.
//
// Tolerance / iteration budget sourced from bench_config:
//   ftol_rel  -> objective_delta_stop_strategy delta (1e-12 default vs 1e-16
//                under publication mode; this is the dlib equivalent of
//                NLopt's set_ftol_rel for an absolute objective-delta gate).
//   max_iter  -> objective_delta_stop_strategy max-iter cap.
//
// Wall-time gap: dlib's find_min_box_constrained has no built-in wall-clock
// budget API. config.max_wall_time_s is intentionally NOT enforced here;
// the gap is documented in the publication-mode methodology write-up.
template <typename Problem>
auto run_dlib_lbfgs_box(std::string_view problem_name,
                        const Problem& prob,
                        int /*max_evals_legacy*/,
                        const bench_config& config) -> benchmark_result
{
    eval_counts counts;
    counts.set_max_f_evals(config.max_f_evals);
    counting_problem<Problem> wrapped{prob, counts};

    dlib_objective<Problem> obj{&wrapped};
    dlib_gradient<Problem> grad{&wrapped};

    auto x0 = to_dlib(prob.initial_point());
    auto lb = to_dlib(prob.lower_bounds());
    auto ub = to_dlib(prob.upper_bounds());
    clamp_to_box(x0, lb, ub);

    auto t0 = std::chrono::steady_clock::now();

    std::string_view status_str = "converged";
    try
    {
        dlib::find_min_box_constrained(
            dlib::lbfgs_search_strategy(10),
            dlib::objective_delta_stop_strategy(config.ftol_rel, config.max_iter),
            obj, grad, x0, lb, ub);
    }
    catch(const std::exception&)
    {
        status_str = "failed";
    }

    auto t1 = std::chrono::steady_clock::now();
    auto wall_us = std::chrono::duration_cast<
        std::chrono::microseconds>(t1 - t0).count();

    double final_obj = prob.value(
        Eigen::Map<const Eigen::VectorXd>(x0.begin(), x0.size()));
    double known_opt = prob.optimal_value();
    Eigen::Map<const Eigen::VectorXd> final_x_map(x0.begin(), x0.size());
    const double final_cv = constraint_violation_at(prob, final_x_map);

    return benchmark_result{
        .solver = "dlib_lbfgs_box",
        .library = "dlib",
        .problem = problem_name,
        .pclass = prob.pclass,
        .dimension = prob.dimension(),
        .seed = config.seed,
        .mode = (config.the_mode == bench_config::mode::publication)
                    ? std::string_view{"publication"}
                    : std::string_view{"library_defaults"},
        .solver_iters = 0,
        .f_evals = counts.f,
        .g_evals = counts.g,
        .c_evals = counts.c,
        .J_evals = counts.J,
        .wall_time_us = wall_us,
        .final_objective = final_obj,
        .known_optimum = known_opt,
        .accuracy = std::abs(final_obj - known_opt),
        .constraint_violation = final_cv,
        .status = status_str,
        .cap_status = cap_status_string(status_str, counts, config),
        .solve_wall_time_us = wall_us,
        .end_to_end_wall_time_us = wall_us,
    };
}

// Run dlib BOBYQA (derivative-free, bound-constrained) on a problem.
//
// Tolerance / eval budget sourced from bench_config:
//   ftol_rel    -> rhoend (BOBYQA final trust-region radius; tightening this
//                  drives the trust-region contraction floor and is BOBYQA's
//                  closest analog to NLopt's ftol_rel for the publication
//                  protocol).
//   max_f_evals -> 7th positional arg of find_min_bobyqa (function-evaluation
//                  cap inside the trust-region loop).
//
// Wall-time gap: dlib's find_min_bobyqa accepts no time-budget argument;
// config.max_wall_time_s is intentionally NOT enforced. Documented in the
// publication-mode methodology write-up.
template <typename Problem>
auto run_dlib_bobyqa(std::string_view problem_name,
                     const Problem& prob,
                     int /*max_evals_legacy*/,
                     const bench_config& config) -> benchmark_result
{
    eval_counts counts;
    counts.set_max_f_evals(config.max_f_evals);
    counting_problem<Problem> wrapped{prob, counts};

    dlib_objective<Problem> obj{&wrapped};

    auto x0 = to_dlib(prob.initial_point());
    auto lb = to_dlib(prob.lower_bounds());
    auto ub = to_dlib(prob.upper_bounds());
    clamp_to_box(x0, lb, ub);

    int n = prob.dimension();
    // BOBYQA initial trust region radius: reasonable fraction of bound range.
    // rhoend tracks config.ftol_rel so publication mode contracts the trust
    // region down to 1e-16 before declaring convergence.
    double rhobeg = 1.0;
    double rhoend = config.ftol_rel;

    auto t0 = std::chrono::steady_clock::now();

    double minf{};
    std::string_view status_str = "converged";
    try
    {
        minf = dlib::find_min_bobyqa(
            obj, x0, 2 * n + 1,
            lb, ub, rhobeg, rhoend, config.max_f_evals);
    }
    catch(const std::exception&)
    {
        // dlib throws on max-evals reached or other failures.
        minf = prob.value(
            Eigen::Map<const Eigen::VectorXd>(x0.begin(), x0.size()));
        status_str = "maxeval_reached";
    }

    auto t1 = std::chrono::steady_clock::now();
    auto wall_us = std::chrono::duration_cast<
        std::chrono::microseconds>(t1 - t0).count();

    double known_opt = prob.optimal_value();
    Eigen::Map<const Eigen::VectorXd> final_x_map(x0.begin(), x0.size());
    const double final_cv = constraint_violation_at(prob, final_x_map);

    return benchmark_result{
        .solver = "dlib_bobyqa",
        .library = "dlib",
        .problem = problem_name,
        .pclass = prob.pclass,
        .dimension = prob.dimension(),
        .seed = config.seed,
        .mode = (config.the_mode == bench_config::mode::publication)
                    ? std::string_view{"publication"}
                    : std::string_view{"library_defaults"},
        .solver_iters = 0,
        .f_evals = counts.f,
        .g_evals = counts.g,
        .c_evals = counts.c,
        .J_evals = counts.J,
        .wall_time_us = wall_us,
        .final_objective = minf,
        .known_optimum = known_opt,
        .accuracy = std::abs(minf - known_opt),
        .constraint_violation = final_cv,
        .status = status_str,
        .cap_status = cap_status_string(status_str, counts, config),
        .solve_wall_time_us = wall_us,
        .end_to_end_wall_time_us = wall_us,
    };
}

// Run dlib global optimizer on a problem.
//
// Eval / wall budgets sourced from bench_config. find_min_global supports
// both a `max_function_calls` cap and a `std::chrono::duration` runtime cap
// natively (the global optimizer is the one dlib API surface that does
// expose a wall-clock budget); both are consumed so the publication-mode
// 10 s budget actually fires for stochastic global search.
template <typename Problem>
auto run_dlib_global(std::string_view problem_name,
                     const Problem& prob,
                     int /*max_evals_legacy*/,
                     const bench_config& config) -> benchmark_result
{
    eval_counts counts;
    counts.set_max_f_evals(config.max_f_evals);
    counting_problem<Problem> wrapped{prob, counts};

    dlib_objective<Problem> obj{&wrapped};

    auto lb = to_dlib(prob.lower_bounds());
    auto ub = to_dlib(prob.upper_bounds());

    auto t0 = std::chrono::steady_clock::now();

    std::string_view status_str = "converged";
    dlib::function_evaluation result;
    double final_obj = std::numeric_limits<double>::quiet_NaN();
    double final_cv = std::numeric_limits<double>::quiet_NaN();
    try
    {
        const auto wall_budget = std::chrono::duration_cast<
            std::chrono::milliseconds>(
                std::chrono::duration<double>(config.max_wall_time_s));
        result = dlib::find_min_global(
            obj, lb, ub,
            dlib::max_function_calls(config.max_f_evals),
            wall_budget);
        final_obj = result.y;
        Eigen::Map<const Eigen::VectorXd> final_x_map(result.x.begin(), result.x.size());
        final_cv = constraint_violation_at(prob, final_x_map);
    }
    catch(const std::exception&)
    {
        status_str = "failed";
    }

    auto t1 = std::chrono::steady_clock::now();
    auto wall_us = std::chrono::duration_cast<
        std::chrono::microseconds>(t1 - t0).count();

    double known_opt = prob.optimal_value();

    return benchmark_result{
        .solver = "dlib_global",
        .library = "dlib",
        .problem = problem_name,
        .pclass = prob.pclass,
        .dimension = prob.dimension(),
        .seed = config.seed,
        .mode = (config.the_mode == bench_config::mode::publication)
                    ? std::string_view{"publication"}
                    : std::string_view{"library_defaults"},
        .solver_iters = 0,
        .f_evals = counts.f,
        .g_evals = counts.g,
        .c_evals = counts.c,
        .J_evals = counts.J,
        .wall_time_us = wall_us,
        .final_objective = final_obj,
        .known_optimum = known_opt,
        .accuracy = std::abs(final_obj - known_opt),
        .constraint_violation = final_cv,
        .status = status_str,
        .cap_status = cap_status_string(status_str, counts, config),
        .solve_wall_time_us = wall_us,
        .end_to_end_wall_time_us = wall_us,
    };
}

}

void run_dlib_benchmarks(std::vector<benchmark_result>& results,
                         std::vector<std::vector<trace_entry>>& traces,
                         const bench_config& config)
{
    // dlib is summary-only per D-D4: no per-iter callback surface in
    // find_min_box_constrained / find_min_bobyqa / find_min_global.
    // We push an empty trace vector per result row to preserve
    // results[i] <-> traces[i] index alignment with the driver's traces[]
    // array; the driver writes nothing for empty trace vectors.
    //
    // Each adapter call wraps prob through counting_problem<P>;
    // {f,g,c,J}_evals come from the wrapper. dlib does not expose its own
    // iter count for any of the find_min_* variants, so solver_iters is
    // populated as 0 (treated as a diagnostic "n/a" by post-processing).
    constexpr int max_evals = 10000;

    for_each_problem([&](std::string_view name, auto&& prob) {
        using P = std::remove_cvref_t<decltype(prob)>;

        constexpr bool is_bound =
            has_class(P::pclass, problem_class::bound_constrained);
        constexpr bool is_ineq =
            has_class(P::pclass, problem_class::inequality);
        constexpr bool is_eq =
            has_class(P::pclass, problem_class::equality);
        constexpr bool is_mixed =
            has_class(P::pclass, problem_class::mixed);
        constexpr bool is_global =
            has_class(P::pclass, problem_class::global);
        constexpr bool has_gradient = differentiable<P>;

        // Skip constrained problems (dlib has no constraint support).
        constexpr bool has_constraints = is_ineq || is_eq || is_mixed;

        // Bound-constrained (no general constraints): L-BFGS box + BOBYQA.
        if constexpr(is_bound && !has_constraints && !is_global && has_gradient)
        {
            results.push_back(
                detail::run_dlib_lbfgs_box(name, prob, max_evals, config));
            traces.push_back(std::vector<trace_entry>{});
        }

        if constexpr(is_bound && !has_constraints && !is_global)
        {
            results.push_back(
                detail::run_dlib_bobyqa(name, prob, max_evals, config));
            traces.push_back(std::vector<trace_entry>{});
        }

        // Global problems with bounds: global optimizer + BOBYQA.
        if constexpr(is_global && is_bound)
        {
            results.push_back(
                detail::run_dlib_global(name, prob, max_evals, config));
            traces.push_back(std::vector<trace_entry>{});
            results.push_back(
                detail::run_dlib_bobyqa(name, prob, max_evals, config));
            traces.push_back(std::vector<trace_entry>{});
        }
    });
}

}
