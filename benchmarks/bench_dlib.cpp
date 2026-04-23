// dlib comparison benchmarks for nablapp benchmark suite.
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
// (controlled by NABLAPP_HAS_DLIB compile definition from CMake).

#include "bench_dlib.h"
#include "counting_problem.h"
#include "problem_registry.h"

#include "nablapp/formulation/concepts.h"

#include <dlib/global_optimization.h>
#include <dlib/matrix.h>
#include <dlib/optimization.h>

#include <chrono>
#include <cmath>
#include <string_view>
#include <vector>

namespace nablapp::bench
{

namespace detail
{

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

// Run dlib L-BFGS with box constraints on a problem.
template <typename Problem>
auto run_dlib_lbfgs_box(std::string_view problem_name,
                        const Problem& prob,
                        int max_evals,
                        const bench_config& config) -> benchmark_result
{
    eval_counts counts;
    counting_problem<Problem> wrapped{prob, counts};

    dlib_objective<Problem> obj{&wrapped};
    dlib_gradient<Problem> grad{&wrapped};

    auto x0 = to_dlib(prob.initial_point());
    auto lb = to_dlib(prob.lower_bounds());
    auto ub = to_dlib(prob.upper_bounds());

    auto t0 = std::chrono::high_resolution_clock::now();

    std::string_view status_str = "converged";
    try
    {
        dlib::find_min_box_constrained(
            dlib::lbfgs_search_strategy(10),
            dlib::objective_delta_stop_strategy(1e-12, max_evals),
            obj, grad, x0, lb, ub);
    }
    catch(const std::exception&)
    {
        status_str = "failed";
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto wall_us = std::chrono::duration_cast<
        std::chrono::microseconds>(t1 - t0).count();

    double final_obj = prob.value(
        Eigen::Map<const Eigen::VectorXd>(x0.begin(), x0.size()));
    double known_opt = prob.optimal_value();

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
        .status = status_str,
    };
}

// Run dlib BOBYQA (derivative-free, bound-constrained) on a problem.
template <typename Problem>
auto run_dlib_bobyqa(std::string_view problem_name,
                     const Problem& prob,
                     int max_evals,
                     const bench_config& config) -> benchmark_result
{
    eval_counts counts;
    counting_problem<Problem> wrapped{prob, counts};

    dlib_objective<Problem> obj{&wrapped};

    auto x0 = to_dlib(prob.initial_point());
    auto lb = to_dlib(prob.lower_bounds());
    auto ub = to_dlib(prob.upper_bounds());

    int n = prob.dimension();
    // BOBYQA initial trust region radius: reasonable fraction of bound range.
    double rhobeg = 1.0;
    double rhoend = 1e-8;

    auto t0 = std::chrono::high_resolution_clock::now();

    double minf{};
    std::string_view status_str = "converged";
    try
    {
        minf = dlib::find_min_bobyqa(
            obj, x0, 2 * n + 1,
            lb, ub, rhobeg, rhoend, max_evals);
    }
    catch(const std::exception&)
    {
        // dlib throws on max-evals reached or other failures.
        minf = prob.value(
            Eigen::Map<const Eigen::VectorXd>(x0.begin(), x0.size()));
        status_str = "max_iterations";
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto wall_us = std::chrono::duration_cast<
        std::chrono::microseconds>(t1 - t0).count();

    double known_opt = prob.optimal_value();

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
        .status = status_str,
    };
}

// Run dlib global optimizer on a problem.
template <typename Problem>
auto run_dlib_global(std::string_view problem_name,
                     const Problem& prob,
                     int max_evals,
                     const bench_config& config) -> benchmark_result
{
    eval_counts counts;
    counting_problem<Problem> wrapped{prob, counts};

    dlib_objective<Problem> obj{&wrapped};

    auto lb = to_dlib(prob.lower_bounds());
    auto ub = to_dlib(prob.upper_bounds());

    auto t0 = std::chrono::high_resolution_clock::now();

    std::string_view status_str = "converged";
    dlib::function_evaluation result;
    try
    {
        result = dlib::find_min_global(
            obj, lb, ub,
            dlib::max_function_calls(max_evals));
    }
    catch(const std::exception&)
    {
        status_str = "failed";
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto wall_us = std::chrono::duration_cast<
        std::chrono::microseconds>(t1 - t0).count();

    double known_opt = prob.optimal_value();
    double final_obj = result.y;

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
        .status = status_str,
    };
}

}

void run_dlib_benchmarks(std::vector<benchmark_result>& results, const bench_config& config)
{
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
            results.push_back(
                detail::run_dlib_lbfgs_box(name, prob, max_evals, config));

        if constexpr(is_bound && !has_constraints && !is_global)
            results.push_back(
                detail::run_dlib_bobyqa(name, prob, max_evals, config));

        // Global problems with bounds: global optimizer + BOBYQA.
        if constexpr(is_global && is_bound)
        {
            results.push_back(
                detail::run_dlib_global(name, prob, max_evals, config));
            results.push_back(
                detail::run_dlib_bobyqa(name, prob, max_evals, config));
        }
    });
}

}
