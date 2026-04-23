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

// Counting objective functor -- wraps a nablapp problem for dlib, tracking
// function evaluation count (dlib does not expose this).
template <typename Problem>
struct counting_objective
{
    Problem prob;
    mutable int f_count{0};

    double operator()(const dlib::matrix<double, 0, 1>& x) const
    {
        ++f_count;
        Eigen::Map<const Eigen::VectorXd> xv(x.begin(), x.size());
        return prob.value(xv);
    }
};

// Counting gradient functor for dlib's find_min_box_constrained.
template <typename Problem>
struct counting_gradient
{
    Problem prob;
    mutable int g_count{0};

    dlib::matrix<double, 0, 1> operator()(
        const dlib::matrix<double, 0, 1>& x) const
    {
        ++g_count;
        Eigen::Map<const Eigen::VectorXd> xmap(x.begin(), x.size());
        Eigen::Vector<double, Problem::problem_dimension> xv(xmap);
        Eigen::Vector<double, Problem::problem_dimension> g;
        prob.gradient(xv, g);

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
                        int max_evals) -> benchmark_result
{
    counting_objective<Problem> obj{prob};
    counting_gradient<Problem> grad{prob};

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
        .f_evals = obj.f_count,
        .g_evals = grad.g_count,
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
                     int max_evals) -> benchmark_result
{
    counting_objective<Problem> obj{prob};

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
        .f_evals = obj.f_count,
        .g_evals = 0,
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
                     int max_evals) -> benchmark_result
{
    counting_objective<Problem> obj{prob};

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
        .f_evals = obj.f_count,
        .g_evals = 0,
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
    // bench_config consumption: mode::library_defaults preserves existing
    // byte-identical behavior (this plan scope). A follow-on plan branches
    // on config.the_mode == mode::publication for tightened tolerances +
    // trace emission and routes problem callbacks through
    // counting_problem<P>.
    (void)config;  // unused in this scaffold — consumed in follow-on plans.

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
                detail::run_dlib_lbfgs_box(name, prob, max_evals));

        if constexpr(is_bound && !has_constraints && !is_global)
            results.push_back(
                detail::run_dlib_bobyqa(name, prob, max_evals));

        // Global problems with bounds: global optimizer + BOBYQA.
        if constexpr(is_global && is_bound)
        {
            results.push_back(
                detail::run_dlib_global(name, prob, max_evals));
            results.push_back(
                detail::run_dlib_bobyqa(name, prob, max_evals));
        }
    });
}

}
