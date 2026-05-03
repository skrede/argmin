// libcmaes (Beyer-Schwefel-Hansen-Benazera, MIT) comparison benchmarks.
//
// Reference target for the argmin CMA-ES policy: a fair head-to-head
// against another C++ CMA-ES implementation rather than against the
// stock NLopt CRS2 / ISRES baseline (which are different algorithm
// classes, not different CMA-ES implementations).
//
// Solver mapping:
//   Global bound-constrained: CMAES_DEFAULT -> "libcmaes_cmaes"
//                              IPOP_CMAES   -> "libcmaes_ipop"
//
// Constrained (equality / inequality) cells are intentionally skipped:
// libcmaes is a black-box global optimizer with no native constraint
// support; routing constrained problems through it would require a
// penalty / augmented-Lagrangian wrapper that confounds the
// implementation comparison with the wrapper's behavior.
//
// Reference: Hansen & Ostermeier (2001); Hansen (2023) arXiv:1604.00772;
//            libcmaes upstream https://github.com/CMA-ES/libcmaes.

#include "bench_libcmaes.h"
#include "trace_entry.h"
#include "counting_problem.h"
#include "problem_registry.h"

#include "argmin/formulation/concepts.h"

#include <libcmaes/cmaes.h>
#include <libcmaes/cmaparameters.h>
#include <libcmaes/cmasolutions.h>
#include <libcmaes/genopheno.h>
#include <libcmaes/pwq_bound_strategy.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

namespace argmin::bench
{

namespace detail
{

using libcmaes_genopheno =
    libcmaes::GenoPheno<libcmaes::pwqBoundStrategy>;

using libcmaes_parameters =
    libcmaes::CMAParameters<libcmaes_genopheno>;

// Trace wrapper bundling the counting_problem<P> with a destination trace
// vector and the timing baseline. libcmaes invokes the FitFunc once per
// candidate evaluation; we use that hook to emit per-eval trace rows in
// publication mode. step_norm and kkt_residual are NaN (libcmaes's
// callback exposes neither directly).
template <typename Problem>
struct libcmaes_trace_wrapper
{
    counting_problem<Problem>* prob{nullptr};
    std::vector<trace_entry>*  trace{nullptr};
    std::int64_t               t0_us{0};
    double                     f_star{};
    double                     f_best_running{std::numeric_limits<double>::infinity()};
    int                        iter_count{0};
};

template <typename Problem>
auto make_objective(counting_problem<Problem>& wrapped)
    -> libcmaes::FitFunc
{
    return [&wrapped](const double* x, const int& n) -> double
    {
        Eigen::Map<const Eigen::VectorXd> xmap(x, n);
        Eigen::Vector<double, Problem::problem_dimension> xv(xmap);
        return wrapped.value(xv);
    };
}

template <typename Problem>
auto make_trace_objective(libcmaes_trace_wrapper<Problem>& tw)
    -> libcmaes::FitFunc
{
    return [&tw](const double* x, const int& n) -> double
    {
        Eigen::Map<const Eigen::VectorXd> xmap(x, n);
        Eigen::Vector<double, Problem::problem_dimension> xv(xmap);
        const double f = tw.prob->value(xv);

        tw.f_best_running = std::min(tw.f_best_running, f);

        const auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        tw.trace->push_back(trace_entry{
            .iter         = tw.iter_count++,
            .f_evals      = tw.prob->counts->f,
            .g_evals      = tw.prob->counts->g,
            .c_evals      = tw.prob->counts->c,
            .J_evals      = tw.prob->counts->J,
            .wall_us      = now_us - tw.t0_us,
            .f_current    = f,
            .f_best       = tw.f_best_running,
            .accuracy     = std::abs(f - tw.f_star),
            .cv           = std::numeric_limits<double>::quiet_NaN(),
            .step_norm    = std::numeric_limits<double>::quiet_NaN(),
            .kkt_residual = std::numeric_limits<double>::quiet_NaN(),
        });

        return f;
    };
}

[[nodiscard]] auto libcmaes_status_string(int run_status) -> std::string_view
{
    // libcmaes CMAStopCriteria codes (cmastopcriteria.h, libcmaes 0.10).
    // Positive codes denote successful termination on a stop criterion;
    // negative codes denote error/abort. Mapping to argmin's status
    // vocabulary keeps the publish_summary `status` column comparable
    // across libraries.
    switch(run_status)
    {
    case  0:  return "running";
    case  1:  return "max_iterations";       // CONT
    case  2:  return "ftol_reached";         // FTARGET_REACHED
    case  3:  return "xtol_reached";         // TOLHISTFUN
    case  4:  return "ftol_reached";         // EQUALFUNVALS
    case  5:  return "xtol_reached";         // TOLX
    case  6:  return "stalled";              // TOLUPSIGMA
    case  7:  return "stalled";              // STAGNATION
    case  8:  return "diverged";             // CONDITIONCOV
    case  9:  return "diverged";             // NOEFFECTAXIS
    case 10:  return "diverged";             // NOEFFECTCOOR
    case 11:  return "max_iterations";       // MAXFEVALS
    case 12:  return "max_iterations";       // MAXITER
    case 13:  return "diverged";             // FTARGET / unused
    default:  return run_status < 0 ? "failed" : "converged";
    }
}

// Run a single libcmaes solver on a global bound-constrained problem.
// Mirrors run_nlopt_solver's structure: capture eval counts via
// counting_problem<P>, time only the optimize() call, return a populated
// benchmark_result on the publish_summary schema.
template <typename Problem>
auto run_libcmaes_solver(int algo,
                         std::string_view solver_name,
                         std::string_view problem_name,
                         Problem& prob,
                         const bench_config& config,
                         std::vector<trace_entry>& local_trace) -> benchmark_result
{
    eval_counts counts;
    counting_problem<Problem> wrapped{prob, counts};

    const int n = prob.dimension();
    auto x0_vec = prob.initial_point();
    std::vector<double> x0(x0_vec.data(), x0_vec.data() + n);

    auto lb = prob.lower_bounds();
    auto ub = prob.upper_bounds();
    std::vector<double> lb_vec(lb.data(), lb.data() + n);
    std::vector<double> ub_vec(ub.data(), ub.data() + n);

    // sigma initialization mirrors argmin's cmaes_policy::init() default
    // for bound-constrained problems: max(ub - lb) / 3, with a floor of
    // 0.3 if every range is non-finite. Hansen (2023) recommends sigma in
    // [search_range / 4, search_range / 3].
    double max_range = 0.0;
    for(int i = 0; i < n; ++i)
    {
        double range = ub_vec[i] - lb_vec[i];
        if(std::isfinite(range))
            max_range = std::max(max_range, range);
    }
    const double sigma0 = max_range > 0.0 ? max_range / 3.0 : 0.3;

    libcmaes_genopheno gp(lb_vec.data(), ub_vec.data(), n);

    // lambda = -1 -> libcmaes auto-computes 4 + floor(3*ln(n)). This
    // matches the unbounded-problem default. argmin's policy uses
    // 4*n on bounded problems; we keep libcmaes on its own default to
    // benchmark each implementation under its own auto-tuning, not
    // argmin's choice imposed externally.
    libcmaes_parameters params(x0, sigma0, /*lambda=*/-1, config.seed, gp);
    params.set_algo(algo);
    params.set_max_iter(config.max_iter);
    params.set_max_fevals(config.max_f_evals);
    params.set_ftolerance(config.ftol_rel);
    params.set_xtolerance(config.xtol_rel);
    params.set_quiet(true);

    libcmaes_trace_wrapper<Problem> tw{
        .prob   = &wrapped,
        .trace  = &local_trace,
        .t0_us  = 0,
        .f_star = prob.optimal_value(),
    };

    libcmaes::FitFunc fit = config.trace_enabled
        ? make_trace_objective(tw)
        : make_objective(wrapped);

    tw.t0_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    auto t0 = std::chrono::high_resolution_clock::now();

    libcmaes::CMASolutions sols;
    std::string_view status_str;
    double minf = std::numeric_limits<double>::infinity();
    try
    {
        sols = libcmaes::cmaes<libcmaes_genopheno>(fit, params);
        status_str = libcmaes_status_string(sols.run_status());
        minf = sols.best_candidate().get_fvalue();
    }
    catch(const std::exception&)
    {
        status_str = "failed";
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto wall_us = std::chrono::duration_cast<
        std::chrono::microseconds>(t1 - t0).count();

    const double known_opt = prob.optimal_value();

    return benchmark_result{
        .solver = solver_name,
        .library = "libcmaes",
        .problem = problem_name,
        .pclass = prob.pclass,
        .dimension = n,
        .seed = config.seed,
        .mode = (config.the_mode == bench_config::mode::publication)
                    ? std::string_view{"publication"}
                    : std::string_view{"library_defaults"},
        .solver_iters = static_cast<int>(sols.niter()),
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

} // detail

void run_libcmaes_benchmarks(std::vector<benchmark_result>& results,
                             std::vector<std::vector<trace_entry>>& traces,
                             const bench_config& config)
{
    auto run_emitting = [&](auto runner_call) {
        std::vector<trace_entry> local_trace;
        results.push_back(runner_call(local_trace));
        traces.push_back(std::move(local_trace));
    };

    for_each_problem([&](std::string_view name, auto&& prob) {
        using P = std::remove_cvref_t<decltype(prob)>;
        auto& p = const_cast<P&>(prob);

        constexpr bool is_global =
            has_class(P::pclass, problem_class::global);

        if constexpr(is_global && bound_constrained<P>)
        {
            run_emitting([&](std::vector<trace_entry>& t) {
                return detail::run_libcmaes_solver(
                    ::CMAES_DEFAULT, "libcmaes_cmaes", name, p, config, t);
            });
            run_emitting([&](std::vector<trace_entry>& t) {
                return detail::run_libcmaes_solver(
                    ::IPOP_CMAES, "libcmaes_ipop", name, p, config, t);
            });
        }
    });
}

}
