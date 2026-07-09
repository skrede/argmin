// Ceres Solver comparison benchmarks for argmin benchmark suite.
//
// Each Ceres solver is benchmarked on applicable problems using Ceres' native
// API (per D-01: no common adapter interface). Results are collected as
// benchmark_result structs with library = "ceres".
//
// Per Research pitfall 2: Ceres has NO constrained optimization. Only
// benchmark on unconstrained problems via GradientProblem + GradientProblemSolver.
//
// Solver mapping (per D-04):
//   Unconstrained: ceres::LBFGS -> "ceres_lbfgs"

#include "bench_ceres.h"
#include "trace_entry.h"
#include "counting_problem.h"
#include "problem_registry.h"

#include <ceres/iteration_callback.h>
#include <ceres/gradient_problem.h>
#include <ceres/gradient_problem_solver.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace argmin::bench
{

namespace detail
{

// Adapts any argmin problem as a ceres::FirstOrderFunction.
//
// Stores a pointer to a caller-owned counting_problem<Problem>; every
// Evaluate() invocation routes through the wrapper so the shared
// eval_counts get bumped on each value/gradient call. Caller must keep
// the counting_problem alive for the full Solve() lifetime.
template <typename Problem>
class ceres_first_order_adapter : public ceres::FirstOrderFunction
{
public:
    explicit ceres_first_order_adapter(counting_problem<Problem>* wrapped)
        : wrapped_{wrapped}
    {
    }

    bool Evaluate(const double* parameters,
                  double* cost,
                  double* gradient) const override
    {
        Eigen::Map<const Eigen::VectorXd> xmap(parameters, wrapped_->dimension());
        Eigen::Vector<double, Problem::problem_dimension> x(xmap);
        *cost = wrapped_->value(x);

        if(gradient)
        {
            Eigen::Vector<double, Problem::problem_dimension> g;
            wrapped_->gradient(x, g);
            Eigen::Map<Eigen::VectorXd>(gradient, wrapped_->dimension()) = g;
        }

        return true;
    }

    int NumParameters() const override
    {
        return wrapped_->dimension();
    }

private:
    counting_problem<Problem>* wrapped_;
};

[[nodiscard]] auto ceres_status_string(ceres::TerminationType t)
    -> std::string_view
{
    switch(t)
    {
    case ceres::CONVERGENCE: return "converged";
    case ceres::NO_CONVERGENCE: return "max_iterations";
    case ceres::FAILURE: return "failed";
    case ceres::USER_SUCCESS: return "user_success";
    case ceres::USER_FAILURE: return "user_failure";
    default: return "unknown";
    }
}

[[nodiscard]] inline auto cap_status_string(std::string_view status,
                                            const eval_counts& counts,
                                            const bench_config& config) -> std::string
{
    if(status == "maxtime_reached")
        return "wall";
    if(config.max_f_evals > 0 && counts.f >= config.max_f_evals)
        return "f_eval";
    return std::string{counts.cap_status()};
}

// Per-iter trace callback (Pattern 4 in 32.8-RESEARCH.md). Registered on
// GradientProblemSolver::Options::callbacks alongside
// update_state_every_iteration=true (Pitfall 4) so the IterationSummary
// reflects the accepted iterate's cost rather than the trial point. cv is
// always 0 because Ceres has no constrained surface; kkt_residual is NaN
// per the D-C3 capability footnote.
struct ceres_trace_callback : public ceres::IterationCallback
{
    std::vector<trace_entry>* trace{nullptr};
    const eval_counts*        counts{nullptr};
    std::int64_t              t0_us{0};
    double                    f_star{};
    double                    f_best_running{std::numeric_limits<double>::infinity()};

    ceres_trace_callback(std::vector<trace_entry>* tr,
                         const eval_counts*        c,
                         std::int64_t              t0,
                         double                    fs)
        : trace{tr}, counts{c}, t0_us{t0}, f_star{fs}
    {
    }

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& s) override
    {
        const auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        f_best_running = std::min(f_best_running, s.cost);

        trace->push_back(trace_entry{
            .iter         = s.iteration,
            .f_evals      = counts->f,
            .g_evals      = counts->g,
            .c_evals      = 0,
            .J_evals      = 0,
            .wall_us      = now_us - t0_us,
            .f_current    = s.cost,
            .f_best       = f_best_running,
            .accuracy     = std::abs(s.cost - f_star),
            .cv           = 0.0,
            .step_norm    = s.step_norm,
            .kkt_residual = std::numeric_limits<double>::quiet_NaN(),
        });
        return ceres::SOLVER_CONTINUE;
    }
};

// Run Ceres LBFGS on a problem and return a benchmark_result.
template <typename Problem>
auto run_ceres_solver(std::string_view problem_name,
                      const Problem& prob,
                      int /*max_iterations_legacy*/,
                      const bench_config& config,
                      std::vector<trace_entry>& local_trace) -> benchmark_result
{
    eval_counts counts;
    counts.set_max_f_evals(config.max_f_evals);
    counting_problem<Problem> wrapped{prob, counts};

    auto x0 = prob.initial_point();

    // GradientProblem takes ownership of the raw adapter pointer; the
    // adapter holds a non-owning pointer to `wrapped` which lives on the
    // local stack frame for the duration of the Solve() call.
    auto* adapter = new ceres_first_order_adapter<Problem>(&wrapped);
    ceres::GradientProblem gradient_problem(adapter);

    // Stopping criteria sourced from bench_config. Ceres' GradientProblemSolver
    // exposes function_tolerance (objective relative change) and
    // parameter_tolerance (iterate relative change); both are tightened to
    // 1e-16 under publication mode so the DM tau-grid down to 1e-12 is
    // observable. max_solver_time_in_seconds caps wall time per
    // (seed, solver, problem) triple at the publication-mode budget.
    ceres::GradientProblemSolver::Options options;
    options.max_num_iterations = config.max_iter;
    options.line_search_direction_type = ceres::LBFGS;
    options.function_tolerance = config.ftol_rel;
    options.parameter_tolerance = config.xtol_rel;
    options.gradient_tolerance = 1e-10;
    options.max_solver_time_in_seconds = config.max_wall_time_s;
    options.logging_type = ceres::SILENT;
    // Pitfall 4 (32.8-RESEARCH.md): without this flag the IterationSummary
    // received by the callback reflects the line-search trial point instead
    // of the accepted iterate. Set unconditionally; behavior only changes
    // when callbacks are also registered, which library_defaults does not do.
    options.update_state_every_iteration = true;

    std::vector<double> x(x0.data(), x0.data() + prob.dimension());

    ceres::GradientProblemSolver::Summary summary;

    // Per-iter trace registration (Pattern 4 in 32.8-RESEARCH.md). Stack
    // lifetime extends across Solve(); the callback is unregistered when
    // options goes out of scope after the call returns. The `t0_us` baseline
    // is captured immediately before ceres::Solve so per-iter wall_us
    // excludes adapter / GradientProblem construction and matches the
    // summary wall_time_us baseline (the t0 captured below).
    auto t0 = std::chrono::steady_clock::now();
    const auto t0_us_for_trace = std::chrono::duration_cast<std::chrono::microseconds>(
        t0.time_since_epoch()).count();
    ceres_trace_callback cb(&local_trace, &counts, t0_us_for_trace,
                            prob.optimal_value());
    if(config.trace_enabled)
        options.callbacks.push_back(&cb);

    ceres::Solve(options, gradient_problem, x.data(), &summary);
    auto t1 = std::chrono::steady_clock::now();

    auto wall_us = std::chrono::duration_cast<
        std::chrono::microseconds>(t1 - t0).count();

    double known_opt = prob.optimal_value();
    const auto status = ceres_status_string(summary.termination_type);

    return benchmark_result{
        .solver = "ceres_lbfgs",
        .library = "ceres",
        .problem = problem_name,
        .pclass = prob.pclass,
        .dimension = prob.dimension(),
        .seed = config.seed,
        .mode = (config.the_mode == bench_config::mode::publication)
                    ? std::string_view{"publication"}
                    : std::string_view{"library_defaults"},
        .solver_iters = static_cast<int>(summary.iterations.size()),
        .f_evals = counts.f,
        .g_evals = counts.g,
        .c_evals = counts.c,
        .J_evals = counts.J,
        .wall_time_us = wall_us,
        .final_objective = summary.final_cost,
        .known_optimum = known_opt,
        .accuracy = std::abs(summary.final_cost - known_opt),
        .constraint_violation = 0.0,
        .status = status,
        .cap_status = cap_status_string(status, counts, config),
        .solve_wall_time_us = wall_us,
        .end_to_end_wall_time_us = wall_us,
    };
}

}

void run_ceres_benchmarks(std::vector<benchmark_result>& results,
                          std::vector<std::vector<trace_entry>>& traces,
                          const bench_config& config)
{
    // bench_config consumption: every adapter call wraps prob through
    // counting_problem<P>; {f,g,c,J}_evals come from the wrapper, while
    // solver_iters reads back Ceres' native iteration count from the
    // GradientProblemSolver::Summary.iterations vector.
    //
    // Under config.trace_enabled, the IterationCallback appends per-iter
    // rows into a per-problem local_trace; that vector is moved into
    // traces[] alongside the result. Under library_defaults, the callback
    // is not registered and local_trace stays empty — the empty vector is
    // pushed to preserve the results[i] <-> traces[i] index invariant.
    constexpr int max_iterations = 10000;

    // Ceres only does unconstrained -- use for_each_problem_of_class.
    for_each_problem_of_class(problem_class::unconstrained,
        [&](std::string_view name, auto&& prob) {
            std::vector<trace_entry> local_trace;
            results.push_back(
                detail::run_ceres_solver(name, prob, max_iterations, config,
                                         local_trace));
            traces.push_back(std::move(local_trace));
        });
}

}
