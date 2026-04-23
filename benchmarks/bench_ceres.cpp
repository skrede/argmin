// Ceres Solver comparison benchmarks for nablapp benchmark suite.
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
#include "counting_problem.h"
#include "problem_registry.h"

#include <ceres/gradient_problem.h>
#include <ceres/gradient_problem_solver.h>

#include <chrono>
#include <cmath>
#include <string_view>
#include <vector>

namespace nablapp::bench
{

namespace detail
{

// Adapts any nablapp problem as a ceres::FirstOrderFunction.
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

// Run Ceres LBFGS on a problem and return a benchmark_result.
template <typename Problem>
auto run_ceres_solver(std::string_view problem_name,
                      const Problem& prob,
                      int max_iterations,
                      const bench_config& config) -> benchmark_result
{
    eval_counts counts;
    counting_problem<Problem> wrapped{prob, counts};

    auto x0 = prob.initial_point();

    // GradientProblem takes ownership of the raw adapter pointer; the
    // adapter holds a non-owning pointer to `wrapped` which lives on the
    // local stack frame for the duration of the Solve() call.
    auto* adapter = new ceres_first_order_adapter<Problem>(&wrapped);
    ceres::GradientProblem gradient_problem(adapter);

    ceres::GradientProblemSolver::Options options;
    options.max_num_iterations = max_iterations;
    options.line_search_direction_type = ceres::LBFGS;
    options.function_tolerance = 1e-12;
    options.gradient_tolerance = 1e-10;
    options.logging_type = ceres::SILENT;

    std::vector<double> x(x0.data(), x0.data() + prob.dimension());

    ceres::GradientProblemSolver::Summary summary;

    auto t0 = std::chrono::high_resolution_clock::now();
    ceres::Solve(options, gradient_problem, x.data(), &summary);
    auto t1 = std::chrono::high_resolution_clock::now();

    auto wall_us = std::chrono::duration_cast<
        std::chrono::microseconds>(t1 - t0).count();

    double known_opt = prob.optimal_value();

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
        .status = ceres_status_string(summary.termination_type),
    };
}

}

void run_ceres_benchmarks(std::vector<benchmark_result>& results, const bench_config& config)
{
    // bench_config consumption: every adapter call wraps prob through
    // counting_problem<P>; {f,g,c,J}_evals come from the wrapper, while
    // solver_iters reads back Ceres' native iteration count from the
    // GradientProblemSolver::Summary.iterations vector.
    constexpr int max_iterations = 10000;

    // Ceres only does unconstrained -- use for_each_problem_of_class.
    for_each_problem_of_class(problem_class::unconstrained,
        [&](std::string_view name, auto&& prob) {
            using P = std::remove_cvref_t<decltype(prob)>;
            P p{};
            results.push_back(
                detail::run_ceres_solver(name, p, max_iterations, config));
        });
}

}
