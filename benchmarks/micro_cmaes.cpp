// Micro-benchmark: nablapp CMA-ES vs NLopt CMA-ES on global problems.
//
// Profiles per-solve wall time and eval counts. IPOP restarts enabled
// for both libraries — essential for multimodal landscape coverage.
// Run under perf for flamegraph analysis:
//   perf record -F 99999 -g -- ./micro_cmaes
//   perf report --stdio --percent-limit=1.0

#include "nablapp/solver/cmaes_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/test_functions/rastrigin.h"
#include "nablapp/test_functions/rosenbrock.h"

#include <Eigen/Core>

#include <nlopt.hpp>

#include <chrono>
#include <cstdint>
#include <print>

namespace
{

struct timing
{
    double wall_us;
    double objective;
    std::uint32_t evals;
};

// NLopt callback for Rastrigin.
double nlopt_rastrigin(unsigned n, const double* x, double*, void*)
{
    constexpr double two_pi = 2.0 * 3.14159265358979323846;
    double f = 10.0 * n;
    for(unsigned i = 0; i < n; ++i)
        f += x[i] * x[i] - 10.0 * std::cos(two_pi * x[i]);
    return f;
}

// NLopt callback for Rosenbrock.
double nlopt_rosenbrock(unsigned n, const double* x, double*, void*)
{
    double f = 0.0;
    for(unsigned i = 0; i + 1 < n; ++i)
    {
        double t1 = 1.0 - x[i];
        double t2 = x[i + 1] - x[i] * x[i];
        f += t1 * t1 + 100.0 * t2 * t2;
    }
    return f;
}

template <typename Problem>
timing bench_nablapp(const Problem& problem, std::uint32_t reps, std::uint32_t max_iter)
{
    auto x0 = problem.initial_point();
    nablapp::solver_options opts;
    opts.max_iterations = max_iter;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    typename nablapp::cmaes_policy<>::options_type cmaes_opts{};
    cmaes_opts.seed = 42;
    cmaes_opts.restart = nablapp::cmaes_policy<>::restart_strategy::ipop;

    // Warmup.
    {
        nablapp::basic_solver solver{nablapp::cmaes_policy<>{}, problem, x0, opts, cmaes_opts};
        solver.solve();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t iters = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nablapp::basic_solver solver{nablapp::cmaes_policy<>{}, problem, x0, opts, cmaes_opts};
        auto result = solver.solve();
        fval = result.objective_value;
        iters = result.iterations;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, iters};
}

timing bench_nlopt_cma(int n, nlopt_func obj, const std::vector<double>& x0,
                       const std::vector<double>& lb, const std::vector<double>& ub,
                       std::uint32_t reps, std::uint32_t max_eval)
{
    // Warmup.
    {
        nlopt::opt opt(nlopt::GN_CRS2_LM, n);
        opt.set_min_objective(obj, nullptr);
        opt.set_lower_bounds(lb);
        opt.set_upper_bounds(ub);
        opt.set_maxeval(max_eval);
        std::vector<double> x = x0;
        double fval;
        try { opt.optimize(x, fval); } catch(...) {}
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t evals = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        // NLopt has no CMA-ES; use GN_CRS2_LM (global, derivative-free)
        // as the closest comparable global optimizer.
        nlopt::opt opt(nlopt::GN_CRS2_LM, n);
        opt.set_min_objective(obj, nullptr);
        opt.set_lower_bounds(lb);
        opt.set_upper_bounds(ub);
        opt.set_maxeval(max_eval);
        opt.set_ftol_rel(1e-14);
        std::vector<double> x = x0;
        try { opt.optimize(x, fval); } catch(...) {}
        evals = static_cast<std::uint32_t>(opt.get_numevals());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, evals};
}

// Bounded Rosenbrock for CMA-ES (unimodal, tests smooth landscape convergence).
struct bounded_rosenbrock
{
    static constexpr int problem_dimension = nablapp::dynamic_dimension;
    static constexpr nablapp::problem_class pclass =
        nablapp::problem_class::global | nablapp::problem_class::bound_constrained;
    int dimension() const { return 2; }
    double value(const Eigen::VectorXd& x) const
    {
        double t1 = 1.0 - x[0];
        double t2 = x[1] - x[0] * x[0];
        return t1 * t1 + 100.0 * t2 * t2;
    }
    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd::Constant(2, -5.0); }
    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd::Constant(2, 5.0); }
    Eigen::VectorXd initial_point() const { return Eigen::VectorXd{{-1.2, 1.0}}; }
};

void print_row(std::string_view solver, const timing& t)
{
    std::println("  {:>16s}  {:10.2f}  {:10d}  {:.6e}", solver, t.wall_us, t.evals, t.objective);
}

}

int main()
{
    constexpr std::uint32_t reps = 100;
    std::println("CMA-ES micro-benchmark, {} repetitions each\n", reps);
    std::println("  {:>16s}  {:>10s}  {:>10s}  {:>12s}", "solver", "wall (us)", "evals", "objective");

    // Rastrigin 2D
    {
        std::println("\n--- Rastrigin 2D (global minimum at 0, multimodal) ---");
        nablapp::rastrigin<double> prob{.n = 2};
        auto na = bench_nablapp(prob, reps, 10000);
        auto nl = bench_nlopt_cma(2, nlopt_rastrigin,
            {2.5, 2.5}, {-5.12, -5.12}, {5.12, 5.12}, reps, 10000);
        print_row("nablapp (IPOP)", na);
        print_row("nlopt (CRS2_LM)", nl);
    }

    // Rastrigin 5D
    {
        std::println("\n--- Rastrigin 5D ---");
        nablapp::rastrigin<double> prob{.n = 5};
        std::vector<double> x0(5, 2.5), lb(5, -5.12), ub(5, 5.12);
        auto na = bench_nablapp(prob, reps, 50000);
        auto nl = bench_nlopt_cma(5, nlopt_rastrigin, x0, lb, ub, reps, 50000);
        print_row("nablapp (IPOP)", na);
        print_row("nlopt (CRS2_LM)", nl);
    }

    // Rosenbrock 2D (bounded, unimodal — tests CMA-ES on smooth landscape)
    {
        std::println("\n--- Rosenbrock 2D bounded [-5,5]^2 ---");
        bounded_rosenbrock prob;
        auto na = bench_nablapp(prob, reps, 10000);
        auto nl = bench_nlopt_cma(2, nlopt_rosenbrock,
            {-1.2, 1.0}, {-5.0, -5.0}, {5.0, 5.0}, reps, 10000);
        print_row("nablapp (IPOP)", na);
        print_row("nlopt (CRS2_LM)", nl);
    }

    std::println("\nProfile with:");
    std::println("  perf stat ./micro_cmaes");
    std::println("  perf record -F 99999 -g -- ./micro_cmaes");
    std::println("  perf report --stdio --percent-limit=1.0");
}
