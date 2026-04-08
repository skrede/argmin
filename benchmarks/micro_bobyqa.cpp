// Micro-benchmark: nablapp BOBYQA vs NLopt BOBYQA on HS problems.
//
// Profiles per-solve wall time and eval counts for direct comparison.
// Run under perf for flamegraph analysis:
//   perf record -F 99999 -g -- ./micro_bobyqa
//   perf report --stdio --percent-limit=1.0

#include "nablapp/solver/bobyqa_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/test_functions/hock_schittkowski.h"

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

// NLopt callback for HS001.
double nlopt_hs001(unsigned, const double* x, double*, void*)
{
    double t1 = x[1] - x[0] * x[0];
    double t2 = 1.0 - x[0];
    return 100.0 * t1 * t1 + t2 * t2;
}

// NLopt callback for HS005.
double nlopt_hs005(unsigned, const double* x, double*, void*)
{
    double d = x[0] - x[1];
    return std::sin(x[0] + x[1]) + d * d - 1.5 * x[0] + 2.5 * x[1] + 1.0;
}

template <typename Problem>
timing bench_nablapp(const Problem& problem, std::uint32_t reps)
{
    auto x0 = problem.initial_point();
    nablapp::solver_options opts;
    opts.max_iterations = 5000;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-12);

    // Warmup.
    {
        nablapp::basic_solver solver{nablapp::bobyqa_policy<2>{}, problem, x0, opts};
        solver.solve();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t iters = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nablapp::basic_solver solver{nablapp::bobyqa_policy<2>{}, problem, x0, opts};
        auto result = solver.solve();
        fval = result.objective_value;
        iters = result.iterations;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, iters};
}

timing bench_nlopt_hs001(std::uint32_t reps)
{
    // Warmup.
    {
        nlopt::opt opt(nlopt::LN_BOBYQA, 2);
        opt.set_min_objective(nlopt_hs001, nullptr);
        opt.set_lower_bounds({-1e20, -1.5});
        opt.set_upper_bounds({1e20, 1e20});
        opt.set_maxeval(5000);
        opt.set_ftol_rel(1e-14);
        opt.set_xtol_rel(1e-14);
        std::vector<double> x = {-2.0, 1.0};
        double fval;
        opt.optimize(x, fval);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t evals = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nlopt::opt opt(nlopt::LN_BOBYQA, 2);
        opt.set_min_objective(nlopt_hs001, nullptr);
        opt.set_lower_bounds({-1e20, -1.5});
        opt.set_upper_bounds({1e20, 1e20});
        opt.set_maxeval(5000);
        opt.set_ftol_rel(1e-14);
        opt.set_xtol_rel(1e-14);
        std::vector<double> x = {-2.0, 1.0};
        opt.optimize(x, fval);
        evals = static_cast<std::uint32_t>(opt.get_numevals());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, evals};
}

timing bench_nlopt_hs005(std::uint32_t reps)
{
    // Warmup.
    {
        nlopt::opt opt(nlopt::LN_BOBYQA, 2);
        opt.set_min_objective(nlopt_hs005, nullptr);
        opt.set_lower_bounds({-1.5, -3.0});
        opt.set_upper_bounds({4.0, 3.0});
        opt.set_maxeval(5000);
        opt.set_ftol_rel(1e-14);
        opt.set_xtol_rel(1e-14);
        std::vector<double> x = {0.0, 0.0};
        double fval;
        opt.optimize(x, fval);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t evals = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nlopt::opt opt(nlopt::LN_BOBYQA, 2);
        opt.set_min_objective(nlopt_hs005, nullptr);
        opt.set_lower_bounds({-1.5, -3.0});
        opt.set_upper_bounds({4.0, 3.0});
        opt.set_maxeval(5000);
        opt.set_ftol_rel(1e-14);
        opt.set_xtol_rel(1e-14);
        std::vector<double> x = {0.0, 0.0};
        opt.optimize(x, fval);
        evals = static_cast<std::uint32_t>(opt.get_numevals());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, evals};
}

void print_row(std::string_view solver, const timing& t)
{
    std::println("  {:>12s}  {:10.2f}  {:10d}  {:.6e}", solver, t.wall_us, t.evals, t.objective);
}

}

int main()
{
    constexpr std::uint32_t reps = 1000;
    std::println("BOBYQA micro-benchmark, {} repetitions each\n", reps);
    std::println("  {:>12s}  {:>10s}  {:>10s}  {:>12s}", "solver", "wall (us)", "evals", "objective");

    // HS001
    std::println("\n--- HS001 (Rosenbrock variant, x1 >= -1.5) ---");
    auto na1 = bench_nablapp(nablapp::hs001<double>{}, reps);
    auto nl1 = bench_nlopt_hs001(reps);
    print_row("nablapp", na1);
    print_row("nlopt", nl1);
    std::println("  ratio: {:.1f}x wall, {:.1f}x evals", na1.wall_us / nl1.wall_us, double(na1.evals) / nl1.evals);

    // HS005
    std::println("\n--- HS005 (trigonometric, tight bounds) ---");
    auto na5 = bench_nablapp(nablapp::hs005<double>{}, reps);
    auto nl5 = bench_nlopt_hs005(reps);
    print_row("nablapp", na5);
    print_row("nlopt", nl5);
    std::println("  ratio: {:.1f}x wall, {:.1f}x evals", na5.wall_us / nl5.wall_us, double(na5.evals) / nl5.evals);

    std::println("\nProfile with:");
    std::println("  perf stat ./micro_bobyqa");
    std::println("  perf record -F 99999 -g -- ./micro_bobyqa");
    std::println("  perf report --stdio --percent-limit=1.0");
}
