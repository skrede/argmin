// Micro-benchmark: argmin BOBYQA vs NLopt BOBYQA on HS problems.
//
// Three-way comparison: argmin<2> (fixed-N), argmin<> (dynamic), NLopt.
// Profiles per-solve wall time and eval counts for direct comparison.
// Run under perf for flamegraph analysis:
//   perf record -F 99999 -g -- ./micro_bobyqa
//   perf report --stdio --percent-limit=1.0

#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <nlopt.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <print>

namespace
{

struct timing
{
    double wall_us;
    double objective;
    std::uint32_t evals;
};

// Dynamic-dimension HS001 wrapper for bobyqa_policy<> benchmarking.
struct hs001_dynamic
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;
    static constexpr argmin::problem_class pclass = argmin::problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return 2; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
    {
        double t1 = x[1] - x[0] * x[0];
        double t2 = 1.0 - x[0];
        return 100.0 * t1 * t1 + t2 * t2;
    }

    [[nodiscard]] Eigen::VectorXd lower_bounds() const
    {
        Eigen::VectorXd lb(2);
        lb << -std::numeric_limits<double>::infinity(), -1.5;
        return lb;
    }

    [[nodiscard]] Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(2, std::numeric_limits<double>::infinity());
    }

    [[nodiscard]] Eigen::VectorXd initial_point() const
    {
        return Eigen::VectorXd{{-2.0, 1.0}};
    }
};

// Dynamic-dimension HS005 wrapper for bobyqa_policy<> benchmarking.
struct hs005_dynamic
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;
    static constexpr argmin::problem_class pclass = argmin::problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return 2; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
    {
        double d = x[0] - x[1];
        return std::sin(x[0] + x[1]) + d * d - 1.5 * x[0] + 2.5 * x[1] + 1.0;
    }

    [[nodiscard]] Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd{{-1.5, -3.0}};
    }

    [[nodiscard]] Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd{{4.0, 3.0}};
    }

    [[nodiscard]] Eigen::VectorXd initial_point() const
    {
        return Eigen::VectorXd{{0.0, 0.0}};
    }
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

template <typename Policy, typename Problem>
timing bench_argmin(Policy policy, const Problem& problem, std::uint32_t reps)
{
    auto x0 = problem.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 5000;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-12);

    // Warmup.
    {
        argmin::step_budget_solver solver{policy, problem, x0, opts};
        solver.solve();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t iters = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        argmin::step_budget_solver solver{policy, problem, x0, opts};
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
    {
        std::println("\n--- HS001 (Rosenbrock variant, x1 >= -1.5) ---");
        auto fixed = bench_argmin(argmin::bobyqa_policy<2>{}, argmin::hs001<double>{}, reps);
        auto dyn   = bench_argmin(argmin::bobyqa_policy<>{},  hs001_dynamic{}, reps);
        auto nlopt = bench_nlopt_hs001(reps);
        print_row("argmin<2>", fixed);
        print_row("argmin<>", dyn);
        print_row("nlopt", nlopt);
        std::println("  ratio fixed/nlopt: {:.1f}x wall, {:.1f}x evals",
            fixed.wall_us / nlopt.wall_us, double(fixed.evals) / nlopt.evals);
        std::println("  ratio dyn/nlopt:   {:.1f}x wall, {:.1f}x evals",
            dyn.wall_us / nlopt.wall_us, double(dyn.evals) / nlopt.evals);
    }

    // HS005
    {
        std::println("\n--- HS005 (trigonometric, tight bounds) ---");
        auto fixed = bench_argmin(argmin::bobyqa_policy<2>{}, argmin::hs005<double>{}, reps);
        auto dyn   = bench_argmin(argmin::bobyqa_policy<>{},  hs005_dynamic{}, reps);
        auto nlopt = bench_nlopt_hs005(reps);
        print_row("argmin<2>", fixed);
        print_row("argmin<>", dyn);
        print_row("nlopt", nlopt);
        std::println("  ratio fixed/nlopt: {:.1f}x wall, {:.1f}x evals",
            fixed.wall_us / nlopt.wall_us, double(fixed.evals) / nlopt.evals);
        std::println("  ratio dyn/nlopt:   {:.1f}x wall, {:.1f}x evals",
            dyn.wall_us / nlopt.wall_us, double(dyn.evals) / nlopt.evals);
    }

    std::println("\nProfile with:");
    std::println("  perf stat ./micro_bobyqa");
    std::println("  perf record -F 99999 -g -- ./micro_bobyqa");
    std::println("  perf report --stdio --percent-limit=1.0");
}
