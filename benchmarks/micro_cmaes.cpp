// Micro-benchmark: nablapp CMA-ES vs NLopt CRS2_LM on global problems.
//
// Three-way comparison: nablapp<N> (fixed-N), nablapp<> (dynamic), NLopt CRS2_LM.
// IPOP restarts enabled for nablapp variants -- essential for multimodal
// landscape coverage. NLopt CRS2_LM is the closest comparable global
// optimizer (NLopt has no CMA-ES implementation).
// Run under perf for flamegraph analysis:
//   perf record -F 99999 -g -- ./micro_cmaes
//   perf report --stdio --percent-limit=1.0

#include "nablapp/solver/cmaes_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/test_functions/rastrigin.h"

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

// Fixed-dimension bounded Rosenbrock for CMA-ES.
// Unimodal, tests smooth landscape convergence with bounds.
template <int N>
struct bounded_rosenbrock
{
    static constexpr int problem_dimension = N;
    static constexpr nablapp::problem_class pclass =
        nablapp::problem_class::global | nablapp::problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return N; }

    [[nodiscard]] double value(const Eigen::Vector<double, N>& x) const
    {
        double f = 0.0;
        for(int i = 0; i + 1 < N; ++i)
        {
            double t1 = 1.0 - x[i];
            double t2 = x[i + 1] - x[i] * x[i];
            f += t1 * t1 + 100.0 * t2 * t2;
        }
        return f;
    }

    [[nodiscard]] Eigen::Vector<double, N> lower_bounds() const
    {
        return Eigen::Vector<double, N>::Constant(N, -5.0);
    }

    [[nodiscard]] Eigen::Vector<double, N> upper_bounds() const
    {
        return Eigen::Vector<double, N>::Constant(N, 5.0);
    }

    [[nodiscard]] Eigen::Vector<double, N> initial_point() const
    {
        Eigen::Vector<double, N> x0;
        x0[0] = -1.2;
        for(int i = 1; i < N; ++i)
            x0[i] = 1.0;
        return x0;
    }
};

// Dynamic-dimension bounded Rosenbrock (for cmaes_policy<>).
struct bounded_rosenbrock_dynamic
{
    static constexpr int problem_dimension = nablapp::dynamic_dimension;
    static constexpr nablapp::problem_class pclass =
        nablapp::problem_class::global | nablapp::problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return 2; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
    {
        double t1 = 1.0 - x[0];
        double t2 = x[1] - x[0] * x[0];
        return t1 * t1 + 100.0 * t2 * t2;
    }

    [[nodiscard]] Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Constant(2, -5.0);
    }

    [[nodiscard]] Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(2, 5.0);
    }

    [[nodiscard]] Eigen::VectorXd initial_point() const
    {
        return Eigen::VectorXd{{-1.2, 1.0}};
    }
};

template <typename Policy, typename Problem>
timing bench_nablapp(Policy policy, const Problem& problem,
                     std::uint32_t reps, std::uint32_t max_iter)
{
    auto x0 = problem.initial_point();
    nablapp::solver_options opts;
    opts.max_iterations = max_iter;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    typename Policy::options_type cmaes_opts{};
    cmaes_opts.seed = 42;
    cmaes_opts.restart = Policy::restart_strategy::ipop;

    // Warmup.
    {
        nablapp::basic_solver solver{policy, problem, x0, opts, cmaes_opts};
        solver.solve();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t iters = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nablapp::basic_solver solver{policy, problem, x0, opts, cmaes_opts};
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
        // NLopt has no CMA-ES; GN_CRS2_LM is the closest comparable
        // global derivative-free optimizer.
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
        nablapp::rastrigin<double, 2> fixed_prob;
        nablapp::rastrigin<double>    dyn_prob{.n = 2};
        auto fixed = bench_nablapp(nablapp::cmaes_policy<2>{}, fixed_prob, reps, 10000);
        auto dyn   = bench_nablapp(nablapp::cmaes_policy<>{},  dyn_prob,  reps, 10000);
        auto nlopt = bench_nlopt_cma(2, nlopt_rastrigin,
            {2.5, 2.5}, {-5.12, -5.12}, {5.12, 5.12}, reps, 10000);
        print_row("nablapp<2>", fixed);
        print_row("nablapp<>", dyn);
        print_row("nlopt CRS2_LM", nlopt);
        std::println("  ratio fixed/nlopt: {:.1f}x wall, {:.1f}x evals",
            fixed.wall_us / nlopt.wall_us, double(fixed.evals) / nlopt.evals);
        std::println("  ratio dyn/nlopt:   {:.1f}x wall, {:.1f}x evals",
            dyn.wall_us / nlopt.wall_us, double(dyn.evals) / nlopt.evals);
    }

    // Rastrigin 5D
    {
        std::println("\n--- Rastrigin 5D (global minimum at 0, multimodal) ---");
        nablapp::rastrigin<double, 5> fixed_prob;
        nablapp::rastrigin<double>    dyn_prob{.n = 5};
        std::vector<double> x0(5, 2.5), lb(5, -5.12), ub(5, 5.12);
        auto fixed = bench_nablapp(nablapp::cmaes_policy<5>{}, fixed_prob, reps, 50000);
        auto dyn   = bench_nablapp(nablapp::cmaes_policy<>{},  dyn_prob,  reps, 50000);
        auto nlopt = bench_nlopt_cma(5, nlopt_rastrigin, x0, lb, ub, reps, 50000);
        print_row("nablapp<5>", fixed);
        print_row("nablapp<>", dyn);
        print_row("nlopt CRS2_LM", nlopt);
        std::println("  ratio fixed/nlopt: {:.1f}x wall, {:.1f}x evals",
            fixed.wall_us / nlopt.wall_us, double(fixed.evals) / nlopt.evals);
        std::println("  ratio dyn/nlopt:   {:.1f}x wall, {:.1f}x evals",
            dyn.wall_us / nlopt.wall_us, double(dyn.evals) / nlopt.evals);
    }

    // Rosenbrock 2D (bounded, unimodal -- tests CMA-ES on smooth landscape)
    {
        std::println("\n--- Rosenbrock 2D bounded [-5,5]^2 ---");
        bounded_rosenbrock<2>       fixed_prob;
        bounded_rosenbrock_dynamic  dyn_prob;
        auto fixed = bench_nablapp(nablapp::cmaes_policy<2>{}, fixed_prob, reps, 10000);
        auto dyn   = bench_nablapp(nablapp::cmaes_policy<>{},  dyn_prob,  reps, 10000);
        auto nlopt = bench_nlopt_cma(2, nlopt_rosenbrock,
            {-1.2, 1.0}, {-5.0, -5.0}, {5.0, 5.0}, reps, 10000);
        print_row("nablapp<2>", fixed);
        print_row("nablapp<>", dyn);
        print_row("nlopt CRS2_LM", nlopt);
        std::println("  ratio fixed/nlopt: {:.1f}x wall, {:.1f}x evals",
            fixed.wall_us / nlopt.wall_us, double(fixed.evals) / nlopt.evals);
        std::println("  ratio dyn/nlopt:   {:.1f}x wall, {:.1f}x evals",
            dyn.wall_us / nlopt.wall_us, double(dyn.evals) / nlopt.evals);
    }

    std::println("\nProfile with:");
    std::println("  perf stat ./micro_cmaes");
    std::println("  perf record -F 99999 -g -- ./micro_cmaes");
    std::println("  perf report --stdio --percent-limit=1.0");
}
