// Micro-benchmark: nablapp L-BFGS-B vs NLopt L-BFGS on a single problem.
//
// Build:
//   g++ -std=c++23 -O3 -march=native -fno-math-errno -fno-trapping-math \
//       -I ../lib/nablapp/include -I <eigen-path> \
//       micro_lbfgsb.cpp -lnlopt -o micro_lbfgsb
//
// Or via CMake target (added below).

#include "nablapp/solver/lbfgsb_policy.h"
#include "nablapp/solver/basic_solver.h"

#include <Eigen/Core>

#include <nlopt.hpp>

#include <chrono>
#include <cstdint>
#include <print>

namespace
{

// Rosenbrock 2D — same problem for both libraries.
struct rosenbrock
{
    static constexpr int problem_dimension = nablapp::dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        double t1 = 1.0 - x[0];
        double t2 = x[1] - x[0] * x[0];
        return t1 * t1 + 100.0 * t2 * t2;
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        g[1] = 200.0 * (x[1] - x[0] * x[0]);
    }

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd{{-5.0, -5.0}}; }
    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd{{5.0, 5.0}}; }
};

// NLopt callback.
double nlopt_rosenbrock(unsigned, const double* x, double* grad, void*)
{
    double t1 = 1.0 - x[0];
    double t2 = x[1] - x[0] * x[0];
    if(grad)
    {
        grad[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        grad[1] = 200.0 * (x[1] - x[0] * x[0]);
    }
    return t1 * t1 + 100.0 * t2 * t2;
}

struct timing
{
    double wall_us;
    double objective;
    std::uint32_t evals;
};

timing bench_nablapp(std::uint32_t reps)
{
    rosenbrock problem;
    Eigen::VectorXd x0{{-1.2, 1.0}};
    nablapp::solver_options opts;
    opts.max_iterations = 1000;
    opts.set_gradient_threshold(1e-12);
    opts.set_objective_threshold(1e-14);
    opts.set_step_threshold(1e-14);

    // Warmup.
    {
        nablapp::basic_solver solver{nablapp::lbfgsb_policy{}, problem, x0, opts};
        solver.solve();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t iters = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nablapp::basic_solver solver{nablapp::lbfgsb_policy{}, problem, x0, opts};
        auto result = solver.solve();
        fval = result.objective_value;
        iters = result.iterations;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, iters};
}

timing bench_nlopt(std::uint32_t reps)
{
    // Warmup.
    {
        nlopt::opt opt(nlopt::LD_LBFGS, 2);
        opt.set_min_objective(nlopt_rosenbrock, nullptr);
        opt.set_lower_bounds({-5.0, -5.0});
        opt.set_upper_bounds({5.0, 5.0});
        opt.set_maxeval(1000);
        opt.set_ftol_rel(1e-14);
        opt.set_xtol_rel(1e-14);
        std::vector<double> x = {-1.2, 1.0};
        double fval;
        opt.optimize(x, fval);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t evals = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nlopt::opt opt(nlopt::LD_LBFGS, 2);
        opt.set_min_objective(nlopt_rosenbrock, nullptr);
        opt.set_lower_bounds({-5.0, -5.0});
        opt.set_upper_bounds({5.0, 5.0});
        opt.set_maxeval(1000);
        opt.set_ftol_rel(1e-14);
        opt.set_xtol_rel(1e-14);
        std::vector<double> x = {-1.2, 1.0};
        opt.optimize(x, fval);
        evals = static_cast<std::uint32_t>(opt.get_numevals());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, evals};
}

}

int main()
{
    constexpr std::uint32_t reps = 10000;

    std::println("Rosenbrock 2D, {} repetitions each\n", reps);

    auto na = bench_nablapp(reps);
    auto nl = bench_nlopt(reps);

    std::println("  {:>12s}  {:>10s}  {:>10s}  {:>12s}", "solver", "wall (us)", "evals", "objective");
    std::println("  {:>12s}  {:10.2f}  {:10d}  {:.6e}", "nablapp", na.wall_us, na.evals, na.objective);
    std::println("  {:>12s}  {:10.2f}  {:10d}  {:.6e}", "nlopt", nl.wall_us, nl.evals, nl.objective);
    std::println("\n  ratio (nablapp/nlopt): {:.1f}x", na.wall_us / nl.wall_us);

    std::println("\nNow profile with:");
    std::println("  perf record -F 99999 -g -- ./micro_lbfgsb");
    std::println("  perf report --stdio --percent-limit=1.0");
}
