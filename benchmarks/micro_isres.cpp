// Micro-benchmark: nablapp ISRES vs NLopt GN_ISRES on global constrained problems.
//
// ISRES is a global stochastic optimizer using evolution strategies with
// stochastic ranking for constraint handling. Tested on multimodal
// bound-constrained problems with trivial constraints to satisfy the
// constrained_values concept.
//
// Reference: Runarsson & Yao (2005), "Search Biases in Constrained
//            Evolutionary Optimization", IEEE Trans. SMC-C.

#include "nablapp/solver/isres_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/test_functions/rastrigin.h"
#include "nablapp/test_functions/schwefel.h"

#include <Eigen/Core>

#include <nlopt.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <print>

namespace
{

struct timing
{
    double wall_us;
    double objective;
    std::uint32_t evals;
};

// Rastrigin wrapper with trivial constraints for ISRES's constrained_values requirement.
// ISRES requires bound_constrained + constrained_values even for unconstrained problems.
struct rastrigin_constrained
{
    static constexpr int problem_dimension = nablapp::dynamic_dimension;
    static constexpr nablapp::problem_class pclass =
        nablapp::problem_class::global | nablapp::problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
    {
        constexpr double two_pi = 2.0 * std::numbers::pi;
        double f = 10.0 * 2.0;
        for(int i = 0; i < 2; ++i)
            f += x[i] * x[i] - 10.0 * std::cos(two_pi * x[i]);
        return f;
    }

    void constraints(const Eigen::VectorXd&, Eigen::VectorXd& c) const
    {
        c.resize(0);
    }

    [[nodiscard]] Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Constant(2, -5.12);
    }

    [[nodiscard]] Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(2, 5.12);
    }

    [[nodiscard]] Eigen::VectorXd initial_point() const
    {
        return Eigen::VectorXd::Constant(2, 2.5);
    }
};

// Schwefel wrapper with trivial constraints for ISRES.
struct schwefel_constrained
{
    static constexpr int problem_dimension = nablapp::dynamic_dimension;
    static constexpr nablapp::problem_class pclass =
        nablapp::problem_class::global | nablapp::problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
    {
        double s = 0.0;
        for(int i = 0; i < 2; ++i)
            s += x[i] * std::sin(std::sqrt(std::abs(x[i])));
        return 418.9829 * 2.0 - s;
    }

    void constraints(const Eigen::VectorXd&, Eigen::VectorXd& c) const
    {
        c.resize(0);
    }

    [[nodiscard]] Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Constant(2, -500.0);
    }

    [[nodiscard]] Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(2, 500.0);
    }

    [[nodiscard]] Eigen::VectorXd initial_point() const
    {
        return Eigen::VectorXd::Constant(2, -200.0);
    }
};

// NLopt callback for Rastrigin.
double nlopt_rastrigin(unsigned n, const double* x, double*, void*)
{
    constexpr double two_pi = 2.0 * 3.14159265358979323846;
    double f = 10.0 * static_cast<double>(n);
    for(unsigned i = 0; i < n; ++i)
        f += x[i] * x[i] - 10.0 * std::cos(two_pi * x[i]);
    return f;
}

// NLopt callback for Schwefel.
double nlopt_schwefel(unsigned n, const double* x, double*, void*)
{
    double s = 0.0;
    for(unsigned i = 0; i < n; ++i)
        s += x[i] * std::sin(std::sqrt(std::abs(x[i])));
    return 418.9829 * static_cast<double>(n) - s;
}

template <typename Problem>
timing bench_nablapp(const Problem& problem, std::uint32_t reps)
{
    auto x0 = problem.initial_point();
    nablapp::solver_options opts;
    opts.max_iterations = 50000;
    opts.set_objective_threshold(1e-6);
    opts.set_step_threshold(1e-12);

    // Set seed for reproducibility.
    nablapp::isres_policy<> policy;
    policy.options.seed = 42;

    // Warmup.
    {
        nablapp::basic_solver solver{policy, problem, x0, opts};
        solver.solve();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t iters = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nablapp::basic_solver solver{policy, problem, x0, opts};
        auto result = solver.solve();
        fval = result.objective_value;
        iters = result.iterations;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, iters};
}

timing bench_nlopt_rastrigin(std::uint32_t reps)
{
    {
        nlopt::opt opt(nlopt::GN_ISRES, 2);
        opt.set_min_objective(nlopt_rastrigin, nullptr);
        opt.set_lower_bounds({-5.12, -5.12});
        opt.set_upper_bounds({5.12, 5.12});
        opt.set_maxeval(50000);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        opt.set_population(0);
        std::vector<double> x = {2.5, 2.5};
        double fval;
        opt.optimize(x, fval);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t evals = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nlopt::opt opt(nlopt::GN_ISRES, 2);
        opt.set_min_objective(nlopt_rastrigin, nullptr);
        opt.set_lower_bounds({-5.12, -5.12});
        opt.set_upper_bounds({5.12, 5.12});
        opt.set_maxeval(50000);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        opt.set_population(0);
        std::vector<double> x = {2.5, 2.5};
        opt.optimize(x, fval);
        evals = static_cast<std::uint32_t>(opt.get_numevals());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, evals};
}

timing bench_nlopt_schwefel(std::uint32_t reps)
{
    {
        nlopt::opt opt(nlopt::GN_ISRES, 2);
        opt.set_min_objective(nlopt_schwefel, nullptr);
        opt.set_lower_bounds({-500.0, -500.0});
        opt.set_upper_bounds({500.0, 500.0});
        opt.set_maxeval(50000);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        opt.set_population(0);
        std::vector<double> x = {-200.0, -200.0};
        double fval;
        opt.optimize(x, fval);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t evals = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nlopt::opt opt(nlopt::GN_ISRES, 2);
        opt.set_min_objective(nlopt_schwefel, nullptr);
        opt.set_lower_bounds({-500.0, -500.0});
        opt.set_upper_bounds({500.0, 500.0});
        opt.set_maxeval(50000);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        opt.set_population(0);
        std::vector<double> x = {-200.0, -200.0};
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
    constexpr std::uint32_t reps = 10;
    std::println("ISRES micro-benchmark, {} repetitions each\n", reps);
    std::println("  {:>12s}  {:>10s}  {:>10s}  {:>12s}", "solver", "wall (us)", "evals", "objective");

    // Rastrigin
    {
        std::println("\n--- Rastrigin (global, n=2, f*=0) ---");
        auto nab  = bench_nablapp(rastrigin_constrained{}, reps);
        auto nlop = bench_nlopt_rastrigin(reps);
        print_row("nablapp", nab);
        print_row("nlopt", nlop);
        std::println("  ratio nablapp/nlopt: {:.1f}x wall, {:.1f}x evals",
            nab.wall_us / nlop.wall_us, double(nab.evals) / nlop.evals);
    }

    // Schwefel
    {
        std::println("\n--- Schwefel (global, n=2, f*~0) ---");
        auto nab  = bench_nablapp(schwefel_constrained{}, reps);
        auto nlop = bench_nlopt_schwefel(reps);
        print_row("nablapp", nab);
        print_row("nlopt", nlop);
        std::println("  ratio nablapp/nlopt: {:.1f}x wall, {:.1f}x evals",
            nab.wall_us / nlop.wall_us, double(nab.evals) / nlop.evals);
    }
}
