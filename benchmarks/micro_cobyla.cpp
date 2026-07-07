// Micro-benchmark: argmin COBYLA vs NLopt COBYLA on constrained HS problems.
//
// COBYLA is a derivative-free constrained optimizer using trust-region
// linear models. Benchmarked on inequality-constrained problems that
// match its design space.
//
// Reference: Powell, M. J. D. (1994) "A direct search optimization
//            method that models the objective and constraint functions
//            by linear interpolation."

#include "argmin/solver/cobyla_policy.h"
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

// Dynamic-dimension HS024 wrapper for cobyla_policy<>.
// HS024: 2D, 3 inequality constraints.
//   min  k*(x0-3)^2*x1^3, k=1/(27*sqrt(3))
//   s.t. x0/sqrt(3) - x1 >= 0
//        x0 + sqrt(3)*x1 >= 0
//        6 - x0 - sqrt(3)*x1 >= 0
//        x0, x1 >= 0
//   x0 = (1, 0.5), f* = -1.
struct hs024_dynamic
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;
    static constexpr argmin::problem_class pclass =
        argmin::problem_class::inequality | argmin::problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 3; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
    {
        double k = 1.0 / (27.0 * std::sqrt(3.0));
        double t = (x[0] - 3.0) * (x[0] - 3.0) - 9.0;
        return k * t * x[1] * x[1] * x[1];
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        double s3 = std::sqrt(3.0);
        c.resize(3);
        c[0] = x[0] / s3 - x[1];
        c[1] = x[0] + s3 * x[1];
        c[2] = 6.0 - x[0] - s3 * x[1];
    }

    [[nodiscard]] Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Zero(2);
    }

    [[nodiscard]] Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(2, std::numeric_limits<double>::infinity());
    }

    [[nodiscard]] Eigen::VectorXd initial_point() const
    {
        return Eigen::VectorXd{{1.0, 0.5}};
    }
};

// Dynamic-dimension HS076 wrapper for cobyla_policy<>.
// HS076: 4D, 3 inequality constraints, box bounds x >= 0.
//   f* = -4.6818...
struct hs076_dynamic
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;
    static constexpr argmin::problem_class pclass =
        argmin::problem_class::inequality | argmin::problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return 4; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 3; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
    {
        return x[0] * x[0] + 0.5 * x[1] * x[1]
               + x[2] * x[2] + 0.5 * x[3] * x[3]
               - x[0] * x[2] + x[2] * x[3]
               - x[0] - 3.0 * x[1] + x[2] - x[3];
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(3);
        c[0] = 5.0 - (x[0] + 2.0 * x[1] + x[2] + x[3]);
        c[1] = 4.0 - (3.0 * x[0] + x[1] + 2.0 * x[2] - x[3]);
        c[2] = x[1] + 4.0 * x[2] - 1.5;
    }

    [[nodiscard]] Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Zero(4);
    }

    [[nodiscard]] Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(4, std::numeric_limits<double>::infinity());
    }

    [[nodiscard]] Eigen::VectorXd initial_point() const
    {
        return Eigen::VectorXd::Constant(4, 0.5);
    }
};

// NLopt callback for HS024 objective.
double nlopt_hs024_obj(unsigned, const double* x, double*, void*)
{
    double k = 1.0 / (27.0 * std::sqrt(3.0));
    double t = (x[0] - 3.0) * (x[0] - 3.0) - 9.0;
    return k * t * x[1] * x[1] * x[1];
}

// NLopt inequality: c(x) <= 0.
// HS024 c0: x0/sqrt(3) - x1 >= 0 => -(x0/sqrt(3) - x1) <= 0
double nlopt_hs024_ineq0(unsigned, const double* x, double*, void*)
{
    return -(x[0] / std::sqrt(3.0) - x[1]);
}

double nlopt_hs024_ineq1(unsigned, const double* x, double*, void*)
{
    return -(x[0] + std::sqrt(3.0) * x[1]);
}

double nlopt_hs024_ineq2(unsigned, const double* x, double*, void*)
{
    return -(6.0 - x[0] - std::sqrt(3.0) * x[1]);
}

// NLopt callback for HS076 objective.
double nlopt_hs076_obj(unsigned, const double* x, double*, void*)
{
    return x[0] * x[0] + 0.5 * x[1] * x[1]
           + x[2] * x[2] + 0.5 * x[3] * x[3]
           - x[0] * x[2] + x[2] * x[3]
           - x[0] - 3.0 * x[1] + x[2] - x[3];
}

// NLopt HS076 inequalities (negated for c <= 0 convention).
double nlopt_hs076_ineq0(unsigned, const double* x, double*, void*)
{
    return (x[0] + 2.0 * x[1] + x[2] + x[3]) - 5.0;
}

double nlopt_hs076_ineq1(unsigned, const double* x, double*, void*)
{
    return (3.0 * x[0] + x[1] + 2.0 * x[2] - x[3]) - 4.0;
}

double nlopt_hs076_ineq2(unsigned, const double* x, double*, void*)
{
    return -(x[1] + 4.0 * x[2] - 1.5);
}

template <typename Problem>
timing bench_argmin(const Problem& problem, std::uint32_t reps)
{
    auto x0 = problem.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 10000;
    opts.set_objective_threshold(1e-8);
    opts.set_step_threshold(1e-12);

    // Warmup.
    {
        argmin::step_budget_solver solver{argmin::cobyla_policy{}, problem, x0, opts};
        solver.solve();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t iters = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        argmin::step_budget_solver solver{argmin::cobyla_policy{}, problem, x0, opts};
        auto result = solver.solve();
        fval = result.objective_value;
        iters = result.iterations;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, iters};
}

timing bench_nlopt_hs024(std::uint32_t reps)
{
    {
        nlopt::opt opt(nlopt::LN_COBYLA, 2);
        opt.set_min_objective(nlopt_hs024_obj, nullptr);
        opt.set_lower_bounds({0.0, 0.0});
        opt.set_upper_bounds({1e20, 1e20});
        opt.add_inequality_constraint(nlopt_hs024_ineq0, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs024_ineq1, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs024_ineq2, nullptr, 1e-10);
        opt.set_maxeval(10000);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        std::vector<double> x = {1.0, 0.5};
        double fval;
        opt.optimize(x, fval);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t evals = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nlopt::opt opt(nlopt::LN_COBYLA, 2);
        opt.set_min_objective(nlopt_hs024_obj, nullptr);
        opt.set_lower_bounds({0.0, 0.0});
        opt.set_upper_bounds({1e20, 1e20});
        opt.add_inequality_constraint(nlopt_hs024_ineq0, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs024_ineq1, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs024_ineq2, nullptr, 1e-10);
        opt.set_maxeval(10000);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        std::vector<double> x = {1.0, 0.5};
        opt.optimize(x, fval);
        evals = static_cast<std::uint32_t>(opt.get_numevals());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, evals};
}

timing bench_nlopt_hs076(std::uint32_t reps)
{
    {
        nlopt::opt opt(nlopt::LN_COBYLA, 4);
        opt.set_min_objective(nlopt_hs076_obj, nullptr);
        opt.set_lower_bounds({0.0, 0.0, 0.0, 0.0});
        opt.set_upper_bounds({1e20, 1e20, 1e20, 1e20});
        opt.add_inequality_constraint(nlopt_hs076_ineq0, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs076_ineq1, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs076_ineq2, nullptr, 1e-10);
        opt.set_maxeval(10000);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        std::vector<double> x = {0.5, 0.5, 0.5, 0.5};
        double fval;
        opt.optimize(x, fval);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t evals = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nlopt::opt opt(nlopt::LN_COBYLA, 4);
        opt.set_min_objective(nlopt_hs076_obj, nullptr);
        opt.set_lower_bounds({0.0, 0.0, 0.0, 0.0});
        opt.set_upper_bounds({1e20, 1e20, 1e20, 1e20});
        opt.add_inequality_constraint(nlopt_hs076_ineq0, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs076_ineq1, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs076_ineq2, nullptr, 1e-10);
        opt.set_maxeval(10000);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        std::vector<double> x = {0.5, 0.5, 0.5, 0.5};
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
    constexpr std::uint32_t reps = 200;
    std::println("COBYLA micro-benchmark, {} repetitions each\n", reps);
    std::println("  {:>12s}  {:>10s}  {:>10s}  {:>12s}", "solver", "wall (us)", "evals", "objective");

    // HS024
    {
        std::println("\n--- HS024 (inequality, n=2, f*=-1) ---");
        auto nab  = bench_argmin(hs024_dynamic{}, reps);
        auto nlop = bench_nlopt_hs024(reps);
        print_row("argmin", nab);
        print_row("nlopt", nlop);
        std::println("  ratio argmin/nlopt: {:.1f}x wall, {:.1f}x evals",
            nab.wall_us / nlop.wall_us, double(nab.evals) / nlop.evals);
    }

    // HS076
    {
        std::println("\n--- HS076 (inequality + box, n=4, f*=-4.68) ---");
        auto nab  = bench_argmin(hs076_dynamic{}, reps);
        auto nlop = bench_nlopt_hs076(reps);
        print_row("argmin", nab);
        print_row("nlopt", nlop);
        std::println("  ratio argmin/nlopt: {:.1f}x wall, {:.1f}x evals",
            nab.wall_us / nlop.wall_us, double(nab.evals) / nlop.evals);
    }
}
