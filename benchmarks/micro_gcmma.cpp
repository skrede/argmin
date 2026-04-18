// Micro-benchmark: nablapp GCMMA vs NLopt LD_MMA on inequality-constrained
// HS problems.
//
// GCMMA wraps MMA with a conservativity loop (Svanberg 2002 Section 4.2)
// that grows per-component regularizers raa_0, raa_i on non-conservative
// trials and returns a null step on max-inner exhaustion, preserving the
// paper's global convergence proof.
//
// LD_MMA is NLopt's only exposed CCSA-family algorithm. It is structurally
// equivalent to gcmma_policy for cross-bench purposes (both solve the
// separable reciprocal approximation subproblem with conservativity-based
// globalization; LD_MMA uses NLopt's quadratic CCSA variant while
// gcmma_policy uses Svanberg 2002's reciprocal variant -- same convergence
// proof family).
//
// Problems covered:
//   HS035 -- inequality + bound, n=3, f* = 1/9. Historic poster-child
//            failure mode pre-fix (reached optimum but classified stalled
//            because step_result.kkt_residual was nullopt).
//   HS076 -- inequality + bound, n=4, f* = -4.6818.
//   HS043 -- inequality-only, n=4, f* = -44. Oscillation-prone under
//            aggressive conservativity growth factors (raa_growth = 2.0).
//
// Threshold convention: NLopt-parity relative objective tolerance via
// slsqp_compatible_convergence (ftol_rel + xtol_rel). absolute ftol
// caused nablapp to exit earlier than NLopt by construction.
//
// Reference: Svanberg 2002, "A class of globally convergent optimization
//            methods based on conservative convex separable approximations",
//            SIAM J. Optim. 12(2);
//            NLopt 2.10.0 src/algs/mma/ccsa_quadratic.c (LD_MMA reference).

#include "nablapp/solver/gcmma_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/test_functions/hock_schittkowski.h"

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

// NLopt callbacks for HS035.
double nlopt_hs035_obj(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = -8.0 + 4.0 * x[0] + 2.0 * x[1] + 2.0 * x[2];
        grad[1] = -6.0 + 2.0 * x[0] + 4.0 * x[1];
        grad[2] = -4.0 + 2.0 * x[0] + 2.0 * x[2];
    }
    return 9.0 - 8.0 * x[0] - 6.0 * x[1] - 4.0 * x[2]
         + 2.0 * x[0] * x[0] + 2.0 * x[1] * x[1] + x[2] * x[2]
         + 2.0 * x[0] * x[1] + 2.0 * x[0] * x[2];
}

// HS035 inequality 0: nablapp form c0 = 3 - (x0 + x1 + 2*x2) >= 0
//                     NLopt  form      (x0 + x1 + 2*x2) - 3 <= 0.
double nlopt_hs035_ineq0(unsigned, const double* x, double* grad, void*)
{
    if(grad) { grad[0] = 1.0; grad[1] = 1.0; grad[2] = 2.0; }
    return (x[0] + x[1] + 2.0 * x[2]) - 3.0;
}

// NLopt callbacks for HS076.
double nlopt_hs076_obj(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = 2.0 * x[0] - x[2] - 1.0;
        grad[1] = x[1] - 3.0;
        grad[2] = 2.0 * x[2] - x[0] + x[3] + 1.0;
        grad[3] = x[3] + x[2] - 1.0;
    }
    return x[0] * x[0] + 0.5 * x[1] * x[1]
         + x[2] * x[2] + 0.5 * x[3] * x[3]
         - x[0] * x[2] + x[2] * x[3]
         - x[0] - 3.0 * x[1] + x[2] - x[3];
}

// HS076 inequality 0: 5 - (x0 + 2*x1 + x2 + x3) >= 0 -> (x0+2*x1+x2+x3) - 5 <= 0.
double nlopt_hs076_ineq0(unsigned, const double* x, double* grad, void*)
{
    if(grad) { grad[0] = 1.0; grad[1] = 2.0; grad[2] = 1.0; grad[3] = 1.0; }
    return (x[0] + 2.0 * x[1] + x[2] + x[3]) - 5.0;
}

// HS076 inequality 1: 4 - (3*x0 + x1 + 2*x2 - x3) >= 0.
double nlopt_hs076_ineq1(unsigned, const double* x, double* grad, void*)
{
    if(grad) { grad[0] = 3.0; grad[1] = 1.0; grad[2] = 2.0; grad[3] = -1.0; }
    return (3.0 * x[0] + x[1] + 2.0 * x[2] - x[3]) - 4.0;
}

// HS076 inequality 2: x1 + 4*x2 - 1.5 >= 0  ->  1.5 - x1 - 4*x2 <= 0.
double nlopt_hs076_ineq2(unsigned, const double* x, double* grad, void*)
{
    if(grad) { grad[0] = 0.0; grad[1] = -1.0; grad[2] = -4.0; grad[3] = 0.0; }
    return 1.5 - x[1] - 4.0 * x[2];
}

// HS043 callbacks (shared semantics with micro_mma.cpp).
double nlopt_hs043_obj(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = 2.0 * x[0] - 5.0;
        grad[1] = 2.0 * x[1] - 5.0;
        grad[2] = 4.0 * x[2] - 21.0;
        grad[3] = 2.0 * x[3] + 7.0;
    }
    return x[0] * x[0] + x[1] * x[1] + 2.0 * x[2] * x[2]
         + x[3] * x[3] - 5.0 * x[0] - 5.0 * x[1]
         - 21.0 * x[2] + 7.0 * x[3];
}

double nlopt_hs043_ineq0(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = 2.0 * x[0] + 1.0;
        grad[1] = 2.0 * x[1] - 1.0;
        grad[2] = 2.0 * x[2] + 1.0;
        grad[3] = 2.0 * x[3] - 1.0;
    }
    return (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
           + x[3] * x[3] + x[0] - x[1] + x[2] - x[3]) - 8.0;
}

double nlopt_hs043_ineq1(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = 2.0 * x[0] - 1.0;
        grad[1] = 4.0 * x[1];
        grad[2] = 2.0 * x[2];
        grad[3] = 4.0 * x[3] - 1.0;
    }
    return (x[0] * x[0] + 2.0 * x[1] * x[1]
           + x[2] * x[2] + 2.0 * x[3] * x[3] - x[0] - x[3]) - 10.0;
}

double nlopt_hs043_ineq2(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = 4.0 * x[0] + 2.0;
        grad[1] = 2.0 * x[1] - 1.0;
        grad[2] = 2.0 * x[2];
        grad[3] = -1.0;
    }
    return (2.0 * x[0] * x[0] + x[1] * x[1]
           + x[2] * x[2] + 2.0 * x[0] - x[1] - x[3]) - 5.0;
}

template <typename Problem>
timing bench_gcmma(const Problem& problem, std::uint32_t reps)
{
    auto x0 = problem.initial_point();
    // NLopt-parity convergence (ftol_rel + xtol_rel + stall) matching
    // LD_MMA's termination convention; absolute ftol exited earlier than
    // NLopt by construction.
    nablapp::solver_options<nablapp::slsqp_compatible_convergence> opts;
    opts.max_iterations = 5000;
    opts.set_objective_threshold_rel(1e-12);
    opts.set_step_threshold_rel(1e-12);

    constexpr int N = Problem::problem_dimension;

    // Warmup.
    {
        nablapp::basic_solver solver{nablapp::gcmma_policy<N>{},
                                     problem, x0, opts};
        solver.solve();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t iters = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nablapp::basic_solver solver{nablapp::gcmma_policy<N>{},
                                     problem, x0, opts};
        auto result = solver.solve();
        fval = result.objective_value;
        iters = result.iterations;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, iters};
}

timing bench_nlopt_hs035(std::uint32_t reps)
{
    {
        nlopt::opt opt(nlopt::LD_MMA, 3);
        opt.set_min_objective(nlopt_hs035_obj, nullptr);
        opt.set_lower_bounds({0.0, 0.0, 0.0});
        opt.set_upper_bounds({1e20, 1e20, 1e20});
        opt.add_inequality_constraint(nlopt_hs035_ineq0, nullptr, 1e-10);
        opt.set_maxeval(5000);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        std::vector<double> x = {0.5, 0.5, 0.5};
        double fval;
        opt.optimize(x, fval);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t evals = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nlopt::opt opt(nlopt::LD_MMA, 3);
        opt.set_min_objective(nlopt_hs035_obj, nullptr);
        opt.set_lower_bounds({0.0, 0.0, 0.0});
        opt.set_upper_bounds({1e20, 1e20, 1e20});
        opt.add_inequality_constraint(nlopt_hs035_ineq0, nullptr, 1e-10);
        opt.set_maxeval(5000);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        std::vector<double> x = {0.5, 0.5, 0.5};
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
        nlopt::opt opt(nlopt::LD_MMA, 4);
        opt.set_min_objective(nlopt_hs076_obj, nullptr);
        opt.set_lower_bounds({0.0, 0.0, 0.0, 0.0});
        opt.set_upper_bounds({1e20, 1e20, 1e20, 1e20});
        opt.add_inequality_constraint(nlopt_hs076_ineq0, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs076_ineq1, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs076_ineq2, nullptr, 1e-10);
        opt.set_maxeval(5000);
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
        nlopt::opt opt(nlopt::LD_MMA, 4);
        opt.set_min_objective(nlopt_hs076_obj, nullptr);
        opt.set_lower_bounds({0.0, 0.0, 0.0, 0.0});
        opt.set_upper_bounds({1e20, 1e20, 1e20, 1e20});
        opt.add_inequality_constraint(nlopt_hs076_ineq0, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs076_ineq1, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs076_ineq2, nullptr, 1e-10);
        opt.set_maxeval(5000);
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

timing bench_nlopt_hs043(std::uint32_t reps)
{
    {
        nlopt::opt opt(nlopt::LD_MMA, 4);
        opt.set_min_objective(nlopt_hs043_obj, nullptr);
        opt.add_inequality_constraint(nlopt_hs043_ineq0, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs043_ineq1, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs043_ineq2, nullptr, 1e-10);
        opt.set_maxeval(5000);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        std::vector<double> x = {0.0, 0.0, 0.0, 0.0};
        double fval;
        opt.optimize(x, fval);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t evals = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nlopt::opt opt(nlopt::LD_MMA, 4);
        opt.set_min_objective(nlopt_hs043_obj, nullptr);
        opt.add_inequality_constraint(nlopt_hs043_ineq0, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs043_ineq1, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs043_ineq2, nullptr, 1e-10);
        opt.set_maxeval(5000);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        std::vector<double> x = {0.0, 0.0, 0.0, 0.0};
        opt.optimize(x, fval);
        evals = static_cast<std::uint32_t>(opt.get_numevals());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, evals};
}

void print_row(std::string_view solver, const timing& t)
{
    std::println("  {:>12s}  {:10.2f}  {:10d}  {:.6e}",
                 solver, t.wall_us, t.evals, t.objective);
}

}

int main()
{
    constexpr std::uint32_t reps = 500;
    std::println("GCMMA micro-benchmark, {} repetitions each\n", reps);
    std::println("  {:>12s}  {:>10s}  {:>10s}  {:>12s}",
                 "solver", "wall (us)", "evals", "objective");

    // HS035 (f* = 1/9 = 0.1111...; historic GCMMA failure mode pre-fix).
    {
        std::println("\n--- HS035 (inequality+bound, n=3, f*=1/9) ---");
        auto nab  = bench_gcmma(nablapp::hs035<>{}, reps);
        auto nlop = bench_nlopt_hs035(reps);
        print_row("nablapp", nab);
        print_row("nlopt", nlop);
        std::println("  ratio nablapp/nlopt: {:.1f}x wall, {:.1f}x evals",
                     nab.wall_us / nlop.wall_us,
                     double(nab.evals) / nlop.evals);
    }

    // HS076 (f* = -4.6818; inequality + bound-constrained, n=4).
    {
        std::println("\n--- HS076 (inequality+bound, n=4, f*=-4.6818) ---");
        auto nab  = bench_gcmma(nablapp::hs076<>{}, reps);
        auto nlop = bench_nlopt_hs076(reps);
        print_row("nablapp", nab);
        print_row("nlopt", nlop);
        std::println("  ratio nablapp/nlopt: {:.1f}x wall, {:.1f}x evals",
                     nab.wall_us / nlop.wall_us,
                     double(nab.evals) / nlop.evals);
    }

    // HS043 (f* = -44; inequality-only, n=4; oscillation-prone under
    // aggressive raa_growth).
    {
        std::println("\n--- HS043 (inequality, n=4, f*=-44) ---");
        auto nab  = bench_gcmma(nablapp::hs043<>{}, reps);
        auto nlop = bench_nlopt_hs043(reps);
        print_row("nablapp", nab);
        print_row("nlopt", nlop);
        std::println("  ratio nablapp/nlopt: {:.1f}x wall, {:.1f}x evals",
                     nab.wall_us / nlop.wall_us,
                     double(nab.evals) / nlop.evals);
    }
}
