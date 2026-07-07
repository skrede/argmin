// Micro-benchmark: argmin byrd_lbfgsb vs NLopt LD_LBFGS on bounded Rosenbrock.
//
// byrd_lbfgsb_policy is structurally identical to lbfgsb_policy but with
// Armijo backtracking (instead of Strong Wolfe) and a shorter curvature
// history depth (5 pairs instead of 10) motivated by non-convex
// landscapes. Both bench configurations run on the same 2D bounded
// Rosenbrock used by micro_lbfgsb so the pair of files produces
// directly comparable timings.
//
// Reference: Byrd, Lu, Nocedal, Zhu (1995) "A Limited Memory Algorithm
//            for Bound Constrained Optimization", SIAM J. Sci. Comput.
//            16(5), pp. 1190-1208.

#ifdef ARGMIN_BENCH_TRACE_ALLOC
#include "argmin/detail/bench/alloc_counter.h"
#endif

#include "argmin/solver/byrd_lbfgsb_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/test_functions/more_garbow_hillstrom.h"

#include <Eigen/Core>

#include <nlopt.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <print>

namespace
{

// Rosenbrock 2D -- same problem for both libraries. Wide bounds
// [-5, 5]^2 let the trajectory run without hitting a bound, exercising
// the all-free (two-loop recursion) fast path.
struct rosenbrock
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

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

// Bounded Rosenbrock 2D with tight bounds that bind during the walk.
// Starting from x0=(-0.9, -0.9) with bounds [-1, 1]^2 exercises the
// multi-breakpoint GCP branch (same pattern as micro_lbfgsb).
struct bounded_rosenbrock
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

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

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd{{-1.0, -1.0}}; }
    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd{{1.0, 1.0}}; }
};

// NLopt callback. Shared by both bounds cases; the bounds difference is
// applied on the nlopt::opt instance, not here.
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

timing bench_argmin(std::uint32_t reps)
{
    rosenbrock problem;
    Eigen::VectorXd x0{{-1.2, 1.0}};
    argmin::solver_options opts;
    opts.max_iterations = 1000;
    opts.set_gradient_threshold(1e-12);
    opts.set_objective_threshold(1e-14);
    opts.set_step_threshold(1e-14);

    // Warmup.
    {
        argmin::step_budget_solver solver{argmin::byrd_lbfgsb_policy{}, problem, x0, opts};
        solver.solve();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t iters = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        argmin::step_budget_solver solver{argmin::byrd_lbfgsb_policy{}, problem, x0, opts};
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

timing bench_argmin_bounded(std::uint32_t reps)
{
    bounded_rosenbrock problem;
    Eigen::VectorXd x0{{-0.9, -0.9}};
    argmin::solver_options opts;
    opts.max_iterations = 1000;
    opts.set_gradient_threshold(1e-12);
    opts.set_objective_threshold(1e-14);
    opts.set_step_threshold(1e-14);

    // Warmup.
    {
        argmin::step_budget_solver solver{argmin::byrd_lbfgsb_policy{}, problem, x0, opts};
        solver.solve();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t iters = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        argmin::step_budget_solver solver{argmin::byrd_lbfgsb_policy{}, problem, x0, opts};
        auto result = solver.solve();
        fval = result.objective_value;
        iters = result.iterations;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, iters};
}

timing bench_nlopt_bounded(std::uint32_t reps)
{
    // Warmup.
    {
        nlopt::opt opt(nlopt::LD_LBFGS, 2);
        opt.set_min_objective(nlopt_rosenbrock, nullptr);
        opt.set_lower_bounds({-1.0, -1.0});
        opt.set_upper_bounds({1.0, 1.0});
        opt.set_maxeval(1000);
        opt.set_ftol_rel(1e-14);
        opt.set_xtol_rel(1e-14);
        std::vector<double> x = {-0.9, -0.9};
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
        opt.set_lower_bounds({-1.0, -1.0});
        opt.set_upper_bounds({1.0, 1.0});
        opt.set_maxeval(1000);
        opt.set_ftol_rel(1e-14);
        opt.set_xtol_rel(1e-14);
        std::vector<double> x = {-0.9, -0.9};
        opt.optimize(x, fval);
        evals = static_cast<std::uint32_t>(opt.get_numevals());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, evals};
}

// kkt_residual regression probe.
//
// byrd_lbfgsb_policy populates step_result::kkt_residual via
// detail::kkt_residual_bound on every step (projected-gradient
// infinity-norm). This probe runs byrd_lbfgsb on the bounded
// Rosenbrock problem via step(), confirms the observed step_result
// carries a populated, non-negative kkt_residual, and prints the
// value for telemetry parity with the existing benchmark output.
// Failure prints FAIL and reports through the caller's return code.
bool probe_kkt_residual()
{
    bounded_rosenbrock problem;
    Eigen::VectorXd x0{{-0.9, -0.9}};
    argmin::solver_options opts;
    opts.max_iterations = 40;
    opts.set_gradient_threshold(1e-12);
    opts.set_objective_threshold(1e-14);
    opts.set_step_threshold(1e-14);

    argmin::step_budget_solver solver{argmin::byrd_lbfgsb_policy{}, problem, x0, opts};

    argmin::step_result<double> last{};
    for(std::uint32_t i = 0; i < opts.max_iterations; ++i)
    {
        last = solver.step();
        if(last.policy_status)
            break;
    }

    if(!last.kkt_residual.has_value())
    {
        std::println(stderr, "FAIL: kkt_residual not populated (byrd_lbfgsb)");
        return false;
    }
    if(last.kkt_residual.value() < 0.0)
    {
        std::println(stderr, "FAIL: kkt_residual is negative: {}",
                     last.kkt_residual.value());
        return false;
    }
    std::println("  byrd_lbfgsb bounded Rosenbrock kkt_residual: {:.6e} "
                 "(gradient_norm: {:.6e})",
                 last.kkt_residual.value(), last.gradient_norm);
    return true;
}

// Phase 31.1 regression probe: byrd_lbfgsb on brown_badly_scaled must
// terminate with solver_status::roundoff_limited in under 30 iters
// (D-D1). Pre-fix: null-step path populated kkt_residual but left
// policy_status unset, leading to silent runs to max_iterations.
//
// Reference: Byrd, Lu, Nocedal, Zhu 1995 Algorithm CP; N&W 2e
//            Section 3.5 (roundoff limitation in line search).
bool probe_regression_brown_badly_scaled()
{
    argmin::brown_badly_scaled<> p;
    Eigen::Vector<double, argmin::brown_badly_scaled<>::problem_dimension> x0 =
        p.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 10000;
    opts.set_gradient_threshold(1e-12);
    opts.set_objective_threshold(1e-14);
    opts.set_step_threshold(1e-14);

    argmin::step_budget_solver solver{
        argmin::byrd_lbfgsb_policy<argmin::brown_badly_scaled<>::problem_dimension>{},
        p, x0, opts};
    argmin::step_result<double> last{};
    std::uint32_t iters = 0;
    for(std::uint32_t i = 0; i < opts.max_iterations; ++i)
    {
        last = solver.step();
        ++iters;
        if(last.policy_status)
            break;
    }

    const argmin::solver_status policy_status =
        last.policy_status.value_or(argmin::solver_status::running);
    const bool status_ok = policy_status == argmin::solver_status::roundoff_limited;
    const bool iter_ok = iters < 30;
    const bool objective_ok =
        std::abs(last.objective_value - 6.627535934050483e-28) < 1e-27;
    const bool ok = status_ok && iter_ok && objective_ok;
    if(!ok)
        std::println(stderr,
                     "FAIL: byrd_lbfgsb brown_badly_scaled status={} iters={} f={:.6e}",
                     static_cast<int>(policy_status),
                     iters,
                     last.objective_value);
    std::println("  byrd_lbfgsb brown_badly_scaled: iters={} status={} f={:.6e}",
                 iters,
                 static_cast<int>(policy_status),
                 last.objective_value);
    return ok;
}

}

#ifdef ARGMIN_BENCH_TRACE_ALLOC
namespace
{

// Fixed-dimension wide-bounds Rosenbrock for the allocation gate (see
// micro_lbfgsb.cpp for the fixed-dimension / pre-allocated-reset rationale).
struct rosenbrock_fixed
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        const double t1 = 1.0 - x[0];
        const double t2 = x[1] - x[0] * x[0];
        return t1 * t1 + 100.0 * t2 * t2;
    }

    void gradient(const Eigen::Vector<double, 2>& x,
                  Eigen::Vector<double, 2>& g) const
    {
        g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        g[1] = 200.0 * (x[1] - x[0] * x[0]);
    }

    Eigen::Vector<double, 2> lower_bounds() const
    {
        return Eigen::Vector<double, 2>{-5.0, -5.0};
    }

    Eigen::Vector<double, 2> upper_bounds() const
    {
        return Eigen::Vector<double, 2>{5.0, 5.0};
    }
};

}

// byrd_lbfgsb shares the L-BFGS-B all-free hot path and is expected
// allocation-free on the steady-state loop and reset(). Held to the
// zero-allocation gate immediately (min_per_step = 0).
int argmin_alloc_trace_probe()
{
    rosenbrock_fixed problem;
    Eigen::Vector<double, 2> x0{-1.2, 1.0};
    const Eigen::VectorXd x0_reset = x0;
    argmin::solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-10);

    argmin::step_budget_solver solver{argmin::byrd_lbfgsb_policy<2>{}, problem, x0, opts};

    solver.step();
    solver.step();

    constexpr std::size_t hot_steps = 10;
    argmin::detail::bench::reset_alloc_count();
    argmin::detail::bench::arm_alloc_trace();
    for(std::size_t i = 0; i < hot_steps; ++i)
        solver.step();
    solver.reset(x0_reset);
    for(std::size_t i = 0; i < hot_steps; ++i)
        solver.step();
    argmin::detail::bench::disarm_alloc_trace();

    return argmin::detail::bench::evaluate_gate("byrd_lbfgsb", 2 * hot_steps, 0);
}
#endif

#ifndef ARGMIN_BENCH_TRACE_ALLOC
int main()
{
    constexpr std::uint32_t reps = 10000;

    if(!probe_kkt_residual())
        return 1;
    if(!probe_regression_brown_badly_scaled())
        return 1;

    std::println("Rosenbrock 2D (wide bounds [-5,5]^2, all-free fast path), "
                 "{} repetitions each\n", reps);

    auto na = bench_argmin(reps);
    auto nl = bench_nlopt(reps);

    std::println("  {:>12s}  {:>10s}  {:>10s}  {:>12s}", "solver", "wall (us)", "evals", "objective");
    std::println("  {:>12s}  {:10.2f}  {:10d}  {:.6e}", "argmin", na.wall_us, na.evals, na.objective);
    std::println("  {:>12s}  {:10.2f}  {:10d}  {:.6e}", "nlopt", nl.wall_us, nl.evals, nl.objective);
    std::println("\n  ratio (argmin/nlopt): {:.1f}x", na.wall_us / nl.wall_us);

    std::println("\nBounded Rosenbrock 2D (tight bounds [-1,1]^2, multi-breakpoint GCP branch), "
                 "{} repetitions each\n", reps);

    auto nab = bench_argmin_bounded(reps);
    auto nlb = bench_nlopt_bounded(reps);

    std::println("  {:>12s}  {:>10s}  {:>10s}  {:>12s}", "solver", "wall (us)", "evals", "objective");
    std::println("  {:>12s}  {:10.2f}  {:10d}  {:.6e}", "argmin", nab.wall_us, nab.evals, nab.objective);
    std::println("  {:>12s}  {:10.2f}  {:10d}  {:.6e}", "nlopt", nlb.wall_us, nlb.evals, nlb.objective);
    std::println("\n  ratio (argmin/nlopt): {:.1f}x", nab.wall_us / nlb.wall_us);

    std::println("\nNow profile with:");
    std::println("  perf record -F 99999 -g -- ./micro_byrd_lbfgsb");
    std::println("  perf report --stdio --percent-limit=1.0");
}
#endif
