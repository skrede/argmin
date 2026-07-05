// Micro-benchmark: argmin L-BFGS-B vs NLopt L-BFGS on a single problem.
//
// Build:
//   g++ -std=c++23 -O3 -march=native -fno-math-errno -fno-trapping-math \
//       -I ../lib/argmin/include -I <eigen-path> \
//       micro_lbfgsb.cpp -lnlopt -o micro_lbfgsb
//
// Or via CMake target (added below).

#ifdef ARGMIN_BENCH_TRACE_ALLOC
#include "argmin/detail/bench/alloc_counter.h"
#endif

#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/basic_solver.h"

#include <Eigen/Core>

#include <nlopt.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <print>

namespace
{

// Rosenbrock 2D — same problem for both libraries.
// Bounds [-5, 5]^2 are wide enough that the trajectory from x0=(-1.2, 1.0)
// to the unconstrained optimum (1, 1) never binds; this case exercises
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
//
// Starting from x0=(-0.9, -0.9) with bounds [-1, 1]^2, the anti-gradient ray
// exits the box from the start (both coords have negative gradients from x0
// values near -1 because the Rosenbrock gradient at (-0.9, -0.9) is
// g = (-2*(1-(-0.9)) - 400*(-0.9)*(-0.9 - 0.81), 200*(-0.9 - 0.81))
//   = (-3.8 + 615.6, -342) = (611.8, -342)
// So the walk hits the coord-0 upper and coord-1 lower breakpoints in early
// iterations and exercises the multi-breakpoint GCP branch that the
// reduced-direction derivative reconstruction fix applies to.
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

// NLopt callback. Shared by both the wide-bounds and tight-bounds cases;
// the bounds difference is applied on the nlopt::opt instance, not here.
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
        argmin::basic_solver solver{argmin::lbfgsb_policy{}, problem, x0, opts};
        solver.solve();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t iters = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        argmin::basic_solver solver{argmin::lbfgsb_policy{}, problem, x0, opts};
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

// Bounded-Rosenbrock timings; exercises the multi-breakpoint GCP branch in
// cauchy_point_solver::solve that is not touched by the wide-bounds case.
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
        argmin::basic_solver solver{argmin::lbfgsb_policy{}, problem, x0, opts};
        solver.solve();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t iters = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        argmin::basic_solver solver{argmin::lbfgsb_policy{}, problem, x0, opts};
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
// lbfgsb_policy populates step_result::kkt_residual via
// detail::kkt_residual_bound on every step (projected-gradient
// infinity-norm). The probe runs lbfgsb on the bounded Rosenbrock
// problem via step(), confirms the observed step_result carries a
// populated, non-negative kkt_residual, and prints the value for
// telemetry parity. Failure prints FAIL and reports through main.
bool probe_kkt_residual()
{
    bounded_rosenbrock problem;
    Eigen::VectorXd x0{{-0.9, -0.9}};
    argmin::solver_options opts;
    opts.max_iterations = 40;
    opts.set_gradient_threshold(1e-12);
    opts.set_objective_threshold(1e-14);
    opts.set_step_threshold(1e-14);

    argmin::basic_solver solver{argmin::lbfgsb_policy{}, problem, x0, opts};

    argmin::step_result<double> last{};
    for(std::uint32_t i = 0; i < opts.max_iterations; ++i)
    {
        last = solver.step();
        if(last.policy_status)
            break;
    }

    if(!last.kkt_residual.has_value())
    {
        std::println(stderr, "FAIL: kkt_residual not populated (lbfgsb)");
        return false;
    }
    if(last.kkt_residual.value() < 0.0)
    {
        std::println(stderr, "FAIL: kkt_residual is negative: {}",
                     last.kkt_residual.value());
        return false;
    }
    std::println("  lbfgsb bounded Rosenbrock kkt_residual: {:.6e} "
                 "(gradient_norm: {:.6e})",
                 last.kkt_residual.value(), last.gradient_norm);
    return true;
}

// Phase 31.1 regression probe: lbfgsb on bounded Rosenbrock is a
// no-op stability check. lbfgsb was touched in Phase 31.1 only to
// remove the dead feasibility_gate = +inf field (D-C3); the
// kkt_residual_bound helper is unchanged per D-B3. Probe asserts
// that the projected-gradient KKT stays below 1e-5 at convergence.
//
// Reference: N&W 2e Section 16.7 (projected gradient optimality).
bool probe_regression_bounded_rosenbrock()
{
    bounded_rosenbrock problem;
    Eigen::VectorXd x0{{-0.9, -0.9}};
    argmin::solver_options opts;
    opts.max_iterations = 100;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    argmin::basic_solver solver{argmin::lbfgsb_policy{}, problem, x0, opts};
    argmin::step_result<double> last{};
    for(std::uint32_t i = 0; i < opts.max_iterations; ++i)
    {
        last = solver.step();
        if(last.policy_status)
            break;
    }

    const double kkt = last.kkt_residual.value_or(-1.0);
    const bool ok = kkt >= 0.0 && kkt < 1e-5;
    if(!ok)
        std::println(stderr,
                     "FAIL: lbfgsb bounded_rosenbrock kkt={:.6e}",
                     kkt);
    std::println("  lbfgsb bounded Rosenbrock: f={:.6e} kkt={:.6e}",
                 last.objective_value, kkt);
    return ok;
}

}

#ifdef ARGMIN_BENCH_TRACE_ALLOC
namespace
{

// Fixed-dimension bounded Rosenbrock for the allocation gate. A compile-time
// dimension keeps basic_solver::reset's internal x0 copy on the stack, and a
// pre-allocated VectorXd reset argument avoids the API's parameter copy, so
// the probe measures only genuine policy heap traffic.
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

// L-BFGS-B is expected allocation-free on its steady-state hot loop and
// reset() (the all-free two-loop recursion path). min_per_step = 0 holds it to
// the zero-allocation gate immediately: the un-blinded gate must not
// false-positive on an already-clean policy.
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

    argmin::basic_solver solver{argmin::lbfgsb_policy<2>{}, problem, x0, opts};

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

    return argmin::detail::bench::evaluate_gate("lbfgsb", 2 * hot_steps, 0);
}
#endif

#ifndef ARGMIN_BENCH_TRACE_ALLOC
int main()
{
    constexpr std::uint32_t reps = 10000;

    if(!probe_kkt_residual())
        return 1;
    if(!probe_regression_bounded_rosenbrock())
        return 1;

    std::println("Rosenbrock 2D (wide bounds [-5,5]^2, all-free fast path), {} repetitions each\n", reps);

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
    std::println("  perf record -F 99999 -g -- ./micro_lbfgsb");
    std::println("  perf report --stdio --percent-limit=1.0");
}
#endif
