// Micro-benchmark: argmin L-BFGS-B vs NLopt L-BFGS on a single problem.
//
// Build:
//   g++ -std=c++23 -O3 -march=native -fno-math-errno -fno-trapping-math \
//       -I ../lib/argmin/include -I <eigen-path> \
//       micro_lbfgsb.cpp -lnlopt -o micro_lbfgsb
//
// Or via CMake target (added below).

#ifdef ARGMIN_BENCH_TRACE_ALLOC
#include "argmin/detail/diagnostics/alloc_counter.h"
#endif

#include "argmin/solver/lbfgsb_policy.h"
#include "argmin/solver/step_budget_solver.h"

#include <Eigen/Core>

#include <nlopt.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include "bench_print.h"

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
        argmin::step_budget_solver solver{argmin::lbfgsb_policy{}, problem, x0, opts};
        solver.solve();
    }

    auto t0 = std::chrono::steady_clock::now();
    double fval = 0.0;
    std::uint32_t iters = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        argmin::step_budget_solver solver{argmin::lbfgsb_policy{}, problem, x0, opts};
        auto result = solver.solve();
        fval = result.objective_value;
        iters = result.iterations;
    }
    auto t1 = std::chrono::steady_clock::now();
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

    auto t0 = std::chrono::steady_clock::now();
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
    auto t1 = std::chrono::steady_clock::now();
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
        argmin::step_budget_solver solver{argmin::lbfgsb_policy{}, problem, x0, opts};
        solver.solve();
    }

    auto t0 = std::chrono::steady_clock::now();
    double fval = 0.0;
    std::uint32_t iters = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        argmin::step_budget_solver solver{argmin::lbfgsb_policy{}, problem, x0, opts};
        auto result = solver.solve();
        fval = result.objective_value;
        iters = result.iterations;
    }
    auto t1 = std::chrono::steady_clock::now();
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

    auto t0 = std::chrono::steady_clock::now();
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
    auto t1 = std::chrono::steady_clock::now();
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

    argmin::step_budget_solver solver{argmin::lbfgsb_policy{}, problem, x0, opts};

    argmin::step_result<double> last{};
    for(std::uint32_t i = 0; i < opts.max_iterations; ++i)
    {
        last = solver.step();
        if(last.policy_status)
            break;
    }

    if(!last.kkt_residual.has_value())
    {
        argmin::bench::println(stderr, "FAIL: kkt_residual not populated (lbfgsb)");
        return false;
    }
    if(last.kkt_residual.value() < 0.0)
    {
        argmin::bench::println(stderr, "FAIL: kkt_residual is negative: {}",
                     last.kkt_residual.value());
        return false;
    }
    argmin::bench::println("  lbfgsb bounded Rosenbrock kkt_residual: {:.6e} "
                 "(gradient_norm: {:.6e})",
                 last.kkt_residual.value(), last.gradient_norm);
    return true;
}

// Regression probe: lbfgsb on bounded Rosenbrock is a no-op stability check.
// The projected-gradient KKT must stay below 1e-5 at convergence.
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

    argmin::step_budget_solver solver{argmin::lbfgsb_policy{}, problem, x0, opts};
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
        argmin::bench::println(stderr,
                     "FAIL: lbfgsb bounded_rosenbrock kkt={:.6e}",
                     kkt);
    argmin::bench::println("  lbfgsb bounded Rosenbrock: f={:.6e} kkt={:.6e}",
                 last.objective_value, kkt);
    return ok;
}

}

#ifdef ARGMIN_BENCH_TRACE_ALLOC
namespace
{

// Fixed-dimension bounded Rosenbrock for the allocation gate. A compile-time
// dimension keeps step_budget_solver::reset's internal x0 copy on the stack, and a
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

// Bound-active fixture for the L-BFGS-B allocation gate.
//
// rosenbrock_fixed above keeps every variable interior, so the solver stays on
// the all-free two-loop-recursion path and never touches the generalized-
// Cauchy-point -> free-variable subspace -> reduced_hessian / multiply branch.
// A zero-allocation gate measured only on that fixture would certify "including
// the bound-active path" vacuously.
//
// This quadratic places its unconstrained minimizer OUTSIDE the box on the
// first axis (x0* = 2, upper bound 1) and interior on the second (x1* = 0.5):
// the constrained solution pins x0 to its upper bound while x1 stays free, so
// every steady-state step enters the bound-active branch with a non-empty free
// set -- exactly the code the claim must exercise.
struct bounded_quadratic_fixed
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        const double a = x[0] - 2.0;
        const double b = x[1] - 0.5;
        return a * a + b * b;
    }

    void gradient(const Eigen::Vector<double, 2>& x,
                  Eigen::Vector<double, 2>& g) const
    {
        g[0] = 2.0 * (x[0] - 2.0);
        g[1] = 2.0 * (x[1] - 0.5);
    }

    Eigen::Vector<double, 2> lower_bounds() const
    {
        return Eigen::Vector<double, 2>{0.0, 0.0};
    }

    Eigen::Vector<double, 2> upper_bounds() const
    {
        return Eigen::Vector<double, 2>{1.0, 1.0};
    }
};

// Pre-hoist witness floor for the bound-active branch. The dynamic
// reduced_hessian plus the multiply rhs/z temporaries allocate every step this
// branch runs; the floor is a non-blindness lower bound (comfortably below the
// measured Release count, comfortably above zero). Under
// ARGMIN_ALLOC_GATE_EXPECT_ZERO this floor is ignored and the same branch is
// held to the zero-allocation gate.
constexpr std::size_t lbfgsb_bound_active_witness = 4;

}

// L-BFGS-B is expected allocation-free on its steady-state hot loop and
// reset(). Two scenarios run: the wide-bounds all-free path (held to the
// zero-allocation gate immediately) and a bound-active path that exercises the
// GCP -> free-variable subspace -> reduced_hessian / multiply branch. The
// bound-active scenario carries a path-entry assertion so the gate can never
// certify the claim without actually running the branch.
int argmin_alloc_trace_probe()
{
    int rc = 0;

    // Scenario 1: wide-bounds all-free fast path -- allocation-free at HEAD and
    // must stay so.
    {
        rosenbrock_fixed problem;
        Eigen::Vector<double, 2> x0{-1.2, 1.0};
        const Eigen::VectorXd x0_reset = x0;
        argmin::solver_options opts;
        opts.max_iterations = 200;
        opts.set_gradient_threshold(1e-8);
        opts.set_objective_threshold(1e-10);
        opts.set_step_threshold(1e-10);

        argmin::step_budget_solver solver{argmin::lbfgsb_policy<2>{}, problem, x0, opts};

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

        rc |= argmin::detail::bench::evaluate_gate("lbfgsb", 2 * hot_steps, 0);
    }

    // Scenario 2: bound-active path (GCP -> free-variable subspace ->
    // reduced_hessian / multiply).
    {
        bounded_quadratic_fixed problem;
        Eigen::Vector<double, 2> x0{0.1, 0.1};
        const Eigen::VectorXd x0_reset = x0;
        argmin::solver_options opts;
        opts.max_iterations = 200;
        opts.set_gradient_threshold(1e-8);
        opts.set_objective_threshold(1e-10);
        opts.set_step_threshold(1e-10);

        argmin::step_budget_solver solver{argmin::lbfgsb_policy<2>{}, problem, x0, opts};

        solver.step();
        solver.step();

        // Path-entry observable: a bound is active exactly when the shared
        // direction predicate rejects the all-free fast path, i.e. when the
        // solver takes the GCP + subspace branch this scenario targets.
        std::size_t bound_active_steps = 0;
        auto observe_path = [&]() {
            const auto& st = solver.state();
            if(!argmin::detail::all_variables_free<double, 2>(
                   st.x, st.g, st.lower, st.upper))
                ++bound_active_steps;
        };

        constexpr std::size_t hot_steps = 10;
        argmin::detail::bench::reset_alloc_count();
        argmin::detail::bench::arm_alloc_trace();
        for(std::size_t i = 0; i < hot_steps; ++i)
        {
            solver.step();
            observe_path();
        }
        solver.reset(x0_reset);
        for(std::size_t i = 0; i < hot_steps; ++i)
        {
            solver.step();
            observe_path();
        }
        argmin::detail::bench::disarm_alloc_trace();

        if(bound_active_steps == 0)
        {
            argmin::bench::println(stderr,
                "  [alloc-gate] lbfgsb_bound_active FAIL: fixture never entered "
                "the bound-active branch (path-entry assertion) -- a zero-mode "
                "gate would certify the claim vacuously");
            return 1;
        }
        argmin::bench::println("  [alloc-gate] lbfgsb_bound_active path-entry: {}/{} armed "
                     "steps had an active bound",
                     bound_active_steps, 2 * hot_steps);

        rc |= argmin::detail::bench::evaluate_gate(
            "lbfgsb_bound_active", 2 * hot_steps, lbfgsb_bound_active_witness);
    }

    return rc;
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

    argmin::bench::println("Rosenbrock 2D (wide bounds [-5,5]^2, all-free fast path), {} repetitions each\n", reps);

    auto na = bench_argmin(reps);
    auto nl = bench_nlopt(reps);

    argmin::bench::println("  {:>12s}  {:>10s}  {:>10s}  {:>12s}", "solver", "wall (us)", "evals", "objective");
    argmin::bench::println("  {:>12s}  {:10.2f}  {:10d}  {:.6e}", "argmin", na.wall_us, na.evals, na.objective);
    argmin::bench::println("  {:>12s}  {:10.2f}  {:10d}  {:.6e}", "nlopt", nl.wall_us, nl.evals, nl.objective);
    argmin::bench::println("\n  ratio (argmin/nlopt): {:.1f}x", na.wall_us / nl.wall_us);

    argmin::bench::println("\nBounded Rosenbrock 2D (tight bounds [-1,1]^2, multi-breakpoint GCP branch), "
                 "{} repetitions each\n", reps);

    auto nab = bench_argmin_bounded(reps);
    auto nlb = bench_nlopt_bounded(reps);

    argmin::bench::println("  {:>12s}  {:>10s}  {:>10s}  {:>12s}", "solver", "wall (us)", "evals", "objective");
    argmin::bench::println("  {:>12s}  {:10.2f}  {:10d}  {:.6e}", "argmin", nab.wall_us, nab.evals, nab.objective);
    argmin::bench::println("  {:>12s}  {:10.2f}  {:10d}  {:.6e}", "nlopt", nlb.wall_us, nlb.evals, nlb.objective);
    argmin::bench::println("\n  ratio (argmin/nlopt): {:.1f}x", nab.wall_us / nlb.wall_us);

    argmin::bench::println("\nNow profile with:");
    argmin::bench::println("  perf record -F 99999 -g -- ./micro_lbfgsb");
    argmin::bench::println("  perf report --stdio --percent-limit=1.0");
}
#endif
