// Micro-benchmark: argmin Levenberg-Marquardt on Rosenbrock least-squares.
//
// lm_policy is unconstrained nonlinear least-squares with Nielsen (1999)
// adaptive damping. NLopt has no direct L-M equivalent (LD_LBFGS /
// LN_BOBYQA are not least-squares solvers), so this bench is
// argmin-only and focuses on per-step timing and the kkt_residual
// regression probe. Profile numbers pair with micro_projected_gn.cpp
// (same Rosenbrock residuals but with bound constraints) and
// micro_projected_gradient_gn.cpp for relative cost of LM vs projected
// variants.
//
// Reference: Nielsen, H. B. (1999) "Damping Parameter in Marquardt's
//            Method", IMM-REP-1999-05.
//            K&W Section 6.3-6.4 Algorithm 6.3 (Levenberg-Marquardt).
//            N&W Section 10.2-10.3 (nonlinear least-squares).

#ifdef ARGMIN_BENCH_TRACE_ALLOC
#include "argmin/detail/diagnostics/alloc_counter.h"
#include "argmin/detail/diagnostics/steady_state_driver.h"
#endif

#include "bench_micro_gate.h"

#include "argmin/solver/lm_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/test_functions/small_dense.h"

#include <Eigen/Core>

#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace
{

// Rosenbrock 2D in least-squares residual form (b=5 per project convention):
//   r_0 = 1 - x_0
//   r_1 = sqrt(5) * (x_1 - x_0^2)
//   f(x) = 0.5 * (r_0^2 + r_1^2)
// Unconstrained minimum at (1, 1), f* = 0.
struct rosenbrock_ls
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return 2; }
    int num_residuals() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        double r0 = 1.0 - x(0);
        double r1 = std::sqrt(5.0) * (x(1) - x(0) * x(0));
        return 0.5 * (r0 * r0 + r1 * r1);
    }

    void residuals(const Eigen::VectorXd& x, Eigen::VectorXd& r) const
    {
        r(0) = 1.0 - x(0);
        r(1) = std::sqrt(5.0) * (x(1) - x(0) * x(0));
    }

    void jacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const
    {
        J(0, 0) = -1.0;
        J(0, 1) = 0.0;
        J(1, 0) = -2.0 * std::sqrt(5.0) * x(0);
        J(1, 1) = std::sqrt(5.0);
    }
};

struct rosenbrock_ls_fixed
{
    static constexpr int problem_dimension = 2;

    int dimension() const { return 2; }
    int num_residuals() const { return 2; }

    double value(const Eigen::Vector<double, 2>& x) const
    {
        double r0 = 1.0 - x(0);
        double r1 = std::sqrt(5.0) * (x(1) - x(0) * x(0));
        return 0.5 * (r0 * r0 + r1 * r1);
    }

    void residuals(const Eigen::Vector<double, 2>& x,
                   Eigen::VectorXd& r) const
    {
        r(0) = 1.0 - x(0);
        r(1) = std::sqrt(5.0) * (x(1) - x(0) * x(0));
    }

    void jacobian(const Eigen::Vector<double, 2>& x,
                  Eigen::MatrixXd& J) const
    {
        J(0, 0) = -1.0;
        J(0, 1) = 0.0;
        J(1, 0) = -2.0 * std::sqrt(5.0) * x(0);
        J(1, 1) = std::sqrt(5.0);
    }

    Eigen::Vector<double, 2> initial_point() const
    {
        return Eigen::Vector<double, 2>{-1.0, 1.0};
    }
};

struct timing
{
    double wall_us;
    double objective;
    double constraint_violation;
    std::uint32_t evals;
    const char* unit;
};

timing bench_argmin(std::uint32_t reps)
{
    rosenbrock_ls problem;
    Eigen::VectorXd x0{{-1.0, 1.0}};
    argmin::solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-12);
    opts.set_objective_threshold(1e-14);
    opts.set_step_threshold(1e-14);

    // Warmup.
    {
        argmin::step_budget_solver solver{argmin::lm_policy{}, problem, x0, opts};
        solver.solve();
    }

    auto t0 = std::chrono::steady_clock::now();
    double fval = 0.0;
    std::uint32_t iters = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        argmin::step_budget_solver solver{argmin::lm_policy{}, problem, x0, opts};
        auto result = solver.solve();
        fval = result.objective_value;
        iters = result.iterations;
    }
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, 0.0, iters, "steps"};
}

// kkt_residual regression probe.
//
// lm_policy populates step_result::kkt_residual as the gradient
// infinity-norm ||J^T r||_inf on every step. Because lm_policy is
// unconstrained least-squares, the first-order KKT condition reduces
// to stationarity and this is the exact KKT residual. This probe
// runs lm_policy on Rosenbrock least-squares via step(), confirms
// the observed step_result carries a populated, non-negative
// kkt_residual, and prints the value for telemetry parity.
// Failure prints FAIL and reports through the caller's return code.
bool probe_kkt_residual()
{
    rosenbrock_ls problem;
    Eigen::VectorXd x0{{-1.0, 1.0}};
    argmin::solver_options opts;
    opts.max_iterations = 40;
    opts.set_gradient_threshold(1e-12);
    opts.set_objective_threshold(1e-14);
    opts.set_step_threshold(1e-14);

    argmin::step_budget_solver solver{argmin::lm_policy{}, problem, x0, opts};

    argmin::step_result<double> last{};
    for(std::uint32_t i = 0; i < opts.max_iterations; ++i)
    {
        last = solver.step();
        if(last.policy_status)
            break;
    }

    if(!last.kkt_residual.has_value())
    {
        argmin::bench::println(stderr, "FAIL: kkt_residual not populated (lm)");
        return false;
    }
    if(last.kkt_residual.value() < 0.0)
    {
        argmin::bench::println(stderr, "FAIL: kkt_residual is negative: {}",
                     last.kkt_residual.value());
        return false;
    }
    argmin::bench::println("  lm Rosenbrock LS kkt_residual: {:.6e} "
                 "(gradient_norm: {:.6e})",
                 last.kkt_residual.value(), last.gradient_norm);
    return true;
}

}

#ifdef ARGMIN_BENCH_TRACE_ALLOC
namespace
{

// Steady-state allocation gate: run the shared returns-data driver over a
// fixed-N Levenberg-Marquardt solve loop, then fail the bench (non-zero) if the
// armed window terminated early or observed any allocation.
template <typename Policy, typename Problem>
int gate_steady(const char* label, Policy policy, const Problem& problem)
{
    argmin::solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-12);
    opts.set_objective_threshold(1e-14);
    opts.set_step_threshold(1e-14);

    const auto r = argmin::detail::bench::measure_steady(policy, problem, opts, 10);

    std::printf("  [alloc-gate] %-18s eigen_malloc=%lu c_alloc=%lu "
                "armed_steps=%lu\n",
                label, static_cast<unsigned long>(r.eigen_malloc),
                static_cast<unsigned long>(r.c_alloc),
                static_cast<unsigned long>(r.armed_steps));

    if(r.terminated_early)
    {
        argmin::bench::println(stderr,
            "  [alloc-gate] {} FAIL: policy signaled termination inside the "
            "armed window -- not a pre-convergence steady state", label);
        return 1;
    }
    if(r.eigen_malloc != 0 || r.c_alloc != 0)
    {
        std::fprintf(stderr,
            "  [alloc-gate] %-18s FAIL: expected allocation-free, saw "
            "eigen_malloc=%lu c_alloc=%lu\n", label,
            static_cast<unsigned long>(r.eigen_malloc),
            static_cast<unsigned long>(r.c_alloc));
        return 1;
    }
    std::printf("  [alloc-gate] %-18s PASS (allocation-free)\n", label);
    return 0;
}

// Construction-window record (informational, never gates).
template <typename Policy, typename Problem>
void record_construction(const char* label, Policy policy, const Problem& problem)
{
    auto x0 = problem.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-12);
    opts.set_objective_threshold(1e-14);
    opts.set_step_threshold(1e-14);

    argmin::detail::bench::reset_alloc_count();
    argmin::detail::bench::arm_alloc_trace();
    argmin::step_budget_solver solver{policy, problem, x0, opts};
    solver.step();
    argmin::detail::bench::disarm_alloc_trace();

    std::printf("  [alloc-gate] %-18s construction+first_step alloc=%zu\n",
                label, argmin::detail::bench::read_alloc_count());
}

}

// Steady-state allocation record for Levenberg-Marquardt on the 2D Rosenbrock
// least-squares fixture and the 12D extended-Rosenbrock least-squares fixture.
int argmin_alloc_trace_probe()
{
    record_construction("lm rosenbrock2", argmin::lm_policy<2>{}, rosenbrock_ls_fixed{});

    int rc = 0;
    rc |= gate_steady("lm rosenbrock2", argmin::lm_policy<2>{},
        rosenbrock_ls_fixed{});
    rc |= gate_steady("lm sd_ls012", argmin::lm_policy<argmin::sd_ls012<>::problem_dimension>{},
        argmin::sd_ls012<>{});
    return rc;
}
#endif

#ifndef ARGMIN_BENCH_TRACE_ALLOC
int main()
{
    constexpr std::uint32_t reps = 10000;

    if(!probe_kkt_residual())
        return 1;

    argmin::bench::println("\nRosenbrock 2D LS (unconstrained, analytic Jacobian), "
                 "{} repetitions each\n", reps);

    auto na = bench_argmin(reps);

    argmin::bench::println("  {:>12s}  {:>10s}  {:>8s}  {:>10s}  {:>10s}  {:>12s}  {:>12s}",
                           "solver", "solve_us", "unit", "units", "unit_us", "objective", "cv");
    constexpr argmin::bench::micro_gate gate{0.0, 1e-10, 0.0};
    const argmin::bench::micro_observation obs{
        "argmin_lm", na.objective, na.constraint_violation};
    if(argmin::bench::observation_passes(obs, gate))
        argmin::bench::println("  {:>12s}  {:10.2f}  {:>8s}  {:10d}  {:10.2f}  {:.6e}  {:.6e}",
                               "argmin_lm",
                               na.wall_us,
                               na.unit,
                               na.evals,
                               argmin::bench::per_unit_us(na.wall_us, na.evals),
                               na.objective,
                               na.constraint_violation);
    else
        argmin::bench::print_gated_comparison("Rosenbrock LS", obs, obs, gate);

    argmin::bench::println("\nNow profile with:");
    argmin::bench::println("  perf record -F 99999 -g -- ./micro_lm");
    argmin::bench::println("  perf report --stdio --percent-limit=1.0");
}
#endif
