// Micro-benchmark: nablapp Levenberg-Marquardt on Rosenbrock least-squares.
//
// lm_policy is unconstrained nonlinear least-squares with Nielsen (1999)
// adaptive damping. NLopt has no direct L-M equivalent (LD_LBFGS /
// LN_BOBYQA are not least-squares solvers), so this bench is
// nablapp-only and focuses on per-step timing and the kkt_residual
// regression probe. Profile numbers pair with micro_projected_gn.cpp
// (same Rosenbrock residuals but with bound constraints) and
// micro_projected_gradient_gn.cpp for relative cost of LM vs projected
// variants.
//
// Reference: Nielsen, H. B. (1999) "Damping Parameter in Marquardt's
//            Method", IMM-REP-1999-05.
//            K&W Section 6.3-6.4 Algorithm 6.3 (Levenberg-Marquardt).
//            N&W Section 10.2-10.3 (nonlinear least-squares).

#include "nablapp/solver/lm_policy.h"
#include "nablapp/solver/basic_solver.h"

#include <Eigen/Core>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <print>

namespace
{

// Rosenbrock 2D in least-squares residual form (b=5 per project convention):
//   r_0 = 1 - x_0
//   r_1 = sqrt(5) * (x_1 - x_0^2)
//   f(x) = 0.5 * (r_0^2 + r_1^2)
// Unconstrained minimum at (1, 1), f* = 0.
struct rosenbrock_ls
{
    static constexpr int problem_dimension = nablapp::dynamic_dimension;

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

struct timing
{
    double wall_us;
    double objective;
    std::uint32_t evals;
};

timing bench_nablapp(std::uint32_t reps)
{
    rosenbrock_ls problem;
    Eigen::VectorXd x0{{-1.0, 1.0}};
    nablapp::solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-12);
    opts.set_objective_threshold(1e-14);
    opts.set_step_threshold(1e-14);

    // Warmup.
    {
        nablapp::basic_solver solver{nablapp::lm_policy{}, problem, x0, opts};
        solver.solve();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t iters = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nablapp::basic_solver solver{nablapp::lm_policy{}, problem, x0, opts};
        auto result = solver.solve();
        fval = result.objective_value;
        iters = result.iterations;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, iters};
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
    nablapp::solver_options opts;
    opts.max_iterations = 40;
    opts.set_gradient_threshold(1e-12);
    opts.set_objective_threshold(1e-14);
    opts.set_step_threshold(1e-14);

    nablapp::basic_solver solver{nablapp::lm_policy{}, problem, x0, opts};

    nablapp::step_result<double> last{};
    for(std::uint32_t i = 0; i < opts.max_iterations; ++i)
    {
        last = solver.step();
        if(last.policy_status)
            break;
    }

    if(!last.kkt_residual.has_value())
    {
        std::println(stderr, "FAIL: kkt_residual not populated (lm)");
        return false;
    }
    if(last.kkt_residual.value() < 0.0)
    {
        std::println(stderr, "FAIL: kkt_residual is negative: {}",
                     last.kkt_residual.value());
        return false;
    }
    std::println("  lm Rosenbrock LS kkt_residual: {:.6e} "
                 "(gradient_norm: {:.6e})",
                 last.kkt_residual.value(), last.gradient_norm);
    return true;
}

}

int main()
{
    constexpr std::uint32_t reps = 10000;

    if(!probe_kkt_residual())
        return 1;

    std::println("\nRosenbrock 2D LS (unconstrained, analytic Jacobian), "
                 "{} repetitions each\n", reps);

    auto na = bench_nablapp(reps);

    std::println("  {:>12s}  {:>10s}  {:>10s}  {:>12s}",
                 "solver", "wall (us)", "iters", "objective");
    std::println("  {:>12s}  {:10.2f}  {:10d}  {:.6e}",
                 "nablapp_lm", na.wall_us, na.evals, na.objective);

    std::println("\nNow profile with:");
    std::println("  perf record -F 99999 -g -- ./micro_lm");
    std::println("  perf report --stdio --percent-limit=1.0");
}
