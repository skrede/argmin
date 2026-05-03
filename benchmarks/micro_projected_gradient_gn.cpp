// Micro-benchmark: argmin projected gradient GN vs NLopt BOBYQA on bounded Rosenbrock LS.
//
// Compares projected_gradient_gn_policy (least-squares residual form with
// analytic Jacobian, projected-gradient inner step) against NLopt LN_BOBYQA
// (derivative-free, scalar objective) on the same bounded Rosenbrock problem.
// The comparison is inherently asymmetric: projected gradient GN exploits
// least-squares structure while BOBYQA is general.

#include "argmin/solver/projected_gradient_gn_policy.h"
#include "argmin/solver/basic_solver.h"

#include <Eigen/Core>

#include <nlopt.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <print>

namespace
{

// Bounded Rosenbrock 2D in least-squares residual form (b=5).
// Bounds: x_0 in [-2, 0.5], x_1 in [-2, 2].
// Constrained optimum at (0.5, 0.25), f* = 0.125.
struct bounded_rosenbrock_ls
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return 2; }
    int num_residuals() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        double r0 = 1.0 - x[0];
        double r1 = std::sqrt(5.0) * (x[1] - x[0] * x[0]);
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

    Eigen::VectorXd lower_bounds() const { return Eigen::VectorXd{{-2.0, -2.0}}; }
    Eigen::VectorXd upper_bounds() const { return Eigen::VectorXd{{0.5, 2.0}}; }
};

// NLopt callback: scalar Rosenbrock (no gradient -- BOBYQA is derivative-free).
double nlopt_rosenbrock(unsigned, const double* x, double*, void*)
{
    double r0 = 1.0 - x[0];
    double r1 = std::sqrt(5.0) * (x[1] - x[0] * x[0]);
    return 0.5 * (r0 * r0 + r1 * r1);
}

struct timing
{
    double wall_us;
    double objective;
    std::uint32_t evals;
};

timing bench_argmin_proj_grad(std::uint32_t reps)
{
    bounded_rosenbrock_ls problem;
    Eigen::VectorXd x0{{-1.0, 1.0}};
    argmin::solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-10);

    // Warmup.
    {
        argmin::basic_solver solver{argmin::projected_gradient_gn_policy{}, problem, x0, opts};
        solver.solve();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t iters = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        argmin::basic_solver solver{argmin::projected_gradient_gn_policy{}, problem, x0, opts};
        auto result = solver.solve();
        fval = result.objective_value;
        iters = result.iterations;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, iters};
}

timing bench_nlopt_bobyqa(std::uint32_t reps)
{
    auto run_nlopt = [](std::vector<double>& x, double& fval, nlopt::opt& opt) {
        try { opt.optimize(x, fval); }
        catch(const nlopt::roundoff_limited&) {}
    };

    // Warmup.
    {
        nlopt::opt opt(nlopt::LN_BOBYQA, 2);
        opt.set_min_objective(nlopt_rosenbrock, nullptr);
        opt.set_lower_bounds({-2.0, -2.0});
        opt.set_upper_bounds({0.5, 2.0});
        opt.set_maxeval(200);
        opt.set_ftol_rel(1e-14);
        opt.set_xtol_rel(1e-14);
        std::vector<double> x = {-1.0, 1.0};
        double fval;
        run_nlopt(x, fval, opt);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t evals = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nlopt::opt opt(nlopt::LN_BOBYQA, 2);
        opt.set_min_objective(nlopt_rosenbrock, nullptr);
        opt.set_lower_bounds({-2.0, -2.0});
        opt.set_upper_bounds({0.5, 2.0});
        opt.set_maxeval(200);
        opt.set_ftol_rel(1e-14);
        opt.set_xtol_rel(1e-14);
        std::vector<double> x = {-1.0, 1.0};
        run_nlopt(x, fval, opt);
        evals = static_cast<std::uint32_t>(opt.get_numevals());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, evals};
}

// kkt_residual regression probe.
//
// projected_gradient_gn_policy populates step_result::kkt_residual via
// detail::kkt_residual_bound on both the backtracking and dogleg
// return paths (projected-gradient infinity-norm). This probe runs
// projected_gradient_gn on the bounded Rosenbrock LS problem via
// step(), confirms the observed step_result carries a populated,
// non-negative kkt_residual, and prints the value for telemetry
// parity. Failure prints FAIL and reports through main.
bool probe_kkt_residual()
{
    bounded_rosenbrock_ls problem;
    Eigen::VectorXd x0{{-1.0, 1.0}};
    argmin::solver_options opts;
    opts.max_iterations = 40;
    opts.set_gradient_threshold(1e-10);

    argmin::basic_solver solver{argmin::projected_gradient_gn_policy{},
                                 problem, x0, opts};

    argmin::step_result<double> last{};
    for(std::uint32_t i = 0; i < opts.max_iterations; ++i)
    {
        last = solver.step();
        if(last.policy_status)
            break;
    }

    if(!last.kkt_residual.has_value())
    {
        std::println(stderr,
                     "FAIL: kkt_residual not populated (projected_gradient_gn)");
        return false;
    }
    if(last.kkt_residual.value() < 0.0)
    {
        std::println(stderr, "FAIL: kkt_residual is negative: {}",
                     last.kkt_residual.value());
        return false;
    }
    std::println("  projected_gradient_gn bounded Rosenbrock LS kkt_residual: "
                 "{:.6e} (gradient_norm: {:.6e})",
                 last.kkt_residual.value(), last.gradient_norm);
    return true;
}

}

int main()
{
    constexpr std::uint32_t reps = 10000;

    if(!probe_kkt_residual())
        return 1;

    std::println("Bounded Rosenbrock 2D LS, {} repetitions each\n", reps);

    auto na_pg = bench_argmin_proj_grad(reps);
    auto nl = bench_nlopt_bobyqa(reps);

    std::println("  {:>22s}  {:>10s}  {:>10s}  {:>12s}",
                 "solver", "wall (us)", "evals", "objective");
    std::println("  {:>22s}  {:10.2f}  {:10d}  {:.6e}",
                 "projected_gradient_gn", na_pg.wall_us, na_pg.evals, na_pg.objective);
    std::println("  {:>22s}  {:10.2f}  {:10d}  {:.6e}",
                 "nlopt_bobyqa", nl.wall_us, nl.evals, nl.objective);
    std::println("\n  ratio (projected_gradient_gn / nlopt): {:.2f}x", na_pg.wall_us / nl.wall_us);

    std::println("\nNow profile with:");
    std::println("  perf record -F 99999 -g -- ./micro_projected_gradient_gn");
    std::println("  perf report --stdio --percent-limit=1.0");
}
