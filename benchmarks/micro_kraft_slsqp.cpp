// Micro-benchmark: nablapp kraft_slsqp vs NLopt LD_SLSQP on HS071.
//
// HS071 is a mixed-constraint problem (n=4, m_eq=1, m_ineq=1) at
// IK-relevant scale, testing per-step wall time for constrained SQP.
//
// Reference: Hock, W. and Schittkowski, K. (1981) "Test Examples for
//            Nonlinear Programming Codes", Lecture Notes in Economics
//            and Mathematical Systems, Vol. 187, Springer.

#include "nablapp/solver/kraft_slsqp_policy.h"
#include "nablapp/solver/nw_sqp_policy.h"
#include "nablapp/solver/basic_solver.h"

#include <Eigen/Core>

#include <nlopt.hpp>

#include <chrono>
#include <cstdint>
#include <print>

namespace
{

// HS071: n=4, 1 equality, 1 inequality, box bounds.
//
//   min   x1*x4*(x1+x2+x3) + x3
//   s.t.  x1*x2*x3*x4 >= 25          (inequality: c_ineq >= 0)
//         x1^2+x2^2+x3^2+x4^2 = 40   (equality:   c_eq  = 0)
//         1 <= xi <= 5
//   x0 = {1, 5, 5, 1}
//   f* ~ 17.014
struct hs071
{
    static constexpr int problem_dimension = nablapp::dynamic_dimension;

    int dimension() const { return 4; }

    double value(const Eigen::VectorXd& x) const
    {
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g[0] = x[3] * (x[0] + x[1] + x[2]) + x[0] * x[3];
        g[1] = x[0] * x[3];
        g[2] = x[0] * x[3] + 1.0;
        g[3] = x[0] * (x[0] + x[1] + x[2]);
    }

    Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd{{1.0, 1.0, 1.0, 1.0}};
    }

    Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd{{5.0, 5.0, 5.0, 5.0}};
    }

    int num_equality() const { return 1; }
    int num_inequality() const { return 1; }

    // Combined constraint vector: [equality, inequality].
    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(2);
        // Equality: x1^2+x2^2+x3^2+x4^2 - 40 = 0
        c[0] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] - 40.0;
        // Inequality: x1*x2*x3*x4 - 25 >= 0
        c[1] = x[0] * x[1] * x[2] * x[3] - 25.0;
    }

    void constraint_jacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const
    {
        J.resize(2, 4);
        // Equality Jacobian
        J(0, 0) = 2.0 * x[0];
        J(0, 1) = 2.0 * x[1];
        J(0, 2) = 2.0 * x[2];
        J(0, 3) = 2.0 * x[3];
        // Inequality Jacobian
        J(1, 0) = x[1] * x[2] * x[3];
        J(1, 1) = x[0] * x[2] * x[3];
        J(1, 2) = x[0] * x[1] * x[3];
        J(1, 3) = x[0] * x[1] * x[2];
    }
};

// NLopt objective callback.
double nlopt_hs071_objective(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = x[3] * (x[0] + x[1] + x[2]) + x[0] * x[3];
        grad[1] = x[0] * x[3];
        grad[2] = x[0] * x[3] + 1.0;
        grad[3] = x[0] * (x[0] + x[1] + x[2]);
    }
    return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
}

// NLopt equality constraint: x1^2+x2^2+x3^2+x4^2 - 40 = 0
double nlopt_hs071_equality(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = 2.0 * x[0];
        grad[1] = 2.0 * x[1];
        grad[2] = 2.0 * x[2];
        grad[3] = 2.0 * x[3];
    }
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] - 40.0;
}

// NLopt inequality constraint: 25 - x1*x2*x3*x4 <= 0
// NLopt convention: c(x) <= 0, so we negate the >= 0 form.
double nlopt_hs071_inequality(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = -x[1] * x[2] * x[3];
        grad[1] = -x[0] * x[2] * x[3];
        grad[2] = -x[0] * x[1] * x[3];
        grad[3] = -x[0] * x[1] * x[2];
    }
    return 25.0 - x[0] * x[1] * x[2] * x[3];
}

struct timing
{
    double wall_us;
    double objective;
    std::uint32_t evals;
    double per_step_us;
    double line_search_calls_per_step;  // kraft_slsqp only; 0 for others
};

timing bench_nablapp(std::uint32_t reps)
{
    hs071 problem;
    Eigen::VectorXd x0{{1.0, 5.0, 5.0, 1.0}};
    nablapp::solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-10);

    // Warmup + convergence check.
    std::uint32_t iters = 0;
    {
        nablapp::basic_solver solver{nablapp::kraft_slsqp_policy{}, problem, x0, opts};
        auto result = solver.solve();
        iters = static_cast<std::uint32_t>(result.iterations);
        if(std::abs(result.objective_value - 17.014) > 0.5)
            std::println("WARNING: kraft_slsqp did not converge correctly, f={:.6e}",
                         result.objective_value);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint64_t total_ls_calls = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nablapp::basic_solver solver{nablapp::kraft_slsqp_policy{}, problem, x0, opts};
        auto result = solver.solve();
        fval = result.objective_value;
        iters = static_cast<std::uint32_t>(result.iterations);
        total_ls_calls += solver.state().line_search_calls;
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double per_solve = total_us / reps;
    double per_step = total_us / (reps * iters);
    double ls_per_step = static_cast<double>(total_ls_calls) / (reps * iters);
    return {per_solve, fval, iters, per_step, ls_per_step};
}

timing bench_nablapp_nw_sqp(std::uint32_t reps)
{
    hs071 problem;
    Eigen::VectorXd x0{{1.0, 5.0, 5.0, 1.0}};
    nablapp::solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-10);

    // Warmup + convergence check.
    std::uint32_t iters = 0;
    {
        nablapp::basic_solver solver{nablapp::nw_sqp_policy{}, problem, x0, opts};
        auto result = solver.solve();
        iters = static_cast<std::uint32_t>(result.iterations);
        if(std::abs(result.objective_value - 17.014) > 0.5)
            std::println("WARNING: nw_sqp did not converge correctly, f={:.6e}",
                         result.objective_value);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nablapp::basic_solver solver{nablapp::nw_sqp_policy{}, problem, x0, opts};
        auto result = solver.solve();
        fval = result.objective_value;
        iters = static_cast<std::uint32_t>(result.iterations);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double per_solve = total_us / reps;
    double per_step = total_us / (reps * iters);
    return {per_solve, fval, iters, per_step, 0.0};
}

timing bench_nlopt(std::uint32_t reps)
{
    // Warmup.
    std::uint32_t evals = 0;
    {
        nlopt::opt opt(nlopt::LD_SLSQP, 4);
        opt.set_min_objective(nlopt_hs071_objective, nullptr);
        opt.set_lower_bounds({1.0, 1.0, 1.0, 1.0});
        opt.set_upper_bounds({5.0, 5.0, 5.0, 5.0});
        opt.add_equality_constraint(nlopt_hs071_equality, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs071_inequality, nullptr, 1e-10);
        opt.set_maxeval(200);
        opt.set_ftol_rel(1e-10);
        opt.set_xtol_rel(1e-10);
        std::vector<double> x = {1.0, 5.0, 5.0, 1.0};
        double fval;
        opt.optimize(x, fval);
        evals = static_cast<std::uint32_t>(opt.get_numevals());
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nlopt::opt opt(nlopt::LD_SLSQP, 4);
        opt.set_min_objective(nlopt_hs071_objective, nullptr);
        opt.set_lower_bounds({1.0, 1.0, 1.0, 1.0});
        opt.set_upper_bounds({5.0, 5.0, 5.0, 5.0});
        opt.add_equality_constraint(nlopt_hs071_equality, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs071_inequality, nullptr, 1e-10);
        opt.set_maxeval(200);
        opt.set_ftol_rel(1e-10);
        opt.set_xtol_rel(1e-10);
        std::vector<double> x = {1.0, 5.0, 5.0, 1.0};
        opt.optimize(x, fval);
        evals = static_cast<std::uint32_t>(opt.get_numevals());
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double per_solve = total_us / reps;
    double per_step = total_us / (reps * evals);
    return {per_solve, fval, evals, per_step, 0.0};
}

}

int main()
{
    constexpr std::uint32_t reps = 10000;

    std::println("HS071 (n=4, m_eq=1, m_ineq=1), {} repetitions each\n", reps);

    auto kraft = bench_nablapp(reps);
    auto nw = bench_nablapp_nw_sqp(reps);
    auto nl = bench_nlopt(reps);

    std::println("  {:>12s}  {:>12s}  {:>12s}  {:>10s}  {:>12s}",
                 "solver", "solve (us)", "step (us)", "iters", "objective");
    std::println("  {:>12s}  {:12.2f}  {:12.2f}  {:10d}  {:.6e}",
                 "kraft_slsqp", kraft.wall_us, kraft.per_step_us, kraft.evals, kraft.objective);
    std::println("  {:>12s}  {:12.2f}  {:12.2f}  {:10d}  {:.6e}",
                 "nw_sqp", nw.wall_us, nw.per_step_us, nw.evals, nw.objective);
    std::println("  {:>12s}  {:12.2f}  {:12.2f}  {:10d}  {:.6e}",
                 "nlopt", nl.wall_us, nl.per_step_us, nl.evals, nl.objective);
    std::println("\n  per-solve ratio kraft_slsqp/nlopt: {:.2f}x", kraft.wall_us / nl.wall_us);
    std::println("  per-step  ratio kraft_slsqp/nlopt: {:.2f}x", kraft.per_step_us / nl.per_step_us);
    std::println("  per-solve ratio nw_sqp/nlopt:      {:.2f}x", nw.wall_us / nl.wall_us);
    std::println("  per-step  ratio nw_sqp/nlopt:      {:.2f}x", nw.per_step_us / nl.per_step_us);

    std::println("\n  kraft_slsqp phi_ls calls per step: {:.3f}",
                 kraft.line_search_calls_per_step);
    std::println("  (average number of merit-function evaluations per kraft_slsqp_policy::step()");
    std::println("   invocation, averaged over {} reps x {} iters. Armijo success on first try = 1.0;",
                 reps, kraft.evals);
    std::println("   2.0 means one backtrack on average; 3.0 means two backtracks.)");

    std::println("\nNow profile with:");
    std::println("  perf record -F 99999 -g -- ./micro_kraft_slsqp");
    std::println("  perf report --stdio --percent-limit=1.0");
}
