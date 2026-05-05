// Micro-benchmark: argmin filter_slsqp vs NLopt LD_SLSQP on constrained HS problems.
//
// Filter SLSQP uses Fletcher-Leyffer 2002 filter acceptance with the
// Kraft 1988 QP solver and adaptive BFGS Hessian. Tested on problems
// with inequality and mixed constraints.
//
// Reference: Fletcher & Leyffer 2002, "Nonlinear programming without a
//            penalty function", Math. Program. 91:239-269.

#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <nlopt.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <print>
#include <string_view>

namespace
{

struct timing
{
    double wall_us;
    double objective;
    std::uint32_t evals;
};

// Dynamic-dimension HS043 wrapper (inequality, n=4).
struct hs043_dynamic
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;
    static constexpr argmin::problem_class pclass = argmin::problem_class::inequality;

    [[nodiscard]] int dimension() const { return 4; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 3; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
    {
        return x[0] * x[0] + x[1] * x[1] + 2.0 * x[2] * x[2]
               + x[3] * x[3] - 5.0 * x[0] - 5.0 * x[1]
               - 21.0 * x[2] + 7.0 * x[3];
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g[0] = 2.0 * x[0] - 5.0;
        g[1] = 2.0 * x[1] - 5.0;
        g[2] = 4.0 * x[2] - 21.0;
        g[3] = 2.0 * x[3] + 7.0;
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(3);
        c[0] = 8.0 - (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
               + x[3] * x[3] + x[0] - x[1] + x[2] - x[3]);
        c[1] = 10.0 - (x[0] * x[0] + 2.0 * x[1] * x[1]
               + x[2] * x[2] + 2.0 * x[3] * x[3] - x[0] - x[3]);
        c[2] = 5.0 - (2.0 * x[0] * x[0] + x[1] * x[1]
               + x[2] * x[2] + 2.0 * x[0] - x[1] - x[3]);
    }

    void constraint_jacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const
    {
        J.resize(3, 4);
        J(0, 0) = -(2.0 * x[0] + 1.0);
        J(0, 1) = -(2.0 * x[1] - 1.0);
        J(0, 2) = -(2.0 * x[2] + 1.0);
        J(0, 3) = -(2.0 * x[3] - 1.0);
        J(1, 0) = -(2.0 * x[0] - 1.0);
        J(1, 1) = -4.0 * x[1];
        J(1, 2) = -2.0 * x[2];
        J(1, 3) = -(4.0 * x[3] - 1.0);
        J(2, 0) = -(4.0 * x[0] + 2.0);
        J(2, 1) = -(2.0 * x[1] - 1.0);
        J(2, 2) = -2.0 * x[2];
        J(2, 3) = 1.0;
    }

    [[nodiscard]] Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Constant(4, -std::numeric_limits<double>::infinity());
    }

    [[nodiscard]] Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(4, std::numeric_limits<double>::infinity());
    }

    [[nodiscard]] Eigen::VectorXd initial_point() const
    {
        return Eigen::VectorXd::Zero(4);
    }
};

// Dynamic-dimension HS071 wrapper (mixed, n=4).
struct hs071_dynamic
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    [[nodiscard]] int dimension() const { return 4; }
    [[nodiscard]] int num_equality() const { return 1; }
    [[nodiscard]] int num_inequality() const { return 1; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
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

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(2);
        c[0] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] - 40.0;
        c[1] = x[0] * x[1] * x[2] * x[3] - 25.0;
    }

    void constraint_jacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const
    {
        J.resize(2, 4);
        J(0, 0) = 2.0 * x[0]; J(0, 1) = 2.0 * x[1];
        J(0, 2) = 2.0 * x[2]; J(0, 3) = 2.0 * x[3];
        J(1, 0) = x[1] * x[2] * x[3]; J(1, 1) = x[0] * x[2] * x[3];
        J(1, 2) = x[0] * x[1] * x[3]; J(1, 3) = x[0] * x[1] * x[2];
    }

    [[nodiscard]] Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd::Constant(4, 1.0);
    }

    [[nodiscard]] Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd::Constant(4, 5.0);
    }

    [[nodiscard]] Eigen::VectorXd initial_point() const
    {
        return Eigen::VectorXd{{1.0, 5.0, 5.0, 1.0}};
    }
};

// Dynamic-dimension HS076 wrapper (inequality + box, n=4).
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

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        g[0] = 2.0 * x[0] - x[2] - 1.0;
        g[1] = x[1] - 3.0;
        g[2] = 2.0 * x[2] - x[0] + x[3] + 1.0;
        g[3] = x[3] + x[2] - 1.0;
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(3);
        c[0] = 5.0 - (x[0] + 2.0 * x[1] + x[2] + x[3]);
        c[1] = 4.0 - (3.0 * x[0] + x[1] + 2.0 * x[2] - x[3]);
        c[2] = x[1] + 4.0 * x[2] - 1.5;
    }

    void constraint_jacobian(const Eigen::VectorXd&, Eigen::MatrixXd& J) const
    {
        J.resize(3, 4);
        J << -1.0, -2.0, -1.0, -1.0,
             -3.0, -1.0, -2.0,  1.0,
              0.0,  1.0,  4.0,  0.0;
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

// NLopt callbacks for HS043.
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
        grad[0] = 2.0 * x[0] + 1.0; grad[1] = 2.0 * x[1] - 1.0;
        grad[2] = 2.0 * x[2] + 1.0; grad[3] = 2.0 * x[3] - 1.0;
    }
    return (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
           + x[3] * x[3] + x[0] - x[1] + x[2] - x[3]) - 8.0;
}

double nlopt_hs043_ineq1(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = 2.0 * x[0] - 1.0; grad[1] = 4.0 * x[1];
        grad[2] = 2.0 * x[2];        grad[3] = 4.0 * x[3] - 1.0;
    }
    return (x[0] * x[0] + 2.0 * x[1] * x[1]
           + x[2] * x[2] + 2.0 * x[3] * x[3] - x[0] - x[3]) - 10.0;
}

double nlopt_hs043_ineq2(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = 4.0 * x[0] + 2.0; grad[1] = 2.0 * x[1] - 1.0;
        grad[2] = 2.0 * x[2];        grad[3] = -1.0;
    }
    return (2.0 * x[0] * x[0] + x[1] * x[1]
           + x[2] * x[2] + 2.0 * x[0] - x[1] - x[3]) - 5.0;
}

// NLopt callbacks for HS071.
double nlopt_hs071_obj(unsigned, const double* x, double* grad, void*)
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

double nlopt_hs071_eq(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = 2.0 * x[0]; grad[1] = 2.0 * x[1];
        grad[2] = 2.0 * x[2]; grad[3] = 2.0 * x[3];
    }
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] - 40.0;
}

double nlopt_hs071_ineq(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = -x[1] * x[2] * x[3]; grad[1] = -x[0] * x[2] * x[3];
        grad[2] = -x[0] * x[1] * x[3]; grad[3] = -x[0] * x[1] * x[2];
    }
    return 25.0 - x[0] * x[1] * x[2] * x[3];
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

double nlopt_hs076_ineq0(unsigned, const double* x, double* grad, void*)
{
    if(grad) { grad[0] = 1.0; grad[1] = 2.0; grad[2] = 1.0; grad[3] = 1.0; }
    return (x[0] + 2.0 * x[1] + x[2] + x[3]) - 5.0;
}

double nlopt_hs076_ineq1(unsigned, const double* x, double* grad, void*)
{
    if(grad) { grad[0] = 3.0; grad[1] = 1.0; grad[2] = 2.0; grad[3] = -1.0; }
    return (3.0 * x[0] + x[1] + 2.0 * x[2] - x[3]) - 4.0;
}

double nlopt_hs076_ineq2(unsigned, const double* x, double* grad, void*)
{
    if(grad) { grad[0] = 0.0; grad[1] = -1.0; grad[2] = -4.0; grad[3] = 0.0; }
    return -(x[1] + 4.0 * x[2] - 1.5);
}

// Sweep result: extends timing with primal feasibility and a feasible
// flag so the envelope-sweep TSV can report (f, cv, outer_iters).
struct sweep_row
{
    double f;
    double cv;
    std::uint32_t outer_iters;
    bool accepted;
};

template <typename Problem>
timing bench_argmin(const Problem& problem,
                    std::uint32_t reps,
                    std::optional<double> gamma_f = std::nullopt,
                    std::optional<double> gamma_h = std::nullopt)
{
    auto x0 = problem.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 5000;
    opts.set_gradient_threshold(1e-10);
    opts.set_objective_threshold(1e-8);
    opts.set_step_threshold(1e-12);

    // Wachter & Biegler 2006 Section 2.3 envelope margins, threaded
    // through filter_slsqp_policy::options_type when set on the CLI.
    typename argmin::filter_slsqp_policy<>::options_type policy_opts;
    if(gamma_f) policy_opts.gamma_f = *gamma_f;
    if(gamma_h) policy_opts.gamma_h = *gamma_h;

    // Warmup.
    {
        argmin::basic_solver solver{argmin::filter_slsqp_policy<>{},
                                    problem, x0, opts, policy_opts};
        solver.solve();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t iters = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        argmin::basic_solver solver{argmin::filter_slsqp_policy<>{},
                                    problem, x0, opts, policy_opts};
        auto result = solver.solve();
        fval = result.objective_value;
        iters = result.iterations;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, iters};
}

// Single-run probe used by the envelope sweep harness. Reports best
// strictly-feasible (f, cv, outer_iters) under the configured envelope.
//
// Rebinds the policy to the problem's compile-time dimension so that
// the matching options_type instantiation can be passed alongside (the
// 5-arg basic_solver ctor has a same_as<Policy::options_type> guard).
//
// Tolerance regime mirrors tests/unit/filter_sqp_test.cpp HS024
// regression guard: tight objective and step thresholds avoid premature
// objective-tolerance termination on slow-progress trajectories where
// the envelope sweep needs the full iteration budget to discriminate
// (gamma_f, gamma_h) cells.
template <typename Problem>
sweep_row sweep_argmin(const Problem& problem,
                       std::optional<double> gamma_f,
                       std::optional<double> gamma_h)
{
    using policy_t = argmin::filter_slsqp_policy<
        argmin::problem_dimension_v<Problem>>;

    auto x0 = problem.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 200;
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    typename policy_t::options_type policy_opts;
    if(gamma_f) policy_opts.gamma_f = *gamma_f;
    if(gamma_h) policy_opts.gamma_h = *gamma_h;

    argmin::basic_solver solver{policy_t{}, problem, x0, opts, policy_opts};
    auto result = solver.solve();
    return {result.objective_value, result.constraint_violation,
            result.iterations,
            result.constraint_violation <= opts.feasibility_tolerance};
}

timing bench_nlopt_hs043(std::uint32_t reps)
{
    {
        nlopt::opt opt(nlopt::LD_SLSQP, 4);
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
        nlopt::opt opt(nlopt::LD_SLSQP, 4);
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

timing bench_nlopt_hs071(std::uint32_t reps)
{
    {
        nlopt::opt opt(nlopt::LD_SLSQP, 4);
        opt.set_min_objective(nlopt_hs071_obj, nullptr);
        opt.set_lower_bounds({1.0, 1.0, 1.0, 1.0});
        opt.set_upper_bounds({5.0, 5.0, 5.0, 5.0});
        opt.add_equality_constraint(nlopt_hs071_eq, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs071_ineq, nullptr, 1e-10);
        opt.set_maxeval(5000);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        std::vector<double> x = {1.0, 5.0, 5.0, 1.0};
        double fval;
        opt.optimize(x, fval);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t evals = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nlopt::opt opt(nlopt::LD_SLSQP, 4);
        opt.set_min_objective(nlopt_hs071_obj, nullptr);
        opt.set_lower_bounds({1.0, 1.0, 1.0, 1.0});
        opt.set_upper_bounds({5.0, 5.0, 5.0, 5.0});
        opt.add_equality_constraint(nlopt_hs071_eq, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs071_ineq, nullptr, 1e-10);
        opt.set_maxeval(5000);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        std::vector<double> x = {1.0, 5.0, 5.0, 1.0};
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
        nlopt::opt opt(nlopt::LD_SLSQP, 4);
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
        nlopt::opt opt(nlopt::LD_SLSQP, 4);
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

void print_row(std::string_view solver, const timing& t)
{
    std::println("  {:>12s}  {:10.2f}  {:10d}  {:.6e}", solver, t.wall_us, t.evals, t.objective);
}

// kkt_residual regression probe.
//
// filter_slsqp uses L-BFGS with least-squares multiplier estimates
// (N&W eq. 18.15), so kkt_residual is populated from those estimates.
// This probe runs the policy on HS071 via step_n() with a small
// budget and confirms the terminal non-restoration step_result
// carries a populated, non-negative kkt_residual.
//
// Reference: N&W 2e Section 12.3 / eq. 12.34.
bool probe_kkt_residual()
{
    hs071_dynamic problem;
    auto x0 = problem.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-10);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    argmin::basic_solver solver{argmin::filter_slsqp_policy<>{}, problem, x0, opts};

    argmin::step_result<double> last{};
    argmin::step_result<double> last_with_kkt{};
    bool any_kkt = false;
    for(std::uint32_t i = 0; i < opts.max_iterations; ++i)
    {
        last = solver.step();
        if(last.kkt_residual.has_value())
        {
            last_with_kkt = last;
            any_kkt = true;
        }
        if(last.policy_status)
            break;
    }

    // filter_slsqp's restoration and zero-step paths intentionally
    // leave kkt_residual unset; accept any step in the run that
    // populated the field.
    if(!any_kkt)
    {
        std::println("FAIL: kkt_residual not populated (filter_slsqp)");
        return false;
    }
    if(last_with_kkt.kkt_residual.value() < 0.0)
    {
        std::println("FAIL: kkt_residual is negative: {}",
                     last_with_kkt.kkt_residual.value());
        return false;
    }
    std::println("  filter_slsqp HS071 kkt_residual: {:.6e} (gradient_norm: {:.6e})",
                 last_with_kkt.kkt_residual.value(),
                 last_with_kkt.gradient_norm);
    return true;
}

// Phase 31.1 regression probe: filter_slsqp on HS026 must reach
// f < 1e-5 after the Full E-measure (N&W 2e Definition 12.1) blocks
// the premature ftol that post-phase31 let fire at iter 12 on
// equality-only problems.
//
// Reference: N&W 2e Definition 12.1; post-phase30 baseline.
bool probe_regression_hs026()
{
    argmin::hs026<> p;
    Eigen::VectorXd x0 = p.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    argmin::basic_solver solver{
        argmin::filter_slsqp_policy<argmin::hs026<>::problem_dimension>{},
        p, x0, opts};
    argmin::step_result<double> last{};
    argmin::step_result<double> last_with_kkt{};
    bool any_kkt = false;
    for(std::uint32_t i = 0; i < opts.max_iterations; ++i)
    {
        last = solver.step();
        if(last.kkt_residual.has_value())
        {
            last_with_kkt = last;
            any_kkt = true;
        }
        if(last.policy_status)
            break;
    }

    const double kkt = any_kkt ? last_with_kkt.kkt_residual.value() : -1.0;
    const bool ok = last.objective_value < 1e-5;
    if(!ok)
        std::println(stderr,
                     "FAIL: filter_slsqp HS026 f={:.6e} kkt={:.6e}",
                     last.objective_value, kkt);
    std::println("  filter_slsqp HS026: f={:.6e} kkt={:.6e}",
                 last.objective_value, kkt);
    return ok;
}

// Regression probe: filter_slsqp HS024 termination closure via the
// composite kkt_residual (N&W 2e eq. 12.34). Prior behaviour: the
// outer constrained-convergence wrapper gated termination on the L1
// sum of inequality violations h_k, which for multi-constraint
// problems (HS024: 3 inequalities) sat above the per-constraint
// feasibility threshold. With the wrapper removed, kkt_residual
// carries primal feasibility at the L-infinity norm and the
// least-squares lambda estimate is mu-projected (cwiseMax(0.0)) so
// the dual-feasibility leg no longer fires spuriously.
//
// Reference: N&W 2e Definition 12.1 (KKT primal + dual feasibility);
//            eq. 12.34 (full first-order optimality E-measure);
//            eq. 18.15 (least-squares lambda). Post-phase30 baseline
//            13 iters; target within 1.
bool probe_regression_hs024()
{
    argmin::hs024<> p;
    Eigen::VectorXd x0 = p.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    argmin::basic_solver solver{
        argmin::filter_slsqp_policy<argmin::hs024<>::problem_dimension>{},
        p, x0, opts};
    auto result = solver.solve(opts);

    const bool ok = result.iterations <= 14
        && std::abs(result.objective_value - (-1.0)) < 1e-6;
    if(!ok)
        std::println(stderr,
                     "FAIL: filter_slsqp HS024 iters={} f={:.6e} (expected <= 14 @ -1.0)",
                     result.iterations, result.objective_value);
    std::println("  filter_slsqp HS024: iters={} f={:.6e}",
                 result.iterations, result.objective_value);
    return ok;
}

}

int main(int argc, char** argv)
{
    // Parse --gamma-f / --gamma-h CLI overrides for the filter envelope
    // sweep. When both are provided the binary switches into sweep mode:
    // the regression probes and NLopt comparison are skipped, and a
    // single-run TSV-friendly line per problem is emitted on stdout
    // ("policy\tgamma_f\tgamma_h\tproblem\tf\tcv\touter_iters\taccepted").
    //
    // Reference: Wachter & Biegler 2006 Section 2.3 (envelope margins).
    std::optional<double> cli_gamma_f;
    std::optional<double> cli_gamma_h;
    for(int i = 1; i + 1 < argc; ++i)
    {
        std::string_view arg{argv[i]};
        if(arg == "--gamma-f")
            cli_gamma_f = std::stod(argv[++i]);
        else if(arg == "--gamma-h")
            cli_gamma_h = std::stod(argv[++i]);
    }

    if(cli_gamma_f && cli_gamma_h)
    {
        const double gf = *cli_gamma_f;
        const double gh = *cli_gamma_h;
        auto emit = [gf, gh](std::string_view name, const sweep_row& r) {
            std::println("filter_slsqp\t{:.0e}\t{:.0e}\t{}\t{:.10e}\t{:.10e}\t{}\t{}",
                         gf, gh, name, r.f, r.cv, r.outer_iters,
                         r.accepted ? 1 : 0);
        };
        emit("HS043", sweep_argmin(hs043_dynamic{}, cli_gamma_f, cli_gamma_h));
        emit("HS024", sweep_argmin(argmin::hs024<>{}, cli_gamma_f, cli_gamma_h));
        emit("HS076", sweep_argmin(hs076_dynamic{}, cli_gamma_f, cli_gamma_h));
        return 0;
    }

    constexpr std::uint32_t reps = 500;

    if(!probe_kkt_residual())
        return 1;
    if(!probe_regression_hs026())
        return 1;
    if(!probe_regression_hs024())
        return 1;

    std::println("Filter SLSQP micro-benchmark, {} repetitions each\n", reps);
    std::println("  {:>12s}  {:>10s}  {:>10s}  {:>12s}", "solver", "wall (us)", "evals", "objective");

    // HS043
    {
        std::println("\n--- HS043 (inequality, n=4, f*=-44) ---");
        auto nab  = bench_argmin(hs043_dynamic{}, reps, cli_gamma_f, cli_gamma_h);
        auto nlop = bench_nlopt_hs043(reps);
        print_row("argmin", nab);
        print_row("nlopt", nlop);
        std::println("  ratio argmin/nlopt: {:.1f}x wall, {:.1f}x evals",
            nab.wall_us / nlop.wall_us, double(nab.evals) / nlop.evals);
    }

    // HS071
    {
        std::println("\n--- HS071 (mixed, n=4, f*~17.014) ---");
        auto nab  = bench_argmin(hs071_dynamic{}, reps, cli_gamma_f, cli_gamma_h);
        auto nlop = bench_nlopt_hs071(reps);
        print_row("argmin", nab);
        print_row("nlopt", nlop);
        std::println("  ratio argmin/nlopt: {:.1f}x wall, {:.1f}x evals",
            nab.wall_us / nlop.wall_us, double(nab.evals) / nlop.evals);
    }

    // HS076
    {
        std::println("\n--- HS076 (inequality + box, n=4, f*=-4.68) ---");
        auto nab  = bench_argmin(hs076_dynamic{}, reps, cli_gamma_f, cli_gamma_h);
        auto nlop = bench_nlopt_hs076(reps);
        print_row("argmin", nab);
        print_row("nlopt", nlop);
        std::println("  ratio argmin/nlopt: {:.1f}x wall, {:.1f}x evals",
            nab.wall_us / nlop.wall_us, double(nab.evals) / nlop.evals);
    }
}
