// Micro-benchmark: nablapp MMA vs NLopt LD_MMA on inequality-constrained HS problems.
//
// MMA handles inequality constraints only, using gradient-based convex
// separable approximations with moving asymptotes.
//
// Threshold convention: relative objective tolerance (1e-12) matches
// NLopt's ftol_rel on LD_MMA; previously an absolute threshold caused
// nablapp to exit earlier than NLopt by construction and made post-fix
// measurements uninterpretable.
//
// Reference: Svanberg 1987, "The method of moving asymptotes --
//            a new method for structural optimization";
//            NLopt 2.10.0 src/algs/mma/mma.c (ftol_rel convention).

#include "nablapp/solver/mma_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/test_functions/hock_schittkowski.h"

#include "counting_problem.h"

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
    std::uint32_t iters;     // outer iters (counts.g for nablapp; grad-bearing obj calls for nlopt)
    std::uint32_t f_evals;   // total obj-callback count
    std::uint32_t inner;     // f_evals - iters: conservativity-loop trials
};

// NLopt obj-callback counter passed via user-data. grad != nullptr means
// the call is on an outer iter (CCSA requests obj+grad once per outer);
// grad == nullptr means an inner conservativity trial.
struct nlopt_counts
{
    std::uint32_t outer{0};
    std::uint32_t inner{0};
};

// Dynamic-dimension HS024 wrapper (inequality, n=2).
struct hs024_dynamic
{
    static constexpr int problem_dimension = nablapp::dynamic_dimension;
    static constexpr nablapp::problem_class pclass =
        nablapp::problem_class::inequality | nablapp::problem_class::bound_constrained;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_equality() const { return 0; }
    [[nodiscard]] int num_inequality() const { return 3; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
    {
        double k = 1.0 / (27.0 * std::sqrt(3.0));
        double t = (x[0] - 3.0) * (x[0] - 3.0) - 9.0;
        return k * t * x[1] * x[1] * x[1];
    }

    void gradient(const Eigen::VectorXd& x, Eigen::VectorXd& g) const
    {
        double k = 1.0 / (27.0 * std::sqrt(3.0));
        double t = (x[0] - 3.0) * (x[0] - 3.0) - 9.0;
        g[0] = k * 2.0 * (x[0] - 3.0) * x[1] * x[1] * x[1];
        g[1] = k * t * 3.0 * x[1] * x[1];
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        double s3 = std::sqrt(3.0);
        c.resize(3);
        c[0] = x[0] / s3 - x[1];
        c[1] = x[0] + s3 * x[1];
        c[2] = 6.0 - x[0] - s3 * x[1];
    }

    void constraint_jacobian(const Eigen::VectorXd&, Eigen::MatrixXd& J) const
    {
        double s3 = std::sqrt(3.0);
        J.resize(3, 2);
        J(0, 0) = 1.0 / s3;  J(0, 1) = -1.0;
        J(1, 0) = 1.0;        J(1, 1) = s3;
        J(2, 0) = -1.0;       J(2, 1) = -s3;
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

// Dynamic-dimension HS043 wrapper (inequality, n=4).
struct hs043_dynamic
{
    static constexpr int problem_dimension = nablapp::dynamic_dimension;
    static constexpr nablapp::problem_class pclass = nablapp::problem_class::inequality;

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

// NLopt callbacks for HS024.
double nlopt_hs024_obj(unsigned, const double* x, double* grad, void* data)
{
    if(data) { auto* c = static_cast<nlopt_counts*>(data); grad ? ++c->outer : ++c->inner; }
    double k = 1.0 / (27.0 * std::sqrt(3.0));
    double t = (x[0] - 3.0) * (x[0] - 3.0) - 9.0;
    if(grad)
    {
        grad[0] = k * 2.0 * (x[0] - 3.0) * x[1] * x[1] * x[1];
        grad[1] = k * t * 3.0 * x[1] * x[1];
    }
    return k * t * x[1] * x[1] * x[1];
}

double nlopt_hs024_ineq0(unsigned, const double* x, double* grad, void*)
{
    double s3 = std::sqrt(3.0);
    if(grad) { grad[0] = -1.0 / s3; grad[1] = 1.0; }
    return -(x[0] / s3 - x[1]);
}

double nlopt_hs024_ineq1(unsigned, const double* x, double* grad, void*)
{
    double s3 = std::sqrt(3.0);
    if(grad) { grad[0] = -1.0; grad[1] = -s3; }
    return -(x[0] + s3 * x[1]);
}

double nlopt_hs024_ineq2(unsigned, const double* x, double* grad, void*)
{
    double s3 = std::sqrt(3.0);
    if(grad) { grad[0] = 1.0; grad[1] = s3; }
    return -(6.0 - x[0] - s3 * x[1]);
}

// NLopt callbacks for HS043.
double nlopt_hs043_obj(unsigned, const double* x, double* grad, void* data)
{
    if(data) { auto* c = static_cast<nlopt_counts*>(data); grad ? ++c->outer : ++c->inner; }
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

// Dynamic-dimension HS076 wrapper (inequality, n=4, x >= 0, f* = -4.6818..).
struct hs076_dynamic
{
    static constexpr int problem_dimension = nablapp::dynamic_dimension;
    static constexpr nablapp::problem_class pclass =
        nablapp::problem_class::inequality | nablapp::problem_class::bound_constrained;

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
        return Eigen::VectorXd::Constant(
            4, std::numeric_limits<double>::infinity());
    }

    [[nodiscard]] Eigen::VectorXd initial_point() const
    {
        return Eigen::VectorXd::Constant(4, 0.5);
    }
};

// NLopt callbacks for HS076.
double nlopt_hs076_obj(unsigned, const double* x, double* grad, void* data)
{
    if(data) { auto* c = static_cast<nlopt_counts*>(data); grad ? ++c->outer : ++c->inner; }
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
    return -(5.0 - (x[0] + 2.0 * x[1] + x[2] + x[3]));
}

double nlopt_hs076_ineq1(unsigned, const double* x, double* grad, void*)
{
    if(grad) { grad[0] = 3.0; grad[1] = 1.0; grad[2] = 2.0; grad[3] = -1.0; }
    return -(4.0 - (3.0 * x[0] + x[1] + 2.0 * x[2] - x[3]));
}

double nlopt_hs076_ineq2(unsigned, const double* x, double* grad, void*)
{
    if(grad) { grad[0] = 0.0; grad[1] = -1.0; grad[2] = -4.0; grad[3] = 0.0; }
    return -(x[1] + 4.0 * x[2] - 1.5);
}

template <typename Problem>
timing bench_nablapp(const Problem& problem, std::uint32_t reps)
{
    auto x0 = problem.initial_point();
    // NLopt-parity convergence: LD_MMA uses ftol_rel on the objective.
    // slsqp_compatible_convergence mirrors NLopt's ftol_rel + xtol_rel
    // convention so cross-bench measurements are meaningful; absolute
    // ftol made nablapp exit earlier than NLopt by construction
    // regardless of fix quality.
    nablapp::solver_options<nablapp::slsqp_compatible_convergence> opts;
    opts.max_iterations = 5000;
    opts.set_objective_threshold_rel(1e-12);
    opts.set_step_threshold_rel(1e-12);

    nablapp::bench::eval_counts counts;
    nablapp::bench::counting_problem<Problem> wrapped{problem, counts};

    // Warmup.
    {
        nablapp::basic_solver solver{nablapp::mma_policy<>{}, wrapped, x0, opts};
        solver.solve();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    std::uint32_t outer_g = 0;
    std::uint32_t f_evals = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        counts.reset();
        nablapp::basic_solver solver{nablapp::mma_policy<>{}, wrapped, x0, opts};
        auto result = solver.solve();
        fval = result.objective_value;
        outer_g = static_cast<std::uint32_t>(counts.g);  // grad calls = outer iters
        f_evals = static_cast<std::uint32_t>(counts.f);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    std::uint32_t inner = f_evals > outer_g ? f_evals - outer_g : 0;
    return {us, fval, outer_g, f_evals, inner};
}

timing bench_nlopt_hs024(std::uint32_t reps)
{
    {
        nlopt::opt opt(nlopt::LD_CCSAQ, 2);
        opt.set_min_objective(nlopt_hs024_obj, nullptr);
        opt.set_lower_bounds({0.0, 0.0});
        opt.set_upper_bounds({1e20, 1e20});
        opt.add_inequality_constraint(nlopt_hs024_ineq0, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs024_ineq1, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs024_ineq2, nullptr, 1e-10);
        opt.set_maxeval(200);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        std::vector<double> x = {1.0, 0.5};
        double fval;
        opt.optimize(x, fval);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    nlopt_counts nl{};
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nl = {};
        nlopt::opt opt(nlopt::LD_CCSAQ, 2);
        opt.set_min_objective(nlopt_hs024_obj, &nl);
        opt.set_lower_bounds({0.0, 0.0});
        opt.set_upper_bounds({1e20, 1e20});
        opt.add_inequality_constraint(nlopt_hs024_ineq0, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs024_ineq1, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs024_ineq2, nullptr, 1e-10);
        opt.set_maxeval(200);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        std::vector<double> x = {1.0, 0.5};
        opt.optimize(x, fval);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, nl.outer, nl.outer + nl.inner, nl.inner};
}

timing bench_nlopt_hs043(std::uint32_t reps)
{
    {
        nlopt::opt opt(nlopt::LD_CCSAQ, 4);
        opt.set_min_objective(nlopt_hs043_obj, nullptr);
        opt.add_inequality_constraint(nlopt_hs043_ineq0, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs043_ineq1, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs043_ineq2, nullptr, 1e-10);
        opt.set_maxeval(200);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        std::vector<double> x = {0.0, 0.0, 0.0, 0.0};
        double fval;
        opt.optimize(x, fval);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    nlopt_counts nl{};
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nl = {};
        nlopt::opt opt(nlopt::LD_CCSAQ, 4);
        opt.set_min_objective(nlopt_hs043_obj, &nl);
        opt.add_inequality_constraint(nlopt_hs043_ineq0, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs043_ineq1, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs043_ineq2, nullptr, 1e-10);
        opt.set_maxeval(200);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        std::vector<double> x = {0.0, 0.0, 0.0, 0.0};
        opt.optimize(x, fval);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, nl.outer, nl.outer + nl.inner, nl.inner};
}

timing bench_nlopt_hs076(std::uint32_t reps)
{
    {
        nlopt::opt opt(nlopt::LD_CCSAQ, 4);
        opt.set_min_objective(nlopt_hs076_obj, nullptr);
        opt.set_lower_bounds({0.0, 0.0, 0.0, 0.0});
        opt.set_upper_bounds({1e20, 1e20, 1e20, 1e20});
        opt.add_inequality_constraint(nlopt_hs076_ineq0, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs076_ineq1, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs076_ineq2, nullptr, 1e-10);
        opt.set_maxeval(200);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        std::vector<double> x = {0.5, 0.5, 0.5, 0.5};
        double fval;
        opt.optimize(x, fval);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double fval = 0.0;
    nlopt_counts nl{};
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nl = {};
        nlopt::opt opt(nlopt::LD_CCSAQ, 4);
        opt.set_min_objective(nlopt_hs076_obj, &nl);
        opt.set_lower_bounds({0.0, 0.0, 0.0, 0.0});
        opt.set_upper_bounds({1e20, 1e20, 1e20, 1e20});
        opt.add_inequality_constraint(nlopt_hs076_ineq0, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs076_ineq1, nullptr, 1e-10);
        opt.add_inequality_constraint(nlopt_hs076_ineq2, nullptr, 1e-10);
        opt.set_maxeval(200);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        std::vector<double> x = {0.5, 0.5, 0.5, 0.5};
        opt.optimize(x, fval);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, nl.outer, nl.outer + nl.inner, nl.inner};
}

void print_row(std::string_view solver, const timing& t)
{
    double per_iter = t.iters > 0 ? double(t.inner) / t.iters : 0.0;
    std::println("  {:>12s}  {:10.2f}  {:8d}  {:8d}  {:8d}  {:>5.2f}  {:.6e}",
        solver, t.wall_us, t.iters, t.f_evals, t.inner, per_iter, t.objective);
}

}

int main()
{
    constexpr std::uint32_t reps = 100;
    std::println("MMA micro-benchmark, {} repetitions each\n", reps);
    std::println("  iters  = outer iters (nablapp counts.g | nlopt grad-bearing obj calls)");
    std::println("  f_evals = total obj-callback count");
    std::println("  inner  = f_evals - iters (conservativity-loop trials)");
    std::println("  in/it  = inner / iters (avg conservativity trials per outer)");
    std::println("  {:>12s}  {:>10s}  {:>8s}  {:>8s}  {:>8s}  {:>5s}  {:>12s}",
        "solver", "wall (us)", "iters", "f_evals", "inner", "in/it", "objective");

    auto print_block = [](std::string_view name, const timing& nab, const timing& nlop) {
        std::println("\n--- {} ---", name);
        print_row("nablapp", nab);
        print_row("nlopt", nlop);
        std::println("  ratio nablapp/nlopt: {:.2f}x wall, {:.2f}x iters, {:.2f}x f_evals, {:.2f}x inner",
            nab.wall_us / nlop.wall_us,
            nlop.iters > 0 ? double(nab.iters) / nlop.iters : 0.0,
            double(nab.f_evals) / nlop.f_evals,
            nlop.inner > 0 ? double(nab.inner) / nlop.inner : 0.0);
    };

    {
        auto nab  = bench_nablapp(hs024_dynamic{}, reps);
        auto nlop = bench_nlopt_hs024(reps);
        print_block("HS024 (inequality, n=2, f*=-1)", nab, nlop);
    }
    {
        auto nab  = bench_nablapp(hs043_dynamic{}, reps);
        auto nlop = bench_nlopt_hs043(reps);
        print_block("HS043 (inequality, n=4, f*=-44)", nab, nlop);
    }
    {
        auto nab  = bench_nablapp(hs076_dynamic{}, reps);
        auto nlop = bench_nlopt_hs076(reps);
        print_block("HS076 (inequality, n=4, f*=-4.6818)", nab, nlop);
    }
}
