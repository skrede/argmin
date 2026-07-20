// Micro-benchmark: argmin NW-SQP vs NLopt LD_SLSQP on equality/mixed HS problems.
//
// NW-SQP implements Nocedal & Wright Chapter 18 line-search SQP with
// damped BFGS and L1 merit function. Tested on problems with equality
// and mixed constraints that exercise the full QP solver path.
//
// Reference: Nocedal & Wright, Chapter 18, Sections 18.1-18.6.

#ifdef ARGMIN_BENCH_TRACE_ALLOC
#include "argmin/detail/bench/alloc_counter.h"
#endif

#include "bench_micro_gate.h"

#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/test_functions/hock_schittkowski.h"
#include "argmin/test_functions/small_dense.h"

#include <Eigen/Core>

#ifndef ARGMIN_BENCH_TRACE_ALLOC
#include <nlopt.hpp>
#endif

#include <cmath>
#include <chrono>
#include <limits>
#include <cstdint>

namespace
{

struct timing
{
    double wall_us;
    double objective;
    double constraint_violation;
    std::uint32_t evals;
    const char* unit;
};

// Dynamic-dimension HS039 wrapper (equality, n=4).
//   min  -x0
//   s.t. x1 - x0^3 - x2^2 = 0
//        x0^2 - x1 - x3^2 = 0
//   x0 = (2,2,2,2), f* = -1.
struct hs039_dynamic
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    [[nodiscard]] int dimension() const { return 4; }
    [[nodiscard]] int num_equality() const { return 2; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] double value(const Eigen::VectorXd& x) const
    {
        return -x[0];
    }

    void gradient(const Eigen::VectorXd&, Eigen::VectorXd& g) const
    {
        g[0] = -1.0; g[1] = 0.0; g[2] = 0.0; g[3] = 0.0;
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(2);
        c[0] = x[1] - x[0] * x[0] * x[0] - x[2] * x[2];
        c[1] = x[0] * x[0] - x[1] - x[3] * x[3];
    }

    void constraint_jacobian(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const
    {
        J.resize(2, 4);
        J(0, 0) = -3.0 * x[0] * x[0]; J(0, 1) = 1.0;
        J(0, 2) = -2.0 * x[2];         J(0, 3) = 0.0;
        J(1, 0) = 2.0 * x[0];          J(1, 1) = -1.0;
        J(1, 2) = 0.0;                  J(1, 3) = -2.0 * x[3];
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
        return Eigen::VectorXd::Constant(4, 2.0);
    }
};

// Dynamic-dimension HS071 wrapper (mixed, n=4).
// Reused from micro_kraft_slsqp pattern.
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

#ifndef ARGMIN_BENCH_TRACE_ALLOC
// NLopt callbacks for HS039.
double nlopt_hs039_obj(unsigned, const double* x, double* grad, void*)
{
    if(grad) { grad[0] = -1.0; grad[1] = 0.0; grad[2] = 0.0; grad[3] = 0.0; }
    return -x[0];
}

double nlopt_hs039_eq0(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = -3.0 * x[0] * x[0]; grad[1] = 1.0;
        grad[2] = -2.0 * x[2];         grad[3] = 0.0;
    }
    return x[1] - x[0] * x[0] * x[0] - x[2] * x[2];
}

double nlopt_hs039_eq1(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = 2.0 * x[0]; grad[1] = -1.0;
        grad[2] = 0.0;         grad[3] = -2.0 * x[3];
    }
    return x[0] * x[0] - x[1] - x[3] * x[3];
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
#endif

template <typename Problem>
timing bench_argmin(const Problem& problem, std::uint32_t reps)
{
    auto x0 = problem.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 5000;
    opts.set_gradient_threshold(1e-10);
    opts.set_objective_threshold(1e-8);
    opts.set_step_threshold(1e-12);

    // Warmup.
    {
        argmin::step_budget_solver solver{argmin::nw_sqp_policy<>{}, problem, x0, opts};
        solver.solve();
    }

    auto t0 = std::chrono::steady_clock::now();
    double fval = 0.0;
    double cv = 0.0;
    std::uint32_t iters = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        argmin::step_budget_solver solver{argmin::nw_sqp_policy<>{}, problem, x0, opts};
        auto result = solver.solve();
        fval = result.objective_value;
        cv = result.constraint_violation;
        iters = result.iterations;
    }
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, cv, iters, "steps"};
}

#ifndef ARGMIN_BENCH_TRACE_ALLOC
timing bench_nlopt_hs039(std::uint32_t reps)
{
    {
        nlopt::opt opt(nlopt::LD_SLSQP, 4);
        opt.set_min_objective(nlopt_hs039_obj, nullptr);
        opt.add_equality_constraint(nlopt_hs039_eq0, nullptr, 1e-10);
        opt.add_equality_constraint(nlopt_hs039_eq1, nullptr, 1e-10);
        opt.set_maxeval(5000);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        std::vector<double> x = {2.0, 2.0, 2.0, 2.0};
        double fval;
        opt.optimize(x, fval);
    }

    auto t0 = std::chrono::steady_clock::now();
    double fval = 0.0;
    double cv = 0.0;
    std::uint32_t evals = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nlopt::opt opt(nlopt::LD_SLSQP, 4);
        opt.set_min_objective(nlopt_hs039_obj, nullptr);
        opt.add_equality_constraint(nlopt_hs039_eq0, nullptr, 1e-10);
        opt.add_equality_constraint(nlopt_hs039_eq1, nullptr, 1e-10);
        opt.set_maxeval(5000);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        std::vector<double> x = {2.0, 2.0, 2.0, 2.0};
        opt.optimize(x, fval);
        hs039_dynamic problem;
        cv = argmin::bench::constraint_violation(
            problem, Eigen::Map<const Eigen::VectorXd>(x.data(), 4));
        evals = static_cast<std::uint32_t>(opt.get_numevals());
    }
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, cv, evals, "evals"};
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

    auto t0 = std::chrono::steady_clock::now();
    double fval = 0.0;
    double cv = 0.0;
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
        hs071_dynamic problem;
        cv = argmin::bench::constraint_violation(
            problem, Eigen::Map<const Eigen::VectorXd>(x.data(), 4));
        evals = static_cast<std::uint32_t>(opt.get_numevals());
    }
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, cv, evals, "evals"};
}
#endif

void print_row(std::string_view solver, const timing& t)
{
    argmin::bench::println("  {:>12s}  {:10.2f}  {:>8s}  {:10d}  {:10.2f}  {:.6e}  {:.6e}",
                           solver,
                           t.wall_us,
                           t.unit,
                           t.evals,
                           argmin::bench::per_unit_us(t.wall_us, t.evals),
                           t.objective,
                           t.constraint_violation);
}

// kkt_residual regression probe.
//
// nw_sqp is gradient-aware and must populate step_result::kkt_residual
// on every non-null step. This probe runs nw_sqp on HS039 via step_n()
// with a small budget and confirms the terminal step_result carries a
// populated, non-negative kkt_residual value.
//
// Reference: N&W 2e Section 12.3 / eq. 12.34.
bool probe_kkt_residual()
{
    hs039_dynamic problem;
    auto x0 = problem.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-10);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    argmin::step_budget_solver solver{argmin::nw_sqp_policy<>{}, problem, x0, opts};

    argmin::step_result<double> last{};
    for(std::uint32_t i = 0; i < opts.max_iterations; ++i)
    {
        last = solver.step();
        if(last.policy_status)
            break;
    }

    if(!last.kkt_residual.has_value())
    {
        argmin::bench::println("FAIL: kkt_residual not populated (nw_sqp)");
        return false;
    }
    if(last.kkt_residual.value() < 0.0)
    {
        argmin::bench::println("FAIL: kkt_residual is negative: {}",
                     last.kkt_residual.value());
        return false;
    }
    argmin::bench::println("  nw_sqp HS039 kkt_residual: {:.6e} (gradient_norm: {:.6e})",
                 last.kkt_residual.value(), last.gradient_norm);
    return true;
}

// Regression probe: nw_sqp on HS026 must reach f < 1e-5 after the
// Full E-measure (N&W 2e Definition 12.1) blocks a historical
// premature ftol at iter 12.
//
// Reference: N&W 2e Definition 12.1; historical baseline 20 iters.
bool probe_regression_hs026()
{
    argmin::hs026<> p;
    Eigen::VectorXd x0 = p.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    argmin::step_budget_solver solver{
        argmin::nw_sqp_policy<argmin::hs026<>::problem_dimension>{},
        p, x0, opts};
    argmin::step_result<double> last{};
    for(std::uint32_t i = 0; i < opts.max_iterations; ++i)
    {
        last = solver.step();
        if(last.policy_status)
            break;
    }

    const double kkt = last.kkt_residual.value_or(-1.0);
    const bool ok = last.objective_value < 1e-5;
    if(!ok)
        argmin::bench::println(stderr,
                     "FAIL: nw_sqp HS026 f={:.6e} kkt={:.6e}",
                     last.objective_value, kkt);
    argmin::bench::println("  nw_sqp HS026: f={:.6e} kkt={:.6e}",
                 last.objective_value, kkt);
    return ok;
}

// Regression probe: nw_sqp HS007 iter bound -- Lagrangian-stationarity
// tail-drift closure via multiplier re-estimation at x_{k+1} (N&W 2e
// Section 18.3 eq. 18.15). Reusing QP multipliers produced a stale
// stationarity leg that oscillated with the remaining problem curvature;
// the least-squares projection replaces them with the lambda best-
// explaining grad_f at the current iterate.
//
// Reference: N&W 2e Section 18.3 (multiplier re-estimation);
//            eq. 18.15 (least-squares lambda).
bool probe_regression_hs007_iter_bound()
{
    argmin::hs007<> p;
    Eigen::VectorXd x0 = p.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    argmin::step_budget_solver solver{
        argmin::nw_sqp_policy<argmin::hs007<>::problem_dimension>{},
        p, x0, opts};
    auto result = solver.solve(opts);

    const bool ok = result.iterations <= 12
        && result.objective_value < -1.7320;
    if(!ok)
        argmin::bench::println(stderr,
                     "FAIL: nw_sqp HS007 iters={} f={:.6e} (expected <= 12 @ f < -1.7320)",
                     result.iterations, result.objective_value);
    argmin::bench::println("  nw_sqp HS007: iters={} f={:.6e}",
                 result.iterations, result.objective_value);
    return ok;
}

}

#ifdef ARGMIN_BENCH_TRACE_ALLOC
namespace
{

// Steady-state allocation measurement for a fixed-N nw_sqp solve loop.
//
// Warmup boundary: a full warmup solve() walks the whole descent trajectory
// unarmed, warming every one-time / lazy allocation (the QP-solver workspace
// built at construction and the per-state buffer resizes). reset(x0) then
// returns to the start OUTSIDE any armed region, a short unarmed transient
// re-enters the steady descent, and only then does the armed window measure
// the pure per-step traffic. reset() never sits inside the armed window, so a
// reset-time allocation can never be miscounted as per-step. The window is
// asserted to be pre-convergence: if the policy signals termination inside it,
// the window is not steady state and the probe fails rather than reporting a
// vacuous zero.
template <typename Policy, typename Problem>
int measure_steady(const char* label, Policy policy, const Problem& problem,
                   std::size_t min_per_step)
{
    auto x0 = problem.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-10);

    argmin::step_budget_solver solver{policy, problem, x0, opts};

    solver.solve();
    solver.reset(x0);
    solver.step();
    solver.step();

    constexpr std::size_t hot_steps = 10;
    argmin::detail::bench::reset_alloc_count();
    argmin::detail::bench::arm_alloc_trace();
    bool terminated = false;
    for(std::size_t i = 0; i < hot_steps; ++i)
    {
        const auto r = solver.step();
        if(r.policy_status.has_value())
            terminated = true;
    }
    argmin::detail::bench::disarm_alloc_trace();

    if(terminated)
    {
        std::fprintf(stderr,
            "  [alloc-gate] %-18s FAIL: policy signaled termination inside "
            "the armed window -- not a pre-convergence steady state\n", label);
        return 1;
    }
    return argmin::detail::bench::evaluate_gate(label, hot_steps, min_per_step);
}

// Construction-window record (informational, never gates): arm before the
// solver is constructed and disarm after the first step, capturing the
// one-time setup allocation count that feeds the zero-after-construction
// characterization.
template <typename Policy, typename Problem>
void record_construction(const char* label, Policy policy, const Problem& problem)
{
    auto x0 = problem.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-10);

    argmin::detail::bench::reset_alloc_count();
    argmin::detail::bench::arm_alloc_trace();
    argmin::step_budget_solver solver{policy, problem, x0, opts};
    solver.step();
    argmin::detail::bench::disarm_alloc_trace();

    std::printf("  [alloc-gate] %-18s construction+first_step alloc=%zu\n",
                label, argmin::detail::bench::read_alloc_count());
}

// 12-D chained Rosenbrock with 2 equality + 4 inequality rows (6 runtime
// constraints) and box bounds, but a declared constraint_count of 12 -- the
// runtime constraint count sits strictly below the compile-time cap, so the
// qp_result multiplier storage is inline-max-bounded while carrying fewer
// active rows than its capacity. (cap + 2N)*8 = (12 + 24)*8 = 288 bytes, far
// inside the EIGEN_STACK_ALLOCATION_LIMIT inline regime. The constrained
// optimum is the all-ones point (feasible, objective 0). Math mirrors sd012;
// only the declared cap differs, so this exercises the runtime-m < cap path
// the sd012/sd024 gates (runtime-m == cap) never reach.
struct undercap_rosenbrock
{
    static constexpr int problem_dimension = 12;
    static constexpr int constraint_count = 12;

    [[nodiscard]] int dimension() const { return problem_dimension; }
    [[nodiscard]] int num_equality() const { return 2; }
    [[nodiscard]] int num_inequality() const { return 4; }

    [[nodiscard]] double value(const Eigen::Vector<double, problem_dimension>& x) const
    {
        double f = 0.0;
        for(int i = 0; i + 1 < problem_dimension; ++i)
        {
            const double a = x[i + 1] - x[i] * x[i];
            const double b = 1.0 - x[i];
            f += 100.0 * a * a + b * b;
        }
        return f;
    }

    void gradient(const Eigen::Vector<double, problem_dimension>& x,
                  Eigen::Vector<double, problem_dimension>& g) const
    {
        g.setZero();
        for(int i = 0; i + 1 < problem_dimension; ++i)
        {
            const double a = x[i + 1] - x[i] * x[i];
            g[i] += -400.0 * a * x[i] - 2.0 * (1.0 - x[i]);
            g[i + 1] += 200.0 * a;
        }
    }

    void constraints(const Eigen::Vector<double, problem_dimension>& x, auto& c) const
    {
        c[0] = x[0] + x[1] + x[2] + x[3] + x[4] + x[5] - 6.0;
        c[1] = x[6] + x[7] + x[8] + x[9] + x[10] + x[11] - 6.0;
        c[2] = 4.0 - x[0];
        c[3] = 4.0 - x[6];
        c[4] = x[3] + 3.0;
        c[5] = x[9] + 3.0;
    }

    void constraint_jacobian(const Eigen::Vector<double, problem_dimension>&, auto& J) const
    {
        J.setZero();
        for(int i = 0; i < 6; ++i)
            J(0, i) = 1.0;
        for(int i = 6; i < 12; ++i)
            J(1, i) = 1.0;
        J(2, 0) = -1.0;
        J(3, 6) = -1.0;
        J(4, 3) = 1.0;
        J(5, 9) = 1.0;
    }

    [[nodiscard]] Eigen::Vector<double, problem_dimension> lower_bounds() const
    {
        return Eigen::Vector<double, problem_dimension>::Constant(problem_dimension, -5.0);
    }

    [[nodiscard]] Eigen::Vector<double, problem_dimension> upper_bounds() const
    {
        return Eigen::Vector<double, problem_dimension>::Constant(problem_dimension, 5.0);
    }

    [[nodiscard]] Eigen::Vector<double, problem_dimension> initial_point() const
    {
        Eigen::Vector<double, problem_dimension> x0;
        for(int i = 0; i < problem_dimension; ++i)
            x0[i] = (i % 2 == 0) ? -1.2 : 1.0;
        return x0;
    }

    [[nodiscard]] double optimal_value() const { return 0.0; }
};

// Fresh un-armed solve witnessing that the under-cap fixture converges to its
// known optimum, so the gate proves runtime-m < cap is numerically correct and
// not merely allocation-free. Returns 0 on success, 1 on a missed optimum.
int witness_undercap_optimum()
{
    undercap_rosenbrock problem;
    auto x0 = problem.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-10);

    argmin::step_budget_solver solver{
        argmin::nw_sqp_policy<undercap_rosenbrock::problem_dimension>{},
        problem, x0, opts};
    const auto result = solver.solve();

    const double f_gap = std::abs(result.objective_value - problem.optimal_value());
    if(f_gap > 1e-6 || result.constraint_violation > 1e-6)
    {
        std::fprintf(stderr,
            "  [alloc-gate] nw_sqp bounded_m FAIL: under-cap solve missed the "
            "optimum (f=%.6e gap=%.6e cv=%.6e)\n",
            result.objective_value, f_gap, result.constraint_violation);
        return 1;
    }
    std::printf("  [alloc-gate] nw_sqp bounded_m witness: f=%.6e cv=%.6e "
                "(optimum reached)\n",
                result.objective_value, result.constraint_violation);
    return 0;
}

}

// Steady-state allocation gate for nw_sqp across the fixed-N fixture family,
// spanning unconstrained (sd006) through mixed-constrained near-ceiling
// (sd024). Each window measures the real per-step number with reset() kept
// outside the armed region; every window is held to zero allocations per
// steady-state step (zero mode in the gate build, min_per_step 0 in a plain
// build).
int argmin_alloc_trace_probe()
{
#ifdef ARGMIN_ALLOC_TRACE_BOUNDED_M
    // Runtime-m-below-cap steady-state gate: the declared constraint cap (12)
    // exceeds the runtime constraint count (6), so this exercises the inline
    // max-bounded multiplier storage carrying fewer active rows than capacity.
    record_construction("nw_sqp bounded_m",
        argmin::nw_sqp_policy<undercap_rosenbrock::problem_dimension>{},
        undercap_rosenbrock{});

    int rc = 0;
    rc |= measure_steady("nw_sqp bounded_m",
        argmin::nw_sqp_policy<undercap_rosenbrock::problem_dimension>{},
        undercap_rosenbrock{}, 0);
    rc |= witness_undercap_optimum();
    return rc;
#else
    record_construction("nw_sqp hs071",
        argmin::nw_sqp_policy<argmin::hs071<>::problem_dimension>{}, argmin::hs071<>{});

    int rc = 0;
    rc |= measure_steady("nw_sqp hs071",
        argmin::nw_sqp_policy<argmin::hs071<>::problem_dimension>{},
        argmin::hs071<>{}, 0);
    rc |= measure_steady("nw_sqp sd006",
        argmin::nw_sqp_policy<argmin::sd006<>::problem_dimension>{},
        argmin::sd006<>{}, 0);
    rc |= measure_steady("nw_sqp sd012",
        argmin::nw_sqp_policy<argmin::sd012<>::problem_dimension>{},
        argmin::sd012<>{}, 0);
    rc |= measure_steady("nw_sqp sd024",
        argmin::nw_sqp_policy<argmin::sd024<>::problem_dimension>{},
        argmin::sd024<>{}, 0);
    return rc;
#endif
}
#endif

#ifndef ARGMIN_BENCH_TRACE_ALLOC
int main()
{

    constexpr std::uint32_t reps = 500;

    if(!probe_kkt_residual())
        return 1;
    if(!probe_regression_hs026())
        return 1;
    if(!probe_regression_hs007_iter_bound())
        return 1;

    argmin::bench::println("NW-SQP micro-benchmark, {} repetitions each\n", reps);
    argmin::bench::println("  {:>12s}  {:>10s}  {:>8s}  {:>10s}  {:>10s}  {:>12s}  {:>12s}",
                           "solver", "solve_us", "unit", "units", "unit_us", "objective", "cv");

    // HS039
    {
        argmin::bench::println("\n--- HS039 (equality, n=4, f*=-1) ---");
        auto nab  = bench_argmin(hs039_dynamic{}, reps);
        auto nlop = bench_nlopt_hs039(reps);
        constexpr argmin::bench::micro_gate gate{-1.0, 1e-6, 1e-6};
        if(argmin::bench::comparison_passes(
               "HS039",
               {"argmin", nab.objective, nab.constraint_violation},
               {"nlopt", nlop.objective, nlop.constraint_violation},
               gate))
        {
            print_row("argmin", nab);
            print_row("nlopt", nlop);
            argmin::bench::println("  per-solve ratio argmin/nlopt: {:.1f}x",
                                   nab.wall_us / nlop.wall_us);
        }
    }

    // HS071
    {
        argmin::bench::println("\n--- HS071 (mixed, n=4, f*~17.014) ---");
        auto nab  = bench_argmin(hs071_dynamic{}, reps);
        auto nlop = bench_nlopt_hs071(reps);
        constexpr argmin::bench::micro_gate gate{17.0140173, 1e-6, 1e-6};
        if(argmin::bench::comparison_passes(
               "HS071",
               {"argmin", nab.objective, nab.constraint_violation},
               {"nlopt", nlop.objective, nlop.constraint_violation},
               gate))
        {
            print_row("argmin", nab);
            print_row("nlopt", nlop);
            argmin::bench::println("  per-solve ratio argmin/nlopt: {:.1f}x",
                                   nab.wall_us / nlop.wall_us);
        }
    }
}
#endif
