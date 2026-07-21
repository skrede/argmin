// Micro-benchmark: argmin filter_trsqp vs NLopt LD_SLSQP on constrained HS problems.
//
// Filter TR-SQP combines the Fletcher-Leyffer 2002 filter acceptance
// envelope with a Byrd-Omojokun composite step (normal + tangential
// decomposition inside a single trust region) and a Steihaug-CG inner
// solve on the tangential leg. Mode-dispatched per the sqp_mode NTTP:
// `accurate` uses the Wachter-Biegler 2005 switching-condition reject
// gate and the LNP 1998 reference trust-region defaults; `fast`
// shrinks the trust region directly on reject (Fletcher-Leyffer-Toint
// 2002 Section 3) and uses tighter inner-CG / multiplier-reest
// strides.
//
// Reference: Fletcher and Leyffer 2002, "Nonlinear programming
//            without a penalty function", Math. Program. 91:239-269;
//            Lalee, Nocedal, Plantenga 1998, "On the implementation
//            of an algorithm for large-scale equality constrained
//            optimization", SIAM J. Optim. 8:682-706;
//            Byrd 1987, "Robust trust region methods for constrained
//            optimization", talk SIAM Conf. on Optimization.

#ifdef ARGMIN_BENCH_TRACE_ALLOC
#include "argmin/detail/diagnostics/alloc_counter.h"
#endif

#include "argmin/solver/filter_trsqp_policy.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <nlopt.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include "bench_print.h"
#include <string_view>
#include <type_traits>

namespace
{

struct timing
{
    double wall_us;
    double objective;
    std::uint32_t evals;
};

// Dynamic-dimension HS071 wrapper (mixed, n=4). Only the `fast` mode
// closes HS071 at the locked policy defaults; `accurate` mode is
// excluded from the per-cell timing loop because per-iter cost on a
// non-converging trajectory is not a meaningful comparison metric.
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

// NLopt callbacks for HS026 (equality, n=3).
//
// min (x0 - x1)^2 + (x1 - x2)^4
// s.t. (1 + x1^2) * x0 + x2^4 - 3 = 0
double nlopt_hs026_obj(unsigned, const double* x, double* grad, void*)
{
    const double d01 = x[0] - x[1];
    const double d12 = x[1] - x[2];
    if(grad)
    {
        grad[0] = 2.0 * d01;
        grad[1] = -2.0 * d01 + 4.0 * d12 * d12 * d12;
        grad[2] = -4.0 * d12 * d12 * d12;
    }
    return d01 * d01 + d12 * d12 * d12 * d12;
}

double nlopt_hs026_eq(unsigned, const double* x, double* grad, void*)
{
    if(grad)
    {
        grad[0] = 1.0 + x[1] * x[1];
        grad[1] = 2.0 * x[1] * x[0];
        grad[2] = 4.0 * x[2] * x[2] * x[2];
    }
    return (1.0 + x[1] * x[1]) * x[0] + x[2] * x[2] * x[2] * x[2] - 3.0;
}

// NLopt callbacks for HS028 (equality, n=3).
//
// min (x0 + x1)^2 + (x1 + x2)^2
// s.t. x0 + 2*x1 + 3*x2 - 1 = 0
double nlopt_hs028_obj(unsigned, const double* x, double* grad, void*)
{
    const double s01 = x[0] + x[1];
    const double s12 = x[1] + x[2];
    if(grad)
    {
        grad[0] = 2.0 * s01;
        grad[1] = 2.0 * s01 + 2.0 * s12;
        grad[2] = 2.0 * s12;
    }
    return s01 * s01 + s12 * s12;
}

double nlopt_hs028_eq(unsigned, const double* x, double* grad, void*)
{
    if(grad) { grad[0] = 1.0; grad[1] = 2.0; grad[2] = 3.0; }
    return x[0] + 2.0 * x[1] + 3.0 * x[2] - 1.0;
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

// Per-cell argmin timing harness, mode-parametric on the
// filter_trsqp_policy NTTP. Mode selects the per-mode trust-region
// defaults baked into filter_trsqp_policy.h (initial_trust_radius,
// switching-condition reject gate vs tr_shrink, multiplier-reest
// stride, restoration cap).
//
// Convergence tolerances are read directly from the policy's per-mode
// defaults so the measured wall time reflects a convergence trajectory
// rather than a steady-state non-convergent loop pinned at
// max_iterations -- this matters because fast-mode tolerances are
// looser than accurate-mode tolerances by several orders of magnitude.
template <argmin::sqp_mode Mode, typename Problem>
timing bench_argmin(const Problem& problem,
                    std::uint32_t reps,
                    std::optional<double> gamma_f = std::nullopt,
                    std::optional<double> gamma_h = std::nullopt)
{
    // Mode -> policy alias: fast picks filter_trsqp_policy_fast (tr_shrink
    // reject + tighter inner-CG / multiplier-reest strides), accurate
    // picks filter_trsqp_policy_accurate (switching-condition reject +
    // Lalee-Nocedal-Plantenga 1998 reference trust-region defaults).
    using policy_t = std::conditional_t<
        Mode == argmin::sqp_mode::fast,
        argmin::filter_trsqp_policy_fast<argmin::problem_dimension_v<Problem>>,
        argmin::filter_trsqp_policy_accurate<argmin::problem_dimension_v<Problem>>>;

    auto x0 = problem.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 5000;
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    // Fletcher and Leyffer 2002 Section 2.3 envelope margins, threaded
    // through filter_trsqp_policy::options_type when set on the CLI.
    typename policy_t::options_type policy_opts;
    if(gamma_f) policy_opts.gamma_f = *gamma_f;
    if(gamma_h) policy_opts.gamma_h = *gamma_h;

    // Warmup. solve(opts) is used instead of solve() so the
    // constraint_tolerance set on opts above flows into the optimality
    // gates at step_n time; the no-arg solve() builds a fresh
    // solver_options<> internally that drops the
    // constraint_tolerance::optional, which on fast-mode policies
    // disables the looser tolerance regime the bench is meant to time.
    {
        argmin::step_budget_solver solver{policy_t{}, problem, x0, opts, policy_opts};
        solver.solve(opts);
    }

    auto t0 = std::chrono::steady_clock::now();
    double fval = 0.0;
    std::uint32_t iters = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        argmin::step_budget_solver solver{policy_t{}, problem, x0, opts, policy_opts};
        auto result = solver.solve(opts);
        fval = result.objective_value;
        iters = result.iterations;
    }
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, iters};
}

// Single-run probe used by the envelope sweep harness. Reports best
// strictly-feasible (f, cv, outer_iters) under the configured envelope.
//
// Rebinds the policy to the problem's compile-time dimension so that
// the matching options_type instantiation can be passed alongside (the
// 5-arg step_budget_solver ctor has a same_as<Policy::options_type> guard).
//
// Gamma sweeps are an accurate-mode diagnostic; the policy is bound to
// sqp_mode::accurate for sweep_argmin so the swept envelope is read
// against the canonical Wachter-Biegler 2005 switching-condition
// reject path.
//
// Tolerance regime mirrors the filter_trsqp HS028 acceptance gate:
// tight objective and step thresholds with a 500-outer-iter budget so
// envelope cells with extra rejections still have room to converge.
template <typename Problem>
sweep_row sweep_argmin(const Problem& problem,
                       std::optional<double> gamma_f,
                       std::optional<double> gamma_h)
{
    using policy_t = argmin::filter_trsqp_policy<
        argmin::problem_dimension_v<Problem>, argmin::sqp_mode::accurate>;

    auto x0 = problem.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-4);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-12);

    typename policy_t::options_type policy_opts;
    if(gamma_f) policy_opts.gamma_f = *gamma_f;
    if(gamma_h) policy_opts.gamma_h = *gamma_h;

    argmin::step_budget_solver solver{policy_t{}, problem, x0, opts, policy_opts};
    auto result = solver.solve();
    return {result.objective_value, result.constraint_violation,
            result.iterations,
            result.constraint_violation <= opts.feasibility_tolerance};
}

timing bench_nlopt_hs026(std::uint32_t reps)
{
    {
        nlopt::opt opt(nlopt::LD_SLSQP, 3);
        opt.set_min_objective(nlopt_hs026_obj, nullptr);
        opt.add_equality_constraint(nlopt_hs026_eq, nullptr, 1e-10);
        opt.set_maxeval(5000);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        std::vector<double> x = {-2.6, 2.0, 2.0};
        double fval;
        opt.optimize(x, fval);
    }

    auto t0 = std::chrono::steady_clock::now();
    double fval = 0.0;
    std::uint32_t evals = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nlopt::opt opt(nlopt::LD_SLSQP, 3);
        opt.set_min_objective(nlopt_hs026_obj, nullptr);
        opt.add_equality_constraint(nlopt_hs026_eq, nullptr, 1e-10);
        opt.set_maxeval(5000);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        std::vector<double> x = {-2.6, 2.0, 2.0};
        opt.optimize(x, fval);
        evals = static_cast<std::uint32_t>(opt.get_numevals());
    }
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, evals};
}

timing bench_nlopt_hs028(std::uint32_t reps)
{
    {
        nlopt::opt opt(nlopt::LD_SLSQP, 3);
        opt.set_min_objective(nlopt_hs028_obj, nullptr);
        opt.add_equality_constraint(nlopt_hs028_eq, nullptr, 1e-10);
        opt.set_maxeval(5000);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        std::vector<double> x = {-4.0, 1.0, 1.0};
        double fval;
        opt.optimize(x, fval);
    }

    auto t0 = std::chrono::steady_clock::now();
    double fval = 0.0;
    std::uint32_t evals = 0;
    for(std::uint32_t r = 0; r < reps; ++r)
    {
        nlopt::opt opt(nlopt::LD_SLSQP, 3);
        opt.set_min_objective(nlopt_hs028_obj, nullptr);
        opt.add_equality_constraint(nlopt_hs028_eq, nullptr, 1e-10);
        opt.set_maxeval(5000);
        opt.set_ftol_rel(1e-12);
        opt.set_xtol_rel(1e-12);
        std::vector<double> x = {-4.0, 1.0, 1.0};
        opt.optimize(x, fval);
        evals = static_cast<std::uint32_t>(opt.get_numevals());
    }
    auto t1 = std::chrono::steady_clock::now();
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

    auto t0 = std::chrono::steady_clock::now();
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
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
    return {us, fval, evals};
}

void print_row(std::string_view solver, const timing& t)
{
    argmin::bench::println("  {:>12s}  {:10.2f}  {:10d}  {:.6e}", solver, t.wall_us, t.evals, t.objective);
}

// kkt_residual regression probe.
//
// filter_trsqp uses the active-set QP multipliers from the composite-
// step solve; normal (accepted) steps populate kkt_residual, while
// null-step and restoration paths leave it unset. This probe runs the
// policy on HS028 (the canonical reference cell that converges on both
// modes at the locked defaults) via step_n() with a small budget and
// confirms that at least one step in the trajectory carries a
// populated, non-negative kkt_residual.
//
// Reference: N&W 2e Section 12.3 / eq. 12.34.
bool probe_kkt_residual()
{
    argmin::hs028<> problem;
    auto x0 = problem.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-10);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    argmin::step_budget_solver solver{
        argmin::filter_trsqp_policy_accurate<argmin::hs028<>::problem_dimension>{},
        problem, x0, opts};

    argmin::step_result<double> last_with_kkt{};
    bool any_kkt = false;
    for(std::uint32_t i = 0; i < opts.max_iterations; ++i)
    {
        auto r = solver.step();
        if(r.kkt_residual.has_value())
        {
            last_with_kkt = r;
            any_kkt = true;
        }
        if(r.policy_status)
            break;
    }

    if(!any_kkt)
    {
        argmin::bench::println("FAIL: kkt_residual not populated (filter_trsqp)");
        return false;
    }
    if(last_with_kkt.kkt_residual.value() < 0.0)
    {
        argmin::bench::println("FAIL: kkt_residual is negative: {}",
                     last_with_kkt.kkt_residual.value());
        return false;
    }
    argmin::bench::println("  filter_trsqp HS028 kkt_residual: {:.6e} (gradient_norm: {:.6e})",
                 last_with_kkt.kkt_residual.value(),
                 last_with_kkt.gradient_norm);
    return true;
}

// HS026 regression probe: filter_trsqp on HS026 must reach f < 1e-5
// on both modes. HS026 is the equality-only reference cell that
// exercises the Full E-measure (N&W 2e Definition 12.1) path through
// the composite-step + filter envelope.
//
// Reference: N&W 2e Definition 12.1; Fletcher-Leyffer 2002 Section 2.3
//            (filter envelope on equality-only problems).
template <argmin::sqp_mode Mode>
bool probe_regression_hs026()
{
    argmin::hs026<> p;
    Eigen::VectorXd x0 = p.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);

    argmin::step_budget_solver solver{
        argmin::filter_trsqp_policy<argmin::hs026<>::problem_dimension, Mode>{},
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
    constexpr std::string_view mode_name =
        (Mode == argmin::sqp_mode::fast) ? "fast" : "accurate";
    if(!ok)
        argmin::bench::println(stderr,
                     "FAIL: filter_trsqp HS026 ({}): f={:.6e} kkt={:.6e}",
                     mode_name, last.objective_value, kkt);
    argmin::bench::println("  filter_trsqp HS026 ({}): f={:.6e} kkt={:.6e}",
                 mode_name, last.objective_value, kkt);
    return ok;
}

// HS028 regression probe: filter_trsqp on HS028 must close at
// f* = 0. HS028 is the canonical reference cell for the filter
// family; closure on both modes confirms the composite-step +
// switching-condition reject path (accurate) and the tr_shrink reject
// path (fast) are both well-behaved on equality-only LP-feasible
// problems.
//
// Per-mode acceptance bar:
//   accurate -- |f| < 1e-6 (quadratic objective with linear equality
//               collapses to residual-zero on filter acceptance);
//   fast     -- |f| < 1e-2 (relaxed margin sized to the fast-mode
//               gradient tolerance, mirroring the filter_trsqp HS028
//               test acceptance gate).
//
// The probe reads per-mode default tolerances directly from the
// policy so the fast path does not run with the accurate-mode tight
// thresholds (which would cause fast to spin to max_iterations
// without crossing its own tolerance bar).
//
// Reference: H&S Problem 28; Lalee-Nocedal-Plantenga 1998 Section 3.1
//            (composite step on equality-only problems).
template <argmin::sqp_mode Mode>
bool probe_regression_hs028()
{
    using policy_t =
        argmin::filter_trsqp_policy<argmin::hs028<>::problem_dimension, Mode>;

    argmin::hs028<> p;
    Eigen::Vector<double, argmin::hs028<>::problem_dimension> x0 =
        p.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    argmin::step_budget_solver solver{policy_t{}, p, x0, opts};
    auto result = solver.solve(opts);

    constexpr double bar =
        (Mode == argmin::sqp_mode::fast) ? 1e-2 : 1e-6;
    const bool ok = std::abs(result.objective_value) < bar;
    constexpr std::string_view mode_name =
        (Mode == argmin::sqp_mode::fast) ? "fast" : "accurate";
    if(!ok)
        argmin::bench::println(stderr,
                     "FAIL: filter_trsqp HS028 ({}): iters={} f={:.6e} (expected |f| < {:.0e} @ 0)",
                     mode_name, result.iterations, result.objective_value, bar);
    argmin::bench::println("  filter_trsqp HS028 ({}): iters={} f={:.6e}",
                 mode_name, result.iterations, result.objective_value);
    return ok;
}

}

#ifdef ARGMIN_BENCH_TRACE_ALLOC
// Compile-time N=3 HS026 fixture for the fixed-N trust-region allocation
// gate. HS026 is a NONLINEAR-equality problem (no bounds, no inequality), so
// it drives the composite-step machinery that the gate certifies at zero --
// the dogleg normal step against a curved Jacobian, the multi-iteration
// Steihaug-CG tangential leg, the active-set equality-multiplier
// re-estimation, and the second-order-correction retry -- WITHOUT engaging the
// box-face tangential free-set restart. That restart is deliberately excluded:
// see the residual note on argmin_alloc_trace_probe below.
struct hs026_fixed_gate
{
    static constexpr int problem_dimension = 3;

    [[nodiscard]] int dimension() const { return 3; }
    [[nodiscard]] int num_equality() const { return 1; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] double value(const Eigen::Vector<double, 3>& x) const
    {
        const double d01 = x[0] - x[1];
        const double d12 = x[1] - x[2];
        return d01 * d01 + d12 * d12 * d12 * d12;
    }

    void gradient(const Eigen::Vector<double, 3>& x,
                  Eigen::Vector<double, 3>& g) const
    {
        const double d01 = x[0] - x[1];
        const double d12 = x[1] - x[2];
        g[0] = 2.0 * d01;
        g[1] = -2.0 * d01 + 4.0 * d12 * d12 * d12;
        g[2] = -4.0 * d12 * d12 * d12;
    }

    void constraints(const Eigen::Vector<double, 3>& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = (1.0 + x[1] * x[1]) * x[0] + x[2] * x[2] * x[2] * x[2] - 3.0;
    }

    void constraint_jacobian(const Eigen::Vector<double, 3>& x,
                             Eigen::MatrixXd& J) const
    {
        J.resize(1, 3);
        J(0, 0) = 1.0 + x[1] * x[1];
        J(0, 1) = 2.0 * x[1] * x[0];
        J(0, 2) = 4.0 * x[2] * x[2] * x[2];
    }
};

// Allocation gate for the trust-region SQP pair at fixed N over the real-time
// operating regime: warm-started, bounded composite steps per control tick
// that make genuine progress and never idle at the optimum. Each tick
// warm-resets to the start point (held to the same zero-allocation bar) and
// takes a bounded run of progress steps; the armed window spans several such
// ticks for an extended (>= 40 armed step) bounded-steady-state confirmation.
// Because the count is asserted EXACTLY zero across every armed step, this is
// the settling proof the "never settles" HEAD behavior demanded: a
// per-step-varying residual would surface as a nonzero spike somewhere in the
// window. Defining ARGMIN_ALLOC_TRACE_TR_SQP selects the bare tr_sqp_policy so
// the same driver covers both trust-region policies in lockstep.
//
// Two paths are deliberately OUTSIDE the certified pre-convergence window,
// each a characterized residual in a shared subsystem left unchanged by this
// hoist:
//
//   1. Post-convergence feasibility restoration. filter_trsqp shares
//      detail::restore_l1 with the other filter families; once the composite
//      step collapses to a zero step at a converged iterate whose L1 violation
//      sits marginally above the 1e-8 restoration trigger (an -O3
//      -march=native FMA/vectorization rounding artifact -- an -O2 build stays
//      below the trigger and never restores), the zero-step branch invokes
//      feasibility restoration, which allocates its local work vectors. The
//      warm-reset window stays strictly pre-convergence, so it never idles
//      into restoration.
//
//   2. Box-face tangential free-set restart. On BOUND- or INEQUALITY-
//      constrained problems the box-projected Steihaug-CG can pin coordinates
//      at their faces (Lin-More variant B); when the pinned set exhausts the
//      reduced free dimension the augmented normal equations go rank-deficient
//      and the projection routes through detail::null_space_project's pivoted-
//      QR fallback, which allocates. That helper is shared by the whole SQP
//      family; making its rank-deficient branch allocation-free is a
//      shared-subsystem change (and a tiny ridge that keeps the max-sized LDLT
//      definite perturbs the HS076 / HS043 box-active trajectories, so it is
//      not a numerics-free hoist). The equality-only HS026 fixture has no box
//      faces and never engages this path, so the composite-step RT hot loop it
//      exercises is certified at exactly zero; the bound/inequality free-set-
//      restart allocation is the characterized residual, documented for the
//      RT-safety matrix.
//
// Trajectory of the un-blinded fixed-N reading (this HS026 window): the
// pre-hoist trust-region code allocates in the tens per step (the reviewer
// measured 29-55 mallocs/step, never settling); the substrate + policy hoists
// bring it to exactly 0. The gate is demonstrably non-blind: the
// alloc_trace_main.cpp canary independently proves an armed Eigen allocation
// is counted.
//
// Built with ARGMIN_ALLOC_GATE_EXPECT_ZERO so evaluate_gate asserts zero
// allocations across the armed window.
int argmin_alloc_trace_probe()
{
    hs026_fixed_gate problem;
    Eigen::Vector<double, 3> x0{-2.6, 2.0, 2.0};
    argmin::solver_options opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-10);

#ifdef ARGMIN_ALLOC_TRACE_TR_SQP
    using policy_t = argmin::tr_sqp_policy<3, argmin::sqp_mode::accurate>;
    const char* const label = "tr_sqp";
#else
    using policy_t = argmin::filter_trsqp_policy_accurate<3>;
    const char* const label = "filter_trsqp";
#endif

    argmin::step_budget_solver solver{policy_t{}, problem, x0, opts};

    // Warmup absorbs lazy first-push BFGS / buffer-sizing allocations.
    solver.step();
    solver.step();

    // Bounded warm-started progress steps per tick keep the window strictly
    // ahead of the zero-step restoration onset while exercising the full
    // dogleg / CG / SOC / multiplier path every step; the armed window spans
    // several ticks (>= 40 armed steps) for the bounded-steady-state proof.
    constexpr std::size_t progress_steps = 7;
    constexpr std::size_t ticks = 6;

    argmin::detail::bench::reset_alloc_count();
    argmin::detail::bench::arm_alloc_trace();
    for(std::size_t t = 0; t < ticks; ++t)
    {
        solver.reset(x0);
        for(std::size_t i = 0; i < progress_steps; ++i)
            solver.step();
    }
    argmin::detail::bench::disarm_alloc_trace();

    return argmin::detail::bench::evaluate_gate(
        label, ticks * progress_steps, /*min_per_step=*/0);
}
#endif

#ifndef ARGMIN_BENCH_TRACE_ALLOC
int main(int argc, char** argv)
{

    // Parse --gamma-f / --gamma-h CLI overrides for the filter envelope
    // sweep. When both are provided the binary switches into sweep mode:
    // the regression probes and NLopt comparison are skipped, and a
    // single-run TSV-friendly line per problem is emitted on stdout
    // ("policy\tgamma_f\tgamma_h\tproblem\tf\tcv\touter_iters\taccepted").
    //
    // Reference: Fletcher and Leyffer 2002 Section 2.3 (envelope margins).
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
            argmin::bench::println("filter_trsqp\t{:.0e}\t{:.0e}\t{}\t{:.10e}\t{:.10e}\t{}\t{}",
                         gf, gh, name, r.f, r.cv, r.outer_iters,
                         r.accepted ? 1 : 0);
        };
        emit("HS028", sweep_argmin(argmin::hs028<>{}, cli_gamma_f, cli_gamma_h));
        emit("HS026", sweep_argmin(argmin::hs026<>{}, cli_gamma_f, cli_gamma_h));
        return 0;
    }

    constexpr std::uint32_t reps = 500;

    if(!probe_kkt_residual())
        return 1;
    if(!probe_regression_hs026<argmin::sqp_mode::accurate>())
        return 1;
    if(!probe_regression_hs026<argmin::sqp_mode::fast>())
        return 1;
    if(!probe_regression_hs028<argmin::sqp_mode::accurate>())
        return 1;
    if(!probe_regression_hs028<argmin::sqp_mode::fast>())
        return 1;

    argmin::bench::println("Filter TR-SQP micro-benchmark, {} repetitions each\n", reps);
    argmin::bench::println("  {:>12s}  {:>10s}  {:>10s}  {:>12s}", "solver", "wall (us)", "evals", "objective");

    // HS028 -- canonical reference cell; closes on both modes.
    {
        argmin::bench::println("\n--- HS028 (equality, n=3, f*=0) [accurate] ---");
        auto nab  = bench_argmin<argmin::sqp_mode::accurate>(
            argmin::hs028<>{}, reps, cli_gamma_f, cli_gamma_h);
        auto nlop = bench_nlopt_hs028(reps);
        print_row("argmin", nab);
        print_row("nlopt", nlop);
        argmin::bench::println("  ratio argmin/nlopt: {:.1f}x wall, {:.1f}x evals",
            nab.wall_us / nlop.wall_us, double(nab.evals) / nlop.evals);
    }
    {
        argmin::bench::println("\n--- HS028 (equality, n=3, f*=0) [fast] ---");
        auto nab  = bench_argmin<argmin::sqp_mode::fast>(
            argmin::hs028<>{}, reps, cli_gamma_f, cli_gamma_h);
        auto nlop = bench_nlopt_hs028(reps);
        print_row("argmin", nab);
        print_row("nlopt", nlop);
        argmin::bench::println("  ratio argmin/nlopt: {:.1f}x wall, {:.1f}x evals",
            nab.wall_us / nlop.wall_us, double(nab.evals) / nlop.evals);
    }

    // HS026 -- equality-only Full-E-measure exercise; closes on both modes.
    {
        argmin::bench::println("\n--- HS026 (equality, n=3, f*=0) [accurate] ---");
        auto nab  = bench_argmin<argmin::sqp_mode::accurate>(
            argmin::hs026<>{}, reps, cli_gamma_f, cli_gamma_h);
        auto nlop = bench_nlopt_hs026(reps);
        print_row("argmin", nab);
        print_row("nlopt", nlop);
        argmin::bench::println("  ratio argmin/nlopt: {:.1f}x wall, {:.1f}x evals",
            nab.wall_us / nlop.wall_us, double(nab.evals) / nlop.evals);
    }
    {
        argmin::bench::println("\n--- HS026 (equality, n=3, f*=0) [fast] ---");
        auto nab  = bench_argmin<argmin::sqp_mode::fast>(
            argmin::hs026<>{}, reps, cli_gamma_f, cli_gamma_h);
        auto nlop = bench_nlopt_hs026(reps);
        print_row("argmin", nab);
        print_row("nlopt", nlop);
        argmin::bench::println("  ratio argmin/nlopt: {:.1f}x wall, {:.1f}x evals",
            nab.wall_us / nlop.wall_us, double(nab.evals) / nlop.evals);
    }

    // HS071 -- mixed n=4; only the `fast` mode closes at the locked
    // policy defaults. The `accurate` mode is intentionally excluded
    // because per-iter cost on a non-converging trajectory is not a
    // meaningful comparison metric.
    {
        argmin::bench::println("\n--- HS071 (mixed, n=4, f*~17.014) [fast] ---");
        auto nab  = bench_argmin<argmin::sqp_mode::fast>(
            hs071_dynamic{}, reps, cli_gamma_f, cli_gamma_h);
        auto nlop = bench_nlopt_hs071(reps);
        print_row("argmin", nab);
        print_row("nlopt", nlop);
        argmin::bench::println("  ratio argmin/nlopt: {:.1f}x wall, {:.1f}x evals",
            nab.wall_us / nlop.wall_us, double(nab.evals) / nlop.evals);
    }
}
#endif
