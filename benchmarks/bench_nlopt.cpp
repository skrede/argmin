// NLopt comparison benchmarks for nablapp benchmark suite.
//
// Each NLopt solver is benchmarked on applicable problems using NLopt's native
// C++ API (per D-01: no common adapter interface). Results are collected as
// benchmark_result structs with library = "nlopt".
//
// Solver mapping (per D-04):
//   Unconstrained:         NLOPT_LD_LBFGS    -> "nlopt_lbfgs"
//   Bound-constrained:     NLOPT_LN_BOBYQA   -> "nlopt_bobyqa"
//   Inequality-constrained: NLOPT_LD_SLSQP   -> "nlopt_slsqp"
//                           NLOPT_LD_MMA      -> "nlopt_mma"
//   Global:                NLOPT_GN_CRS2_LM  -> "nlopt_crs2"
//                           NLOPT_GN_ISRES    -> "nlopt_isres"
//
// IMPORTANT: NLOPT_LD_LBFGS is NOT L-BFGS-B (Byrd-Lu-Nocedal-Zhu).
// NLopt's LD_LBFGS wraps Nocedal's original unconstrained L-BFGS.
// Bound handling is via external clamping, not the B-L-N-Z strategy.
// For bound-constrained comparison, use NLOPT_LN_BOBYQA instead.
// Reference: NLopt documentation, Algorithms section.

#include "bench_nlopt.h"
#include "trace_entry.h"
#include "counting_problem.h"
#include "problem_registry.h"

#include "nablapp/formulation/concepts.h"

#include <nlopt.hpp>

#include <chrono>
#include <cmath>
#include <limits>
#include <string_view>
#include <vector>

namespace nablapp::bench
{

namespace detail
{

// Trace wrapper bundling the counting_problem<P> with a destination trace
// vector and the timing baseline so the publication-mode objective callback
// can append per-invocation trace rows while still bumping the four
// problem-level eval counters (Pattern 5 in 32.8-RESEARCH.md).
//
// NLopt has no native per-iter callback hook; the objective callback fires
// once per (value, gradient) invocation pair, so "iter" in the NLopt trace
// becomes a monotonic eval index. Constraint callbacks still go through the
// bare counting_problem<P>* and do not emit trace rows; their f/g/c/J counter
// bumps are visible in the NEXT objective-callback trace row's counter
// fields, which is consistent with the eval-budget interpretation used by
// the post-processing dm_profile.py.
template <typename Problem>
struct nlopt_trace_wrapper
{
    counting_problem<Problem>* prob{nullptr};
    std::vector<trace_entry>*  trace{nullptr};
    std::int64_t               t0_us{0};
    double                     f_star{};
    double                     f_best_running{std::numeric_limits<double>::infinity()};
    int                        iter_count{0};
};

// Wraps any nablapp problem as an NLopt objective callback.
// Problem must provide value() and gradient().
//
// `data` points at a counting_problem<Problem>; every value()/gradient()
// invocation bumps the shared eval_counts so summary CSV columns
// {f,g,c,J}_evals are populated independently of NLopt's native
// get_numevals() (which folds value+gradient into one invocation).
template <typename Problem>
double nlopt_objective(unsigned n, const double* x, double* grad, void* data)
{
    auto* prob = static_cast<counting_problem<Problem>*>(data);
    Eigen::Map<const Eigen::VectorXd> xmap(x, n);
    Eigen::Vector<double, Problem::problem_dimension> xv(xmap);

    if(grad)
    {
        Eigen::Vector<double, Problem::problem_dimension> g;
        prob->gradient(xv, g);
        Eigen::Map<Eigen::VectorXd>(grad, n) = g;
    }

    return prob->value(xv);
}

// Trace-emitting objective callback used under publication mode.
// `data` points at an nlopt_trace_wrapper<Problem>; every invocation
// appends a trace_entry row populated from the four problem-level counters
// plus the running-min objective. step_norm and kkt_residual are NaN per
// CONTEXT D-C3 capability footnote (NLopt's objective callback does not
// surface either quantity).
template <typename Problem>
double nlopt_trace_objective(unsigned n, const double* x, double* grad, void* data)
{
    auto* w = static_cast<nlopt_trace_wrapper<Problem>*>(data);
    Eigen::Map<const Eigen::VectorXd> xmap(x, n);
    Eigen::Vector<double, Problem::problem_dimension> xv(xmap);

    if(grad)
    {
        Eigen::Vector<double, Problem::problem_dimension> g;
        w->prob->gradient(xv, g);
        Eigen::Map<Eigen::VectorXd>(grad, n) = g;
    }
    const double f = w->prob->value(xv);

    w->f_best_running = std::min(w->f_best_running, f);

    const auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    w->trace->push_back(trace_entry{
        .iter         = w->iter_count++,
        .f_evals      = w->prob->counts->f,
        .g_evals      = w->prob->counts->g,
        .c_evals      = w->prob->counts->c,
        .J_evals      = w->prob->counts->J,
        .wall_us      = now_us - w->t0_us,
        .f_current    = f,
        .f_best       = w->f_best_running,
        .accuracy     = std::abs(f - w->f_star),
        .cv           = std::numeric_limits<double>::quiet_NaN(),
        .step_norm    = std::numeric_limits<double>::quiet_NaN(),
        .kkt_residual = std::numeric_limits<double>::quiet_NaN(),
    });

    return f;
}

// Wraps nablapp vectorised constraints for NLopt's mconstraint interface.
// NLopt inequality: c(x) <= 0.  nablapp convention: c(x) >= 0.
// We negate: result[i] = -c_nablapp[i].
template <typename Problem>
void nlopt_ineq_mconstraint(unsigned m, double* result,
                            unsigned n, const double* x,
                            double* grad, void* data)
{
    auto* prob = static_cast<counting_problem<Problem>*>(data);
    Eigen::Map<const Eigen::VectorXd> xmap(x, n);
    Eigen::Vector<double, Problem::problem_dimension> xv(xmap);

    int n_eq = prob->num_equality();
    int total_constraints = n_eq + static_cast<int>(m);
    Eigen::VectorXd c(total_constraints);
    prob->constraints(xv, c);

    // Only inequality constraints (skip leading equalities).
    for(unsigned i = 0; i < m; ++i)
        result[i] = -c[n_eq + static_cast<int>(i)];

    if(grad)
    {
        Eigen::MatrixXd J(total_constraints, static_cast<int>(n));
        prob->constraint_jacobian(xv, J);
        // grad layout: grad[i*n + j] = d(result[i])/dx[j]
        for(unsigned i = 0; i < m; ++i)
            for(unsigned j = 0; j < n; ++j)
                grad[i * n + j] = -J(n_eq + static_cast<int>(i),
                                     static_cast<int>(j));
    }
}

// Wraps nablapp equality constraints for NLopt's mconstraint interface.
// NLopt equality: h(x) = 0.  nablapp convention: c_eq(x) = 0 (same).
template <typename Problem>
void nlopt_eq_mconstraint(unsigned m, double* result,
                          unsigned n, const double* x,
                          double* grad, void* data)
{
    auto* prob = static_cast<counting_problem<Problem>*>(data);
    Eigen::Map<const Eigen::VectorXd> xmap(x, n);
    Eigen::Vector<double, Problem::problem_dimension> xv(xmap);

    int total_constraints = static_cast<int>(m) + prob->num_inequality();
    Eigen::VectorXd c(total_constraints);
    prob->constraints(xv, c);

    // Equality constraints are the leading rows.
    for(unsigned i = 0; i < m; ++i)
        result[i] = c[static_cast<int>(i)];

    if(grad)
    {
        Eigen::MatrixXd J(total_constraints, static_cast<int>(n));
        prob->constraint_jacobian(xv, J);
        for(unsigned i = 0; i < m; ++i)
            for(unsigned j = 0; j < n; ++j)
                grad[i * n + j] = J(static_cast<int>(i),
                                    static_cast<int>(j));
    }
}

[[nodiscard]] auto nlopt_result_string(nlopt::result r) -> std::string_view
{
    switch(r)
    {
    case nlopt::SUCCESS:          return "converged";
    case nlopt::STOPVAL_REACHED:  return "stopval_reached";
    case nlopt::FTOL_REACHED:     return "ftol_reached";
    case nlopt::XTOL_REACHED:     return "xtol_reached";
    case nlopt::MAXEVAL_REACHED:  return "max_iterations";
    case nlopt::MAXTIME_REACHED:  return "maxtime_reached";
    case nlopt::ROUNDOFF_LIMITED: return "roundoff_limited";
    default:                      return "failed";
    }
}

// Run a single NLopt solver on a problem and return a benchmark_result.
//
// Under config.trace_enabled, each objective-callback invocation appends a
// trace_entry row into local_trace via nlopt_trace_wrapper<Problem>; the
// caller pushes the populated local_trace into the per-problem traces[].
template <typename Problem>
auto run_nlopt_solver(nlopt::algorithm algo,
                      std::string_view solver_name,
                      std::string_view problem_name,
                      Problem& prob,
                      int /*max_evals_legacy*/,
                      const bench_config& config,
                      std::vector<trace_entry>& local_trace) -> benchmark_result
{
    eval_counts counts;
    counting_problem<Problem> wrapped{prob, counts};

    auto n = static_cast<unsigned>(prob.dimension());
    nlopt::opt opt(algo, n);

    // Trace wrapper is only consulted when config.trace_enabled is true; the
    // bare counting_problem<Problem>* path preserves byte-identical
    // library_defaults behavior. The `t0_us` baseline is captured AFTER all
    // setup completes (right before opt.optimize() below) so that per-iter
    // wall_us measures only solve time, matching the summary wall_time_us
    // baseline. The wrapper is constructed here with t0_us=0 and the field
    // is patched right before optimize() is called.
    nlopt_trace_wrapper<Problem> tw{
        .prob           = &wrapped,
        .trace          = &local_trace,
        .t0_us          = 0,
        .f_star         = prob.optimal_value(),
    };

    // Set objective; route through the trace wrapper only under publication
    // mode, otherwise keep the legacy data pointer wiring.
    if(config.trace_enabled)
        opt.set_min_objective(nlopt_trace_objective<Problem>, &tw);
    else
        opt.set_min_objective(nlopt_objective<Problem>, &wrapped);

    // Set bounds if the problem provides them.
    if constexpr(bound_constrained<Problem>)
    {
        auto lb = prob.lower_bounds();
        auto ub = prob.upper_bounds();
        std::vector<double> lb_vec(lb.data(), lb.data() + n);
        std::vector<double> ub_vec(ub.data(), ub.data() + n);
        opt.set_lower_bounds(lb_vec);
        opt.set_upper_bounds(ub_vec);
    }

    // Add constraints if the problem provides them.
    if constexpr(constrained<Problem>)
    {
        int n_ineq = prob.num_inequality();
        int n_eq = prob.num_equality();

        if(n_ineq > 0)
        {
            std::vector<double> tol(static_cast<std::size_t>(n_ineq), 1e-8);
            opt.add_inequality_mconstraint(
                nlopt_ineq_mconstraint<Problem>, &wrapped, tol);
        }
        if(n_eq > 0)
        {
            std::vector<double> tol(static_cast<std::size_t>(n_eq), 1e-8);
            opt.add_equality_mconstraint(
                nlopt_eq_mconstraint<Problem>, &wrapped, tol);
        }
    }

    // Stopping criteria sourced from bench_config so library_defaults and
    // publication modes drive distinct solver behavior. The 1e-12 / 1e-16
    // ftol gap between the two modes is the central methodology lever
    // making every tau in the publication-grade DM grid observable.
    opt.set_maxeval(config.max_f_evals);
    opt.set_ftol_rel(config.ftol_rel);
    opt.set_xtol_rel(config.xtol_rel);
    opt.set_maxtime(config.max_wall_time_s);

    // Initial point.
    auto x0 = prob.initial_point();
    std::vector<double> x(x0.data(), x0.data() + n);
    double minf{};

    // Patch the trace baseline immediately before solve so per-iter wall_us
    // matches the summary's t0 reference frame (both exclude setup overhead).
    tw.t0_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    // Solve with timing.
    auto t0 = std::chrono::high_resolution_clock::now();

    nlopt::result res{};
    std::string_view status_str;
    try
    {
        res = opt.optimize(x, minf);
        status_str = nlopt_result_string(res);
    }
    catch(const nlopt::roundoff_limited&)
    {
        minf = opt.last_optimum_value();
        status_str = "roundoff_limited";
    }
    catch(const std::exception&)
    {
        minf = opt.last_optimum_value();
        status_str = "failed";
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto wall_us = std::chrono::duration_cast<
        std::chrono::microseconds>(t1 - t0).count();

    double known_opt = prob.optimal_value();

    return benchmark_result{
        .solver = solver_name,
        .library = "nlopt",
        .problem = problem_name,
        .pclass = prob.pclass,
        .dimension = prob.dimension(),
        .seed = config.seed,
        .mode = (config.the_mode == bench_config::mode::publication)
                    ? std::string_view{"publication"}
                    : std::string_view{"library_defaults"},
        .solver_iters = opt.get_numevals(),
        .f_evals = counts.f,
        .g_evals = counts.g,
        .c_evals = counts.c,
        .J_evals = counts.J,
        .wall_time_us = wall_us,
        .final_objective = minf,
        .known_optimum = known_opt,
        .accuracy = std::abs(minf - known_opt),
        .status = status_str,
    };
}

// Run a single NLopt solver with a subsidiary local optimizer (for AUGLAG).
template <typename Problem>
auto run_nlopt_auglag(std::string_view problem_name,
                      Problem& prob,
                      int /*max_evals_legacy*/,
                      const bench_config& config,
                      std::vector<trace_entry>& local_trace) -> benchmark_result
{
    eval_counts counts;
    counting_problem<Problem> wrapped{prob, counts};

    auto n = static_cast<unsigned>(prob.dimension());
    nlopt::opt opt(nlopt::LD_AUGLAG, n);

    // Subsidiary solver for augmented Lagrangian inner loop. Inner ftol/xtol
    // mirror the outer config so the inner L-BFGS does not stop short of the
    // outer publication-mode tolerance regime.
    nlopt::opt local_opt(nlopt::LD_LBFGS, n);
    local_opt.set_ftol_rel(config.ftol_rel);
    local_opt.set_xtol_rel(config.xtol_rel);
    local_opt.set_maxeval(config.max_f_evals);
    opt.set_local_optimizer(local_opt);

    nlopt_trace_wrapper<Problem> tw{
        .prob           = &wrapped,
        .trace          = &local_trace,
        .t0_us          = 0,
        .f_star         = prob.optimal_value(),
    };

    if(config.trace_enabled)
        opt.set_min_objective(nlopt_trace_objective<Problem>, &tw);
    else
        opt.set_min_objective(nlopt_objective<Problem>, &wrapped);

    if constexpr(bound_constrained<Problem>)
    {
        auto lb = prob.lower_bounds();
        auto ub = prob.upper_bounds();
        std::vector<double> lb_vec(lb.data(), lb.data() + n);
        std::vector<double> ub_vec(ub.data(), ub.data() + n);
        opt.set_lower_bounds(lb_vec);
        opt.set_upper_bounds(ub_vec);
    }

    if constexpr(constrained<Problem>)
    {
        int n_ineq = prob.num_inequality();
        int n_eq = prob.num_equality();
        if(n_ineq > 0)
        {
            std::vector<double> tol(static_cast<std::size_t>(n_ineq), 1e-8);
            opt.add_inequality_mconstraint(
                nlopt_ineq_mconstraint<Problem>, &wrapped, tol);
        }
        if(n_eq > 0)
        {
            std::vector<double> tol(static_cast<std::size_t>(n_eq), 1e-8);
            opt.add_equality_mconstraint(
                nlopt_eq_mconstraint<Problem>, &wrapped, tol);
        }
    }

    opt.set_maxeval(config.max_f_evals);
    opt.set_ftol_rel(config.ftol_rel);
    opt.set_xtol_rel(config.xtol_rel);
    opt.set_maxtime(config.max_wall_time_s);

    auto x0 = prob.initial_point();
    std::vector<double> x(x0.data(), x0.data() + n);
    double minf{};

    tw.t0_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    auto t0 = std::chrono::high_resolution_clock::now();

    std::string_view status_str;
    try
    {
        auto res = opt.optimize(x, minf);
        status_str = nlopt_result_string(res);
    }
    catch(const nlopt::roundoff_limited&)
    {
        minf = opt.last_optimum_value();
        status_str = "roundoff_limited";
    }
    catch(const std::exception&)
    {
        minf = opt.last_optimum_value();
        status_str = "failed";
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto wall_us = std::chrono::duration_cast<
        std::chrono::microseconds>(t1 - t0).count();

    double known_opt = prob.optimal_value();

    return benchmark_result{
        .solver = "nlopt_auglag",
        .library = "nlopt",
        .problem = problem_name,
        .pclass = prob.pclass,
        .dimension = prob.dimension(),
        .seed = config.seed,
        .mode = (config.the_mode == bench_config::mode::publication)
                    ? std::string_view{"publication"}
                    : std::string_view{"library_defaults"},
        .solver_iters = opt.get_numevals(),
        .f_evals = counts.f,
        .g_evals = counts.g,
        .c_evals = counts.c,
        .J_evals = counts.J,
        .wall_time_us = wall_us,
        .final_objective = minf,
        .known_optimum = known_opt,
        .accuracy = std::abs(minf - known_opt),
        .status = status_str,
    };
}

// Run NLopt GN_ISRES (derivative-free evolutionary constrained optimizer)
// on a problem. ISRES requires bounds and handles both inequality and
// equality constraints via individual callbacks. It needs relaxed stopping
// criteria and deterministic seeding.
//
// Reference: Runarsson & Yao, "Search biases in constrained evolutionary
//            optimization", IEEE TSMC-C, 35(2), 2005.
template <typename Problem>
auto run_nlopt_isres(std::string_view problem_name,
                     Problem& prob,
                     int /*max_evals_legacy*/,
                     const bench_config& config,
                     std::vector<trace_entry>& local_trace) -> benchmark_result
{
    eval_counts counts;
    counting_problem<Problem> wrapped{prob, counts};

    auto n = static_cast<unsigned>(prob.dimension());
    nlopt::opt opt(nlopt::GN_ISRES, n);

    // Deterministic seed for reproducible benchmarks.
    nlopt::srand(static_cast<unsigned long>(config.seed));

    nlopt_trace_wrapper<Problem> tw{
        .prob           = &wrapped,
        .trace          = &local_trace,
        .t0_us          = 0,
        .f_star         = prob.optimal_value(),
    };

    // ISRES uses derivative-free objective evaluation.
    if(config.trace_enabled)
        opt.set_min_objective(nlopt_trace_objective<Problem>, &tw);
    else
        opt.set_min_objective(nlopt_objective<Problem>, &wrapped);

    // Bounds are required for ISRES.
    if constexpr(bound_constrained<Problem>)
    {
        auto lb = prob.lower_bounds();
        auto ub = prob.upper_bounds();
        std::vector<double> lb_vec(lb.data(), lb.data() + n);
        std::vector<double> ub_vec(ub.data(), ub.data() + n);
        opt.set_lower_bounds(lb_vec);
        opt.set_upper_bounds(ub_vec);
    }

    // Register constraints individually (not mconstraint) for maximum
    // compatibility with ISRES. NLopt's GN_ISRES supports both interfaces
    // but individual callbacks are more reliable across versions.
    if constexpr(constrained<Problem>)
    {
        int n_ineq = prob.num_inequality();
        int n_eq = prob.num_equality();

        if(n_ineq > 0)
        {
            std::vector<double> tol(static_cast<std::size_t>(n_ineq), 1e-6);
            opt.add_inequality_mconstraint(
                nlopt_ineq_mconstraint<Problem>, &wrapped, tol);
        }
        if(n_eq > 0)
        {
            std::vector<double> tol(static_cast<std::size_t>(n_eq), 1e-6);
            opt.add_equality_mconstraint(
                nlopt_eq_mconstraint<Problem>, &wrapped, tol);
        }
    }

    // ISRES is evolutionary; tolerances and budgets sourced from bench_config
    // so library_defaults preserves the prior 1e-12 stopping behavior while
    // publication mode tightens to 1e-16 + 10 s wall budget.
    opt.set_maxeval(config.max_f_evals);
    opt.set_ftol_rel(config.ftol_rel);
    opt.set_xtol_rel(config.xtol_rel);
    opt.set_maxtime(config.max_wall_time_s);

    auto x0 = prob.initial_point();
    std::vector<double> x(x0.data(), x0.data() + n);
    double minf{};

    tw.t0_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    auto t0 = std::chrono::high_resolution_clock::now();

    nlopt::result res{};
    std::string_view status_str;
    try
    {
        res = opt.optimize(x, minf);
        status_str = nlopt_result_string(res);
    }
    catch(const nlopt::roundoff_limited&)
    {
        minf = opt.last_optimum_value();
        status_str = "roundoff_limited";
    }
    catch(const std::exception&)
    {
        minf = opt.last_optimum_value();
        status_str = "failed";
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto wall_us = std::chrono::duration_cast<
        std::chrono::microseconds>(t1 - t0).count();

    double known_opt = prob.optimal_value();

    return benchmark_result{
        .solver = "nlopt_isres",
        .library = "nlopt",
        .problem = problem_name,
        .pclass = prob.pclass,
        .dimension = prob.dimension(),
        .seed = config.seed,
        .mode = (config.the_mode == bench_config::mode::publication)
                    ? std::string_view{"publication"}
                    : std::string_view{"library_defaults"},
        .solver_iters = opt.get_numevals(),
        .f_evals = counts.f,
        .g_evals = counts.g,
        .c_evals = counts.c,
        .J_evals = counts.J,
        .wall_time_us = wall_us,
        .final_objective = minf,
        .known_optimum = known_opt,
        .accuracy = std::abs(minf - known_opt),
        .status = status_str,
    };
}

} // detail

void run_nlopt_benchmarks(std::vector<benchmark_result>& results,
                          std::vector<std::vector<trace_entry>>& traces,
                          const bench_config& config)
{
    // bench_config consumption: every adapter call wraps prob through
    // counting_problem<P>. Under config.trace_enabled, each runner appends a
    // per-invocation trace_entry row into a per-problem local_trace via the
    // nlopt_trace_wrapper<Problem>; that vector is moved into traces[]
    // alongside the corresponding result. Under library_defaults, an empty
    // trace vector is pushed to preserve the results[i] <-> traces[i]
    // index invariant.
    constexpr int max_evals = 10000;

    auto run_emitting = [&](auto runner_call) {
        std::vector<trace_entry> local_trace;
        results.push_back(runner_call(local_trace));
        traces.push_back(std::move(local_trace));
    };

    for_each_problem([&](std::string_view name, auto&& prob) {
        using P = std::remove_cvref_t<decltype(prob)>;
        auto& p = const_cast<P&>(prob);

        constexpr bool is_unconstrained =
            has_class(P::pclass, problem_class::unconstrained);
        constexpr bool is_bound =
            has_class(P::pclass, problem_class::bound_constrained);
        constexpr bool is_ineq =
            has_class(P::pclass, problem_class::inequality);
        constexpr bool is_eq =
            has_class(P::pclass, problem_class::equality);
        constexpr bool is_mixed =
            has_class(P::pclass, problem_class::mixed);
        constexpr bool is_global =
            has_class(P::pclass, problem_class::global);
        constexpr bool has_gradient = differentiable<P>;

        // Unconstrained: LD_LBFGS.
        if constexpr(is_unconstrained && has_gradient)
            run_emitting([&](std::vector<trace_entry>& t) {
                return detail::run_nlopt_solver(
                    nlopt::LD_LBFGS, "nlopt_lbfgs", name, p, max_evals, config, t);
            });

        // Bound-constrained (non-global): BOBYQA.
        if constexpr(is_bound && !is_ineq && !is_eq && !is_mixed && !is_global)
            run_emitting([&](std::vector<trace_entry>& t) {
                return detail::run_nlopt_solver(
                    nlopt::LN_BOBYQA, "nlopt_bobyqa", name, p, max_evals, config, t);
            });

        // Inequality-constrained: SLSQP and MMA.
        if constexpr((is_ineq || is_mixed) && has_gradient && bound_constrained<P>)
        {
            run_emitting([&](std::vector<trace_entry>& t) {
                return detail::run_nlopt_solver(
                    nlopt::LD_SLSQP, "nlopt_slsqp", name, p, max_evals, config, t);
            });

            // MMA only supports inequality (no equality).
            if constexpr(!is_eq && !is_mixed)
                run_emitting([&](std::vector<trace_entry>& t) {
                    return detail::run_nlopt_solver(
                        nlopt::LD_MMA, "nlopt_mma", name, p, max_evals, config, t);
                });
        }

        // Equality or mixed constrained: AUGLAG with LBFGS subsidiary.
        if constexpr((is_eq || is_mixed) && has_gradient && bound_constrained<P>)
            run_emitting([&](std::vector<trace_entry>& t) {
                return detail::run_nlopt_auglag(name, p, max_evals, config, t);
            });

        // COBYLA: derivative-free constrained (no gradient needed).
        // Disabled: dominates perf profiles (~98% CPU), masking nablapp data.

        // Global: CRS2 and ISRES (require bounds).
        if constexpr(is_global && bound_constrained<P>)
        {
            run_emitting([&](std::vector<trace_entry>& t) {
                return detail::run_nlopt_solver(
                    nlopt::GN_CRS2_LM, "nlopt_crs2", name, p, max_evals, config, t);
            });
            run_emitting([&](std::vector<trace_entry>& t) {
                return detail::run_nlopt_isres(name, p, max_evals, config, t);
            });
        }

        // ISRES on constrained problems with bounds (evolutionary
        // constrained optimizer). Dispatched separately from gradient-
        // based constrained solvers since ISRES is derivative-free.
        if constexpr((is_ineq || is_eq || is_mixed) && bound_constrained<P>
                     && !is_global)
            run_emitting([&](std::vector<trace_entry>& t) {
                return detail::run_nlopt_isres(name, p, max_evals, config, t);
            });
    });
}

}
