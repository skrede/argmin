// kthohr/optim comparison benchmarks for nablapp benchmark suite.
//
// Per D-01: native API adapter, no common interface. Wires kthohr/optim
// gradient-based and derivative-free algorithms through the
// problem_registry iteration. Every problem callback routes through a
// counting_problem<P> wrapper; the bench summary reads back the four
// counters into the {f,g,c,J}_evals columns.
//
// Algorithm coverage:
//   Unconstrained:        L-BFGS, BFGS, CG, Nelder-Mead, GD
//   Bound-constrained:    L-BFGS / BFGS / CG / NM / GD with vals_bound=true
//   Constrained:          SUMT (Sequential Unconstrained Minimization
//                         Technique) wrapping an inner unconstrained pass
//                         with penalty terms on the constraint residuals.
//   Global (bounded):     DE (Differential Evolution), PSO (Particle Swarm).
//
// Newton is intentionally skipped: kthohr/optim's newton() requires a
// problem-supplied Hessian, which the nablapp test_functions/ structs do
// not advertise.
//
// Sign convention: kthohr/optim treats all inequality constraints as
// c(x) <= 0; nablapp uses c_ineq(x) >= 0 feasible. The constraint
// adapter negates inequality rows on the boundary; equality rows are
// duplicated as (c_eq, -c_eq) since SUMT's penalty model expects scalar
// inequalities.
//
// OPTIM_ENABLE_EIGEN_WRAPPERS is supplied as a compile definition by the
// optim::optim INTERFACE target wired in benchmarks/CMakeLists.txt.

#include "bench_optim.h"
#include "counting_problem.h"
#include "benchmark_result.h"
#include "problem_registry.h"

#include "nablapp/formulation/concepts.h"

#include <optim.hpp>

#include <chrono>
#include <cmath>
#include <limits>
#include <string_view>
#include <vector>

namespace nablapp::bench
{

namespace detail
{

// Build an algo_settings_t respecting the bench_config tolerance regime
// without forcing any particular algorithm-specific knob.
[[nodiscard]] inline auto make_optim_settings(const bench_config& config)
    -> optim::algo_settings_t
{
    optim::algo_settings_t settings;
    settings.iter_max = static_cast<size_t>(config.max_iter);
    settings.print_level = 0;
    settings.rng_seed_value = static_cast<size_t>(config.seed);

    if(config.the_mode == bench_config::mode::publication)
    {
        settings.grad_err_tol = 1e-16;
        settings.rel_objfn_change_tol = 1e-16;
        settings.rel_sol_change_tol = 1e-16;
    }
    else
    {
        settings.grad_err_tol = 1e-10;
        settings.rel_objfn_change_tol = 1e-12;
        settings.rel_sol_change_tol = 1e-12;
    }
    return settings;
}

// kthohr/optim objective signature:
//   fp_t fn(const ColVec_t&, ColVec_t* grad_out, void*)
// `data` is a counting_problem<Problem>*; every invocation bumps the
// shared counters. Gradient is computed only when grad_out != nullptr.
template <typename Problem>
double optim_objective(const Eigen::VectorXd& x,
                       Eigen::VectorXd* grad_out,
                       void* data)
{
    auto* w = static_cast<counting_problem<Problem>*>(data);
    Eigen::Vector<double, Problem::problem_dimension> xv(x);

    if(grad_out)
    {
        Eigen::Vector<double, Problem::problem_dimension> g;
        w->gradient(xv, g);
        *grad_out = g;
    }
    return w->value(xv);
}

// Build a SUMT-compatible constraint function. Returns a column vector
// of constraint values c(x) <= 0; nablapp's c_ineq >= 0 rows are
// negated, equality rows c_eq = 0 are emitted as (c_eq, -c_eq) so SUMT
// drives both sides to zero via the penalty model.
template <typename Problem>
Eigen::VectorXd optim_constraints(const Eigen::VectorXd& x,
                                  Eigen::MatrixXd* jacob_out,
                                  void* data)
{
    auto* w = static_cast<counting_problem<Problem>*>(data);
    Eigen::Vector<double, Problem::problem_dimension> xv(x);

    int n_eq   = w->num_equality();
    int n_ineq = w->num_inequality();
    int n_total = n_eq + n_ineq;

    Eigen::VectorXd c(n_total);
    w->constraints(xv, c);

    int rows_out = 2 * n_eq + n_ineq;
    Eigen::VectorXd out(rows_out);

    // Equality block: emit (c_eq, -c_eq) to bracket each equation.
    for(int i = 0; i < n_eq; ++i)
    {
        out[2 * i]     =  c[i];
        out[2 * i + 1] = -c[i];
    }
    // Inequality block: nablapp c_ineq >= 0 -> optim -c_ineq <= 0.
    for(int i = 0; i < n_ineq; ++i)
        out[2 * n_eq + i] = -c[n_eq + i];

    if(jacob_out)
    {
        Eigen::MatrixXd J(n_total, x.size());
        w->constraint_jacobian(xv, J);

        Eigen::MatrixXd Jout(rows_out, x.size());
        for(int i = 0; i < n_eq; ++i)
        {
            Jout.row(2 * i)     =  J.row(i);
            Jout.row(2 * i + 1) = -J.row(i);
        }
        for(int i = 0; i < n_ineq; ++i)
            Jout.row(2 * n_eq + i) = -J.row(n_eq + i);
        *jacob_out = Jout;
    }
    return out;
}

[[nodiscard]] inline auto status_string(bool ok) -> std::string_view
{
    return ok ? std::string_view{"converged"}
              : std::string_view{"max_iterations"};
}

// Snapshot of the per-run output the dispatcher loop fills in.
struct optim_run_inputs
{
    std::string_view solver_name;
    std::string_view problem_name;
    int dimension;
    ::nablapp::problem_class pclass;
    double known_optimum;
    double final_objective;
    bool ok;
    std::int64_t wall_us;
    size_t opt_iter;
};

template <typename Problem>
[[nodiscard]] auto make_result(const optim_run_inputs& in,
                               const eval_counts& counts,
                               const bench_config& config)
    -> benchmark_result
{
    return benchmark_result{
        .solver = in.solver_name,
        .library = "kthohr_optim",
        .problem = in.problem_name,
        .pclass = in.pclass,
        .dimension = in.dimension,
        .seed = config.seed,
        .mode = (config.the_mode == bench_config::mode::publication)
                    ? std::string_view{"publication"}
                    : std::string_view{"library_defaults"},
        .solver_iters = static_cast<int>(in.opt_iter),
        .f_evals = counts.f,
        .g_evals = counts.g,
        .c_evals = counts.c,
        .J_evals = counts.J,
        .wall_time_us = in.wall_us,
        .final_objective = in.final_objective,
        .known_optimum = in.known_optimum,
        .accuracy = std::abs(in.final_objective - in.known_optimum),
        .status = status_string(in.ok),
    };
}

// Run a single optim algorithm on an unconstrained / bound-constrained
// problem. AlgoFn is a unary lambda invoking the chosen optim::* call
// against the prepared (x, objective, &wrapped, settings) tuple.
template <typename Problem, typename AlgoFn>
auto run_optim_algo(std::string_view solver_name,
                    std::string_view problem_name,
                    const Problem& prob,
                    const bench_config& config,
                    AlgoFn algo) -> benchmark_result
{
    eval_counts counts;
    counting_problem<Problem> wrapped{prob, counts};

    auto x0 = prob.initial_point();
    Eigen::VectorXd x = x0;

    auto settings = make_optim_settings(config);
    if constexpr(bound_constrained<Problem>)
    {
        auto lb = prob.lower_bounds();
        auto ub = prob.upper_bounds();
        Eigen::VectorXd lb_dyn = lb;
        Eigen::VectorXd ub_dyn = ub;
        // Replace +/-inf with a finite sentinel to keep optim's penalty
        // mapping numerically well-defined; matches optim's internal
        // sentinel range used by the bounds-transform.
        constexpr double inf_replace = 1e20;
        for(int i = 0; i < lb_dyn.size(); ++i)
        {
            if(!std::isfinite(lb_dyn[i])) lb_dyn[i] = -inf_replace;
            if(!std::isfinite(ub_dyn[i])) ub_dyn[i] =  inf_replace;
        }
        settings.vals_bound = true;
        settings.lower_bounds = lb_dyn;
        settings.upper_bounds = ub_dyn;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = algo(x, optim_objective<Problem>, &wrapped, settings);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto wall_us = std::chrono::duration_cast<
        std::chrono::microseconds>(t1 - t0).count();

    Eigen::Vector<double, Problem::problem_dimension> xv(x);
    double f_final = prob.value(xv);   // intentionally bypasses counts.

    optim_run_inputs in{
        .solver_name = solver_name,
        .problem_name = problem_name,
        .dimension = prob.dimension(),
        .pclass = prob.pclass,
        .known_optimum = prob.optimal_value(),
        .final_objective = f_final,
        .ok = ok,
        .wall_us = wall_us,
        .opt_iter = settings.opt_iter,
    };
    return make_result<Problem>(in, counts, config);
}

// Run optim::sumt on a constrained problem.
template <typename Problem>
auto run_optim_sumt(std::string_view problem_name,
                    const Problem& prob,
                    const bench_config& config) -> benchmark_result
{
    eval_counts counts;
    counting_problem<Problem> wrapped{prob, counts};

    auto x0 = prob.initial_point();
    Eigen::VectorXd x = x0;
    auto settings = make_optim_settings(config);

    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = optim::sumt(x,
                          optim_objective<Problem>, &wrapped,
                          optim_constraints<Problem>, &wrapped,
                          settings);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto wall_us = std::chrono::duration_cast<
        std::chrono::microseconds>(t1 - t0).count();

    Eigen::Vector<double, Problem::problem_dimension> xv(x);
    double f_final = prob.value(xv);

    optim_run_inputs in{
        .solver_name = "optim_sumt",
        .problem_name = problem_name,
        .dimension = prob.dimension(),
        .pclass = prob.pclass,
        .known_optimum = prob.optimal_value(),
        .final_objective = f_final,
        .ok = ok,
        .wall_us = wall_us,
        .opt_iter = settings.opt_iter,
    };
    return make_result<Problem>(in, counts, config);
}

// Dispatch tables: for a given (Problem, config) emit one benchmark_result
// per applicable algorithm. The outer dispatch by problem_class lives in
// run_optim_benchmarks below.

template <typename Problem>
void dispatch_unconstrained_or_bound(std::string_view name,
                                     const Problem& prob,
                                     const bench_config& config,
                                     std::vector<benchmark_result>& results)
{
    results.push_back(run_optim_algo("optim_lbfgs", name, prob, config,
        [](Eigen::VectorXd& x, auto fn, void* d, optim::algo_settings_t& s) {
            return optim::lbfgs(x, fn, d, s);
        }));
    results.push_back(run_optim_algo("optim_bfgs", name, prob, config,
        [](Eigen::VectorXd& x, auto fn, void* d, optim::algo_settings_t& s) {
            return optim::bfgs(x, fn, d, s);
        }));
    results.push_back(run_optim_algo("optim_cg", name, prob, config,
        [](Eigen::VectorXd& x, auto fn, void* d, optim::algo_settings_t& s) {
            return optim::cg(x, fn, d, s);
        }));
    results.push_back(run_optim_algo("optim_nm", name, prob, config,
        [](Eigen::VectorXd& x, auto fn, void* d, optim::algo_settings_t& s) {
            return optim::nm(x, fn, d, s);
        }));
    results.push_back(run_optim_algo("optim_gd", name, prob, config,
        [](Eigen::VectorXd& x, auto fn, void* d, optim::algo_settings_t& s) {
            return optim::gd(x, fn, d, s);
        }));
}

template <typename Problem>
void dispatch_global(std::string_view name,
                     const Problem& prob,
                     const bench_config& config,
                     std::vector<benchmark_result>& results)
{
    results.push_back(run_optim_algo("optim_de", name, prob, config,
        [](Eigen::VectorXd& x, auto fn, void* d, optim::algo_settings_t& s) {
            return optim::de(x, fn, d, s);
        }));
    results.push_back(run_optim_algo("optim_pso", name, prob, config,
        [](Eigen::VectorXd& x, auto fn, void* d, optim::algo_settings_t& s) {
            return optim::pso(x, fn, d, s);
        }));
}

}

void run_optim_benchmarks(std::vector<benchmark_result>& results, const bench_config& config)
{
    for_each_problem([&](std::string_view name, auto&& prob) {
        using P = std::remove_cvref_t<decltype(prob)>;

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
        constexpr bool has_constraints = is_ineq || is_eq || is_mixed;
        constexpr bool has_gradient = differentiable<P>;

        // Unconstrained or bound-constrained without general constraints:
        // run the gradient-based + Nelder-Mead suite. Skip if neither
        // gradient nor derivative-free can attempt the problem.
        if constexpr(has_gradient && !has_constraints && !is_global
                     && (is_unconstrained || is_bound))
            detail::dispatch_unconstrained_or_bound(name, prob, config, results);

        // General constrained problems: SUMT routes through the
        // counting wrapper for both objective and constraint evaluations.
        if constexpr(has_constraints && has_gradient && bound_constrained<P>)
            results.push_back(detail::run_optim_sumt(name, prob, config));

        // Global (stochastic) problems with bounds: DE + PSO.
        if constexpr(is_global && is_bound)
            detail::dispatch_global(name, prob, config, results);
    });
}

}
