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

// Wraps any nablapp problem as an NLopt objective callback.
// Problem must provide value() and gradient().
template <typename Problem>
double nlopt_objective(unsigned n, const double* x, double* grad, void* data)
{
    auto* prob = static_cast<Problem*>(data);
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

// Wraps nablapp vectorised constraints for NLopt's mconstraint interface.
// NLopt inequality: c(x) <= 0.  nablapp convention: c(x) >= 0.
// We negate: result[i] = -c_nablapp[i].
template <typename Problem>
void nlopt_ineq_mconstraint(unsigned m, double* result,
                            unsigned n, const double* x,
                            double* grad, void* data)
{
    auto* prob = static_cast<Problem*>(data);
    Eigen::Map<const Eigen::VectorXd> xmap(x, n);
    Eigen::Vector<double, Problem::problem_dimension> xv(xmap);

    Eigen::VectorXd c(m);
    prob->constraints(xv, c);

    // Only inequality constraints (skip leading equalities).
    int n_eq = prob->num_equality();
    for(unsigned i = 0; i < m; ++i)
        result[i] = -c[n_eq + static_cast<int>(i)];

    if(grad)
    {
        Eigen::MatrixXd J;
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
    auto* prob = static_cast<Problem*>(data);
    Eigen::Map<const Eigen::VectorXd> xv(x, n);

    Eigen::VectorXd c;
    prob->constraints(xv, c);

    // Equality constraints are the leading rows.
    for(unsigned i = 0; i < m; ++i)
        result[i] = c[static_cast<int>(i)];

    if(grad)
    {
        Eigen::MatrixXd J;
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
template <typename Problem>
auto run_nlopt_solver(nlopt::algorithm algo,
                      std::string_view solver_name,
                      std::string_view problem_name,
                      Problem& prob,
                      int max_evals) -> benchmark_result
{
    auto n = static_cast<unsigned>(prob.dimension());
    nlopt::opt opt(algo, n);

    // Set objective.
    opt.set_min_objective(nlopt_objective<Problem>, &prob);

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
                nlopt_ineq_mconstraint<Problem>, &prob, tol);
        }
        if(n_eq > 0)
        {
            std::vector<double> tol(static_cast<std::size_t>(n_eq), 1e-8);
            opt.add_equality_mconstraint(
                nlopt_eq_mconstraint<Problem>, &prob, tol);
        }
    }

    // Stopping criteria.
    opt.set_maxeval(max_evals);
    opt.set_ftol_rel(1e-12);

    // Initial point.
    auto x0 = prob.initial_point();
    std::vector<double> x(x0.data(), x0.data() + n);
    double minf{};

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

    int f_evals = opt.get_numevals();
    double known_opt = prob.optimal_value();

    return benchmark_result{
        .solver = solver_name,
        .library = "nlopt",
        .problem = problem_name,
        .pclass = prob.pclass,
        .dimension = prob.dimension(),
        .f_evals = f_evals,
        .g_evals = f_evals,
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
                      int max_evals) -> benchmark_result
{
    auto n = static_cast<unsigned>(prob.dimension());
    nlopt::opt opt(nlopt::LD_AUGLAG, n);

    // Subsidiary solver for augmented Lagrangian inner loop.
    nlopt::opt local_opt(nlopt::LD_LBFGS, n);
    local_opt.set_ftol_rel(1e-12);
    opt.set_local_optimizer(local_opt);

    opt.set_min_objective(nlopt_objective<Problem>, &prob);

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
                nlopt_ineq_mconstraint<Problem>, &prob, tol);
        }
        if(n_eq > 0)
        {
            std::vector<double> tol(static_cast<std::size_t>(n_eq), 1e-8);
            opt.add_equality_mconstraint(
                nlopt_eq_mconstraint<Problem>, &prob, tol);
        }
    }

    opt.set_maxeval(max_evals);
    opt.set_ftol_rel(1e-12);

    auto x0 = prob.initial_point();
    std::vector<double> x(x0.data(), x0.data() + n);
    double minf{};

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

    int f_evals = opt.get_numevals();
    double known_opt = prob.optimal_value();

    return benchmark_result{
        .solver = "nlopt_auglag",
        .library = "nlopt",
        .problem = problem_name,
        .pclass = prob.pclass,
        .dimension = prob.dimension(),
        .f_evals = f_evals,
        .g_evals = f_evals,
        .wall_time_us = wall_us,
        .final_objective = minf,
        .known_optimum = known_opt,
        .accuracy = std::abs(minf - known_opt),
        .status = status_str,
    };
}

} // detail

void run_nlopt_benchmarks(std::vector<benchmark_result>& results)
{
    constexpr int max_evals = 10000;

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
            results.push_back(
                detail::run_nlopt_solver(nlopt::LD_LBFGS, "nlopt_lbfgs",
                                         name, p, max_evals));

        // Bound-constrained (non-global): BOBYQA.
        if constexpr(is_bound && !is_ineq && !is_eq && !is_mixed && !is_global)
            results.push_back(
                detail::run_nlopt_solver(nlopt::LN_BOBYQA, "nlopt_bobyqa",
                                         name, p, max_evals));

        // Inequality-constrained: SLSQP and MMA.
        if constexpr((is_ineq || is_mixed) && has_gradient && bound_constrained<P>)
        {
            results.push_back(
                detail::run_nlopt_solver(nlopt::LD_SLSQP, "nlopt_slsqp",
                                         name, p, max_evals));

            // MMA only supports inequality (no equality).
            if constexpr(!is_eq && !is_mixed)
                results.push_back(
                    detail::run_nlopt_solver(nlopt::LD_MMA, "nlopt_mma",
                                             name, p, max_evals));
        }

        // Equality or mixed constrained: AUGLAG with LBFGS subsidiary.
        if constexpr((is_eq || is_mixed) && has_gradient && bound_constrained<P>)
            results.push_back(
                detail::run_nlopt_auglag(name, p, max_evals));

        // COBYLA: derivative-free constrained (no gradient needed).
        // Disabled: dominates perf profiles (~98% CPU), masking nablapp data.
        // if constexpr((is_ineq || is_mixed || is_eq) && bound_constrained<P>)
        //     results.push_back(
        //         detail::run_nlopt_solver(nlopt::LN_COBYLA, "nlopt_cobyla",
        //                                  name, p, max_evals));

        // ISRES: global constrained -- also on constrained bounded problems
        // for comparison with nablapp's isres_policy.
        if constexpr(constrained<P> && bound_constrained<P>)
            results.push_back(
                detail::run_nlopt_solver(nlopt::GN_ISRES, "nlopt_isres",
                                         name, p, max_evals));

        // Global: CRS2 and ISRES (require bounds).
        if constexpr(is_global && bound_constrained<P>)
        {
            results.push_back(
                detail::run_nlopt_solver(nlopt::GN_CRS2_LM, "nlopt_crs2",
                                         name, p, max_evals));
            // nlopt_isres on global problems already dispatched above
            // when constrained<P> is also satisfied.
            if constexpr(!constrained<P>)
                results.push_back(
                    detail::run_nlopt_solver(nlopt::GN_ISRES, "nlopt_isres",
                                             name, p, max_evals));
        }
    });
}

}
