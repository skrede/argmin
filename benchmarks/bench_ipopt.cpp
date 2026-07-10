// IPOPT comparison benchmarks for argmin benchmark suite.
//
// IPOPT (Interior Point OPTimizer, COIN-OR) solves large-scale nonlinear
// programs of the form:
//
//     min  f(x)       s.t.   g_l <= g(x) <= g_u,   x_l <= x <= x_u
//
// via a primal-dual interior-point filter line-search algorithm
// (Waechter & Biegler 2006, Math. Programming 106(1)).
//
// Single solver mapping: "ipopt" -> nlopt_constrained | bound | unconstrained,
// dispatched by problem_class in run_ipopt_benchmarks. We use IPOPT's
// quasi-Newton (L-BFGS) Hessian approximation so the adapter does not need
// to supply second derivatives. IPOPT's L-BFGS is not the same as
// Byrd-Lu-Nocedal-Zhu L-BFGS-B; it is Nocedal's original unconstrained
// L-BFGS applied inside the interior-point subproblem.
//
// Sign convention note:
//   argmin inequalities:  c_ineq(x) >= 0 feasible (mu_ineq >= 0)
//   IPOPT g bounds:        g_l <= g(x) <= g_u
//   Mapping for inequality c_ineq(x) >= 0:  g_l = 0, g_u = 2e19 (IPOPT_INF).
//   Mapping for equality c_eq(x) = 0:       g_l = g_u = 0.
//
// Hessian approximation: quasi-newton (L-BFGS). Analytical Hessian not
// supplied because argmin problem types do not currently advertise one.

#include "bench_ipopt.h"
#include "trace_entry.h"
#include "counting_problem.h"
#include "problem_registry.h"

#include "argmin/formulation/concepts.h"

#include <IpIpoptApplication.hpp>
#include <IpSolveStatistics.hpp>
#include <IpTNLP.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace argmin::bench
{

namespace detail
{

template <typename Problem, typename Vec>
[[nodiscard]] auto constraint_violation_at(const Problem& prob,
                                           const Vec& x) -> double
{
    double violation = 0.0;
    if constexpr(bound_constrained<Problem>)
    {
        const auto lb = prob.lower_bounds();
        const auto ub = prob.upper_bounds();
        for(int i = 0; i < prob.dimension(); ++i)
        {
            if(std::isfinite(lb[i]))
                violation = std::max(violation, static_cast<double>(lb[i] - x[i]));
            if(std::isfinite(ub[i]))
                violation = std::max(violation, static_cast<double>(x[i] - ub[i]));
        }
    }
    if constexpr(constrained<Problem>)
    {
        const int n_eq = prob.num_equality();
        const int n_ineq = prob.num_inequality();
        Eigen::VectorXd c(n_eq + n_ineq);
        prob.constraints(x, c);
        for(int i = 0; i < n_eq; ++i)
            violation = std::max(violation, std::abs(static_cast<double>(c[i])));
        for(int i = 0; i < n_ineq; ++i)
            violation = std::max(violation, std::max(0.0, -static_cast<double>(c[n_eq + i])));
    }
    return violation;
}

[[nodiscard]] inline auto cap_status_string(std::string_view status,
                                            const eval_counts& counts,
                                            const bench_config& config) -> std::string
{
    if(status == "maxtime_reached")
        return "wall";
    if(status == "maxeval_reached")
        return "f_eval";
    if(config.max_f_evals > 0 && counts.f >= config.max_f_evals)
        return "f_eval";
    return std::string{counts.cap_status()};
}

// TNLP adapter wrapping any argmin constrained problem.
// Problem must satisfy the differentiable + constrained concepts (or
// bound_constrained for bound-only subset).
//
// `Problem` is a counting_problem<InnerProblem>; every value/gradient/
// constraints/constraint_jacobian invocation bumps the shared counters
// the bench summary CSV reads back into the {f,g,c,J}_evals columns.
template <typename Problem>
class ipopt_tnlp : public Ipopt::TNLP
{
public:
    ipopt_tnlp(Problem& prob, std::string_view problem_name, int max_evals)
        : prob_{&prob}, problem_name_{problem_name}, max_evals_{max_evals}
    {
    }

    // Captured solution state for the run_ipopt_solver wrapper.
    [[nodiscard]] auto solution() const -> const std::vector<double>& { return x_final_; }
    [[nodiscard]] auto objective() const -> double { return obj_final_; }
    [[nodiscard]] auto status() const -> Ipopt::SolverReturn { return status_; }

    bool get_nlp_info(Ipopt::Index& n, Ipopt::Index& m,
                      Ipopt::Index& nnz_jac_g, Ipopt::Index& nnz_h_lag,
                      IndexStyleEnum& index_style) override
    {
        n = prob_->dimension();
        if constexpr(constrained<Problem>)
            m = prob_->num_equality() + prob_->num_inequality();
        else
            m = 0;
        nnz_jac_g = m * n;  // dense Jacobian (small problems).
        nnz_h_lag = 0;       // quasi-Newton — no Hessian supplied.
        index_style = TNLP::C_STYLE;
        return true;
    }

    bool get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u,
                         Ipopt::Index m, Ipopt::Number* g_l, Ipopt::Number* g_u) override
    {
        // Variable bounds. Problems that satisfy bound_constrained expose
        // lower_bounds/upper_bounds; otherwise fall back to +/-IPOPT_INF.
        constexpr double inf = 2e19;
        if constexpr(bound_constrained<Problem>)
        {
            auto lb = prob_->lower_bounds();
            auto ub = prob_->upper_bounds();
            for(Ipopt::Index j = 0; j < n; ++j)
            {
                x_l[j] = lb[j];
                x_u[j] = ub[j];
            }
        }
        else
        {
            for(Ipopt::Index j = 0; j < n; ++j)
            {
                x_l[j] = -inf;
                x_u[j] = inf;
            }
        }

        // Constraint bounds. Equality first, then inequality, matching the
        // layout of prob_->constraints(x, c). Unconstrained problems (m = 0)
        // fall through without touching g_l / g_u.
        if constexpr(constrained<Problem>)
        {
            const int n_eq = prob_->num_equality();
            for(Ipopt::Index i = 0; i < m; ++i)
            {
                if(i < n_eq)
                {
                    g_l[i] = 0.0;
                    g_u[i] = 0.0;
                }
                else
                {
                    g_l[i] = 0.0;    // argmin: c_ineq(x) >= 0 feasible
                    g_u[i] = inf;
                }
            }
        }
        return true;
    }

    bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number* x,
                            bool /*init_z*/, Ipopt::Number* /*z_L*/,
                            Ipopt::Number* /*z_U*/, Ipopt::Index /*m*/,
                            bool /*init_lambda*/, Ipopt::Number* /*lambda*/) override
    {
        if(init_x)
        {
            auto x0 = prob_->initial_point();
            for(Ipopt::Index j = 0; j < n; ++j)
                x[j] = x0[j];
        }
        return true;
    }

    bool eval_f(Ipopt::Index n, const Ipopt::Number* x, bool /*new_x*/,
                Ipopt::Number& obj_value) override
    {
        Eigen::Map<const Eigen::VectorXd> xmap(x, n);
        Eigen::Vector<double, Problem::problem_dimension> xv(xmap);
        obj_value = prob_->value(xv);
        return true;
    }

    bool eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool /*new_x*/,
                     Ipopt::Number* grad_f) override
    {
        Eigen::Map<const Eigen::VectorXd> xmap(x, n);
        Eigen::Vector<double, Problem::problem_dimension> xv(xmap);
        Eigen::Vector<double, Problem::problem_dimension> g;
        prob_->gradient(xv, g);
        for(Ipopt::Index j = 0; j < n; ++j)
            grad_f[j] = g[j];
        return true;
    }

    bool eval_g(Ipopt::Index n, const Ipopt::Number* x, bool /*new_x*/,
                Ipopt::Index m, Ipopt::Number* g) override
    {
        if constexpr(constrained<Problem>)
        {
            Eigen::Map<const Eigen::VectorXd> xmap(x, n);
            Eigen::Vector<double, Problem::problem_dimension> xv(xmap);
            Eigen::VectorXd c(m);
            prob_->constraints(xv, c);
            for(Ipopt::Index i = 0; i < m; ++i)
                g[i] = c[i];
        }
        return true;
    }

    bool eval_jac_g(Ipopt::Index n, const Ipopt::Number* x, bool /*new_x*/,
                    Ipopt::Index m, Ipopt::Index nele_jac, Ipopt::Index* iRow,
                    Ipopt::Index* jCol, Ipopt::Number* values) override
    {
        if(values == nullptr)
        {
            // Structure: dense row-major.
            Ipopt::Index idx = 0;
            for(Ipopt::Index i = 0; i < m; ++i)
                for(Ipopt::Index j = 0; j < n; ++j)
                {
                    iRow[idx] = i;
                    jCol[idx] = j;
                    ++idx;
                }
        }
        else if constexpr(constrained<Problem>)
        {
            Eigen::Map<const Eigen::VectorXd> xmap(x, n);
            Eigen::Vector<double, Problem::problem_dimension> xv(xmap);
            Eigen::MatrixXd J(m, n);
            prob_->constraint_jacobian(xv, J);
            Ipopt::Index idx = 0;
            for(Ipopt::Index i = 0; i < m; ++i)
                for(Ipopt::Index j = 0; j < n; ++j)
                    values[idx++] = J(i, j);
        }
        (void)nele_jac;
        return true;
    }

    bool eval_h(Ipopt::Index /*n*/, const Ipopt::Number* /*x*/, bool /*new_x*/,
                Ipopt::Number /*obj_factor*/, Ipopt::Index /*m*/,
                const Ipopt::Number* /*lambda*/, bool /*new_lambda*/,
                Ipopt::Index /*nele_hess*/, Ipopt::Index* /*iRow*/,
                Ipopt::Index* /*jCol*/, Ipopt::Number* /*values*/) override
    {
        return false;  // quasi-Newton; no exact Hessian.
    }

    void finalize_solution(Ipopt::SolverReturn status, Ipopt::Index n,
                           const Ipopt::Number* x, const Ipopt::Number* /*z_L*/,
                           const Ipopt::Number* /*z_U*/, Ipopt::Index /*m*/,
                           const Ipopt::Number* /*g*/,
                           const Ipopt::Number* /*lambda*/,
                           Ipopt::Number obj_value,
                           const Ipopt::IpoptData* /*ip_data*/,
                           Ipopt::IpoptCalculatedQuantities* /*ip_cq*/) override
    {
        status_ = status;
        obj_final_ = obj_value;
        x_final_.assign(x, x + n);
    }

    // Per-iter trace hook. Wired by the
    // run_ipopt_solver wrapper before OptimizeTNLP() is invoked; trace_out_
    // points at a per-problem local_trace owned by the caller. Maps the
    // IPOPT-native composite (obj_value, inf_pr, d_norm, max(inf_pr, inf_du))
    // into the D-C3 12-column schema. ALWAYS returns true: under the
    // publication protocol, stopping is governed by max_iter / max_wall_time
    // configured on the IpoptApplication, never by USER_REQUESTED_STOP.
    bool intermediate_callback(
        Ipopt::AlgorithmMode  /*mode*/,
        Ipopt::Index          iter,
        Ipopt::Number         obj_value,
        Ipopt::Number         inf_pr,
        Ipopt::Number         inf_du,
        Ipopt::Number         /*mu*/,
        Ipopt::Number         d_norm,
        Ipopt::Number         /*regularization_size*/,
        Ipopt::Number         /*alpha_du*/,
        Ipopt::Number         /*alpha_pr*/,
        Ipopt::Index          /*ls_trials*/,
        const Ipopt::IpoptData*           /*ip_data*/,
        Ipopt::IpoptCalculatedQuantities* /*ip_cq*/
    ) override
    {
        if(!trace_enabled_)
            return true;

        f_best_running_ = std::min(f_best_running_, obj_value);

        const auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        trace_out_->push_back(trace_entry{
            .iter         = static_cast<int>(iter),
            .f_evals      = prob_->counts->f,
            .g_evals      = prob_->counts->g,
            .c_evals      = prob_->counts->c,
            .J_evals      = prob_->counts->J,
            .wall_us      = now_us - t0_us_,
            .f_current    = obj_value,
            .f_best       = f_best_running_,
            .accuracy     = std::abs(obj_value - f_star_),
            .cv           = inf_pr,
            .step_norm    = d_norm,
            .kkt_residual = std::max(inf_pr, inf_du),
        });
        return true;
    }

    // Trace wiring; populated by the run_ipopt_solver wrapper after
    // construction and before OptimizeTNLP() is invoked.
    void enable_trace(std::vector<trace_entry>* trace_out,
                      std::int64_t              t0_us,
                      double                    f_star)
    {
        trace_out_     = trace_out;
        t0_us_         = t0_us;
        f_star_        = f_star;
        trace_enabled_ = (trace_out_ != nullptr);
    }

private:
    Problem* prob_;
    std::string_view problem_name_;
    int max_evals_;
    Ipopt::SolverReturn status_{Ipopt::INTERNAL_ERROR};
    double obj_final_{std::numeric_limits<double>::quiet_NaN()};
    std::vector<double> x_final_;

    // Per-iter trace state.
    std::vector<trace_entry>* trace_out_{nullptr};
    std::int64_t              t0_us_{0};
    double                    f_star_{};
    double                    f_best_running_{std::numeric_limits<double>::infinity()};
    bool                      trace_enabled_{false};
};

[[nodiscard]] inline auto ipopt_status_string(Ipopt::SolverReturn s) -> std::string_view
{
    using Ipopt::SolverReturn;
    switch(s)
    {
    case SolverReturn::SUCCESS:                         return "converged";
    case SolverReturn::MAXITER_EXCEEDED:                return "max_iterations";
    case SolverReturn::CPUTIME_EXCEEDED:                return "maxtime_reached";
    case SolverReturn::STOP_AT_TINY_STEP:               return "stalled";
    case SolverReturn::STOP_AT_ACCEPTABLE_POINT:        return "ftol_reached";
    case SolverReturn::LOCAL_INFEASIBILITY:             return "infeasible";
    case SolverReturn::USER_REQUESTED_STOP:             return "stopped";
    case SolverReturn::FEASIBLE_POINT_FOUND:            return "feasible";
    case SolverReturn::DIVERGING_ITERATES:              return "diverged";
    case SolverReturn::RESTORATION_FAILURE:             return "restoration_failed";
    case SolverReturn::ERROR_IN_STEP_COMPUTATION:       return "failed";
    case SolverReturn::INVALID_NUMBER_DETECTED:         return "failed";
    case SolverReturn::TOO_FEW_DEGREES_OF_FREEDOM:      return "failed";
    case SolverReturn::INVALID_OPTION:                  return "failed";
    case SolverReturn::OUT_OF_MEMORY:                   return "failed";
    case SolverReturn::INTERNAL_ERROR:                  return "failed";
    case SolverReturn::UNASSIGNED:                      return "failed";
    default:                                            return "failed";
    }
}

// Configurable option bundle describing an IPOPT solver variant. Each
// variant runs the same interior-point filter line-search core but tunes
// the barrier-update strategy and the limited-memory Hessian update.
struct ipopt_variant
{
    std::string_view solver_name;
    std::string_view mu_strategy;         // "adaptive" (default) | "monotone"
    std::string_view lm_update_type;      // "bfgs" (default) | "sr1"
};

template <typename Problem>
auto run_ipopt_solver(std::string_view problem_name,
                      Problem& prob,
                      int /*max_evals_legacy*/,
                      const ipopt_variant& variant,
                      const bench_config& config,
                      std::vector<trace_entry>& local_trace) -> benchmark_result
{
    eval_counts counts;
    counts.set_max_f_evals(config.max_f_evals);
    counting_problem<Problem> wrapped{prob, counts};
    using wrapped_t = counting_problem<Problem>;

    auto tnlp = Ipopt::SmartPtr<ipopt_tnlp<wrapped_t>>(
        new ipopt_tnlp<wrapped_t>(wrapped, problem_name, config.max_iter));

    Ipopt::SmartPtr<Ipopt::IpoptApplication> app =
        IpoptApplicationFactory();

    // Quasi-Newton (limited-memory) Hessian approximation — argmin problem
    // types do not currently expose analytical second derivatives.
    app->Options()->SetStringValue("hessian_approximation", "limited-memory");
    app->Options()->SetStringValue("limited_memory_update_type",
                                   std::string(variant.lm_update_type));

    // Stopping criteria sourced from bench_config so library_defaults keeps
    // the prior 1e-12 / 10 000-iter regime byte-identical while publication
    // mode tightens to 1e-16 + 10 s wall budget. IPOPT's `max_wall_time`
    // option enforces a hard wall-clock budget per (seed, solver, problem)
    // triple per CONTEXT D-C2.
    app->Options()->SetNumericValue("tol", config.ftol_rel);
    app->Options()->SetIntegerValue("max_iter", config.max_iter);
    app->Options()->SetNumericValue("max_wall_time", config.max_wall_time_s);

    // Silence stdout spam during bench; interior-point is chatty by default.
    app->Options()->SetIntegerValue("print_level", 0);
    app->Options()->SetStringValue("sb", "yes");  // suppress banner.

    app->Options()->SetStringValue("mu_strategy",
                                   std::string(variant.mu_strategy));

    auto init_status = app->Initialize();
    if(init_status != Ipopt::Solve_Succeeded)
    {
        return benchmark_result{
            .solver = variant.solver_name,
            .library = "ipopt",
            .problem = problem_name,
            .pclass = prob.pclass,
            .dimension = prob.dimension(),
            .seed = config.seed,
            .mode = (config.the_mode == bench_config::mode::publication)
                        ? std::string_view{"publication"}
                        : std::string_view{"library_defaults"},
            .solver_iters = 0,
            .f_evals = 0,
            .g_evals = 0,
            .c_evals = 0,
            .J_evals = 0,
            .wall_time_us = 0,
            .final_objective = std::numeric_limits<double>::quiet_NaN(),
            .known_optimum = prob.optimal_value(),
            .accuracy = std::numeric_limits<double>::quiet_NaN(),
            .status = "failed",
            .cap_status = "none",
            .solve_wall_time_us = 0,
            .end_to_end_wall_time_us = 0,
        };
    }

    // Per-iter trace wiring. Under library_defaults, trace_out is null and
    // the intermediate_callback short-circuits; under publication, the
    // callback appends rows into local_trace using the IPOPT-native
    // (obj_value, inf_pr, d_norm, max(inf_pr, inf_du)) composite. The
    // `t0_us` baseline is captured immediately before OptimizeTNLP() so
    // per-iter wall_us excludes IpoptApplicationFactory + Initialize setup,
    // matching the summary wall_time_us baseline (the t0 captured here).
    auto t0 = std::chrono::steady_clock::now();
    if(config.trace_enabled)
    {
        const auto t0_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t0.time_since_epoch()).count();
        tnlp->enable_trace(&local_trace, t0_us, prob.optimal_value());
    }
    auto app_status = app->OptimizeTNLP(tnlp);
    auto t1 = std::chrono::steady_clock::now();
    auto wall_us = std::chrono::duration_cast<
        std::chrono::microseconds>(t1 - t0).count();

    double known_opt = prob.optimal_value();
    double final_obj = tnlp->objective();
    double final_cv = std::numeric_limits<double>::quiet_NaN();
    const auto& solution = tnlp->solution();
    if(!solution.empty())
    {
        Eigen::Map<const Eigen::VectorXd> final_x_map(
            solution.data(), static_cast<Eigen::Index>(solution.size()));
        Eigen::Vector<double, Problem::problem_dimension> final_x(final_x_map);
        final_cv = constraint_violation_at(prob, final_x);
    }

    // Map Ipopt::ApplicationReturnStatus -> status_string when the TNLP
    // wasn't finalized (e.g. initialization failure short-circuited).
    //
    // Under publication-mode tolerances (1e-16) IPOPT routinely returns
    // Search_Direction_Becomes_Too_Small or Solved_To_Acceptable_Level
    // because tol=1e-16 is below typical machine-precision floors for
    // the inf_pr / inf_du composites. Both are operationally a
    // converged-at-machine-precision event, not a failure; map them to
    // "ftol_reached" (the publication grade analog of "tightened past
    // what the algorithm can deliver"). Genuine failure modes
    // (Diverging_Iterates, Restoration_Failed, Invalid_Number_Detected,
    // Insufficient_Memory, Internal_Error, Unrecoverable_Exception,
    // NonIpopt_Exception_Thrown) continue to map to "failed".
    std::string_view status_str;
    if(app_status == Ipopt::Solve_Succeeded
       || app_status == Ipopt::Solved_To_Acceptable_Level)
    {
        status_str = ipopt_status_string(tnlp->status());
    }
    else if(app_status == Ipopt::Search_Direction_Becomes_Too_Small)
    {
        status_str = "ftol_reached";
    }
    else if(app_status == Ipopt::Maximum_Iterations_Exceeded)
    {
        status_str = "max_iterations";
    }
    else if(app_status == Ipopt::Maximum_CpuTime_Exceeded
            || app_status == Ipopt::Maximum_WallTime_Exceeded)
    {
        status_str = "maxtime_reached";
    }
    else if(app_status == Ipopt::Infeasible_Problem_Detected)
    {
        status_str = "infeasible";
    }
    else if(app_status == Ipopt::Diverging_Iterates)
    {
        status_str = "diverged";
    }
    else
    {
        status_str = "failed";
    }

    int solver_iters = 0;
    auto stats = app->Statistics();
    if(IsValid(stats))
        solver_iters = stats->IterationCount();

    return benchmark_result{
        .solver = variant.solver_name,
        .library = "ipopt",
        .problem = problem_name,
        .pclass = prob.pclass,
        .dimension = prob.dimension(),
        .seed = config.seed,
        .mode = (config.the_mode == bench_config::mode::publication)
                    ? std::string_view{"publication"}
                    : std::string_view{"library_defaults"},
        .solver_iters = solver_iters,
        .f_evals = counts.f,
        .g_evals = counts.g,
        .c_evals = counts.c,
        .J_evals = counts.J,
        .wall_time_us = wall_us,
        .final_objective = final_obj,
        .known_optimum = known_opt,
        .accuracy = std::abs(final_obj - known_opt),
        .constraint_violation = final_cv,
        .status = status_str,
        .cap_status = cap_status_string(status_str, counts, config),
        .solve_wall_time_us = wall_us,
        .end_to_end_wall_time_us = wall_us,
    };
}

// IPOPT option-variant registry. All three use the same interior-point
// filter line-search core; they differ in the barrier-parameter update
// strategy (adaptive Nocedal–Wächter vs monotone Fiacco–McCormick) and
// the limited-memory secant update (BFGS vs SR1).
constexpr std::array<ipopt_variant, 3> ipopt_variants{{
    {.solver_name = "ipopt",          .mu_strategy = "adaptive", .lm_update_type = "bfgs"},
    {.solver_name = "ipopt_monotone", .mu_strategy = "monotone", .lm_update_type = "bfgs"},
    {.solver_name = "ipopt_sr1",      .mu_strategy = "adaptive", .lm_update_type = "sr1"},
}};

} // detail

void run_ipopt_benchmarks(std::vector<benchmark_result>& results,
                          std::vector<std::vector<trace_entry>>& traces,
                          const bench_config& config)
{
    // Each adapter call wraps prob through counting_problem<P>, populating
    // problem-level {f,g,c,J}_evals. solver_iters reads back IPOPT's
    // native iteration counter via app->Statistics()->IterationCount().
    //
    // Under config.trace_enabled, each variant invocation appends per-iter
    // trace rows into local_trace via TNLP::intermediate_callback; that
    // vector is moved into traces[] alongside the corresponding result.
    // Under library_defaults, an empty trace vector is pushed to preserve
    // the results[i] <-> traces[i] index invariant.
    constexpr int max_evals = 10000;

    for_each_problem([&](std::string_view name, auto&& prob) {
        using P = std::remove_cvref_t<decltype(prob)>;
        auto& p = const_cast<P&>(prob);

        constexpr bool has_gradient = differentiable<P>;
        constexpr bool is_global =
            has_class(P::pclass, problem_class::global);

        // IPOPT is a local gradient-based interior-point NLP solver.
        // Appropriate on any differentiable problem that is not a global
        // (stochastic / evolutionary) case. This covers unconstrained
        // (box-free Newton on the barrier), bound-constrained (the L-BFGS-B
        // peer case vs argmin's lbfgsb / byrd_lbfgsb), and any combination
        // of equality / inequality constraints with optional bounds.
        if constexpr(has_gradient && !is_global)
        {
            for(const auto& variant : detail::ipopt_variants)
            {
                std::vector<trace_entry> local_trace;
                results.push_back(
                    detail::run_ipopt_solver(name, p, max_evals, variant,
                                             config, local_trace));
                traces.push_back(std::move(local_trace));
            }
        }
    });
}

}
