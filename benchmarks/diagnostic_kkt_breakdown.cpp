// Throwaway diagnostic harness: emits per-iteration 5-leg KKT residual
// breakdown across the post-Phase-31 regression cases so the convergence
// fix in the following plan can be authored against measured evidence
// rather than hypothesis.
//
// The six target cases (4 SQP premature-ftol regressions, 1 byrd_lbfgsb
// null-step termination gap, 1 hs071 diagnostic byproduct) are defined
// at the top of main(). Each case drives basic_solver::step() in a
// loop, inspects solver.state() + the returned step_result + the
// per-criterion last_check_results(), and emits one CSV row per step
// on stdout.
//
// The 5-leg breakdown lives in kkt_residual_breakdown() below. It is
// deliberately a local free function (NOT exported to
// lib/nablapp/include/nablapp/detail/) -- this file is deletable once
// the permanent convergence fix lands. The canonical kkt_residual
// rewrite in detail/kkt_residual.h is a separate plan's deliverable.
//
// Reference: Nocedal & Wright, "Numerical Optimization" 2nd ed. (2006),
//            Definition 12.1 (KKT conditions: stationarity + primal
//            feasibility + dual feasibility + complementarity);
//            eq. 12.34 (Lagrangian stationarity leg);
//            Byrd, Lu, Nocedal, Zhu (1995), "A Limited Memory Algorithm
//            for Bound Constrained Optimization", SIAM J. Sci. Comput.
//            16(5), 1190-1208 (byrd_lbfgsb GCP path).
//
// ---------------------------------------------------------------------
// Sign convention for the primal-inequality leg
// ---------------------------------------------------------------------
//
// Per lib/nablapp/include/nablapp/detail/lagrangian.h lines 42-60 and
// lib/nablapp/include/nablapp/formulation/concepts.h lines 87-93, the
// canonical project convention is `c_ineq >= 0` feasible; a negative
// entry is the violation magnitude. Verbatim from lagrangian.h:
//
//   "Equality constraints must be zero; inequality constraints must be
//    non-negative (c_ineq >= 0 convention). Violated inequalities
//    contribute their negative part."
//
// The primal-inequality leg therefore reads `max_i max(-c_ineq[i], 0)`
// so a feasible c_ineq[i] = +0.3 contributes 0, while an infeasible
// c_ineq[i] = -0.2 contributes +0.2. The dual-feasibility leg mirrors
// this: `max_i max(-mu_ineq[i], 0)` captures the KKT requirement
// mu_ineq >= 0 under the same sign convention.

#include "nablapp/solver/byrd_lbfgsb_policy.h"
#include "nablapp/solver/kraft_slsqp_policy.h"
#include "nablapp/solver/basic_solver.h"
#include "nablapp/solver/nw_sqp_policy.h"
#include "nablapp/solver/convergence.h"
#include "nablapp/solver/options.h"
#include "nablapp/test_functions/hock_schittkowski.h"
#include "nablapp/test_functions/more_garbow_hillstrom.h"

#include <Eigen/Core>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <cmath>
#include <print>
#include <string_view>
#include <type_traits>

namespace
{

// Local 5-tuple breakdown of the first-order optimality error.
//
// Returned unreduced so the diagnostic harness can log every leg
// independently and see which one is small vs. which one fires at the
// premature-termination iterate.
//
// Reference: N&W 2e Definition 12.1 / eq. 12.34.
struct kkt_leg_breakdown
{
    double stationarity{};
    double primal_eq{};
    double primal_ineq{};
    double dual_feas{};
    double complementarity{};
};

// Diagnostic-only 5-leg composition of the first-order KKT error,
// returning each leg unreduced. Mirrors the existing 6-arg
// detail::kkt_residual (stationarity + complementarity) plus the
// three missing legs (primal_eq, primal_ineq, dual_feas) that Plan 02
// will fold into the canonical helper. L-infinity norm throughout so
// the legs compose with a plain std::max in the caller.
//
// Templated on every argument so fixed-size solver state vectors
// (e.g. Eigen::Vector<double, N> for s.g) flow through without being
// widened into dynamic Eigen::VectorXd copies; the project convention
// is no dynamic Eigen types in new code.
//
// Reference: N&W 2e Definition 12.1 (KKT conditions); N&W 2e
//            eq. 12.34 (Lagrangian stationarity).
template<class GradF, class JeqT, class JineqT,
         class LambdaEq, class MuIneq, class Ceq, class Cineq>
kkt_leg_breakdown kkt_residual_breakdown(
    const GradF& grad_f,
    const JeqT& J_eq,
    const JineqT& J_ineq,
    const LambdaEq& lambda_eq,
    const MuIneq& mu_ineq,
    const Ceq& c_eq,
    const Cineq& c_ineq)
{
    kkt_leg_breakdown out{};

    // Leg 1: stationarity ||grad_f - J_eq^T lambda_eq - J_ineq^T mu_ineq||_inf.
    typename std::remove_cvref_t<GradF>::PlainObject grad_L = grad_f;
    if(J_eq.rows() > 0 && lambda_eq.size() == J_eq.rows())
        grad_L -= J_eq.transpose() * lambda_eq;
    if(J_ineq.rows() > 0 && mu_ineq.size() == J_ineq.rows())
        grad_L -= J_ineq.transpose() * mu_ineq;
    out.stationarity = grad_L.size() > 0
        ? grad_L.template lpNorm<Eigen::Infinity>()
        : 0.0;

    // Leg 2: primal equality ||c_eq||_inf.
    out.primal_eq = c_eq.size() > 0
        ? c_eq.template lpNorm<Eigen::Infinity>()
        : 0.0;

    // Leg 3: primal inequality ||max(-c_ineq, 0)||_inf under the
    // project convention c_ineq >= 0 feasible (see sign-convention
    // block comment at file top).
    out.primal_ineq = 0.0;
    for(Eigen::Index i = 0; i < c_ineq.size(); ++i)
        out.primal_ineq = std::max(out.primal_ineq, std::max(-c_ineq[i], 0.0));

    // Leg 4: dual feasibility ||max(-mu_ineq, 0)||_inf (mu_ineq >= 0
    // required by the KKT conditions).
    out.dual_feas = 0.0;
    for(Eigen::Index i = 0; i < mu_ineq.size(); ++i)
        out.dual_feas = std::max(out.dual_feas, std::max(-mu_ineq[i], 0.0));

    // Leg 5: complementarity max_i min(|mu_ineq_i|, |c_ineq_i|).
    out.complementarity = 0.0;
    const Eigen::Index m = std::min(mu_ineq.size(), c_ineq.size());
    for(Eigen::Index i = 0; i < m; ++i)
    {
        double local = std::min(std::abs(mu_ineq[i]), std::abs(c_ineq[i]));
        out.complementarity = std::max(out.complementarity, local);
    }

    return out;
}

// Returns the index of the first fired criterion in last_check_results,
// or -1 if none fired on this step.
template <std::size_t K>
int first_fired(const std::array<std::optional<nablapp::solver_status>, K>& results)
{
    for(std::size_t i = 0; i < results.size(); ++i)
    {
        if(results[i])
            return static_cast<int>(i);
    }
    return -1;
}

// Emits the CSV column header a case block expects. Kept in one place
// so every case block stays consistent.
void print_header(std::string_view case_label)
{
    std::println("# case: {}", case_label);
    std::println("iter,f,c_eq_norm,c_ineq_violation,stationarity_leg,"
                 "complementarity_leg,dual_feasibility_leg,primal_eq_leg,"
                 "primal_ineq_leg,kkt_current,gradient_norm,x_norm,"
                 "step_size,feasibility_gate_effective,fired_criterion,"
                 "policy_status");
}

// Emits one CSV data row. `fired_criterion` is the index of the first
// criterion that fired on the last check; `policy_status_code` is the
// solver_status numeric code returned by the policy (or -1 if none).
void print_row(std::uint32_t iter,
               double f, double c_eq_norm, double c_ineq_violation,
               const kkt_leg_breakdown& legs,
               double kkt_current, double gradient_norm,
               double x_norm, double step_size,
               double feasibility_gate_effective,
               int fired_criterion, int policy_status_code)
{
    std::println("{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},"
                 "{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{},{}",
                 iter, f, c_eq_norm, c_ineq_violation,
                 legs.stationarity, legs.complementarity, legs.dual_feas,
                 legs.primal_eq, legs.primal_ineq,
                 kkt_current, gradient_norm, x_norm, step_size,
                 feasibility_gate_effective, fired_criterion,
                 policy_status_code);
}

// -----------------------------------------------------------------
// Case runners for the four SQP regressions + hs071 diagnostic case.
// Each runner constructs the problem + x0 via initial_point(), builds
// a basic_solver with tight convergence thresholds matching the
// existing regression guards in tests/unit/kraft_slsqp_test.cpp line
// 369-389, loops step() up to max_iterations, and emits one CSV row
// per step.
// -----------------------------------------------------------------

template <typename Problem, typename Policy>
void run_sqp_case(std::string_view case_label, Problem problem, Policy policy)
{
    print_header(case_label);

    auto x0 = problem.initial_point();
    nablapp::solver_options opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-6);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    nablapp::basic_solver solver{policy, problem, x0, opts};

    const double feasibility_gate_effective = opts.constraint_tolerance
        ? *opts.constraint_tolerance
        : 1e-6;

    nablapp::step_result<double> last{};
    std::uint32_t final_iter = 0;
    int final_status_code = -1;

    for(std::uint32_t i = 0; i < opts.max_iterations; ++i)
    {
        last = solver.step();
        final_iter = i + 1;

        // Drive the stored convergence policy so last_check_results()
        // reflects this step. basic_solver::step() does NOT invoke
        // convergence.check(); that happens inside solve()/step_n().
        // Call it explicitly here so the diagnostic row carries the
        // same fired-criterion index that step_n would report.
        const auto& conv = solver.convergence();
        auto maybe = conv.check(last, final_iter);
        const int fired = first_fired(conv.last_check_results());

        const auto& s = solver.state();

        // Split s.lambda into (lambda_eq, mu_ineq) using n_eq/n_ineq.
        // Use segment expressions directly -- no VectorXd materialization.
        // If s.lambda is undersized on the first step, pass zero-sized
        // head/segments so the breakdown helper skips those legs.
        const bool lambda_ready = s.lambda.size() >= s.n_eq + s.n_ineq;
        const auto lambda_eq = lambda_ready
            ? s.lambda.head(s.n_eq)
            : s.lambda.head(0);
        const auto mu_ineq = lambda_ready
            ? s.lambda.segment(s.n_eq, s.n_ineq)
            : s.lambda.segment(0, 0);

        kkt_leg_breakdown legs = kkt_residual_breakdown(
            s.g, s.J_eq, s.J_ineq,
            lambda_eq, mu_ineq,
            s.c_eq, s.c_ineq);

        const double c_eq_norm = s.c_eq.size() > 0
            ? s.c_eq.template lpNorm<Eigen::Infinity>()
            : 0.0;
        double c_ineq_violation = 0.0;
        for(Eigen::Index k = 0; k < s.c_ineq.size(); ++k)
            c_ineq_violation = std::max(c_ineq_violation,
                                        std::max(-s.c_ineq[k], 0.0));

        const double kkt_current = last.kkt_residual.value_or(-1.0);
        const int policy_status_code = last.policy_status
            ? static_cast<int>(*last.policy_status)
            : -1;

        print_row(final_iter,
                  last.objective_value,
                  c_eq_norm, c_ineq_violation,
                  legs,
                  kkt_current, last.gradient_norm,
                  last.x_norm, last.step_size,
                  feasibility_gate_effective,
                  fired, policy_status_code);

        if(last.policy_status)
        {
            final_status_code = policy_status_code;
            break;
        }
        if(maybe)
        {
            final_status_code = static_cast<int>(*maybe);
            break;
        }
    }

    std::println("# terminated: status={} iter={}",
                 final_status_code, final_iter);
    std::println("");
}

// -----------------------------------------------------------------
// byrd_lbfgsb on brown_badly_scaled: only s.x, s.g, s.lower, s.upper
// are available on the state. No multipliers, no constraints.
// The per-leg breakdown collapses to stationarity-only (== ||g||_inf
// at an interior point, or the projected-gradient norm at a bound),
// which is what kkt_residual_bound already returns in last.kkt_residual.
// The row is still emitted so the null-step termination gap is visible
// -- final_status=-1 and fired_criterion=-1 for every iteration is the
// regression signal per CONTEXT.md §D.
// -----------------------------------------------------------------

void run_byrd_brown_case()
{
    print_header("byrd_lbfgsb brown_badly_scaled");

    using Problem = nablapp::brown_badly_scaled<double>;
    Problem problem;
    auto x0 = problem.initial_point();
    nablapp::solver_options opts;
    // The post-Phase-31 regression runs to max_iterations silently;
    // cap at 10000 to reproduce the snapshot behaviour. Reading 10000
    // rows per-iter is overkill -- subsample by emitting only every
    // 250th row plus the last. CSV interpretation still covers the
    // whole trajectory via the subsampled view.
    opts.max_iterations = 10000;
    opts.set_gradient_threshold(1e-12);
    opts.set_objective_threshold(1e-14);
    opts.set_step_threshold(1e-14);

    nablapp::basic_solver solver{nablapp::byrd_lbfgsb_policy{}, problem, x0, opts};

    // byrd_lbfgsb default feasibility_gate is +inf (no primal-feasibility
    // gate; bound-constrained projected-gradient already absorbs it).
    constexpr double feasibility_gate_effective =
        std::numeric_limits<double>::infinity();

    nablapp::step_result<double> last{};
    std::uint32_t final_iter = 0;
    int final_status_code = -1;
    constexpr std::uint32_t subsample = 250;

    for(std::uint32_t i = 0; i < opts.max_iterations; ++i)
    {
        last = solver.step();
        final_iter = i + 1;

        const auto& conv = solver.convergence();
        auto maybe = conv.check(last, final_iter);
        const int fired = first_fired(conv.last_check_results());

        const int policy_status_code = last.policy_status
            ? static_cast<int>(*last.policy_status)
            : -1;

        bool terminating = last.policy_status.has_value() || maybe.has_value();
        bool emit = terminating || (i % subsample == 0) || (i < 15);

        if(emit)
        {
            // Unconstrained bound-constrained case: no equality /
            // inequality constraints, no multipliers. Route through
            // kkt_residual_breakdown with zero-sized constraint vectors
            // so the primal / dual / complementarity legs collapse to
            // zero and stationarity reduces to ||g||_inf; keeps the
            // helper the single source of truth for the CSV row.
            //
            // The empty matrices / vectors are built as zero-size slices
            // of s.g so no dynamic Eigen types are introduced. Column
            // count on J_eq / J_ineq does not matter when rows() == 0;
            // the breakdown helper short-circuits before using them.
            const auto& s = solver.state();
            const auto empty_vec = s.g.head(0);
            const auto empty_mat = s.g.transpose().topRows(0);
            kkt_leg_breakdown legs = kkt_residual_breakdown(
                s.g, empty_mat, empty_mat,
                empty_vec, empty_vec,
                empty_vec, empty_vec);

            const double kkt_current = last.kkt_residual.value_or(-1.0);

            print_row(final_iter,
                      last.objective_value,
                      0.0, 0.0,
                      legs,
                      kkt_current, last.gradient_norm,
                      last.x_norm, last.step_size,
                      feasibility_gate_effective,
                      fired, policy_status_code);
        }

        if(last.policy_status)
        {
            final_status_code = policy_status_code;
            break;
        }
        if(maybe)
        {
            final_status_code = static_cast<int>(*maybe);
            break;
        }
    }

    std::println("# terminated: status={} iter={}",
                 final_status_code, final_iter);
    std::println("");
}

}

int main()
{
    using Hs006 = nablapp::hs006<double>;
    using Hs007 = nablapp::hs007<double>;
    using Hs026 = nablapp::hs026<double>;
    using Hs071 = nablapp::hs071<double>;

    // 1. kraft_slsqp on hs006 -- post-phase31 6 iters / acc 2.16e-06 vs
    //    post-phase30 7 iters / acc 9.24e-08 (23x worse).
    run_sqp_case("kraft_slsqp hs006",
                 Hs006{},
                 nablapp::kraft_slsqp_policy<Hs006::problem_dimension>{});

    // 2. kraft_slsqp on hs026 -- post-phase31 12 iters / 1.57e-04 vs
    //    post-phase30 20 iters / 2.90e-07 (542x worse).
    run_sqp_case("kraft_slsqp hs026",
                 Hs026{},
                 nablapp::kraft_slsqp_policy<Hs026::problem_dimension>{});

    // 3. nw_sqp on hs007 -- post-phase31 6 iters / 1.70e-04 vs
    //    post-phase30 7 iters / 1.19e-06 (143x worse).
    run_sqp_case("nw_sqp hs007",
                 Hs007{},
                 nablapp::nw_sqp_policy<Hs007::problem_dimension>{});

    // 4. nw_sqp on hs026 -- byte-identical mechanism to (2).
    run_sqp_case("nw_sqp hs026",
                 Hs026{},
                 nablapp::nw_sqp_policy<Hs026::problem_dimension>{});

    // 5. byrd_lbfgsb on brown_badly_scaled -- 13 iters stalled pre vs
    //    10000 iters max_iterations post, identical acc 6.628e-28.
    run_byrd_brown_case();

    // 6. nw_sqp on hs071 -- D-E4 diagnostic-only sixth case. Post-phase31
    //    ftol_reached at acc=3.56 iter 8; unchanged from pre-phase31
    //    accuracy. Logs the 5-leg breakdown at the iter-8 ftol iterate
    //    so the Plan 02 author can judge whether the Full E-measure
    //    will close it as a side effect.
    run_sqp_case("nw_sqp hs071",
                 Hs071{},
                 nablapp::nw_sqp_policy<Hs071::problem_dimension>{});

    return 0;
}
