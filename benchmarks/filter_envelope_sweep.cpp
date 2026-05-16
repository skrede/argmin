// HS043 envelope sweep for filter_slsqp + filter_nw_sqp policies.
//
// Sweeps the asymmetric (gamma_f, gamma_h) envelope across a 4x4
// grid {1e-3, 1e-4, 1e-5, 1e-6}^2 on HS043 (the canonical
// strictly-feasible-descent over-rejection cell). Each combo is
// replayed against HS024 (LS-failure-prone) and HS076 (multi-
// constraint inequality) regression guards; combos that fail either
// guard are excluded from the candidate set. The selected default
// per policy minimises |objective - (-44.0)| on HS043 subject to
// passing both guards.
//
// Output: per-policy table of (gamma_f, gamma_h, hs043_f, hs043_cv,
// hs043_iters, hs024_passes, hs076_passes), followed by the
// SELECTED line.
//
// Reference: Hock & Schittkowski 1981, Problems 24, 43, 76;
//            Wachter & Biegler 2006 Section 2.3 (filter envelope);
//            Fletcher & Leyffer 2002 Section 5 (slack-margin gamma
//            tuning).

#include "argmin/solver/basic_solver.h"
#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace
{

constexpr std::array<double, 4> gamma_grid{1e-3, 1e-4, 1e-5, 1e-6};

struct sweep_row
{
    double gamma_f{};
    double gamma_h{};
    double hs043_f{};
    double hs043_cv{};
    std::uint32_t hs043_iters{};
    bool hs024_passes{};
    bool hs076_passes{};
    double hs024_f{};
    double hs024_cv{};
    std::uint32_t hs024_iters{};
    double hs076_f{};
    double hs076_cv{};
    std::uint32_t hs076_iters{};
};

// Tag selector to dispatch policy-specific solver-option shapes; the
// two filter policies use slightly different convergence thresholds in
// their existing unit tests (e.g. filter_slsqp HS024 omits the
// gradient threshold to avoid premature termination from the
// least-squares lambda zeroing out ||grad f - A^T lambda||, while
// filter_nw_sqp HS024 sets it to 1e-8).
struct slsqp_tag {};
struct nw_sqp_tag {};

template <template <int> class Policy> struct policy_tag;
template <> struct policy_tag<argmin::filter_slsqp_policy>  { using type = slsqp_tag;  };
template <> struct policy_tag<argmin::filter_nw_sqp_policy> { using type = nw_sqp_tag; };

// HS043 options aligned to per-policy unit-test shape.
inline argmin::solver_options<> hs043_opts(slsqp_tag)
{
    argmin::solver_options<> opts;
    opts.max_iterations = 500;
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-10);
    return opts;
}
inline argmin::solver_options<> hs043_opts(nw_sqp_tag)
{
    argmin::solver_options<> opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-4);
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-15);
    return opts;
}

// HS024 options aligned to per-policy unit-test shape.
inline argmin::solver_options<> hs024_opts(slsqp_tag)
{
    argmin::solver_options<> opts;
    opts.max_iterations = 50;
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);
    return opts;
}
inline argmin::solver_options<> hs024_opts(nw_sqp_tag)
{
    argmin::solver_options<> opts;
    opts.max_iterations = 50;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-12);
    opts.set_step_threshold(1e-12);
    return opts;
}

// HS076 options aligned to per-policy unit-test shape (both policies
// use identical shape on this cell).
inline argmin::solver_options<> hs076_opts()
{
    argmin::solver_options<> opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-6);
    opts.set_step_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    return opts;
}

template <template <int> class Policy>
sweep_row run_combo(double gamma_f, double gamma_h)
{
    using tag = typename policy_tag<Policy>::type;
    sweep_row row{};
    row.gamma_f = gamma_f;
    row.gamma_h = gamma_h;

    // HS043 primary cell.
    {
        argmin::hs043<> problem;
        auto x0 = problem.initial_point();
        auto opts = hs043_opts(tag{});

        typename Policy<argmin::hs043<>::problem_dimension>::options_type policy_opts;
        policy_opts.gamma_f = gamma_f;
        policy_opts.gamma_h = gamma_h;

        argmin::basic_solver solver{
            Policy<argmin::hs043<>::problem_dimension>{},
            problem, x0, opts, policy_opts};
        auto result = solver.solve(opts);
        row.hs043_f = result.objective_value;
        row.hs043_cv = result.constraint_violation;
        row.hs043_iters = result.iterations;
    }

    // HS024 regression guard. Bar: f == Approx(-1.0).margin(1e-6),
    // iters <= 14, cv < 1e-6 (existing test).
    {
        argmin::hs024<> problem;
        auto x0 = problem.initial_point();
        auto opts = hs024_opts(tag{});

        typename Policy<argmin::hs024<>::problem_dimension>::options_type policy_opts;
        policy_opts.gamma_f = gamma_f;
        policy_opts.gamma_h = gamma_h;

        argmin::basic_solver solver{
            Policy<argmin::hs024<>::problem_dimension>{},
            problem, x0, opts, policy_opts};
        auto result = solver.solve(opts);

        const bool f_ok = std::abs(result.objective_value - (-1.0)) < 1e-6;
        const bool cv_ok = result.constraint_violation < 1e-6;
        const bool iters_ok = result.iterations <= 14u;
        row.hs024_passes = f_ok && cv_ok && iters_ok;
        row.hs024_f = result.objective_value;
        row.hs024_cv = result.constraint_violation;
        row.hs024_iters = result.iterations;
    }

    // HS076 regression guard. Bar: f == Approx(-4.68).margin(0.1),
    // cv < 1e-4, iters <= 200 (existing test).
    {
        argmin::hs076<> problem;
        auto x0 = problem.initial_point();
        auto opts = hs076_opts();

        typename Policy<argmin::hs076<>::problem_dimension>::options_type policy_opts;
        policy_opts.gamma_f = gamma_f;
        policy_opts.gamma_h = gamma_h;

        argmin::basic_solver solver{
            Policy<argmin::hs076<>::problem_dimension>{},
            problem, x0, opts, policy_opts};
        auto result = solver.solve(opts);

        const bool f_ok = std::abs(result.objective_value - (-4.68)) < 0.1;
        const bool cv_ok = result.constraint_violation < 1e-4;
        const bool iters_ok = result.iterations <= 200u;
        row.hs076_passes = f_ok && cv_ok && iters_ok;
        row.hs076_f = result.objective_value;
        row.hs076_cv = result.constraint_violation;
        row.hs076_iters = result.iterations;
    }

    return row;
}

template <template <int> class Policy>
void run_policy_sweep(const char* policy_name)
{
    std::printf("\n=== %s envelope sweep ===\n", policy_name);
    std::printf("%-9s %-9s %-12s %-11s %-6s %-9s %-9s %-12s %-11s %-6s %-12s %-11s %-6s\n",
                "gamma_f", "gamma_h",
                "hs043_f", "hs043_cv", "iters",
                "hs024_pass", "hs076_pass",
                "hs024_f", "hs024_cv", "iters",
                "hs076_f", "hs076_cv", "iters");

    std::vector<sweep_row> rows;
    rows.reserve(16);
    for(double gf : gamma_grid)
        for(double gh : gamma_grid)
            rows.push_back(run_combo<Policy>(gf, gh));

    for(const auto& r : rows)
    {
        std::printf(
            "%-9.0e %-9.0e %-12.6f %-11.4e %-6u %-9d %-9d %-12.6f %-11.4e %-6u %-12.6f %-11.4e %-6u\n",
            r.gamma_f, r.gamma_h,
            r.hs043_f, r.hs043_cv, r.hs043_iters,
            r.hs024_passes ? 1 : 0,
            r.hs076_passes ? 1 : 0,
            r.hs024_f, r.hs024_cv, r.hs024_iters,
            r.hs076_f, r.hs076_cv, r.hs076_iters);
    }

    // Selection: minimise |hs043_f - (-44.0)| over combos passing both
    // regression guards. Tie-break on lower hs043_cv.
    sweep_row best{};
    double best_dist = std::numeric_limits<double>::infinity();
    bool have_candidate = false;
    for(const auto& r : rows)
    {
        if(!r.hs024_passes || !r.hs076_passes)
            continue;
        const double dist = std::abs(r.hs043_f - (-44.0));
        if(!have_candidate || dist < best_dist
           || (dist == best_dist && r.hs043_cv < best.hs043_cv))
        {
            best_dist = dist;
            best = r;
            have_candidate = true;
        }
    }

    if(have_candidate)
    {
        std::printf(
            "\nSELECTED %s default: gamma_f=%.0e, gamma_h=%.0e (HS043 f=%.6f, cv=%.4e, iters=%u)\n",
            policy_name, best.gamma_f, best.gamma_h,
            best.hs043_f, best.hs043_cv, best.hs043_iters);
    }
    else
    {
        std::printf(
            "\nSELECTED %s default: NONE (no combo passed both regression guards) -- fallback gamma_f=1e-5, gamma_h=1e-5\n",
            policy_name);
    }
}

}

int main()
{
    run_policy_sweep<argmin::filter_slsqp_policy>("filter_slsqp");
    run_policy_sweep<argmin::filter_nw_sqp_policy>("filter_nw_sqp");
    return 0;
}
