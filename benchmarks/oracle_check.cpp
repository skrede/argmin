// oracle_check: independent analytic correctness validator.
//
// The publication regression gate (regression_check) is a pure CSV bound
// comparison whose baseline bounds and pass/expected_fail dispositions were
// historically derived from the very run it then checks. That makes the
// gate's "green" self-referential: it cannot falsify a genuine regression.
//
// oracle_check supplies the independent witness that gate lacks. It links the
// library and the analytic test-function definitions, reads the publication
// summary plus the returned-point sidecar (publish_returned_points.csv), and
// recomputes correctness ANALYTICALLY at the point the solver actually
// returned:
//
//   * Closed-form cells (Hock-Schittkowski and friends): the objective
//     distance |f(x*) - f*| against the analytic optimum, combined with a
//     feasibility residual recomputed from the problem's own constraint and
//     bound definitions at x* -- NOT the solver's self-reported
//     constraint_violation column.
//
//   * Control cells with no closed-form optimum (the nmpc_lqr family): an
//     analytic KKT residual at x* covering all four KKT conditions
//     (stationarity, primal feasibility, complementarity, and dual
//     feasibility). Constraint multipliers are recovered by least squares
//     from the analytic constraint gradients when the sidecar does not carry
//     them; a wrong-sign multiplier at an active bound (a non-minimizer) is
//     caught by the dual-feasibility leg rather than passed.
//
// The per-cell verdict is written to oracle_verdict.csv. Its witness column
// names the analytic oracle and is never the regression-gate pass token, so
// the downstream disposition source of truth is demonstrably independent of
// the run being judged.
//
// The control-cell KKT pass bound is EMPIRICALLY CALIBRATED, not invented:
// --calibrate solves each control cell with the tightest-tolerance
// filter_trsqp_accurate reference and reports the achievable KKT floor; the
// committed pass bound is that measured floor times a small stated multiplier
// (see kControlKktBound below).
//
// The binary is C++20 and builds -fno-exceptions -fno-rtti clean: all error
// paths are return-based, and no reachable code path throws.
//
// Usage:
//   oracle_check <publish_summary.csv> <publish_returned_points.csv>
//                <oracle_verdict.csv>
//                [--accuracy-cutoff DOUBLE] [--cv-cutoff DOUBLE]
//   oracle_check --calibrate
//   oracle_check --self-test

#include "problem_registry.h"

#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/filter_trsqp_policy.h"
#include "argmin/solver/convergence.h"
#include "argmin/formulation/concepts.h"

#include <Eigen/Dense>

#include <map>
#include <set>
#include <cmath>
#include <tuple>
#include <array>
#include <limits>
#include <string>
#include <vector>
#include <format>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <charconv>
#include <iostream>
#include <optional>
#include <algorithm>
#include <filesystem>
#include <string_view>

namespace
{

// Publication bar mirrored from regression_check's committed cutoffs so the
// oracle's closed-form verdict is comparable to the CSV bound gate: the cell
// meets the oracle when the analytic objective distance is within the
// accuracy cutoff and the recomputed feasibility residual is within the cv
// cutoff.
constexpr double kAccuracyCutoff = 1e-12;
constexpr double kCvCutoff       = 1e-8;

// Active-set identification tolerance for the control-cell KKT oracle: a box
// bound is treated as active when the returned control sits within this
// distance of it. This is a structural set-membership tolerance, not a
// pass/fail numeric -- it decides which bound gradients enter the multiplier
// recovery, not whether a cell passes.
constexpr double kBoundActiveTol = 1e-7;

// Control-cell KKT pass bound, per cell. EMPIRICALLY CALIBRATED (reproduce
// with --calibrate), never invented. Each floor is the KKT residual this
// oracle recomputes at the returned point of the tightest-tolerance
// filter_trsqp_accurate reference solve of that cell (the high-accuracy
// reference), under the bench publication constrained-gate configuration
// (gradient 1e-8, objective/step 1e-6, stationarity 1e-8), measured on this
// host:
//
//   nmpc_lqr_h10:  floor = 9.601343e-09  (stationarity-dominated;
//                                          primal 3.4e-15, complementarity
//                                          and dual at machine zero)
//   nmpc_lqr_h20:  floor = 2.676681e-08  (stationarity-dominated;
//                                          primal 5.4e-15)
//
// The reference SQP is deterministic, so each floor is an exact single
// measurement, not a seed-averaged estimate. The pass bound is the measured
// floor times a one-decade multiplier: the band absorbs the modest
// stationarity-residual spread among the other converged SQP variants on the
// same cell, while a genuinely non-converged NMPC iterate sits several
// decades higher (an unconverged trajectory carries ||KKT|| ~ 1e-1..1e0),
// so the bound catches non-convergence without laundering it.
constexpr double kControlKktMultiplier = 10.0;
constexpr double kControlKktFloor_h10  = 9.601343e-09;
constexpr double kControlKktFloor_h20  = 2.676681e-08;

[[nodiscard]] auto control_kkt_bound(std::string_view problem) -> double
{
    if(problem == "nmpc_lqr_h20")
        return kControlKktFloor_h20 * kControlKktMultiplier;
    // Default to the coarser-horizon floor for any control cell not
    // separately calibrated (the wider bound is the conservative choice).
    return kControlKktFloor_h10 * kControlKktMultiplier;
}

// ---- CSV helpers (std-only, fail-closed) --------------------------------

[[nodiscard]] auto trim(std::string_view s) -> std::string
{
    auto is_ws = [](unsigned char c)
    {
        return c == ' ' || c == '\t' || c == '\r' || c == '\n';
    };
    while(!s.empty() && is_ws(static_cast<unsigned char>(s.front())))
        s.remove_prefix(1);
    while(!s.empty() && is_ws(static_cast<unsigned char>(s.back())))
        s.remove_suffix(1);
    return std::string{s};
}

[[nodiscard]] auto split_csv_line(std::string_view line) -> std::vector<std::string>
{
    std::vector<std::string> out;
    std::size_t start = 0;
    for(std::size_t i = 0; i <= line.size(); ++i)
    {
        if(i == line.size() || line[i] == ',')
        {
            out.emplace_back(trim(line.substr(start, i - start)));
            start = i + 1;
        }
    }
    return out;
}

struct csv_table
{
    std::vector<std::string>              header;
    std::vector<std::vector<std::string>> rows;
};

[[nodiscard]] auto read_csv(const std::filesystem::path& path,
                            csv_table&                   out,
                            std::string&                 err) -> bool
{
    std::ifstream in(path);
    if(!in)
    {
        err = "cannot open '" + path.string() + "' for reading";
        return false;
    }
    std::string line;
    bool have_header = false;
    while(std::getline(in, line))
    {
        std::string stripped = trim(line);
        if(stripped.empty()) continue;
        if(stripped.front() == '#') continue;
        if(!have_header)
        {
            out.header  = split_csv_line(stripped);
            have_header = true;
            continue;
        }
        out.rows.emplace_back(split_csv_line(stripped));
    }
    if(!have_header)
    {
        err = "no header line in '" + path.string() + "'";
        return false;
    }
    return true;
}

[[nodiscard]] auto column_index(const std::vector<std::string>& header,
                                std::string_view                name)
    -> std::optional<std::size_t>
{
    for(std::size_t i = 0; i < header.size(); ++i)
        if(header[i] == name) return i;
    return std::nullopt;
}

[[nodiscard]] auto parse_packed_vector(std::string_view field)
    -> std::vector<double>
{
    std::vector<double> out;
    std::size_t start = 0;
    std::string trimmed = trim(field);
    std::string_view s{trimmed};
    if(s.empty()) return out;
    for(std::size_t i = 0; i <= s.size(); ++i)
    {
        if(i == s.size() || s[i] == ';')
        {
            std::string tok = trim(s.substr(start, i - start));
            double v{};
            auto* first = tok.data();
            auto* last  = tok.data() + tok.size();
            auto [ptr, ec] = std::from_chars(first, last, v);
            if(ec != std::errc{} || ptr != last)
            {
                out.clear();
                return out;  // signal parse failure via empty
            }
            out.push_back(v);
            start = i + 1;
        }
    }
    return out;
}

// ---- Cell key ------------------------------------------------------------

using cell_key = std::tuple<std::string, std::string, std::string, std::string>;

[[nodiscard]] auto is_control_cell(std::string_view problem) -> bool
{
    return problem.size() >= 4 && problem.substr(0, 4) == "nmpc";
}

// ---- Analytic recompute primitives (templated on Problem) ----------------

template <typename P, typename X, typename C>
concept has_constraints = requires(const P& p, const X& x, C& c) { p.constraints(x, c); };

template <typename P>
concept has_constraint_counts = requires(const P& p)
{
    { p.num_equality() } -> std::convertible_to<int>;
    { p.num_inequality() } -> std::convertible_to<int>;
};

template <typename P, typename X>
concept has_bounds = requires(const P& p) { p.lower_bounds(); p.upper_bounds(); };

// Feasibility residual recomputed analytically at x: the worst of the
// absolute equality residuals, the worst inequality violation (c_ineq >= 0
// convention), and the worst box-bound violation.
template <typename Problem, typename Vec>
[[nodiscard]] auto feasibility_residual(const Problem& prob, const Vec& x) -> double
{
    double resid = 0.0;

    if constexpr(has_constraint_counts<Problem> &&
                 has_constraints<Problem, Vec, Eigen::VectorXd>)
    {
        const int n_eq   = prob.num_equality();
        const int n_ineq = prob.num_inequality();
        const int nc     = n_eq + n_ineq;
        if(nc > 0)
        {
            Eigen::VectorXd c(nc);
            c.setZero();
            prob.constraints(x, c);
            for(int i = 0; i < n_eq; ++i)
                resid = std::max(resid, std::abs(c[i]));
            for(int i = n_eq; i < nc; ++i)
                resid = std::max(resid, std::max(0.0, -c[i]));
        }
    }

    if constexpr(has_bounds<Problem, Vec>)
    {
        const auto lb = prob.lower_bounds();
        const auto ub = prob.upper_bounds();
        for(Eigen::Index j = 0; j < x.size(); ++j)
        {
            if(std::isfinite(lb[j]))
                resid = std::max(resid, std::max(0.0, lb[j] - x[j]));
            if(std::isfinite(ub[j]))
                resid = std::max(resid, std::max(0.0, x[j] - ub[j]));
        }
    }

    return resid;
}

struct kkt_breakdown
{
    double stationarity{};
    double primal_feasibility{};
    double complementarity{};
    double dual_feasibility{};  // magnitude of the worst nonnegativity breach
    [[nodiscard]] auto worst() const -> double
    {
        return std::max(std::max(stationarity, primal_feasibility),
                        std::max(complementarity, dual_feasibility));
    }
};

// Analytic KKT residual at x for an equality-plus-box control cell. The
// equality multipliers (and the multipliers of any active box bound) are
// recovered by least squares from the analytic constraint gradients:
//   grad_f(x) = J_eq^T lambda + sum_active sigma_j grad_g_j
// with the Lagrangian sign convention L = f - lambda^T c_eq - sigma^T g_bound
// and g_bound >= 0. Dual feasibility requires sigma >= 0; a recovered
// negative bound multiplier (a non-minimizer parked at a bound) is surfaced
// through the dual_feasibility leg.
template <typename Problem, typename Vec>
[[nodiscard]] auto compute_kkt(const Problem& prob, const Vec& x) -> kkt_breakdown
{
    kkt_breakdown out;
    const int n = static_cast<int>(x.size());

    Vec g = x;
    g.setZero();
    prob.gradient(x, g);

    const int n_eq = prob.num_equality();
    Eigen::MatrixXd J_eq(n_eq, n);
    J_eq.setZero();
    prob.constraint_jacobian(x, J_eq);

    // Equality residual (primal feasibility) and its magnitude.
    Eigen::VectorXd c_eq(n_eq);
    c_eq.setZero();
    prob.constraints(x, c_eq);
    double primal = 0.0;
    for(int i = 0; i < n_eq; ++i)
        primal = std::max(primal, std::abs(c_eq[i]));

    // Active box bounds -> extra constraint-gradient columns.
    std::vector<int>    active_index;      // decision index of the active bound
    std::vector<double> active_slack;      // |x_j - bound|
    std::vector<double> active_sign;       // +1 lower (grad +e_j), -1 upper (grad -e_j)
    if constexpr(has_bounds<Problem, Vec>)
    {
        const auto lb = prob.lower_bounds();
        const auto ub = prob.upper_bounds();
        for(int j = 0; j < n; ++j)
        {
            if(std::isfinite(lb[j]) && std::abs(x[j] - lb[j]) <= kBoundActiveTol)
            {
                active_index.push_back(j);
                active_slack.push_back(std::abs(x[j] - lb[j]));
                active_sign.push_back(+1.0);
            }
            else if(std::isfinite(ub[j]) && std::abs(ub[j] - x[j]) <= kBoundActiveTol)
            {
                active_index.push_back(j);
                active_slack.push_back(std::abs(ub[j] - x[j]));
                active_sign.push_back(-1.0);
            }
            // Box violations also feed primal feasibility.
            if(std::isfinite(lb[j]))
                primal = std::max(primal, std::max(0.0, lb[j] - x[j]));
            if(std::isfinite(ub[j]))
                primal = std::max(primal, std::max(0.0, x[j] - ub[j]));
        }
    }

    const int n_active = static_cast<int>(active_index.size());
    const int ncols    = n_eq + n_active;

    // Multiplier recovery A y ~= g, columns = [ grad c_eq_i | grad g_active_j ].
    Eigen::MatrixXd A(n, ncols);
    A.setZero();
    for(int i = 0; i < n_eq; ++i)
        A.col(i) = J_eq.row(i).transpose();
    for(int a = 0; a < n_active; ++a)
        A(active_index[static_cast<std::size_t>(a)], n_eq + a) =
            active_sign[static_cast<std::size_t>(a)];

    Eigen::VectorXd y = Eigen::VectorXd::Zero(ncols);
    if(ncols > 0)
        y = A.colPivHouseholderQr().solve(g);

    Eigen::VectorXd stat = g - A * y;
    out.stationarity = (n > 0) ? stat.cwiseAbs().maxCoeff() : 0.0;
    out.primal_feasibility = primal;

    // Complementarity: sigma_j * slack_j for active bounds (small because
    // active slack ~ 0); inactive-bound multipliers are structurally zero.
    double compl_resid = 0.0;
    double dual_breach = 0.0;
    for(int a = 0; a < n_active; ++a)
    {
        const double sigma = y[n_eq + a];
        compl_resid = std::max(compl_resid,
                               std::abs(sigma * active_slack[static_cast<std::size_t>(a)]));
        dual_breach = std::max(dual_breach, std::max(0.0, -sigma));
    }
    out.complementarity = compl_resid;
    out.dual_feasibility = dual_breach;
    return out;
}

// ---- Verdict -------------------------------------------------------------

struct verdict
{
    bool                  pass{false};
    std::optional<double> objective_distance;
    std::optional<double> feasibility_residual;
    std::optional<double> kkt_residual;
    std::string           witness;
};

[[nodiscard]] auto fmt_opt(const std::optional<double>& v) -> std::string
{
    if(!v) return "";
    return std::format("{:.15e}", *v);
}

// Evaluate one cell analytically given the concrete problem and the parsed
// returned point. Returns std::nullopt only on a structural mismatch (wrong
// vector length) so the caller can fail closed.
template <typename Problem, typename Vec>
[[nodiscard]] auto evaluate_cell(std::string_view problem_name,
                                 const Problem&   prob,
                                 const Vec&       x,
                                 double           acc_cutoff,
                                 double           cv_cutoff) -> verdict
{
    verdict v;
    if(is_control_cell(problem_name))
    {
        if constexpr(requires(const Problem& p, Vec gg) {
                         p.gradient(x, gg); p.constraint_jacobian(x, std::declval<Eigen::MatrixXd&>());
                         p.num_equality(); })
        {
            const kkt_breakdown kkt = compute_kkt(prob, x);
            const double f = static_cast<double>(prob.value(x));
            v.objective_distance   = std::abs(f - static_cast<double>(prob.optimal_value()));
            v.feasibility_residual = kkt.primal_feasibility;
            v.kkt_residual         = kkt.worst();
            v.pass                 = kkt.worst() <= control_kkt_bound(problem_name);
            v.witness              = "kkt-at-x*";
        }
        else
        {
            v.witness = "kkt-at-x*-unavailable";
        }
        return v;
    }

    // Closed-form cell: objective distance + recomputed feasibility.
    const double f  = static_cast<double>(prob.value(x));
    const double od = std::abs(f - static_cast<double>(prob.optimal_value()));
    const double fr = feasibility_residual(prob, x);
    v.objective_distance   = od;
    v.feasibility_residual = fr;
    v.pass                 = (od <= acc_cutoff) && (fr <= cv_cutoff);
    v.witness              = "feasibility-at-x*";
    return v;
}

// Build the compile-time-correct decision vector for a problem from parsed
// tokens; returns false on a length mismatch.
template <typename Problem>
[[nodiscard]] auto make_point(const Problem&             prob,
                              const std::vector<double>& tokens,
                              Eigen::Vector<double, argmin::problem_dimension_v<Problem>>& x)
    -> bool
{
    constexpr int Ndim = argmin::problem_dimension_v<Problem>;
    const int expected = prob.dimension();
    if(static_cast<int>(tokens.size()) != expected)
        return false;
    if constexpr(Ndim == Eigen::Dynamic)
        x.resize(expected);
    for(int i = 0; i < expected; ++i)
        x[i] = tokens[static_cast<std::size_t>(i)];
    return true;
}

// ---- Verdict CSV production ----------------------------------------------

struct sidecar_row
{
    std::string solver;
    std::string problem;
    std::string mode;
    std::string seed;
    std::vector<double> point;
};

[[nodiscard]] auto run_oracle(const std::filesystem::path& summary_path,
                              const std::filesystem::path& sidecar_path,
                              const std::filesystem::path& verdict_path,
                              double                       acc_cutoff,
                              double                       cv_cutoff) -> int
{
    std::string err;

    // Summary: build the set of argmin cell keys so only argmin rows are
    // judged (the informative cross-solver rows carry no returned point).
    csv_table summary;
    if(!read_csv(summary_path, summary, err))
    {
        std::cerr << "oracle_check: " << err << '\n';
        return 1;
    }
    auto s_solver  = column_index(summary.header, "solver");
    auto s_problem = column_index(summary.header, "problem");
    auto s_mode    = column_index(summary.header, "mode");
    auto s_seed    = column_index(summary.header, "seed");
    auto s_library = column_index(summary.header, "library");
    if(!s_solver || !s_problem || !s_mode || !s_seed || !s_library)
    {
        std::cerr << "oracle_check: publication summary missing a required column"
                     " (solver/problem/mode/seed/library)\n";
        return 1;
    }
    std::set<cell_key> argmin_keys;
    for(const auto& r : summary.rows)
    {
        const std::size_t need =
            std::max({*s_solver, *s_problem, *s_mode, *s_seed, *s_library});
        if(r.size() <= need) continue;
        if(r[*s_library] != "argmin") continue;
        argmin_keys.emplace(r[*s_solver], r[*s_problem], r[*s_mode], r[*s_seed]);
    }

    // Sidecar: the returned points.
    csv_table sidecar;
    if(!read_csv(sidecar_path, sidecar, err))
    {
        std::cerr << "oracle_check: " << err << '\n';
        return 1;
    }
    auto p_solver  = column_index(sidecar.header, "solver");
    auto p_problem = column_index(sidecar.header, "problem");
    auto p_mode    = column_index(sidecar.header, "mode");
    auto p_seed    = column_index(sidecar.header, "seed");
    auto p_point   = column_index(sidecar.header, "returned_point");
    if(!p_solver || !p_problem || !p_mode || !p_seed || !p_point)
    {
        std::cerr << "oracle_check: returned-point sidecar missing a required"
                     " column (solver/problem/mode/seed/returned_point)\n";
        return 1;
    }

    // Group sidecar rows by problem so each concrete problem type is
    // dispatched once through for_each_problem.
    std::map<std::string, std::vector<sidecar_row>> by_problem;
    const std::size_t need =
        std::max({*p_solver, *p_problem, *p_mode, *p_seed, *p_point});
    for(const auto& r : sidecar.rows)
    {
        if(r.size() <= need) continue;
        cell_key key{r[*p_solver], r[*p_problem], r[*p_mode], r[*p_seed]};
        if(!argmin_keys.contains(key)) continue;   // only judge argmin cells
        std::string field = r[*p_point];
        if(trim(field).empty()) continue;          // no point emitted
        sidecar_row sr;
        sr.solver  = r[*p_solver];
        sr.problem = r[*p_problem];
        sr.mode    = r[*p_mode];
        sr.seed    = r[*p_seed];
        sr.point   = parse_packed_vector(field);
        by_problem[sr.problem].push_back(std::move(sr));
    }

    // Emit the verdict CSV.
    std::ofstream out(verdict_path);
    if(!out)
    {
        std::cerr << "oracle_check: cannot open '" << verdict_path.string()
                  << "' for writing\n";
        return 1;
    }
    out << "solver,problem,mode,seed,oracle_pass,objective_distance,"
           "feasibility_residual,kkt_residual,witness\n";

    int structural_failures = 0;
    int emitted = 0;

    argmin::bench::for_each_problem([&](std::string_view name, auto&& prob) {
        using Problem = std::remove_cvref_t<decltype(prob)>;
        auto it = by_problem.find(std::string{name});
        if(it == by_problem.end()) return;
        for(const auto& sr : it->second)
        {
            Eigen::Vector<double, argmin::problem_dimension_v<Problem>> x;
            if(sr.point.empty() || !make_point(prob, sr.point, x))
            {
                std::cerr << "oracle_check: returned-point length mismatch for "
                          << sr.solver << ',' << sr.problem << ',' << sr.mode
                          << ',' << sr.seed << '\n';
                ++structural_failures;
                continue;
            }
            verdict v = evaluate_cell(name, prob, x, acc_cutoff, cv_cutoff);
            out << std::format("{},{},{},{},{},{},{},{},{}\n",
                               sr.solver, sr.problem, sr.mode, sr.seed,
                               v.pass ? "pass" : "fail",
                               fmt_opt(v.objective_distance),
                               fmt_opt(v.feasibility_residual),
                               fmt_opt(v.kkt_residual),
                               v.witness);
            ++emitted;
        }
    });

    out.close();
    std::cout << "oracle_check: wrote " << emitted << " verdicts to "
              << verdict_path.string() << '\n';
    if(structural_failures > 0)
    {
        std::cerr << "oracle_check: " << structural_failures
                  << " returned-point structural mismatch(es); failing closed\n";
        return 1;
    }
    return 0;
}

// ---- Calibration ---------------------------------------------------------

// Solve one control cell with the tightest-tolerance filter_trsqp_accurate
// reference and report the KKT residual recomputed by the oracle routine at
// the returned point. The reference SQP is deterministic, so a single
// measurement establishes the achievable floor.
template <int H>
[[nodiscard]] auto calibrate_control_cell(std::string_view name) -> double
{
    using Problem = argmin::nmpc_lqr<H>;
    constexpr int Ndim = Problem::problem_dimension;
    using Policy = argmin::filter_trsqp_policy_accurate<Ndim>;

    Problem prob;
    auto x0 = prob.initial_point();
    // Mirror the bench publication constrained-gate configuration exactly so
    // the reference point (and therefore the measured floor) matches the
    // high-accuracy reference the publication run actually produces:
    // gradient 1e-8, objective/step 1e-6, stationarity ftol_rel(1e-16) * 1e8.
    argmin::solver_options<> opts{};
    opts.max_iterations = 10000;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-6);
    opts.set_step_threshold(1e-6);
    opts.set_stationarity_threshold(1e-8);

    argmin::step_budget_solver<Policy, Ndim, Problem> solver(prob, x0, opts);
    auto result = solver.solve(opts);
    (void)result;

    Eigen::Vector<double, Ndim> x = solver.state().x;
    const kkt_breakdown kkt = compute_kkt(prob, x);
    std::cout << std::format(
        "{}: kkt_floor={:.6e} (stationarity={:.3e} primal={:.3e} "
        "complementarity={:.3e} dual={:.3e})\n",
        name, kkt.worst(), kkt.stationarity, kkt.primal_feasibility,
        kkt.complementarity, kkt.dual_feasibility);
    return kkt.worst();
}

[[nodiscard]] auto run_calibrate() -> int
{
    std::cout << "oracle_check: control-cell KKT floor calibration"
                 " (filter_trsqp_accurate reference)\n";
    const double f10 = calibrate_control_cell<10>("nmpc_lqr_h10");
    const double f20 = calibrate_control_cell<20>("nmpc_lqr_h20");
    const double worst = std::max(f10, f20);
    std::cout << std::format(
        "oracle_check: worst measured floor = {:.6e}; suggested bound at x{:g}"
        " = {:.6e}\n",
        worst, kControlKktMultiplier, worst * kControlKktMultiplier);
    return 0;
}

void print_usage(std::ostream& os)
{
    os << "Usage: oracle_check <publish_summary.csv> "
          "<publish_returned_points.csv> <oracle_verdict.csv>\n"
          "                    [--accuracy-cutoff DOUBLE] [--cv-cutoff DOUBLE]\n"
          "       oracle_check --calibrate\n"
          "       oracle_check --self-test\n";
}

}  // namespace

int main(int argc, char** argv)
{
    double acc_cutoff = kAccuracyCutoff;
    double cv_cutoff  = kCvCutoff;
    bool   calibrate  = false;
    bool   self_test  = false;
    std::vector<std::string_view> positional;

    for(int i = 1; i < argc; ++i)
    {
        std::string_view arg = argv[i];
        if(arg == "--help" || arg == "-h")
        {
            print_usage(std::cout);
            return 0;
        }
        if(arg == "--calibrate") { calibrate = true; continue; }
        if(arg == "--self-test") { self_test = true; continue; }
        if(arg == "--accuracy-cutoff" && i + 1 < argc)
        {
            std::string_view v = argv[++i];
            std::string t = trim(v);
            auto [ptr, ec] = std::from_chars(t.data(), t.data() + t.size(), acc_cutoff);
            if(ec != std::errc{} || ptr != t.data() + t.size())
            {
                std::cerr << "oracle_check: invalid --accuracy-cutoff value\n";
                return 1;
            }
            continue;
        }
        if(arg == "--cv-cutoff" && i + 1 < argc)
        {
            std::string_view v = argv[++i];
            std::string t = trim(v);
            auto [ptr, ec] = std::from_chars(t.data(), t.data() + t.size(), cv_cutoff);
            if(ec != std::errc{} || ptr != t.data() + t.size())
            {
                std::cerr << "oracle_check: invalid --cv-cutoff value\n";
                return 1;
            }
            continue;
        }
        if(!arg.empty() && arg.front() == '-')
        {
            std::cerr << "oracle_check: unknown or incomplete option '" << arg << "'\n";
            print_usage(std::cerr);
            return 1;
        }
        positional.push_back(arg);
    }

    if(calibrate)
        return run_calibrate();
    if(self_test)
    {
        std::cerr << "oracle_check: --self-test not yet wired\n";
        return 1;
    }

    if(positional.size() != 3)
    {
        print_usage(std::cerr);
        return 1;
    }
    return run_oracle(std::filesystem::path{std::string{positional[0]}},
                      std::filesystem::path{std::string{positional[1]}},
                      std::filesystem::path{std::string{positional[2]}},
                      acc_cutoff, cv_cutoff);
}
