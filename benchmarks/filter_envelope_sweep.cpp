// Filter-envelope (gamma_f, gamma_h) sweep for filter_slsqp + filter_nw_sqp.
//
// Sweeps the asymmetric (gamma_f, gamma_h) envelope across a 4x4 grid
// {1e-3, 1e-4, 1e-5, 1e-6}^2 and evaluates every combo against the
// publication bar (objective distance to the analytic optimum within
// 1e-8, feasibility within 1e-6) over the full filter-lineage cohort:
// HS006, HS007, HS024, HS026, HS028, HS035, HS039, HS040, HS043, HS048,
// HS050, HS051, HS071, HS076. Feasibility is recomputed analytically at
// the returned point (constraint residual at x*), never taken from the
// solver's self-reported constraint_violation.
//
// A combo "clears the bar" only if every cohort cell meets both the
// objective and feasibility tolerance. The shipped default follows a
// burden-of-proof rule: keep gamma_f = gamma_h = 1e-3 only if it clears
// the bar across the whole cohort; otherwise revert to 1e-5.
//
// Output: per-policy per-combo pass table plus the SELECTED line, and an
// optional provenance JSON carrying the evaluated cohort list, a
// committed minimum cell count, the per-cell results, the selected value,
// and the publication-bar correctness witness.
//
// Reference: Hock & Schittkowski 1981 (the HS test cohort);
//            Wachter & Biegler 2006 Section 2.3 (filter envelope);
//            Fletcher & Leyffer 2002 Section 5 (slack-margin gamma
//            tuning).

#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <string_view>

namespace
{

constexpr std::array<double, 4> gamma_grid{1e-3, 1e-4, 1e-5, 1e-6};

// Publication bar (Phase-context locked): objective distance to the
// analytic optimum and recomputed feasibility residual at the returned
// point must both fall within these tolerances.
constexpr double publication_obj_tol = 1e-8;
constexpr double publication_cv_tol = 1e-6;

// The enumerated filter-lineage cohort. Recorded explicitly (with a
// committed minimum cell count) so a future shrink to a smaller subset is
// detectable in the provenance record.
constexpr std::array<std::string_view, 14> cohort_names{
    "hs006", "hs007", "hs024", "hs026", "hs028", "hs035", "hs039",
    "hs040", "hs043", "hs048", "hs050", "hs051", "hs071", "hs076"};
constexpr std::size_t committed_min_cell_count = cohort_names.size();

struct cell_result
{
    std::string name;
    double f{};
    double f_star{};
    double obj_dist{};
    double cv{};
    std::uint32_t iters{};
    bool passes{};
};

struct combo_result
{
    double gamma_f{};
    double gamma_h{};
    std::vector<cell_result> cells;
    bool clears_bar{};
    std::size_t cells_passed{};
};

struct policy_sweep_result
{
    std::string policy;
    std::vector<combo_result> combos;
    bool keep_1e3{};
    double selected_gamma_f{};
    double selected_gamma_h{};
    std::size_t cohort_size{};
    std::size_t combo_1e3_passed{};
    std::size_t combo_1e5_passed{};
    std::string correctness_witness;
};

std::string fmt_num(double v)
{
    if(!std::isfinite(v))
    {
        return std::isnan(v) ? "\"nan\"" : (v < 0.0 ? "\"-inf\"" : "\"inf\"");
    }
    std::ostringstream os;
    os << std::setprecision(17) << v;
    return os.str();
}

// Tag selector to dispatch policy-specific solver-option shapes; the two
// filter policies use slightly different convergence thresholds (the
// filter_slsqp lineage omits the gradient threshold to avoid premature
// termination from the least-squares lambda zeroing out
// ||grad f - A^T lambda||, while filter_nw_sqp sets one).
struct slsqp_tag {};
struct nw_sqp_tag {};

template <template <int> class Policy> struct policy_tag;
template <> struct policy_tag<argmin::filter_slsqp_policy>  { using type = slsqp_tag;  };
template <> struct policy_tag<argmin::filter_nw_sqp_policy> { using type = nw_sqp_tag; };

// Convergence-favorable options per policy: a generous iteration budget
// and tight thresholds so each combo gets its best shot at the bar (the
// question is whether an envelope CAN reach the bar, not how quickly).
inline argmin::solver_options<> cohort_opts(slsqp_tag)
{
    argmin::solver_options<> opts;
    opts.max_iterations = 500;
    opts.set_step_threshold(1e-12);
    opts.set_objective_threshold(1e-12);
    return opts;
}
inline argmin::solver_options<> cohort_opts(nw_sqp_tag)
{
    argmin::solver_options<> opts;
    opts.max_iterations = 500;
    opts.set_gradient_threshold(1e-6);
    opts.set_step_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    return opts;
}

// Independent feasibility witness: recompute the constraint residual at
// the returned point. Equality rows come first (residual |c_eq|);
// inequality rows follow the c_ineq >= 0 convention (residual
// max(0, -c_ineq)). This does NOT consult the solver's self-reported
// constraint_violation.
template <typename Problem, typename Vec>
double feasibility_at(const Problem& prob, const Vec& x)
{
    constexpr int M = Problem::constraint_count;
    Eigen::Vector<double, M> c;
    prob.constraints(x, c);
    const int neq = prob.num_equality();
    double viol = 0.0;
    for(int i = 0; i < M; ++i)
    {
        const double ci = c[i];
        if(i < neq)
            viol = std::max(viol, std::abs(ci));
        else
            viol = std::max(viol, std::max(0.0, -ci));
    }
    return viol;
}

template <template <int> class Policy, typename Problem>
cell_result run_cell(std::string_view name, double gamma_f, double gamma_h)
{
    using tag = typename policy_tag<Policy>::type;
    constexpr int N = Problem::problem_dimension;

    Problem problem;
    auto x0 = problem.initial_point();
    auto opts = cohort_opts(tag{});

    typename Policy<N>::options_type policy_opts;
    policy_opts.gamma_f = gamma_f;
    policy_opts.gamma_h = gamma_h;

    argmin::step_budget_solver solver{Policy<N>{}, problem, x0, opts, policy_opts};
    auto result = solver.solve(opts);

    cell_result cr{};
    cr.name = std::string{name};
    cr.f = result.objective_value;
    cr.f_star = problem.optimal_value();
    cr.obj_dist = std::abs(cr.f - cr.f_star);
    cr.cv = feasibility_at(problem, result.x);
    cr.iters = result.iterations;
    cr.passes = (cr.obj_dist < publication_obj_tol) && (cr.cv < publication_cv_tol);
    return cr;
}

template <template <int> class Policy>
combo_result run_combo(double gf, double gh)
{
    combo_result combo{};
    combo.gamma_f = gf;
    combo.gamma_h = gh;
    combo.cells.reserve(cohort_names.size());

    combo.cells.push_back(run_cell<Policy, argmin::hs006<>>("hs006", gf, gh));
    combo.cells.push_back(run_cell<Policy, argmin::hs007<>>("hs007", gf, gh));
    combo.cells.push_back(run_cell<Policy, argmin::hs024<>>("hs024", gf, gh));
    combo.cells.push_back(run_cell<Policy, argmin::hs026<>>("hs026", gf, gh));
    combo.cells.push_back(run_cell<Policy, argmin::hs028<>>("hs028", gf, gh));
    combo.cells.push_back(run_cell<Policy, argmin::hs035<>>("hs035", gf, gh));
    combo.cells.push_back(run_cell<Policy, argmin::hs039<>>("hs039", gf, gh));
    combo.cells.push_back(run_cell<Policy, argmin::hs040<>>("hs040", gf, gh));
    combo.cells.push_back(run_cell<Policy, argmin::hs043<>>("hs043", gf, gh));
    combo.cells.push_back(run_cell<Policy, argmin::hs048<>>("hs048", gf, gh));
    combo.cells.push_back(run_cell<Policy, argmin::hs050<>>("hs050", gf, gh));
    combo.cells.push_back(run_cell<Policy, argmin::hs051<>>("hs051", gf, gh));
    combo.cells.push_back(run_cell<Policy, argmin::hs071<>>("hs071", gf, gh));
    combo.cells.push_back(run_cell<Policy, argmin::hs076<>>("hs076", gf, gh));

    combo.cells_passed = static_cast<std::size_t>(
        std::count_if(combo.cells.begin(), combo.cells.end(),
                      [](const cell_result& c) { return c.passes; }));
    combo.clears_bar = combo.cells_passed == combo.cells.size();
    return combo;
}

bool is_value(double a, double b)
{
    return std::abs(a - b) <= std::abs(b) * 1e-12;
}

template <template <int> class Policy>
policy_sweep_result run_policy_sweep(const char* policy_name)
{
    std::printf("\n=== %s envelope sweep (publication bar: |f-f*| < %.0e, cv < %.0e) ===\n",
                policy_name, publication_obj_tol, publication_cv_tol);
    std::printf("cohort: %zu cells\n", cohort_names.size());

    std::vector<combo_result> combos;
    combos.reserve(gamma_grid.size() * gamma_grid.size());
    for(double gf : gamma_grid)
        for(double gh : gamma_grid)
            combos.push_back(run_combo<Policy>(gf, gh));

    std::printf("%-9s %-9s %-11s %-10s  %s\n",
                "gamma_f", "gamma_h", "cells_pass", "clears", "failing_cells(obj_dist,cv)");
    for(const auto& c : combos)
    {
        std::string fails;
        for(const auto& cell : c.cells)
        {
            if(!cell.passes)
            {
                std::ostringstream os;
                os << " " << cell.name << "(" << std::setprecision(3)
                   << cell.obj_dist << "," << cell.cv << ")";
                fails += os.str();
            }
        }
        std::printf("%-9.0e %-9.0e %-11s %-10s %s\n",
                    c.gamma_f, c.gamma_h,
                    (std::to_string(c.cells_passed) + "/"
                     + std::to_string(c.cells.size())).c_str(),
                    c.clears_bar ? "yes" : "no",
                    fails.empty() ? " (none)" : fails.c_str());
    }

    // Burden-of-proof selection: keep 1e-3 only if the (1e-3, 1e-3) combo
    // clears the bar across the whole cohort; otherwise revert to 1e-5.
    const combo_result* combo_1e3 = nullptr;
    const combo_result* combo_1e5 = nullptr;
    for(const auto& c : combos)
    {
        if(is_value(c.gamma_f, 1e-3) && is_value(c.gamma_h, 1e-3)) combo_1e3 = &c;
        if(is_value(c.gamma_f, 1e-5) && is_value(c.gamma_h, 1e-5)) combo_1e5 = &c;
    }

    policy_sweep_result result;
    result.policy = policy_name;
    result.combos = std::move(combos);
    result.cohort_size = cohort_names.size();
    result.combo_1e3_passed = combo_1e3 ? combo_1e3->cells_passed : 0;
    result.combo_1e5_passed = combo_1e5 ? combo_1e5->cells_passed : 0;
    result.keep_1e3 = combo_1e3 && combo_1e3->clears_bar;
    result.selected_gamma_f = result.keep_1e3 ? 1e-3 : 1e-5;
    result.selected_gamma_h = result.keep_1e3 ? 1e-3 : 1e-5;

    std::ostringstream witness;
    if(result.keep_1e3)
    {
        witness << "gamma_f = gamma_h = 1e-3 cleared the publication bar "
                   "(|f-f*| < 1e-8 and recomputed feasibility < 1e-6) across all "
                << result.cohort_size << " filter-lineage cohort cells";
    }
    else
    {
        witness << "gamma_f = gamma_h = 1e-3 cleared only "
                << result.combo_1e3_passed << "/" << result.cohort_size
                << " cohort cells at the publication bar; reverted to 1e-5 "
                   "(which cleared " << result.combo_1e5_passed << "/"
                << result.cohort_size << ") per the burden-of-proof rule";
    }
    result.correctness_witness = witness.str();

    std::printf(
        "\nSELECTED %s default: gamma_f=%.0e, gamma_h=%.0e (%s)\n"
        "  witness: %s\n",
        policy_name, result.selected_gamma_f, result.selected_gamma_h,
        result.keep_1e3 ? "kept 1e-3" : "reverted to 1e-5",
        result.correctness_witness.c_str());

    return result;
}

void write_json(const std::vector<policy_sweep_result>& results,
                std::string_view output_path,
                std::string_view provenance_id)
{
    std::ofstream out{std::string{output_path}};
    if(!out)
    {
        std::cerr << "filter_envelope_sweep: cannot open " << output_path
                  << " for writing\n";
        std::exit(2);
    }

    out << "{\n";
    out << "  \"provenance_id\": \"" << provenance_id << "\",\n";
    out << "  \"publication_bar\": {\"obj_tol\": " << fmt_num(publication_obj_tol)
        << ", \"cv_tol\": " << fmt_num(publication_cv_tol) << "},\n";
    out << "  \"cohort\": [";
    for(std::size_t i = 0; i < cohort_names.size(); ++i)
    {
        if(i) out << ", ";
        out << "\"" << cohort_names[i] << "\"";
    }
    out << "],\n";
    out << "  \"cohort_size\": " << cohort_names.size() << ",\n";
    out << "  \"committed_min_cell_count\": " << committed_min_cell_count << ",\n";
    out << "  \"grid\": {\n";
    out << "    \"gamma_f\": [";
    for(std::size_t i = 0; i < gamma_grid.size(); ++i)
    {
        if(i) out << ", ";
        out << fmt_num(gamma_grid[i]);
    }
    out << "],\n";
    out << "    \"gamma_h\": [";
    for(std::size_t i = 0; i < gamma_grid.size(); ++i)
    {
        if(i) out << ", ";
        out << fmt_num(gamma_grid[i]);
    }
    out << "]\n";
    out << "  },\n";

    out << "  \"policies\": [\n";
    for(std::size_t p = 0; p < results.size(); ++p)
    {
        const auto& result = results[p];
        out << "    {\n";
        out << "      \"policy\": \"" << result.policy << "\",\n";
        out << "      \"combos\": [\n";
        for(std::size_t i = 0; i < result.combos.size(); ++i)
        {
            const auto& combo = result.combos[i];
            out << "        {\"gamma_f\": " << fmt_num(combo.gamma_f)
                << ", \"gamma_h\": " << fmt_num(combo.gamma_h)
                << ", \"clears_bar\": " << (combo.clears_bar ? "true" : "false")
                << ", \"cells_passed\": " << combo.cells_passed
                << ", \"cells\": [";
            for(std::size_t j = 0; j < combo.cells.size(); ++j)
            {
                const auto& cell = combo.cells[j];
                out << "{\"name\": \"" << cell.name << "\""
                    << ", \"f\": " << fmt_num(cell.f)
                    << ", \"f_star\": " << fmt_num(cell.f_star)
                    << ", \"obj_dist\": " << fmt_num(cell.obj_dist)
                    << ", \"cv\": " << fmt_num(cell.cv)
                    << ", \"iters\": " << cell.iters
                    << ", \"passes\": " << (cell.passes ? "true" : "false")
                    << "}";
                if(j + 1 < combo.cells.size()) out << ", ";
            }
            out << "]}";
            if(i + 1 < result.combos.size()) out << ",";
            out << "\n";
        }
        out << "      ],\n";
        out << "      \"selected_default\": {\n";
        out << "        \"item\": \"filter_envelope\",\n";
        out << "        \"grid_id\": \"gamma_f_x_gamma_h\",\n";
        out << "        \"policy\": \"" << result.policy << "\",\n";
        out << "        \"selected_value\": {\"gamma_f\": "
            << fmt_num(result.selected_gamma_f) << ", \"gamma_h\": "
            << fmt_num(result.selected_gamma_h) << "},\n";
        out << "        \"kept_1e3\": " << (result.keep_1e3 ? "true" : "false") << ",\n";
        out << "        \"combo_1e3_cells_passed\": " << result.combo_1e3_passed << ",\n";
        out << "        \"combo_1e5_cells_passed\": " << result.combo_1e5_passed << ",\n";
        out << "        \"cohort_size\": " << result.cohort_size << ",\n";
        out << "        \"committed_min_cell_count\": " << committed_min_cell_count << ",\n";
        out << "        \"correctness_witness\": \""
            << result.correctness_witness << "\",\n";
        out << "        \"provenance_id\": \"" << provenance_id << "\"\n";
        out << "      }\n";
        out << "    }";
        if(p + 1 < results.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
}

}

int main(int argc, char** argv)
{
    std::string output_path;
    std::string provenance_id = "filter-envelope-sweep-local";

    for(int i = 1; i < argc; ++i)
    {
        std::string_view a{argv[i]};
        if(a == "--quick")
        {
            // The envelope grid is already smoke-sized.
        }
        else if(a == "--output" && i + 1 < argc)
        {
            output_path = argv[++i];
        }
        else if(a == "--provenance-id" && i + 1 < argc)
        {
            provenance_id = argv[++i];
        }
        else if(a == "--help" || a == "-h")
        {
            std::cout << "filter_envelope_sweep: filter_slsqp/filter_nw_sqp "
                         "gamma_f x gamma_h sweep at the publication bar.\n"
                         "  --quick             run the smoke-sized grid\n"
                         "  --output PATH       JSON output path\n"
                         "  --provenance-id S   evidence record identifier\n";
            return 0;
        }
        else
        {
            std::cerr << "filter_envelope_sweep: unknown arg " << a << "\n";
            return 2;
        }
    }

    std::vector<policy_sweep_result> results;
    results.push_back(run_policy_sweep<argmin::filter_slsqp_policy>("filter_slsqp"));
    results.push_back(run_policy_sweep<argmin::filter_nw_sqp_policy>("filter_nw_sqp"));
    if(!output_path.empty())
    {
        write_json(results, output_path, provenance_id);
        std::cout << "\nJSON written to: " << output_path << "\n";
    }
    return 0;
}
