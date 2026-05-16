// Sweep harness for the joint (penalty_factor, soc_max_iterations) tuning
// of tr_sqp_policy.
//
// Drives the trust-region SQP policy across the cross product of:
//   - penalty_factor:        {0.0, 0.01, 0.05, 0.1, 0.3}
//   - soc_max_iterations:    {0, 1, 2, 3}
//   - HS-suite cell:         HS024, HS026, HS028, HS035, HS039, HS040,
//                            HS043, HS050, HS071, HS076
//   - mode:                  sqp_mode::accurate, sqp_mode::fast
//
// Both knobs are runtime — the harness reuses one Release build of the
// library and toggles `policy.options.penalty_factor` /
// `policy.options.soc_max_iterations` per cell. For every
// (pf, soc_iter, cell, mode) tuple the harness records iterations, final
// objective, final constraint violation (inf-norm), terminal solver
// status, wall microseconds, `f_err = |f - f*| / max(|f*|, 1.0)`, and a
// `within_strict_bar` flag derived from the tr_sqp_test.cpp acceptance
// shapes.
//
// The cell axis partitions into two roles:
//
//   - Reference cells (non-regression gate): HS026, HS028, HS071, HS076.
//     These pass under (penalty_factor = 0, soc_max_iterations = 0) at
//     the current HEAD and must continue to pass under any swept
//     configuration for that configuration to remain a default
//     candidate.
//   - Closure-target cells (improvement axis): HS024, HS035, HS039,
//     HS040, HS043, HS050. These fail at the per-mode strict bar under
//     the (0, 0) baseline. The sweep asks whether any swept
//     configuration closes at least one of them strictly without
//     regressing the reference set.
//
// Joint selection rule (applied in main()):
//
//   Step 1 — Reference-set gate: a (pf, soc_iter) configuration is a
//            Step-1 survivor iff every HS026 / HS028 / HS071 / HS076
//            cell (both modes) reports `within_strict_bar = true`.
//   Step 2 — Closure-count metric: among Step-1 survivors, count how
//            many closure-target cells × both modes report
//            `within_strict_bar = true` at that configuration. Returns
//            a value in [0, 12].
//   Step 3 — Improvement-or-hold:
//             (a) If the configuration with the highest closure_count
//                 has strictly greater closure_count than (0, 0), select
//                 that configuration as the new default. Branch label
//                 "close_>=1_cell". Tie-break on largest penalty_factor,
//                 then smallest soc_max_iter, then lowest summed wall
//                 across closure-target cells.
//             (b) Else if Step-1 has survivors but no closure-count
//                 improvement over (0, 0), keep (0, 0) as the default
//                 and tag every non-passing closure-target cell as
//                 known-failure. Branch label "no_improvement_over_default".
//             (c) Else if Step-1 is empty (every joint regresses at
//                 least one reference cell), keep (0, 0). Branch label
//                 "reference_regression_unavoidable".
//
// Output: a single JSON document with a per-cell `by_config` block plus
// a `selection` block. Default output path is `soc_budget_sweep.json`
// in the current working directory; override with `--output PATH`.
// Stdout prints a per-row table and the selection trace.
//
// Non-ctest, one-shot. The harness is read-only on argmin; consumers
// do not pick it up via FetchContent.
//
// Reference: Hock, W. and Schittkowski, K. (1981) "Test Examples for
//            Nonlinear Programming Codes", Lecture Notes in Economics
//            and Mathematical Systems vol. 187, Springer.
//            Lalee, Nocedal, Plantenga 1998 SIAM J. Optim.
//            8(3):682-706 Section 3.3 (adaptive penalty heuristic) +
//            Section 3.1 (SOC retry shape). Nocedal and Wright 2e
//            Section 18.5 Algorithm 18.4 (Byrd-Omojokun composite step).

#include "argmin/solver/basic_solver.h"
#include "argmin/solver/sqp_mode.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace
{

using argmin::sqp_mode;
using argmin::tr_sqp_policy;

constexpr double      kPenaltyFactors[] = {0.0, 0.01, 0.05, 0.1, 0.3};
constexpr std::size_t kSocMaxIters[]    = {0, 1, 2, 3};

std::string_view status_name(argmin::solver_status s)
{
    using argmin::solver_status;
    switch(s)
    {
        case solver_status::running:                    return "running";
        case solver_status::converged:                  return "converged";
        case solver_status::max_iterations:             return "max_iterations";
        case solver_status::budget_exhausted:           return "budget_exhausted";
        case solver_status::stalled:                    return "stalled";
        case solver_status::diverged:                   return "diverged";
        case solver_status::xtol_reached:               return "xtol_reached";
        case solver_status::ftol_reached:               return "ftol_reached";
        case solver_status::maxeval_reached:            return "maxeval_reached";
        case solver_status::roundoff_limited:           return "roundoff_limited";
        case solver_status::trust_region_step_rejected: return "tr_step_rejected";
        case solver_status::objective_stalled:          return "objective_stalled";
        case solver_status::time_limit_reached:         return "time_limit_reached";
        case solver_status::aborted:                    return "aborted";
    }
    return "unknown";
}

std::string fmt_num(double v)
{
    if(!std::isfinite(v))
    {
        return std::isnan(v) ? "\"nan\"" : (v < 0 ? "\"-inf\"" : "\"inf\"");
    }
    std::ostringstream os;
    os << std::setprecision(17) << v;
    return os.str();
}

// Per-(pf, soc_iter, cell, mode) outcome record.
struct cell_record
{
    double      penalty_factor;
    std::size_t soc_max_iter;
    double      objective_value;
    double      constraint_violation;
    double      f_err;            // |f - f*| / max(|f*|, 1.0)
    std::uint32_t iterations;
    double      wall_us;
    std::string status;
    bool        within_strict_bar;
};

// Strict-bar predicate matching the tr_sqp_test.cpp acceptance shapes.
// Per-mode bars are 1% f_err / 1e-4 cv for accurate mode and 5% f_err /
// 1e-2 cv for fast mode. Cells with f* = 0 use absolute-objective bars
// because the relative f_err formula is ill-posed at zero.
//
// Per-cell bar shapes (matching tr_sqp_test.cpp where the cell exists
// and the Plan-45 trace-classification table for the new closure-target
// cells):
//   - HS024 (f* = -1):       relative-against-1, both modes.
//   - HS026 (f* = 0):        absolute, both modes.
//   - HS028 (f* = 0):        absolute (1e-6 accurate, 1e-2 fast) — long
//                            shallow ridge with relaxed fast-mode bar.
//   - HS035 (f* = 1/9):      relative-against-|1/9|, both modes.
//   - HS039 (f* = -1):       relative-against-1, both modes.
//   - HS040 (f* = -0.25):    relative-against-0.25, both modes.
//   - HS043 (f* = -44):      relative, both modes.
//   - HS050 (f* = 0):        absolute (1e-6 accurate, 1e-2 fast),
//                            mirroring HS028's f* = 0 convention.
//   - HS071 (f* ~ 17.01):    relative, both modes.
//   - HS076 (f* ~ -4.68):    relative, both modes.
bool within_strict_bar(std::string_view problem_name,
                       sqp_mode          mode,
                       double            f,
                       double            f_star,
                       double            cv)
{
    if(problem_name == "hs026")
    {
        const double bar    = (mode == sqp_mode::fast) ? 0.05  : 0.01;
        const double cv_bar = (mode == sqp_mode::fast) ? 1e-2  : 1e-4;
        return std::abs(f - 0.0) <= bar && cv < cv_bar;
    }
    if(problem_name == "hs028")
    {
        if(mode == sqp_mode::fast)
            return std::abs(f - 0.0) <= 1e-2 && cv < 1e-2;
        return std::abs(f - 0.0) <= 1e-6 && cv < 1e-4;
    }
    if(problem_name == "hs050")
    {
        // HS050 f* = 0; reuse the HS028 absolute-bar convention because
        // the relative f_err is ill-posed at zero.
        if(mode == sqp_mode::fast)
            return std::abs(f - 0.0) <= 1e-2 && cv < 1e-2;
        return std::abs(f - 0.0) <= 1e-6 && cv < 1e-4;
    }
    if(problem_name == "hs024" || problem_name == "hs035"
       || problem_name == "hs039" || problem_name == "hs040"
       || problem_name == "hs043" || problem_name == "hs071"
       || problem_name == "hs076")
    {
        // Relative-against-|f*| bar. HS035 uses |f* = 1/9|, HS040 uses
        // |f* = 0.25|, and the rest use |f*| at its native magnitude.
        const double f_err = std::abs(f - f_star) / std::abs(f_star);
        if(mode == sqp_mode::fast)
            return f_err < 0.05 && cv < 1e-2;
        return f_err < 0.01 && cv < 1e-4;
    }
    return false;
}

// Drive one (pf, soc_iter, problem, mode) cell.
template <typename Policy, typename Problem>
cell_record run_cell(double pf, std::size_t soc_iter, const Problem& problem,
                     std::uint32_t max_iterations)
{
    using policy_t = Policy;

    auto x0 = problem.initial_point();
    argmin::solver_options opts;
    opts.max_iterations = max_iterations;
    opts.set_gradient_threshold(policy_t::default_gradient_tolerance);
    opts.set_step_threshold(policy_t::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_t::default_feasibility_tolerance;

    policy_t policy;
    policy.options.penalty_factor      = pf;
    policy.options.soc_max_iterations  = soc_iter;

    argmin::basic_solver solver{policy, problem, x0, opts};

    const auto t0 = std::chrono::steady_clock::now();
    auto result = solver.solve(opts);
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();

    const double f_star = static_cast<double>(problem.optimal_value());
    const double f_err = std::abs(result.objective_value - f_star)
                       / std::max(std::abs(f_star), 1.0);

    cell_record rec;
    rec.penalty_factor       = pf;
    rec.soc_max_iter         = soc_iter;
    rec.objective_value      = result.objective_value;
    rec.constraint_violation = solver.constraint_violation();
    rec.f_err                = f_err;
    rec.iterations           = static_cast<std::uint32_t>(result.iterations);
    rec.wall_us              = wall_us;
    rec.status               = std::string(status_name(result.status));
    return rec;
}

// Cell descriptor for the per-problem dispatch.
struct cell_spec
{
    std::string name;
    std::string mode;        // "accurate" or "fast"
    double      f_star;
};

// One full row of records (joint pf x soc_iter grid) for one cell.
struct cell_block
{
    cell_spec               spec;
    std::vector<cell_record> records;
};

// Sweep dispatch — instantiates Policy per (problem dimension, mode)
// and runs the joint loop on it. Records are appended to the block in
// row-major order over (pf, soc_iter).
template <typename PolicyAccurate, typename PolicyFast, typename Problem>
void run_cell_block(const std::string& problem_name,
                    double             f_star,
                    std::uint32_t      max_iterations,
                    const Problem&     problem,
                    std::vector<cell_block>& out_blocks)
{
    cell_block acc_block{ cell_spec{problem_name, "accurate", f_star}, {} };
    cell_block fst_block{ cell_spec{problem_name, "fast",     f_star}, {} };

    for(double pf : kPenaltyFactors)
    {
        for(std::size_t soc : kSocMaxIters)
        {
            auto a = run_cell<PolicyAccurate>(pf, soc, problem, max_iterations);
            a.within_strict_bar = within_strict_bar(problem_name, sqp_mode::accurate,
                                                    a.objective_value, f_star,
                                                    a.constraint_violation);
            acc_block.records.push_back(std::move(a));

            auto f = run_cell<PolicyFast>(pf, soc, problem, max_iterations);
            f.within_strict_bar = within_strict_bar(problem_name, sqp_mode::fast,
                                                    f.objective_value, f_star,
                                                    f.constraint_violation);
            fst_block.records.push_back(std::move(f));
        }
    }

    out_blocks.push_back(std::move(acc_block));
    out_blocks.push_back(std::move(fst_block));
}

// Cell-name partitioning for the selection rule. The reference set is
// the existing-passing cells under (pf=0, soc=0); the closure-target
// set is the failing cells routed to this sweep.
constexpr std::string_view kReferenceCells[] = {
    "hs026", "hs028", "hs071", "hs076"};
constexpr std::string_view kClosureTargets[] = {
    "hs024", "hs035", "hs039", "hs040", "hs043", "hs050"};
constexpr std::string_view kModes[] = {"accurate", "fast"};

struct selection
{
    std::optional<double>      penalty_factor;
    std::optional<std::size_t> soc_max_iter;
    std::string                branch;        // see selection rule
    std::size_t                closure_count{0};
    std::string                rationale;
};

void write_json(const std::vector<cell_block>& blocks,
                const selection&               sel,
                const std::vector<std::tuple<double, std::size_t, std::size_t, bool>>& closure_count_grid,
                std::string_view               output_path,
                std::string_view               build_type,
                std::string_view               head_sha)
{
    std::ofstream out{std::string{output_path}};
    if(!out)
    {
        std::cerr << "soc_budget_sweep: cannot open " << output_path
                  << " for writing\n";
        std::exit(2);
    }

    out << "{\n";
    out << "  \"head_sha\": \"" << head_sha << "\",\n";
    out << "  \"build_type\": \"" << build_type << "\",\n";
    out << "  \"grid\": {\n";
    out << "    \"penalty_factor\": [";
    for(std::size_t i = 0; i < std::size(kPenaltyFactors); ++i)
    {
        if(i) out << ", ";
        out << fmt_num(kPenaltyFactors[i]);
    }
    out << "],\n";
    out << "    \"soc_max_iter\": [";
    for(std::size_t i = 0; i < std::size(kSocMaxIters); ++i)
    {
        if(i) out << ", ";
        out << kSocMaxIters[i];
    }
    out << "]\n";
    out << "  },\n";

    // Cell-role partitioning for downstream consumers.
    out << "  \"reference_cells\": [";
    for(std::size_t i = 0; i < std::size(kReferenceCells); ++i)
    {
        if(i) out << ", ";
        out << "\"" << kReferenceCells[i] << "\"";
    }
    out << "],\n";
    out << "  \"closure_targets\": [";
    for(std::size_t i = 0; i < std::size(kClosureTargets); ++i)
    {
        if(i) out << ", ";
        out << "\"" << kClosureTargets[i] << "\"";
    }
    out << "],\n";

    out << "  \"cells\": [\n";
    for(std::size_t b = 0; b < blocks.size(); ++b)
    {
        const auto& block = blocks[b];
        out << "    {\n";
        out << "      \"name\": \"" << block.spec.name << "\",\n";
        out << "      \"mode\": \"" << block.spec.mode << "\",\n";
        out << "      \"f_star\": " << fmt_num(block.spec.f_star) << ",\n";
        out << "      \"by_config\": {\n";
        for(std::size_t i = 0; i < block.records.size(); ++i)
        {
            const auto& r = block.records[i];
            std::ostringstream key;
            key << "pf=" << std::fixed << std::setprecision(2)
                << r.penalty_factor
                << ",soc=" << r.soc_max_iter;
            out << "        \"" << key.str() << "\": {"
                << "\"f\": "      << fmt_num(r.objective_value)
                << ", \"cv\": "   << fmt_num(r.constraint_violation)
                << ", \"f_err\": "<< fmt_num(r.f_err)
                << ", \"iters\": "<< r.iterations
                << ", \"wall_us\": "<< fmt_num(r.wall_us)
                << ", \"status\": \"" << r.status << "\""
                << ", \"within_strict_bar\": " << (r.within_strict_bar ? "true" : "false")
                << "}";
            if(i + 1 < block.records.size()) out << ",";
            out << "\n";
        }
        out << "      }\n";
        out << "    }";
        if(b + 1 < blocks.size()) out << ",";
        out << "\n";
    }
    out << "  ],\n";

    // Per-(pf, soc) closure-count summary; downstream MD consumes this.
    out << "  \"closure_count_by_config\": [\n";
    for(std::size_t i = 0; i < closure_count_grid.size(); ++i)
    {
        const auto& [pf, soc, cnt, passes_ref] = closure_count_grid[i];
        out << "    {\"pf\": " << fmt_num(pf)
            << ", \"soc\": " << soc
            << ", \"closure_count\": " << cnt
            << ", \"passes_reference_gate\": " << (passes_ref ? "true" : "false")
            << "}";
        if(i + 1 < closure_count_grid.size()) out << ",";
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"selection\": {\n";
    out << "    \"reference_cells\": [";
    for(std::size_t i = 0; i < std::size(kReferenceCells); ++i)
    {
        if(i) out << ", ";
        out << "\"" << kReferenceCells[i] << "\"";
    }
    out << "],\n";
    out << "    \"closure_targets\": [";
    for(std::size_t i = 0; i < std::size(kClosureTargets); ++i)
    {
        if(i) out << ", ";
        out << "\"" << kClosureTargets[i] << "\"";
    }
    out << "],\n";
    out << "    \"branch\": \"" << sel.branch << "\",\n";
    out << "    \"closure_count\": " << sel.closure_count << ",\n";
    out << "    \"penalty_factor\": ";
    if(sel.penalty_factor) out << fmt_num(*sel.penalty_factor);
    else                   out << "null";
    out << ",\n";
    out << "    \"soc_max_iter\": ";
    if(sel.soc_max_iter) out << *sel.soc_max_iter;
    else                 out << "null";
    out << ",\n";
    out << "    \"rationale\": \"" << sel.rationale << "\"\n";
    out << "  }\n";
    out << "}\n";
}

// Look up a record in the cell block by (pf, soc_iter).
const cell_record* find_record(const cell_block& block,
                               double pf, std::size_t soc_iter)
{
    for(const auto& r : block.records)
    {
        if(r.penalty_factor == pf && r.soc_max_iter == soc_iter)
            return &r;
    }
    return nullptr;
}

const cell_block* find_block(const std::vector<cell_block>& blocks,
                             std::string_view name,
                             std::string_view mode)
{
    for(const auto& b : blocks)
    {
        if(b.spec.name == name && b.spec.mode == mode)
            return &b;
    }
    return nullptr;
}

}  // namespace

int main(int argc, char** argv)
{
    std::string output_path = "soc_budget_sweep.json";
    std::string build_type = "Release";
    std::string head_sha   = "unknown";

    for(int i = 1; i < argc; ++i)
    {
        std::string_view a{argv[i]};
        if(a == "--output" && i + 1 < argc)
        {
            output_path = argv[++i];
        }
        else if(a == "--build-type" && i + 1 < argc)
        {
            build_type = argv[++i];
        }
        else if(a == "--head-sha" && i + 1 < argc)
        {
            head_sha = argv[++i];
        }
        else if(a == "--help" || a == "-h")
        {
            std::cout <<
                "soc_budget_sweep: drive tr_sqp_policy across the joint\n"
                "  (penalty_factor in {0.0, 0.01, 0.05, 0.1, 0.3},\n"
                "   soc_max_iterations in {0, 1, 2, 3}) grid on the\n"
                "  HS-suite cells:\n"
                "    - reference (non-regression gate): HS026, HS028, HS071, HS076\n"
                "    - closure targets (improvement axis): HS024, HS035, HS039,\n"
                "      HS040, HS043, HS050\n"
                "  in both accurate and fast sqp_mode.\n"
                "\n"
                "Selection rule: a configuration that strictly improves the\n"
                "closure-count over (pf=0, soc=0) while preserving the\n"
                "reference set becomes the new default. Otherwise (0, 0)\n"
                "stays as the opt-in-zero default and non-passing closure\n"
                "targets are tagged known-failure.\n"
                "\n"
                "  --output PATH       JSON output path\n"
                "  --build-type S      label only\n"
                "  --head-sha S        label only\n";
            return 0;
        }
        else
        {
            std::cerr << "soc_budget_sweep: unknown arg " << a << "\n";
            return 2;
        }
    }

    std::vector<cell_block> blocks;

    using hs024_t = argmin::hs024<>;
    using hs026_t = argmin::hs026<>;
    using hs028_t = argmin::hs028<>;
    using hs035_t = argmin::hs035<>;
    using hs039_t = argmin::hs039<>;
    using hs040_t = argmin::hs040<>;
    using hs043_t = argmin::hs043<>;
    using hs050_t = argmin::hs050<>;
    using hs071_t = argmin::hs071<>;
    using hs076_t = argmin::hs076<>;

    hs024_t p024; hs026_t p026; hs028_t p028; hs035_t p035; hs039_t p039;
    hs040_t p040; hs043_t p043; hs050_t p050; hs071_t p071; hs076_t p076;

    // Per-problem max_iterations matches the existing tr_sqp_test.cpp
    // convention. HS028 gets 500 (long shallow ridge on the joint
    // slack-augmented primal); the rest get 200.
    run_cell_block<
        tr_sqp_policy<hs024_t::problem_dimension, sqp_mode::accurate>,
        tr_sqp_policy<hs024_t::problem_dimension, sqp_mode::fast>>(
        "hs024", static_cast<double>(p024.optimal_value()), 200, p024, blocks);

    run_cell_block<
        tr_sqp_policy<hs026_t::problem_dimension, sqp_mode::accurate>,
        tr_sqp_policy<hs026_t::problem_dimension, sqp_mode::fast>>(
        "hs026", static_cast<double>(p026.optimal_value()), 200, p026, blocks);

    run_cell_block<
        tr_sqp_policy<hs028_t::problem_dimension, sqp_mode::accurate>,
        tr_sqp_policy<hs028_t::problem_dimension, sqp_mode::fast>>(
        "hs028", static_cast<double>(p028.optimal_value()), 500, p028, blocks);

    run_cell_block<
        tr_sqp_policy<hs035_t::problem_dimension, sqp_mode::accurate>,
        tr_sqp_policy<hs035_t::problem_dimension, sqp_mode::fast>>(
        "hs035", static_cast<double>(p035.optimal_value()), 200, p035, blocks);

    run_cell_block<
        tr_sqp_policy<hs039_t::problem_dimension, sqp_mode::accurate>,
        tr_sqp_policy<hs039_t::problem_dimension, sqp_mode::fast>>(
        "hs039", static_cast<double>(p039.optimal_value()), 200, p039, blocks);

    run_cell_block<
        tr_sqp_policy<hs040_t::problem_dimension, sqp_mode::accurate>,
        tr_sqp_policy<hs040_t::problem_dimension, sqp_mode::fast>>(
        "hs040", static_cast<double>(p040.optimal_value()), 200, p040, blocks);

    run_cell_block<
        tr_sqp_policy<hs043_t::problem_dimension, sqp_mode::accurate>,
        tr_sqp_policy<hs043_t::problem_dimension, sqp_mode::fast>>(
        "hs043", static_cast<double>(p043.optimal_value()), 200, p043, blocks);

    run_cell_block<
        tr_sqp_policy<hs050_t::problem_dimension, sqp_mode::accurate>,
        tr_sqp_policy<hs050_t::problem_dimension, sqp_mode::fast>>(
        "hs050", static_cast<double>(p050.optimal_value()), 200, p050, blocks);

    run_cell_block<
        tr_sqp_policy<hs071_t::problem_dimension, sqp_mode::accurate>,
        tr_sqp_policy<hs071_t::problem_dimension, sqp_mode::fast>>(
        "hs071", static_cast<double>(p071.optimal_value()), 200, p071, blocks);

    run_cell_block<
        tr_sqp_policy<hs076_t::problem_dimension, sqp_mode::accurate>,
        tr_sqp_policy<hs076_t::problem_dimension, sqp_mode::fast>>(
        "hs076", static_cast<double>(p076.optimal_value()), 200, p076, blocks);

    // Print the sweep table.
    std::cout << "cell           mode      pf     soc   iters     f             cv            f_err         bar status\n";
    std::cout << "-------------  --------  -----  ---   -----     ------------  ------------  ------------  --- ------------------\n";
    for(const auto& b : blocks)
    {
        for(const auto& r : b.records)
        {
            std::cout << std::left << std::setw(13) << b.spec.name << "  "
                      << std::setw(8) << b.spec.mode << "  "
                      << std::right << std::setw(5) << std::fixed
                      << std::setprecision(2) << r.penalty_factor << "  "
                      << std::setw(3) << r.soc_max_iter << "   "
                      << std::setw(5) << r.iterations << "    "
                      << std::scientific << std::setprecision(4)
                      << std::setw(12) << r.objective_value << "  "
                      << std::setw(12) << r.constraint_violation << "  "
                      << std::setw(12) << r.f_err << "  "
                      << (r.within_strict_bar ? " Y " : " N ") << " "
                      << std::left << r.status << "\n";
        }
    }

    // Selection rule predicates and metric.
    auto passes_reference_gate = [&](double pf, std::size_t soc) -> bool {
        for(auto cell : kReferenceCells)
        {
            for(auto mode : kModes)
            {
                const cell_block* blk = find_block(blocks, cell, mode);
                if(!blk) return false;
                const cell_record* rec = find_record(*blk, pf, soc);
                if(!rec) return false;
                if(!rec->within_strict_bar) return false;
            }
        }
        return true;
    };

    auto closure_count = [&](double pf, std::size_t soc) -> std::size_t {
        std::size_t cnt = 0;
        for(auto cell : kClosureTargets)
        {
            for(auto mode : kModes)
            {
                const cell_block* blk = find_block(blocks, cell, mode);
                if(!blk) continue;
                const cell_record* rec = find_record(*blk, pf, soc);
                if(!rec) continue;
                if(rec->within_strict_bar) ++cnt;
            }
        }
        return cnt;
    };

    auto closure_target_wall_sum = [&](double pf, std::size_t soc) -> double {
        double sum = 0.0;
        for(auto cell : kClosureTargets)
        {
            for(auto mode : kModes)
            {
                const cell_block* blk = find_block(blocks, cell, mode);
                if(!blk) continue;
                const cell_record* rec = find_record(*blk, pf, soc);
                if(!rec) continue;
                sum += rec->wall_us;
            }
        }
        return sum;
    };

    // Print the reference gate + closure-count table.
    std::cout << "\nReference gate (hs026/hs028/hs071/hs076 both modes within strict bar)\n";
    std::cout << "    + closure_count over hs024/hs035/hs039/hs040/hs043/hs050 both modes:\n";
    std::vector<std::tuple<double, std::size_t, std::size_t, bool>> closure_count_grid;
    for(double pf : kPenaltyFactors)
    {
        for(std::size_t soc : kSocMaxIters)
        {
            const bool        ref_ok = passes_reference_gate(pf, soc);
            const std::size_t cnt    = closure_count(pf, soc);
            closure_count_grid.emplace_back(pf, soc, cnt, ref_ok);
            std::cout << "  pf=" << std::fixed << std::setprecision(2) << pf
                      << " soc=" << soc << " : reference="
                      << (ref_ok ? "PASS   " : "REGRESS")
                      << " ; closure_count=" << cnt << " / 12\n";
        }
    }

    // Step-1 survivors of the reference gate.
    struct candidate
    {
        double      pf;
        std::size_t soc;
        std::size_t closure_count;
    };
    std::vector<candidate> survivors;
    for(double pf : kPenaltyFactors)
    {
        for(std::size_t soc : kSocMaxIters)
        {
            if(passes_reference_gate(pf, soc))
            {
                survivors.push_back({pf, soc, closure_count(pf, soc)});
            }
        }
    }

    const std::size_t baseline_closure = closure_count(0.0, 0);
    const bool baseline_passes_reference = passes_reference_gate(0.0, 0);

    selection sel;
    if(survivors.empty())
    {
        // Step-1 empty: every joint regresses at least one reference
        // cell. Hold (0, 0) and tag every non-passing closure target.
        sel.branch         = "reference_regression_unavoidable";
        sel.closure_count  = baseline_passes_reference ? baseline_closure : 0;
        sel.rationale =
            "Every swept configuration regresses at least one reference "
            "cell in {hs026, hs028, hs071, hs076} on at least one mode. "
            "The opt-in-zero default (penalty_factor=0, soc_max_iterations=0) "
            "stays; non-passing closure-target cells are tagged "
            "known-failure with their trace-derived mechanism family.";
        std::cout << "\nBranch (reference_regression_unavoidable): no configuration "
                     "holds the reference gate. Keep (0, 0).\n";
    }
    else
    {
        // Step 2: find the survivor with the highest closure_count.
        candidate best = survivors[0];
        for(const auto& c : survivors)
        {
            const bool better_count = c.closure_count > best.closure_count;
            const bool same_count   = c.closure_count == best.closure_count;
            const bool better_pf    = c.pf > best.pf;
            const bool same_pf      = c.pf == best.pf;
            const bool better_soc   = c.soc < best.soc;
            const bool same_soc     = c.soc == best.soc;
            if(better_count
               || (same_count && better_pf)
               || (same_count && same_pf && better_soc)
               || (same_count && same_pf && same_soc
                   && closure_target_wall_sum(c.pf, c.soc)
                      < closure_target_wall_sum(best.pf, best.soc)))
            {
                best = c;
            }
        }

        if(best.closure_count > baseline_closure)
        {
            sel.penalty_factor = best.pf;
            sel.soc_max_iter   = best.soc;
            sel.branch         = "close_>=1_cell";
            sel.closure_count  = best.closure_count;
            std::ostringstream os;
            os << "Joint (penalty_factor, soc_max_iterations) = ("
               << std::fixed << std::setprecision(2) << best.pf << ", "
               << best.soc << ") closes " << best.closure_count
               << " closure-target cell-mode tuples (vs " << baseline_closure
               << " at the (0, 0) baseline) while preserving the reference "
                  "set on hs026/hs028/hs071/hs076 across modes. Selected "
                  "as the highest-closure-count survivor, tie-broken by "
                  "largest penalty_factor then smallest soc_max_iterations.";
            sel.rationale = os.str();
            std::cout << "\nBranch (close_>=1_cell): pf=" << std::fixed
                      << std::setprecision(2) << best.pf
                      << "  soc=" << best.soc
                      << "  closure_count=" << best.closure_count
                      << "  baseline_closure=" << baseline_closure << "\n";
        }
        else
        {
            sel.branch        = "no_improvement_over_default";
            sel.closure_count = baseline_closure;
            std::ostringstream os;
            os << "No configuration in the swept grid strictly improves "
                  "the closure_count over the (0, 0) baseline (best survivor "
                  "closure_count = " << best.closure_count
               << ", baseline closure_count = " << baseline_closure
               << "). The opt-in-zero default (penalty_factor=0, "
                  "soc_max_iterations=0) stays; non-passing closure-target "
                  "cells are tagged known-failure with their trace-derived "
                  "mechanism family.";
            sel.rationale = os.str();
            std::cout << "\nBranch (no_improvement_over_default): best survivor "
                         "pf=" << std::fixed << std::setprecision(2) << best.pf
                      << " soc=" << best.soc
                      << " closure_count=" << best.closure_count
                      << " <= baseline " << baseline_closure
                      << ". Keep (0, 0).\n";
        }
    }

    write_json(blocks, sel, closure_count_grid, output_path, build_type, head_sha);
    std::cout << "\nJSON written to: " << output_path << "\n";
    return 0;
}
