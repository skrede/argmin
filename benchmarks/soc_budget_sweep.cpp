// Sweep harness for the joint (penalty_factor, soc_max_iterations) tuning
// of tr_sqp_policy.
//
// Drives the trust-region SQP policy across the cross product of:
//   - penalty_factor:        {0.0, 0.01, 0.05, 0.1, 0.3}
//   - soc_max_iterations:    {0, 1, 2, 3}
//   - HS-suite cell:         HS026, HS028, HS043, HS071, HS076
//   - mode:                  sqp_mode::accurate, sqp_mode::fast
//
// Both knobs are runtime — the harness reuses one Release build of the
// library and toggles `policy.options.penalty_factor` /
// `policy.options.soc_max_iterations` per cell. For every
// (pf, soc_iter, cell, mode) tuple the harness records iterations, final
// objective, final constraint violation (inf-norm), terminal solver
// status, wall microseconds, `f_err = |f - f*| / max(|f*|, 1.0)`, the
// summed SOC retry count across the trajectory, and a
// `within_strict_bar` flag derived from the tr_sqp_test.cpp acceptance
// shapes.
//
// Joint selection rule (applied in main()):
//
//   Step 1 — D-04 hard gate: a (pf, soc_iter) configuration is a Step-1
//            survivor iff every HS026 / HS028 / HS071 / HS076 cell (both
//            modes) reports `within_strict_bar = true`.
//   Step 2 — D-02 strict-pass on HS043: among Step-1 survivors, identify
//            configurations where HS043 (both modes) also reports
//            `within_strict_bar = true`. If non-empty, pick the
//            configuration with the LARGEST penalty_factor (more
//            aggressive penalty growth) AND the SMALLEST soc_max_iter
//            (cheapest retry budget); tie-break on the lowest summed
//            wall on HS043 across modes.
//   Step 3 — D-03 partial branch: if Step 2 is empty, pick the Step-1
//            survivor with the lowest HS043 f_err (summed across modes).
//   Step 4 — Land+revert fallback: if Step 1 is empty (every joint
//            regresses at least one D-04 cell), report null selection
//            with the rationale that the SOC retry surgery is
//            net-negative.
//
// Output: a single JSON document with a per-cell `by_config` block plus
// a `selection` block recording the joint disposition (a / b / c).
// Default output path is
// .planning/phases/44-sqp-family-polish-test-stabilization/
// 44-03-soc-budget-sweep.json; override with `--output PATH`.
// Stdout prints a per-row table and the joint disposition.
//
// Non-ctest, one-shot. The harness is read-only on argmin; consumers
// do not pick it up via FetchContent.
//
// Reference: Hock, W. and Schittkowski, K. (1981) "Test Examples for
//            Nonlinear Programming Codes", Lecture Notes in Economics
//            and Mathematical Systems vol. 187, Springer.
//            Lalee, Nocedal, Plantenga 1998 SIAM J. Optim.
//            8(3):682-706 Section 3.3 (adaptive penalty heuristic) +
//            Section 3.1 (SOC retry shape).

#include "argmin/solver/basic_solver.h"
#include "argmin/solver/sqp_mode.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
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
// HS026 / HS028 use absolute objective bars (f* = 0 for both); HS043 /
// HS071 / HS076 use relative f_err bars.
bool within_strict_bar(std::string_view problem_name,
                       sqp_mode          mode,
                       double            f,
                       double            f_star,
                       double            cv)
{
    if(problem_name == "hs026")
    {
        const double bar = (mode == sqp_mode::fast) ? 0.05 : 0.01;
        const double cv_bar = (mode == sqp_mode::fast) ? 1e-2 : 1e-4;
        return std::abs(f - 0.0) <= bar && cv < cv_bar;
    }
    if(problem_name == "hs028")
    {
        if(mode == sqp_mode::fast)
            return std::abs(f - 0.0) <= 1e-2 && cv < 1e-2;
        return std::abs(f - 0.0) <= 1e-6 && cv < 1e-4;
    }
    if(problem_name == "hs071" || problem_name == "hs076"
       || problem_name == "hs043")
    {
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

struct selection
{
    std::optional<double>      penalty_factor;
    std::optional<std::size_t> soc_max_iter;
    std::string                branch;        // "a", "b", or "c"
    std::string                rationale;
};

void write_json(const std::vector<cell_block>& blocks,
                const selection&               sel,
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

    out << "  \"selection\": {\n";
    out << "    \"d04_gate_cells\": [\"hs026\", \"hs028\", \"hs071\", \"hs076\"],\n";
    out << "    \"branch\": \"" << sel.branch << "\",\n";
    out << "    \"selected_penalty_factor\": ";
    if(sel.penalty_factor) out << fmt_num(*sel.penalty_factor);
    else                   out << "null";
    out << ",\n";
    out << "    \"selected_soc_max_iter\": ";
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
    std::string output_path =
        ".planning/phases/44-sqp-family-polish-test-stabilization/"
        "44-03-soc-budget-sweep.json";
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
            std::cout << "soc_budget_sweep: drive tr_sqp_policy across "
                         "the joint (penalty_factor, soc_max_iter) grid on "
                         "HS026/HS028/HS043/HS071/HS076 (accurate + fast).\n"
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

    using hs026_t = argmin::hs026<>;
    using hs028_t = argmin::hs028<>;
    using hs043_t = argmin::hs043<>;
    using hs071_t = argmin::hs071<>;
    using hs076_t = argmin::hs076<>;

    hs026_t p026; hs028_t p028; hs043_t p043; hs071_t p071; hs076_t p076;

    // Per-problem max_iterations matches tr_sqp_test.cpp setup. HS028
    // gets 500 (long shallow ridge on the joint slack-augmented primal);
    // the rest get 200.
    run_cell_block<
        tr_sqp_policy<hs026_t::problem_dimension, sqp_mode::accurate>,
        tr_sqp_policy<hs026_t::problem_dimension, sqp_mode::fast>>(
        "hs026", static_cast<double>(p026.optimal_value()), 200, p026, blocks);

    run_cell_block<
        tr_sqp_policy<hs028_t::problem_dimension, sqp_mode::accurate>,
        tr_sqp_policy<hs028_t::problem_dimension, sqp_mode::fast>>(
        "hs028", static_cast<double>(p028.optimal_value()), 500, p028, blocks);

    run_cell_block<
        tr_sqp_policy<hs043_t::problem_dimension, sqp_mode::accurate>,
        tr_sqp_policy<hs043_t::problem_dimension, sqp_mode::fast>>(
        "hs043", static_cast<double>(p043.optimal_value()), 200, p043, blocks);

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

    // Joint selection rule. See header comment.
    static constexpr std::string_view kD04Cells[] = {
        "hs026", "hs028", "hs071", "hs076"};
    static constexpr std::string_view kModes[] = {"accurate", "fast"};

    auto passes_d04 = [&](double pf, std::size_t soc) -> bool {
        for(auto cell : kD04Cells)
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

    auto passes_hs043 = [&](double pf, std::size_t soc) -> bool {
        for(auto mode : kModes)
        {
            const cell_block* blk = find_block(blocks, "hs043", mode);
            if(!blk) return false;
            const cell_record* rec = find_record(*blk, pf, soc);
            if(!rec) return false;
            if(!rec->within_strict_bar) return false;
        }
        return true;
    };

    auto hs043_f_err_sum = [&](double pf, std::size_t soc) -> double {
        double sum = 0.0;
        for(auto mode : kModes)
        {
            const cell_block* blk = find_block(blocks, "hs043", mode);
            if(!blk) return std::numeric_limits<double>::infinity();
            const cell_record* rec = find_record(*blk, pf, soc);
            if(!rec) return std::numeric_limits<double>::infinity();
            sum += rec->f_err;
        }
        return sum;
    };

    auto hs043_wall_sum = [&](double pf, std::size_t soc) -> double {
        double sum = 0.0;
        for(auto mode : kModes)
        {
            const cell_block* blk = find_block(blocks, "hs043", mode);
            if(!blk) return std::numeric_limits<double>::infinity();
            const cell_record* rec = find_record(*blk, pf, soc);
            if(!rec) return std::numeric_limits<double>::infinity();
            sum += rec->wall_us;
        }
        return sum;
    };

    // Print the D-04 gate table.
    std::cout << "\nD-04 gate (hs026/hs028/hs071/hs076, both modes):\n";
    for(double pf : kPenaltyFactors)
    {
        for(std::size_t soc : kSocMaxIters)
        {
            std::cout << "  pf=" << std::fixed << std::setprecision(2) << pf
                      << " soc=" << soc << " : "
                      << (passes_d04(pf, soc) ? "PASS" : "REGRESS")
                      << " ; hs043 D-02="
                      << (passes_hs043(pf, soc) ? "PASS" : "FAIL")
                      << " ; hs043 f_err_sum="
                      << std::scientific << std::setprecision(3)
                      << hs043_f_err_sum(pf, soc) << "\n";
        }
    }

    // Step 2: D-02 strict-pass winners.
    struct candidate
    {
        double      pf;
        std::size_t soc;
    };
    std::vector<candidate> d02_winners;
    for(double pf : kPenaltyFactors)
    {
        for(std::size_t soc : kSocMaxIters)
        {
            if(passes_d04(pf, soc) && passes_hs043(pf, soc))
                d02_winners.push_back({pf, soc});
        }
    }

    selection sel;
    if(!d02_winners.empty())
    {
        // Largest pf, smallest soc; tie-break on lowest hs043 wall sum.
        auto best = d02_winners[0];
        for(const auto& c : d02_winners)
        {
            const bool better_pf  = c.pf  >  best.pf;
            const bool same_pf    = c.pf  == best.pf;
            const bool better_soc = c.soc <  best.soc;
            const bool same_soc   = c.soc == best.soc;
            if(better_pf
               || (same_pf && better_soc)
               || (same_pf && same_soc
                   && hs043_wall_sum(c.pf, c.soc)
                      < hs043_wall_sum(best.pf, best.soc)))
            {
                best = c;
            }
        }
        sel.penalty_factor = best.pf;
        sel.soc_max_iter   = best.soc;
        sel.branch         = "a";
        std::ostringstream os;
        os << "Joint (penalty_factor, soc_max_iter) = (" << std::fixed
           << std::setprecision(2) << best.pf << ", " << best.soc
           << ") closes HS043 under D-02 strict bars across modes and "
              "preserves D-04 on {hs026, hs028, hs071, hs076}. Selected "
              "as the largest non-regressing penalty_factor paired with "
              "the smallest non-regressing soc_max_iter.";
        sel.rationale = os.str();
        std::cout << "\nBranch (a) D-02 strict-pass: pf=" << std::fixed
                  << std::setprecision(2) << best.pf
                  << "  soc=" << best.soc << "\n";
    }
    else
    {
        // Step 3: D-03 partial branch. Among Step-1 survivors (D-04 only),
        // pick the one with the lowest HS043 f_err sum.
        std::vector<candidate> d04_survivors;
        for(double pf : kPenaltyFactors)
        {
            for(std::size_t soc : kSocMaxIters)
            {
                if(passes_d04(pf, soc))
                    d04_survivors.push_back({pf, soc});
            }
        }
        if(!d04_survivors.empty())
        {
            auto best = d04_survivors[0];
            double best_f_err = hs043_f_err_sum(best.pf, best.soc);
            for(const auto& c : d04_survivors)
            {
                const double fe = hs043_f_err_sum(c.pf, c.soc);
                if(fe < best_f_err)
                {
                    best = c;
                    best_f_err = fe;
                }
            }
            sel.penalty_factor = best.pf;
            sel.soc_max_iter   = best.soc;
            sel.branch         = "b";
            std::ostringstream os;
            os << "Joint (penalty_factor, soc_max_iter) = (" << std::fixed
               << std::setprecision(2) << best.pf << ", " << best.soc
               << ") preserves D-04 on {hs026, hs028, hs071, hs076} and "
                  "minimizes HS043 f_err (residual " << std::scientific
               << std::setprecision(3) << best_f_err << " summed across "
                  "modes); strict D-02 bars on HS043 are not met. The "
                  "HS043 [!shouldfail] tag stays in place pending the "
                  "filter_trsqp_policy structural closure.";
            sel.rationale = os.str();
            std::cout << "\nBranch (b) D-03 partial-closure: pf=" << std::fixed
                      << std::setprecision(2) << best.pf
                      << "  soc=" << best.soc
                      << "  hs043_f_err_sum=" << std::scientific
                      << std::setprecision(3) << best_f_err << "\n";
        }
        else
        {
            // Step 4: land+revert fallback. Every joint regresses at
            // least one D-04 cell.
            sel.branch = "c";
            sel.rationale =
                "NO (penalty_factor, soc_max_iter) configuration in the "
                "swept grid holds D-04 on {hs026, hs028, hs071, hs076} "
                "across modes. The SOC retry surgery is net-negative as "
                "a standalone landing; the two-commit land+revert "
                "protocol applies. The plan-2 LNP plumbing stays in tree.";
            std::cout << "\nBranch (c) land+revert: every joint regresses D-04.\n";
        }
    }

    write_json(blocks, sel, output_path, build_type, head_sha);
    std::cout << "\nJSON written to: " << output_path << "\n";
    return 0;
}
