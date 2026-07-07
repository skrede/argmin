// Sweep harness for the adaptive L2-merit penalty growth factor on
// tr_sqp_policy.
//
// Drives the trust-region SQP policy across the cross product of:
//   - penalty_factor:  {0.0, 0.01, 0.05, 0.1, 0.3}
//   - HS-suite cell:   HS026, HS028, HS043, HS071, HS076
//   - mode:            sqp_mode::accurate, sqp_mode::fast
//
// Each cell solves the problem with policy.options.penalty_factor set
// to the swept value (no rebuild between values: the penalty_factor is
// a runtime knob). For every (penalty_factor, cell, mode) tuple the
// harness records iterations, final objective, final constraint
// violation (inf-norm), final gradient norm, terminal solver status,
// wall microseconds, `f_err = |f - f*| / max(|f*|, 1.0)`, and a
// `within_strict_bar` flag derived from the tr_sqp_test.cpp acceptance
// bars. The selection rule for the production default penalty_factor is
// the LARGEST value that does not regress HS026/HS028/HS071/HS076
// against those bars (HS043 is informational only).
//
// Predicate parity: within_strict_bar is held byte-for-byte in step
// with the SOC sweep's predicate, including the HS028-accurate
// gradient_norm gate. An earlier revision omitted that gate here; a
// sweep that scores a default under a softer predicate than the peer
// sweep (and the unit gate) re-locks an unsound default no matter how
// correct the kernel is, so the gate is restored for the re-run.
//
// Output: a single JSON document with a per-cell `by_penalty_factor`
// block plus a summary. Default output path is
// `penalty_factor_sweep.json` in the current working directory;
// override with `--output PATH`. Stdout prints a per-row table.
//
// Non-ctest, one-shot. The harness is read-only on argmin; consumers
// do not pick it up via FetchContent.
//
// Reference: Hock, W. and Schittkowski, K. (1981) "Test Examples for
//            Nonlinear Programming Codes", Lecture Notes in Economics
//            and Mathematical Systems vol. 187, Springer.
//            Lalee, Nocedal, Plantenga 1998 SIAM J. Optim.
//            8(3):682-706 Section 3.3 (adaptive penalty heuristic).

#include "argmin/solver/step_budget_solver.h"
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

constexpr double kPenaltyFactors[] = {0.0, 0.01, 0.05, 0.1, 0.3};

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

// Per-cell outcome record.
struct cell_record
{
    double      penalty_factor;
    double      objective_value;
    double      constraint_violation;
    double      gradient_norm;
    double      f_err;            // |f - f*| / max(|f*|, 1.0)
    std::uint32_t iterations;
    double      wall_us;
    std::string status;
    bool        within_strict_bar;
};

// Strict-bar predicate matching the tr_sqp_test.cpp acceptance shapes.
// This predicate is kept byte-for-byte in step with the SOC sweep's
// within_strict_bar (accuracy bar, cv bar, AND the gradient_norm gate
// for the HS028 accurate ridge). A previous revision of this harness
// omitted the gradient_norm gate that the SOC sweep enforced; the two
// sweeps must share one predicate so a default re-locked from one is
// never softer than the bar the other (and the unit test) applies.
//
// HS026 / HS028 use absolute objective bars (f* = 0); HS043 / HS071 /
// HS076 use relative f_err bars. HS028 accurate additionally asserts
// gradient_norm < 1e-4 to mirror the tr_sqp_test.cpp HS028 acceptance.
bool within_strict_bar(std::string_view problem_name,
                       sqp_mode          mode,
                       double            f,
                       double            f_star,
                       double            cv,
                       double            grad_norm)
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
        // Gradient-norm gate mirrors the SOC sweep and the HS028
        // accurate unit cell; a penalty that ships an above-threshold
        // gradient must not read as a passing default.
        return std::abs(f - 0.0) <= 1e-6 && cv < 1e-4 && grad_norm < 1e-4;
    }
    if(problem_name == "hs071")
    {
        const double f_err = std::abs(f - f_star) / std::abs(f_star);
        if(mode == sqp_mode::fast)
            return f_err < 0.05 && cv < 1e-2;
        return f_err < 0.01 && cv < 1e-4;
    }
    if(problem_name == "hs076")
    {
        const double f_err = std::abs(f - f_star) / std::abs(f_star);
        if(mode == sqp_mode::fast)
            return f_err < 0.05 && cv < 1e-2;
        return f_err < 0.01 && cv < 1e-4;
    }
    if(problem_name == "hs043")
    {
        const double f_err = std::abs(f - f_star) / std::abs(f_star);
        if(mode == sqp_mode::fast)
            return f_err < 0.05 && cv < 1e-2;
        return f_err < 0.01 && cv < 1e-4;
    }
    return false;
}

// Drive one (penalty_factor, problem, mode) cell.
//
// max_iterations is supplied per problem to match the existing
// tr_sqp_test.cpp acceptance setup — HS028 uses 500 (long shallow ridge
// on the joint slack-augmented primal); the rest use 200.
template <typename Policy, typename Problem>
cell_record run_cell(double pf, const Problem& problem,
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
    policy.options.penalty_factor = pf;

    argmin::step_budget_solver solver{policy, problem, x0, opts};

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
    rec.objective_value      = result.objective_value;
    rec.constraint_violation = solver.constraint_violation();
    rec.gradient_norm        = result.gradient_norm;
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

// One full row of records (penalty_factor sweep) for one cell.
struct cell_block
{
    cell_spec               spec;
    std::vector<cell_record> records;
};

// Sweep dispatch — instantiates Policy per (problem dimension, mode)
// and runs the penalty_factor loop on it. Records are appended to the
// block in the order penalty_factor values appear in kPenaltyFactors.
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
        auto a = run_cell<PolicyAccurate>(pf, problem, max_iterations);
        a.within_strict_bar = within_strict_bar(problem_name, sqp_mode::accurate,
                                                a.objective_value, f_star,
                                                a.constraint_violation,
                                                a.gradient_norm);
        acc_block.records.push_back(std::move(a));

        auto f = run_cell<PolicyFast>(pf, problem, max_iterations);
        f.within_strict_bar = within_strict_bar(problem_name, sqp_mode::fast,
                                                f.objective_value, f_star,
                                                f.constraint_violation,
                                                f.gradient_norm);
        fst_block.records.push_back(std::move(f));
    }

    out_blocks.push_back(std::move(acc_block));
    out_blocks.push_back(std::move(fst_block));
}

void write_json(const std::vector<cell_block>& blocks,
                std::optional<double>          selected_default,
                std::string_view               selection_rationale,
                std::string_view               output_path,
                std::string_view               build_type,
                std::string_view               head_sha)
{
    std::ofstream out{std::string{output_path}};
    if(!out)
    {
        std::cerr << "penalty_factor_sweep: cannot open " << output_path
                  << " for writing\n";
        std::exit(2);
    }

    out << "{\n";
    out << "  \"head_sha\": \"" << head_sha << "\",\n";
    out << "  \"build_type\": \"" << build_type << "\",\n";
    out << "  \"penalty_factor_set\": [";
    for(std::size_t i = 0; i < std::size(kPenaltyFactors); ++i)
    {
        if(i) out << ", ";
        out << fmt_num(kPenaltyFactors[i]);
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
        out << "      \"by_penalty_factor\": {\n";
        for(std::size_t i = 0; i < block.records.size(); ++i)
        {
            const auto& r = block.records[i];
            std::ostringstream key;
            key << std::fixed << std::setprecision(2) << r.penalty_factor;
            out << "        \"" << key.str() << "\": {"
                << "\"f\": "      << fmt_num(r.objective_value)
                << ", \"cv\": "   << fmt_num(r.constraint_violation)
                << ", \"grad_norm\": " << fmt_num(r.gradient_norm)
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
    out << "    \"reference_gate_cells\": [\"hs026\", \"hs028\", \"hs071\", \"hs076\"],\n";
    out << "    \"selected_penalty_factor\": ";
    if(selected_default) out << fmt_num(*selected_default);
    else                 out << "null";
    out << ",\n";
    out << "    \"rationale\": \"" << selection_rationale << "\"\n";
    out << "  }\n";
    out << "}\n";
}

}  // namespace

int main(int argc, char** argv)
{
    std::string output_path = "penalty_factor_sweep.json";
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
            std::cout << "penalty_factor_sweep: drive tr_sqp_policy across "
                         "penalty_factor in {0.0, 0.01, 0.05, 0.1, 0.3} on "
                         "HS026/HS028/HS043/HS071/HS076 (accurate + fast).\n"
                         "  --output PATH       JSON output path\n"
                         "  --build-type S      label only\n"
                         "  --head-sha S        label only\n";
            return 0;
        }
        else
        {
            std::cerr << "penalty_factor_sweep: unknown arg " << a << "\n";
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
    std::cout << "cell           mode      pf      iters     f             cv            f_err         bar status\n";
    std::cout << "-------------  --------  -----   -----     ------------  ------------  ------------  --- ------------------\n";
    for(const auto& b : blocks)
    {
        for(const auto& r : b.records)
        {
            std::cout << std::left << std::setw(13) << b.spec.name << "  "
                      << std::setw(8) << b.spec.mode << "  "
                      << std::right << std::setw(5) << std::fixed
                      << std::setprecision(2) << r.penalty_factor << "   "
                      << std::setw(5) << r.iterations << "    "
                      << std::scientific << std::setprecision(4)
                      << std::setw(12) << r.objective_value << "  "
                      << std::setw(12) << r.constraint_violation << "  "
                      << std::setw(12) << r.f_err << "  "
                      << (r.within_strict_bar ? " Y " : " N ") << " "
                      << std::left << r.status << "\n";
        }
    }

    // Selection rule:
    //   Reference-gate cells = {hs026, hs028, hs071, hs076} (both modes).
    //   Pass iff every reference cell's `within_strict_bar` is true at
    //   the given penalty_factor. Among the passers, pick the LARGEST
    //   penalty_factor — the most aggressive non-regressing growth
    //   shape. If no penalty_factor passes the reference gate (even
    //   0.0), surface that fact in the JSON and stdout.
    auto passes_reference_gate = [&](double pf) -> bool {
        for(const auto& b : blocks)
        {
            if(b.spec.name == "hs043") continue;
            for(const auto& r : b.records)
            {
                if(r.penalty_factor != pf) continue;
                if(!r.within_strict_bar) return false;
            }
        }
        return true;
    };

    std::optional<double> selected;
    for(auto it = std::rbegin(kPenaltyFactors); it != std::rend(kPenaltyFactors); ++it)
    {
        if(passes_reference_gate(*it))
        {
            selected = *it;
            break;
        }
    }

    std::cout << "\nReference gate (hs026/hs028/hs071/hs076, both modes):\n";
    for(double pf : kPenaltyFactors)
    {
        std::cout << "  penalty_factor=" << std::fixed << std::setprecision(2)
                  << pf << " : " << (passes_reference_gate(pf) ? "PASS" : "REGRESS")
                  << "\n";
    }

    std::string rationale;
    if(selected)
    {
        std::ostringstream os;
        os << "Largest penalty_factor in the swept grid that holds the "
              "reference gate on {hs026, hs028, hs071, hs076} across modes: "
           << std::fixed << std::setprecision(2) << *selected << ".";
        rationale = os.str();
        std::cout << "\nselected default penalty_factor = "
                  << std::fixed << std::setprecision(2) << *selected << "\n";
    }
    else
    {
        rationale =
            "NO penalty_factor in {0.0, 0.01, 0.05, 0.1, 0.3} holds the "
            "reference gate on the four existing-passing cells across modes. "
            "The adaptive-"
            "penalty plumbing is net-negative as a standalone landing; the "
            "freeze-on-feasibility regression is structural and needs a "
            "paired second-order correction retry on the inequality leg to "
            "absorb it. Selection deferred to the joint sweep that revises "
            "this default once SOC retry lands.";
        std::cout << "\nNO penalty_factor passes D-04. See JSON `rationale`.\n";
    }

    write_json(blocks, selected, rationale, output_path, build_type, head_sha);
    std::cout << "\nJSON written to: " << output_path << "\n";
    return 0;
}
