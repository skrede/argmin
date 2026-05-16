// ISRES seed-stability sweep harness for the `isres_policy: simple
// constrained 2D` regression test case.
//
// Drives ISRES across seeds 1..100 on the EXACT problem setup used by
// the regression test in tests/unit/isres_test.cpp (the
// `simple constrained 2D` TEST_CASE: minimise x0^2 + x1^2 subject to
// x0 + x1 >= 1, bounds [-10, 10]^2, with the same `opts.max_iterations =
// 1000` and the same three 1e-15 thresholds the test uses). For each
// seed the binary records the terminal objective_value, constraint
// violation, x.maxCoeff (informational), solver status, and a
// `converged` flag defined as `result.objective_value < 1.0` — matching
// the test's CHECK condition at tests/unit/isres_test.cpp:149.
//
// Output: a single JSON document with per-seed rows and a summary
// block including `lowest_converging_seed`. Default output path is
// .planning/phases/44-sqp-family-polish-test-stabilization/44-04-isres-seed-sweep.json;
// override with `--output PATH`. Stdout prints a one-line-per-seed
// table plus a top-5 lowest-converging-seeds summary.
//
// Non-ctest, one-shot. The harness is read-only on argmin; consumers
// do not pick it up via FetchContent.

#include "argmin/solver/isres_policy.h"
#include "argmin/solver/basic_solver.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
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

// ---------------------------------------------------------------------
// Problem fixture: replicates tests/unit/isres_test.cpp:35-64
// (`simple_constrained`) verbatim. Minimise x0^2 + x1^2 subject to
// x0 + x1 >= 1, with bounds [-10, 10]^2. Optimal: x* = (0.5, 0.5),
// f* = 0.5.
// ---------------------------------------------------------------------
struct simple_constrained
{
    static constexpr int problem_dimension = argmin::dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        return x[0] * x[0] + x[1] * x[1];
    }

    void constraints(const Eigen::VectorXd& x, Eigen::VectorXd& c) const
    {
        c.resize(1);
        c[0] = x[0] + x[1] - 1.0;
    }

    int num_equality() const { return 0; }
    int num_inequality() const { return 1; }

    Eigen::VectorXd lower_bounds() const
    {
        return Eigen::VectorXd{{-10.0, -10.0}};
    }

    Eigen::VectorXd upper_bounds() const
    {
        return Eigen::VectorXd{{10.0, 10.0}};
    }
};

// Map solver_status to a short string for JSON / stdout.
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

struct seed_record
{
    std::uint64_t seed;
    double        objective_value;
    double        constraint_violation;
    double        x_max_coeff;
    bool          converged;
    std::string   status;
};

// Run a single ISRES solve on the test problem at the given seed and
// record the result. Re-instantiates problem / policy / solver per
// call so the sweep stays independent of any cross-seed carry.
seed_record run_one(std::uint64_t seed)
{
    simple_constrained problem;
    Eigen::VectorXd x0{{2.0, 2.0}};
    argmin::solver_options opts;
    // Mirror tests/unit/isres_test.cpp:138-141 verbatim.
    opts.max_iterations = 1000;
    opts.set_gradient_threshold(1e-15);
    opts.set_objective_threshold(1e-15);
    opts.set_step_threshold(1e-15);

    argmin::isres_policy<> policy;
    // The field is std::optional<std::uint64_t>; pass the explicit
    // std::uint64_t to avoid any silent narrowing on assignment.
    policy.options.seed = seed;

    argmin::basic_solver solver{policy, problem, x0, opts};
    auto result = solver.solve();

    return seed_record{
        .seed                 = seed,
        .objective_value      = result.objective_value,
        .constraint_violation = result.constraint_violation,
        .x_max_coeff          = result.x.size() > 0
                                    ? result.x.cwiseAbs().maxCoeff()
                                    : 0.0,
        .converged            = result.objective_value < 1.0,
        .status               = std::string(status_name(result.status)),
    };
}

// Minimal JSON-number formatter that preserves enough precision for
// downstream analysis without trailing garbage zeros. We keep the
// stream defaults (defaultfloat) and bump precision; std::isfinite
// guards NaN / inf since JSON has no representation for them.
std::string fmt_num(double v)
{
    if(!std::isfinite(v))
    {
        // Use a JSON-legal placeholder; downstream readers should
        // treat non-finite as a solver pathology, not a real number.
        return std::isnan(v) ? "\"nan\"" : (v < 0 ? "\"-inf\"" : "\"inf\"");
    }
    std::ostringstream os;
    os << std::setprecision(17) << v;
    return os.str();
}

void write_json(const std::vector<seed_record>& rows,
                std::optional<std::uint64_t>    lowest_converging,
                std::optional<double>           lowest_converging_obj,
                std::string_view                output_path,
                std::string_view                build_type,
                std::string_view                head_sha)
{
    std::ofstream out{std::string{output_path}};
    if(!out)
    {
        std::cerr << "isres_seed_sweep: cannot open " << output_path
                  << " for writing\n";
        std::exit(2);
    }

    int converged_count = 0;
    for(const auto& r : rows) if(r.converged) ++converged_count;

    // Top 5 lowest-numbered converging seeds for the tiebreak summary.
    std::vector<const seed_record*> top5;
    for(const auto& r : rows)
    {
        if(r.converged) top5.push_back(&r);
        if(top5.size() >= 5) break;
    }

    out << "{\n";
    out << "  \"head_sha\": \"" << head_sha << "\",\n";
    out << "  \"build_type\": \"" << build_type << "\",\n";
    out << "  \"problem\": \"isres_simple_constrained_2d\",\n";
    out << "  \"seed_range\": [1, 100],\n";
    out << "  \"convergence_metric\": \"objective_value < 1.0\",\n";
    out << "  \"results\": [\n";
    for(std::size_t i = 0; i < rows.size(); ++i)
    {
        const auto& r = rows[i];
        out << "    { \"seed\": "    << r.seed
            << ", \"objective_value\": " << fmt_num(r.objective_value)
            << ", \"cv\": "              << fmt_num(r.constraint_violation)
            << ", \"x_max_coeff\": "     << fmt_num(r.x_max_coeff)
            << ", \"converged\": "       << (r.converged ? "true" : "false")
            << ", \"status\": \""        << r.status << "\" }";
        if(i + 1 < rows.size()) out << ",";
        out << "\n";
    }
    out << "  ],\n";

    out << "  \"summary\": {\n";
    out << "    \"converged_count\": " << converged_count << ",\n";
    out << "    \"total_count\": "     << rows.size()     << ",\n";
    out << "    \"convergence_rate\": "
        << fmt_num(rows.empty() ? 0.0
                                : static_cast<double>(converged_count)
                                      / static_cast<double>(rows.size()))
        << ",\n";
    out << "    \"lowest_converging_seed\": ";
    if(lowest_converging) out << *lowest_converging; else out << "null";
    out << ",\n";
    out << "    \"lowest_converging_seed_objective_value\": ";
    if(lowest_converging_obj) out << fmt_num(*lowest_converging_obj);
    else                      out << "null";
    out << ",\n";
    out << "    \"top5_lowest_converging_seeds\": [\n";
    for(std::size_t i = 0; i < top5.size(); ++i)
    {
        out << "      { \"seed\": "    << top5[i]->seed
            << ", \"objective_value\": " << fmt_num(top5[i]->objective_value)
            << " }";
        if(i + 1 < top5.size()) out << ",";
        out << "\n";
    }
    out << "    ]\n";
    out << "  }\n";
    out << "}\n";
}

}  // namespace

int main(int argc, char** argv)
{
    std::string output_path =
        ".planning/phases/44-sqp-family-polish-test-stabilization/"
        "44-04-isres-seed-sweep.json";
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
            std::cout << "isres_seed_sweep: drive ISRES across seeds 1..100 "
                         "on the simple_constrained 2D test problem.\n"
                         "  --output PATH       JSON output path "
                         "(default: .planning/phases/.../44-04-isres-seed-sweep.json)\n"
                         "  --build-type S      label only (default Release)\n"
                         "  --head-sha S        label only (default unknown)\n";
            return 0;
        }
        else
        {
            std::cerr << "isres_seed_sweep: unknown arg " << a << "\n";
            return 2;
        }
    }

    std::vector<seed_record> rows;
    rows.reserve(100);

    std::cout << "seed  status              cv          obj          converged\n";
    std::cout << "----  ------------------  ----------  -----------  ---------\n";

    for(std::uint64_t seed = 1; seed <= 100; ++seed)
    {
        seed_record r = run_one(seed);
        std::cout << std::setw(4) << r.seed
                  << "  " << std::left << std::setw(18) << r.status
                  << std::right
                  << "  " << std::setw(10) << std::scientific
                  << std::setprecision(2) << r.constraint_violation
                  << "  " << std::setw(11) << std::fixed
                  << std::setprecision(6) << r.objective_value
                  << "  " << (r.converged ? "yes" : "no")
                  << "\n";
        rows.push_back(std::move(r));
    }

    // Selection rule (per the plan + CONTEXT.md D-08):
    //   LOWEST seed in 1..100 where `objective_value < 1.0`.
    //   The 3-run robustness tiebreak is performed by the caller
    //   (re-running the binary thrice on the selected seed) — ISRES
    //   is seed-deterministic, so this should pass trivially, but the
    //   3-run check guards against non-determinism from parallel
    //   reductions or floating-point reproducibility issues.
    std::optional<std::uint64_t> lowest_seed;
    std::optional<double>        lowest_obj;
    for(const auto& r : rows)
    {
        if(r.converged)
        {
            lowest_seed = r.seed;
            lowest_obj  = r.objective_value;
            break;
        }
    }

    int converged_count = 0;
    for(const auto& r : rows) if(r.converged) ++converged_count;

    std::cout << "\nconverged: " << converged_count << " / " << rows.size()
              << "\n";
    if(lowest_seed)
    {
        std::cout << "lowest converging seed: " << *lowest_seed
                  << " (objective_value = " << std::fixed << std::setprecision(6)
                  << *lowest_obj << ")\n";
    }
    else
    {
        std::cout << "NO seed in 1..100 converged below objective_value = 1.0\n";
    }

    std::cout << "\nTop 5 lowest-numbered converging seeds:\n";
    int shown = 0;
    for(const auto& r : rows)
    {
        if(r.converged)
        {
            std::cout << "  seed=" << r.seed
                      << "  obj=" << std::fixed << std::setprecision(6)
                      << r.objective_value
                      << "  cv=" << std::scientific << std::setprecision(2)
                      << r.constraint_violation
                      << "  status=" << r.status << "\n";
            if(++shown >= 5) break;
        }
    }

    write_json(rows, lowest_seed, lowest_obj, output_path, build_type,
               head_sha);
    std::cout << "\nJSON written to: " << output_path << "\n";

    return 0;
}
