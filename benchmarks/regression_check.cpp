// regression_check: standalone CSV-bound-comparison gate.
//
// Consumes publish_summary.csv (one row per (solver, problem, mode, seed))
// and a regression baseline (one row per (solver, problem, mode) with
// per-cell bounds and expected disposition), aggregates per-seed rows
// into per-cell aggregates, and decides whether every expected:pass cell
// respects its bounds and every expected:shouldfail cell fails at least
// one convergence gate.
//
// Inputs are pure CSV; the binary has no Eigen, no argmin, no Catch2,
// no NLopt link dependency. C++23, std-only.
//
// Usage:
//   regression_check <publish_summary.csv> <regression_baseline.csv>
//
// Exit codes:
//   0  every expected:pass cell satisfies every bound, every
//      expected:shouldfail cell violates at least one convergence gate.
//   1  argument or I/O error (missing files, malformed header).
//   2  at least one expected:pass cell breached a bound. Stderr lists
//      breaches as `BREACH: solver=... problem=... mode=...
//      metric=... measured=... bound=...`.
//   3  at least one expected:shouldfail cell unexpectedly converged.
//      Stderr lists as `UNEXPECTED_PASS: solver=... problem=...
//      mode=...`.
//
// The publish_summary `mode` column reports the bench config mode
// (e.g., "publication"), not the per-solver fast/accurate dispatch
// mode. The fast/accurate axis is encoded in the solver-label suffix
// (e.g., `kraft_slsqp_fast`, `kraft_slsqp_accurate`); the gate joins
// publish_summary rows against the baseline by deriving the solver-mode
// suffix from the solver column.
//
// Env-var REGRESSION_CHECK_DISABLE_WALL_GATE=1 disables the per-step
// wall-clock gate (max_us_per_step). Useful for small-seed-count CI
// runs where per-step timing variance overwhelms the pad. The
// iteration, accuracy, status, and cv gates remain active. The
// production 11-seed sweep leaves this unset so the wall gate fires.

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace
{

// ---- CSV reader ----------------------------------------------------------

struct csv_table
{
    std::vector<std::string>              header;
    std::vector<std::vector<std::string>> rows;
};

[[nodiscard]] auto trim(std::string_view s) -> std::string
{
    auto is_ws = [](unsigned char c){ return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
    while(!s.empty() && is_ws(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
    while(!s.empty() && is_ws(static_cast<unsigned char>(s.back())))  s.remove_suffix(1);
    return std::string(s);
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
    bool        have_header = false;
    while(std::getline(in, line))
    {
        std::string trimmed = trim(line);
        if(trimmed.empty()) continue;
        if(trimmed.front() == '#') continue;
        if(!have_header)
        {
            out.header  = split_csv_line(trimmed);
            have_header = true;
            continue;
        }
        out.rows.emplace_back(split_csv_line(trimmed));
    }
    if(!have_header)
    {
        err = "no header line in '" + path.string() + "'";
        return false;
    }
    return true;
}

[[nodiscard]] auto column_index(const std::vector<std::string>& header,
                                std::string_view                name) -> std::optional<std::size_t>
{
    for(std::size_t i = 0; i < header.size(); ++i)
        if(header[i] == name) return i;
    return std::nullopt;
}

[[nodiscard]] auto require_column(const std::vector<std::string>& header,
                                  std::string_view                name,
                                  const std::filesystem::path&    path) -> std::optional<std::size_t>
{
    auto idx = column_index(header, name);
    if(!idx)
    {
        std::cerr << "regression_check: required column '" << name
                  << "' missing from " << path.string() << "\n";
    }
    return idx;
}

[[nodiscard]] auto parse_double(std::string_view s, double& out) -> bool
{
    // std::from_chars for double is C++17-but-spotty; std::stod is sufficient.
    try
    {
        std::size_t consumed = 0;
        std::string tmp(s);
        out = std::stod(tmp, &consumed);
        return consumed == tmp.size();
    }
    catch(...)
    {
        return false;
    }
}

[[nodiscard]] auto parse_int(std::string_view s, std::int64_t& out) -> bool
{
    auto* first = s.data();
    auto* last  = s.data() + s.size();
    auto [ptr, ec] = std::from_chars(first, last, out);
    return ec == std::errc{} && ptr == last;
}

// ---- Solver-name -> dispatch mode ---------------------------------------

[[nodiscard]] auto solver_dispatch_mode(std::string_view solver) -> std::string
{
    constexpr std::string_view fast_sfx = "_fast";
    constexpr std::string_view acc_sfx  = "_accurate";
    if(solver.size() >= fast_sfx.size()
       && solver.substr(solver.size() - fast_sfx.size()) == fast_sfx)
        return "fast";
    if(solver.size() >= acc_sfx.size()
       && solver.substr(solver.size() - acc_sfx.size()) == acc_sfx)
        return "accurate";
    return "default";
}

// ---- Per-cell aggregation -----------------------------------------------

using cell_key = std::tuple<std::string, std::string, std::string>;  // solver, problem, mode

struct aggregate_row
{
    std::vector<double>       per_step_us;       // wall_time_us / max(solver_iters, 1)
    std::vector<std::int64_t> solver_iters_seeds;
    std::vector<double>       accuracy_seeds;
    std::vector<double>       cv_seeds;           // empty if cv column absent
    bool                      cv_available{false};
    std::vector<std::string>  status_seeds;
};

[[nodiscard]] auto median(std::vector<double> v) -> double
{
    if(v.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    std::size_t n = v.size();
    if(n % 2 == 1) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

[[nodiscard]] auto max_int(const std::vector<std::int64_t>& v) -> std::int64_t
{
    std::int64_t m = std::numeric_limits<std::int64_t>::min();
    for(auto x : v) if(x > m) m = x;
    return m;
}

[[nodiscard]] auto max_double(const std::vector<double>& v) -> double
{
    double m = -std::numeric_limits<double>::infinity();
    for(auto x : v) if(x > m) m = x;
    return m;
}

}  // namespace

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        std::cerr << "Usage: regression_check <publish_summary.csv> <regression_baseline.csv>\n";
        return 1;
    }

    std::filesystem::path summary_path  = argv[1];
    std::filesystem::path baseline_path = argv[2];

    const char* disable_wall_env = std::getenv("REGRESSION_CHECK_DISABLE_WALL_GATE");
    const bool  disable_wall_gate =
        disable_wall_env != nullptr && std::string_view{disable_wall_env} == "1";
    if(disable_wall_gate)
        std::cerr << "INFO: REGRESSION_CHECK_DISABLE_WALL_GATE=1; max_us_per_step gate disabled\n";

    csv_table   summary;
    csv_table   baseline;
    std::string err;
    if(!read_csv(summary_path, summary, err))
    {
        std::cerr << "regression_check: " << err << '\n';
        return 1;
    }
    if(!read_csv(baseline_path, baseline, err))
    {
        std::cerr << "regression_check: " << err << '\n';
        return 1;
    }

    // ---- Summary column indices --------------------------------------
    auto idx_solver  = require_column(summary.header, "solver",        summary_path);
    auto idx_problem = require_column(summary.header, "problem",       summary_path);
    auto idx_iters   = require_column(summary.header, "solver_iters",  summary_path);
    auto idx_wall    = require_column(summary.header, "wall_time_us",  summary_path);
    auto idx_acc     = require_column(summary.header, "accuracy",      summary_path);
    auto idx_status  = require_column(summary.header, "status",        summary_path);
    auto idx_cv      = column_index(summary.header,   "constraint_violation");
    if(!idx_solver || !idx_problem || !idx_iters || !idx_wall || !idx_acc || !idx_status)
        return 1;
    if(!idx_cv)
        std::cerr << "INFO: constraint_violation column missing from "
                     "publish_summary; max_cv_log10 gate disabled\n";

    // ---- Baseline column indices -------------------------------------
    auto b_solver = require_column(baseline.header, "solver",             baseline_path);
    auto b_prob   = require_column(baseline.header, "problem",            baseline_path);
    auto b_mode   = require_column(baseline.header, "mode",               baseline_path);
    auto b_us     = require_column(baseline.header, "max_us_per_step",    baseline_path);
    auto b_acc    = require_column(baseline.header, "min_accuracy_log10", baseline_path);
    auto b_iters  = require_column(baseline.header, "max_outer_iters",    baseline_path);
    auto b_cv     = require_column(baseline.header, "max_cv_log10",       baseline_path);
    auto b_exp    = require_column(baseline.header, "expected",           baseline_path);
    if(!b_solver || !b_prob || !b_mode || !b_us || !b_acc || !b_iters || !b_cv || !b_exp)
        return 1;

    // ---- Aggregate summary rows by (solver, problem, dispatch_mode) ---
    std::map<cell_key, aggregate_row> aggregates;
    for(const auto& row : summary.rows)
    {
        if(row.size() <= std::max({*idx_solver, *idx_problem, *idx_iters,
                                   *idx_wall, *idx_acc, *idx_status}))
            continue;

        const auto& solver  = row[*idx_solver];
        const auto& problem = row[*idx_problem];
        auto        mode    = solver_dispatch_mode(solver);

        std::int64_t iters = 0;
        if(!parse_int(row[*idx_iters], iters)) continue;
        std::int64_t wall_us = 0;
        if(!parse_int(row[*idx_wall], wall_us)) continue;
        double acc = std::numeric_limits<double>::quiet_NaN();
        if(!parse_double(row[*idx_acc], acc)) continue;
        const std::int64_t denom = iters > 0 ? iters : 1;
        const double per_step_us = static_cast<double>(wall_us)
                                 / static_cast<double>(denom);

        auto& agg = aggregates[std::make_tuple(solver, problem, mode)];
        agg.per_step_us.push_back(per_step_us);
        agg.solver_iters_seeds.push_back(iters);
        agg.accuracy_seeds.push_back(acc);
        agg.status_seeds.emplace_back(row[*idx_status]);
        if(idx_cv && row.size() > *idx_cv)
        {
            double cv = std::numeric_limits<double>::quiet_NaN();
            if(parse_double(row[*idx_cv], cv))
            {
                agg.cv_seeds.push_back(cv);
                agg.cv_available = true;
            }
        }
    }

    // ---- Walk baseline; gate each row --------------------------------
    int  breach_count           = 0;
    int  unexpected_pass_count  = 0;
    int  warn_missing           = 0;
    auto emit_breach = [&](std::string_view solver, std::string_view problem,
                           std::string_view mode, std::string_view metric,
                           double measured, double bound)
    {
        std::cerr << "BREACH: solver=" << solver
                  << " problem=" << problem
                  << " mode=" << mode
                  << " metric=" << metric
                  << " measured=" << measured
                  << " bound=" << bound << '\n';
        ++breach_count;
    };

    // Track which (solver, problem, mode) cells from summary were
    // matched against the baseline; the leftover are emitted as WARN
    // diagnostics.
    std::map<cell_key, bool> matched;
    for(const auto& kv : aggregates) matched[kv.first] = false;

    for(const auto& row : baseline.rows)
    {
        if(row.size() <= std::max({*b_solver, *b_prob, *b_mode, *b_us, *b_acc,
                                   *b_iters, *b_cv, *b_exp}))
            continue;
        const auto& solver  = row[*b_solver];
        const auto& problem = row[*b_prob];
        const auto& mode    = row[*b_mode];
        const auto& expect  = row[*b_exp];
        double max_us = 0.0, min_acc_log10 = 0.0, max_cv_log10 = 0.0;
        std::int64_t max_iters = 0;
        if(!parse_double(row[*b_us],  max_us)) continue;
        if(!parse_double(row[*b_acc], min_acc_log10)) continue;
        if(!parse_int   (row[*b_iters], max_iters)) continue;
        if(!parse_double(row[*b_cv],  max_cv_log10)) continue;

        cell_key key{solver, problem, mode};
        auto it = aggregates.find(key);
        if(it == aggregates.end())
        {
            std::cerr << "WARN: baseline cell absent from summary: solver="
                      << solver << " problem=" << problem
                      << " mode=" << mode << '\n';
            continue;
        }
        matched[key] = true;
        const auto& agg = it->second;

        if(expect == "skip")
            continue;

        bool any_nonconverged = false;
        for(const auto& s : agg.status_seeds)
            if(s != "converged") { any_nonconverged = true; break; }

        const double measured_per_step  = median(agg.per_step_us);
        const std::int64_t measured_max_iters = max_int(agg.solver_iters_seeds);
        const double measured_max_acc   = max_double(agg.accuracy_seeds);  // worst-case
        const double measured_acc_log10 = measured_max_acc > 0.0
            ? std::log10(measured_max_acc)
            : -std::numeric_limits<double>::infinity();

        const bool pass_us    = measured_per_step  <= max_us;
        const bool pass_iters = measured_max_iters <= max_iters;
        const bool pass_acc   = measured_acc_log10 <= min_acc_log10;
        bool pass_cv = true;
        double measured_cv_log10 = -std::numeric_limits<double>::infinity();
        if(agg.cv_available && !agg.cv_seeds.empty())
        {
            const double measured_max_cv = max_double(agg.cv_seeds);
            measured_cv_log10 = measured_max_cv > 0.0
                ? std::log10(measured_max_cv)
                : -std::numeric_limits<double>::infinity();
            pass_cv = measured_cv_log10 <= max_cv_log10;
        }
        const bool pass_us_effective = disable_wall_gate || pass_us;
        const bool numeric_all_pass  = pass_us_effective && pass_iters && pass_acc && pass_cv;

        if(expect == "pass")
        {
            if(!pass_us && !disable_wall_gate)
                emit_breach(solver, problem, mode, "max_us_per_step",
                            measured_per_step, max_us);
            if(!pass_iters)
                emit_breach(solver, problem, mode, "max_outer_iters",
                            static_cast<double>(measured_max_iters),
                            static_cast<double>(max_iters));
            if(!pass_acc)
                emit_breach(solver, problem, mode, "min_accuracy_log10",
                            measured_acc_log10, min_acc_log10);
            if(agg.cv_available && !pass_cv)
                emit_breach(solver, problem, mode, "max_cv_log10",
                            measured_cv_log10, max_cv_log10);
            if(any_nonconverged)
            {
                std::cerr << "BREACH: solver=" << solver
                          << " problem=" << problem
                          << " mode=" << mode
                          << " metric=status measured=non-converged-seed-present"
                          << " bound=converged-all-seeds\n";
                ++breach_count;
            }
        }
        else if(expect == "shouldfail")
        {
            // shouldfail cell unexpectedly converges if every seed
            // converged AND every numeric gate held.
            if(!any_nonconverged && numeric_all_pass)
            {
                std::cerr << "UNEXPECTED_PASS: solver=" << solver
                          << " problem=" << problem
                          << " mode=" << mode << '\n';
                ++unexpected_pass_count;
            }
        }
        else
        {
            std::cerr << "WARN: unknown expected disposition '" << expect
                      << "' for solver=" << solver
                      << " problem=" << problem
                      << " mode=" << mode
                      << "; row treated as skip\n";
        }
    }

    for(const auto& [key, was_matched] : matched)
    {
        if(was_matched) continue;
        const auto& [solver, problem, mode] = key;
        std::cerr << "WARN: cell not in baseline: solver=" << solver
                  << " problem=" << problem
                  << " mode=" << mode << '\n';
        ++warn_missing;
    }

    if(breach_count > 0)
    {
        std::cerr << "regression_check: " << breach_count
                  << " breach(es); exit 2\n";
        return 2;
    }
    if(unexpected_pass_count > 0)
    {
        std::cerr << "regression_check: " << unexpected_pass_count
                  << " unexpected-pass cell(s); exit 3\n";
        return 3;
    }
    if(warn_missing > 0)
        std::cerr << "regression_check: " << warn_missing
                  << " summary cell(s) not in baseline (informational)\n";
    return 0;
}
