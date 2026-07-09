// regression_check: standalone CSV-bound-comparison gate.
//
// Consumes publish_summary.csv and a regression baseline keyed by
// (solver, problem, mode). Baseline rows use the disposition vocabulary
// pass, expected_fail, and excluded. Pass rows are gated by status,
// accuracy, feasibility, iteration, and wall-time bounds. Expected-fail
// rows become unexpected passes when correctness gates pass; wall time is
// deliberately ignored for that detection so a correct but slower fix is
// not hidden by an inherited broken-run timing bound.
//
// Inputs are pure CSV; the binary has no Eigen, no argmin, no Catch2, and
// no optional-solver link dependency. It is C++20 std-only.
//
// Usage:
//   regression_check <publish_summary.csv> <regression_baseline.csv>
//                    [--accuracy-cutoff <DOUBLE>] [--cv-cutoff <DOUBLE>]
//                    [--self-test-baseline]

#include <map>
#include <set>
#include <array>
#include <cmath>
#include <tuple>
#include <limits>
#include <string>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <charconv>
#include <iostream>
#include <optional>
#include <algorithm>
#include <filesystem>
#include <string_view>

namespace
{

constexpr double kAccuracyCutoff = 1e-12;
constexpr double kCvCutoff       = 1e-8;

constexpr std::array<std::string_view, 3> kProhibitedPerStepSolvers{
    "ipopt",
    "ipopt_monotone",
    "ipopt_sr1",
};

struct csv_table
{
    std::vector<std::string>              header;
    std::vector<std::vector<std::string>> rows;
};

struct gate_options
{
    double accuracy_cutoff{kAccuracyCutoff};
    double cv_cutoff{kCvCutoff};
    bool disable_wall_gate{false};
    bool promote_log_enabled{false};
};

using cell_key = std::tuple<std::string, std::string, std::string>;

enum class baseline_disposition : std::uint8_t
{
    pass,
    expected_fail,
    excluded,
    invalid,
};

struct aggregate_row
{
    std::vector<double>       per_step_us;
    std::vector<std::int64_t> solver_iters_seeds;
    std::vector<double>       accuracy_seeds;
    std::vector<double>       cv_seeds;
    std::vector<std::string>  status_seeds;
    std::vector<std::string>  cap_status_seeds;
};

struct baseline_indices
{
    std::size_t solver{};
    std::size_t problem{};
    std::size_t mode{};
    std::size_t max_us{};
    std::size_t min_acc{};
    std::size_t max_iters{};
    std::size_t max_cv{};
    std::size_t disposition{};
    std::size_t provenance_id{};
    std::size_t wall_gate_policy{};
    std::size_t exclusion_reason{};
};

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
                  << "' missing from " << path.string() << '\n';
    }
    return idx;
}

[[nodiscard]] auto parse_finite_double(std::string_view s, double& out) -> bool
{
    std::string tmp = trim(s);
    if(tmp.empty()) return false;
    auto* first = tmp.data();
    auto* last  = tmp.data() + tmp.size();
    auto [ptr, ec] = std::from_chars(first, last, out);
    return ec == std::errc{} && ptr == last && std::isfinite(out);
}

[[nodiscard]] auto parse_summary_metric(std::string_view s, double& out) -> bool
{
    std::string tmp = trim(s);
    if(tmp.empty()) return false;
    auto* first = tmp.data();
    auto* last  = tmp.data() + tmp.size();
    auto [ptr, ec] = std::from_chars(first, last, out);
    if(ec == std::errc{} && ptr == last)
    {
        if(std::isnan(out))
        {
            out = std::numeric_limits<double>::infinity();
            return true;
        }
        if(out < 0.0) return false;
        return true;
    }
    if(tmp == "inf" || tmp == "+inf" || tmp == "infinity" || tmp == "+infinity")
    {
        out = std::numeric_limits<double>::infinity();
        return true;
    }
    if(tmp == "nan" || tmp == "+nan" || tmp == "-nan")
    {
        out = std::numeric_limits<double>::infinity();
        return true;
    }
    return false;
}

[[nodiscard]] auto parse_int(std::string_view s, std::int64_t& out) -> bool
{
    std::string tmp = trim(s);
    if(tmp.empty()) return false;
    auto* first = tmp.data();
    auto* last  = tmp.data() + tmp.size();
    auto [ptr, ec] = std::from_chars(first, last, out);
    return ec == std::errc{} && ptr == last;
}

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

[[nodiscard]] auto parse_disposition(std::string_view s) -> baseline_disposition
{
    if(s == "pass") return baseline_disposition::pass;
    if(s == "expected_fail") return baseline_disposition::expected_fail;
    if(s == "excluded") return baseline_disposition::excluded;
    return baseline_disposition::invalid;
}

[[nodiscard]] auto median(std::vector<double> v) -> double
{
    if(v.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    if(n % 2 == 1) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

[[nodiscard]] auto max_int(const std::vector<std::int64_t>& v) -> std::int64_t
{
    std::int64_t m = std::numeric_limits<std::int64_t>::min();
    for(auto x : v)
        if(x > m) m = x;
    return m;
}

[[nodiscard]] auto max_double(const std::vector<double>& v) -> double
{
    double m = -std::numeric_limits<double>::infinity();
    for(auto x : v)
        if(x > m) m = x;
    return m;
}

[[nodiscard]] auto log10_or_neg_inf(double x) -> double
{
    if(x <= 0.0) return -std::numeric_limits<double>::infinity();
    return std::log10(x);
}

[[nodiscard]] auto cap_status_clear(std::string_view s) -> bool
{
    return s.empty() || s == "none";
}

[[nodiscard]] auto correctness_status_passes(const aggregate_row& agg) -> bool
{
    if(agg.status_seeds.empty()) return false;
    for(const auto& status : agg.status_seeds)
        if(status != "converged") return false;
    for(const auto& cap_status : agg.cap_status_seeds)
        if(!cap_status_clear(cap_status)) return false;
    return true;
}

[[nodiscard]] auto status_summary(const aggregate_row& agg) -> std::string
{
    std::string out;
    for(const auto& status : agg.status_seeds)
    {
        if(!out.empty()) out += '|';
        out += status;
    }
    return out.empty() ? "none" : out;
}

[[nodiscard]] auto cap_summary(const aggregate_row& agg) -> std::string
{
    std::string out;
    for(const auto& cap_status : agg.cap_status_seeds)
    {
        if(!out.empty()) out += '|';
        out += cap_status.empty() ? "none" : cap_status;
    }
    return out.empty() ? "none" : out;
}

[[nodiscard]] auto row_has_columns(const std::vector<std::string>& row,
                                   std::size_t                    max_idx,
                                   const std::filesystem::path&    path) -> bool
{
    if(row.size() > max_idx) return true;
    std::cerr << "regression_check: short row in " << path.string()
              << " has " << row.size() << " column(s), needs at least "
              << (max_idx + 1) << '\n';
    return false;
}

[[nodiscard]] auto load_baseline_indices(const csv_table&                baseline,
                                         const std::filesystem::path&    path,
                                         baseline_indices&               out) -> bool
{
    auto solver = require_column(baseline.header, "solver", path);
    auto prob   = require_column(baseline.header, "problem", path);
    auto mode   = require_column(baseline.header, "mode", path);
    auto us     = require_column(baseline.header, "max_us_per_step", path);
    auto acc    = require_column(baseline.header, "min_accuracy_log10", path);
    auto iters  = require_column(baseline.header, "max_outer_iters", path);
    auto cv     = require_column(baseline.header, "max_cv_log10", path);
    auto disp   = require_column(baseline.header, "disposition", path);
    auto prov   = require_column(baseline.header, "provenance_id", path);
    auto wall   = require_column(baseline.header, "wall_gate_policy", path);
    auto excl   = require_column(baseline.header, "exclusion_reason", path);
    if(!solver || !prob || !mode || !us || !acc || !iters || !cv
       || !disp || !prov || !wall || !excl)
        return false;

    out = baseline_indices{
        .solver = *solver,
        .problem = *prob,
        .mode = *mode,
        .max_us = *us,
        .min_acc = *acc,
        .max_iters = *iters,
        .max_cv = *cv,
        .disposition = *disp,
        .provenance_id = *prov,
        .wall_gate_policy = *wall,
        .exclusion_reason = *excl,
    };
    return true;
}

[[nodiscard]] auto validate_baseline_schema(const std::filesystem::path& baseline_path) -> int
{
    csv_table baseline;
    std::string err;
    if(!read_csv(baseline_path, baseline, err))
    {
        std::cerr << "regression_check: " << err << '\n';
        return 1;
    }

    baseline_indices b{};
    if(!load_baseline_indices(baseline, baseline_path, b))
        return 1;

    const std::size_t max_idx = std::max({
        b.solver, b.problem, b.mode, b.max_us, b.min_acc, b.max_iters,
        b.max_cv, b.disposition, b.provenance_id, b.wall_gate_policy,
        b.exclusion_reason,
    });

    for(const auto& row : baseline.rows)
    {
        if(!row_has_columns(row, max_idx, baseline_path))
            return 1;
        const auto disp = parse_disposition(row[b.disposition]);
        if(disp == baseline_disposition::invalid)
        {
            std::cerr << "regression_check: invalid baseline disposition '"
                      << row[b.disposition] << "'\n";
            return 1;
        }
        if(row[b.provenance_id].empty())
        {
            std::cerr << "regression_check: baseline row has empty provenance_id"
                      << " for solver=" << row[b.solver]
                      << " problem=" << row[b.problem]
                      << " mode=" << row[b.mode] << '\n';
            return 1;
        }
        if(disp == baseline_disposition::excluded
           && row[b.exclusion_reason].empty())
        {
            std::cerr << "regression_check: excluded baseline row lacks"
                      << " exclusion_reason for solver=" << row[b.solver]
                      << " problem=" << row[b.problem]
                      << " mode=" << row[b.mode] << '\n';
            return 1;
        }

        double max_us = 0.0;
        double min_acc = 0.0;
        double max_cv = 0.0;
        std::int64_t max_iters = 0;
        if(!parse_finite_double(row[b.max_us], max_us)
           || !parse_finite_double(row[b.min_acc], min_acc)
           || !parse_int(row[b.max_iters], max_iters)
           || !parse_finite_double(row[b.max_cv], max_cv))
        {
            std::cerr << "regression_check: invalid numeric field in baseline"
                      << " row for solver=" << row[b.solver]
                      << " problem=" << row[b.problem]
                      << " mode=" << row[b.mode] << '\n';
            return 1;
        }
    }
    return 0;
}

[[nodiscard]] auto load_summary_aggregates(const std::filesystem::path&      summary_path,
                                           std::map<cell_key, aggregate_row>& aggregates) -> bool
{
    csv_table summary;
    std::string err;
    if(!read_csv(summary_path, summary, err))
    {
        std::cerr << "regression_check: " << err << '\n';
        return false;
    }

    auto idx_solver  = require_column(summary.header, "solver", summary_path);
    auto idx_problem = require_column(summary.header, "problem", summary_path);
    auto idx_iters   = require_column(summary.header, "solver_iters", summary_path);
    auto idx_wall    = require_column(summary.header, "wall_time_us", summary_path);
    auto idx_acc     = require_column(summary.header, "accuracy", summary_path);
    auto idx_cv      = require_column(summary.header, "constraint_violation", summary_path);
    auto idx_status  = require_column(summary.header, "status", summary_path);
    auto idx_disp    = require_column(summary.header, "row_disposition", summary_path);
    auto idx_cap     = require_column(summary.header, "cap_status", summary_path);
    if(!idx_solver || !idx_problem || !idx_iters || !idx_wall || !idx_acc
       || !idx_cv || !idx_status || !idx_disp || !idx_cap)
        return false;

    const std::size_t max_idx = std::max({
        *idx_solver, *idx_problem, *idx_iters, *idx_wall, *idx_acc,
        *idx_cv, *idx_status, *idx_disp, *idx_cap,
    });

    for(const auto& row : summary.rows)
    {
        if(!row_has_columns(row, max_idx, summary_path))
            return false;

        const auto& row_disp = row[*idx_disp];
        if(row_disp == "excluded")
            continue;
        if(row_disp != "included")
        {
            std::cerr << "regression_check: invalid summary row_disposition '"
                      << row_disp << "'\n";
            return false;
        }

        std::int64_t iters = 0;
        std::int64_t wall_us = 0;
        double acc = 0.0;
        double cv = 0.0;
        if(!parse_int(row[*idx_iters], iters)
           || !parse_int(row[*idx_wall], wall_us)
           || !parse_summary_metric(row[*idx_acc], acc)
           || !parse_summary_metric(row[*idx_cv], cv))
        {
            std::cerr << "regression_check: invalid numeric field in summary"
                      << " row for solver=" << row[*idx_solver]
                      << " problem=" << row[*idx_problem] << '\n';
            return false;
        }
        if(iters < 0 || wall_us < 0)
        {
            std::cerr << "regression_check: negative iteration or wall value"
                      << " in summary row for solver=" << row[*idx_solver]
                      << " problem=" << row[*idx_problem] << '\n';
            return false;
        }

        const auto& solver = row[*idx_solver];
        const auto& problem = row[*idx_problem];
        auto mode = solver_dispatch_mode(solver);
        const std::int64_t denom = iters > 0 ? iters : 1;
        const double per_step_us = static_cast<double>(wall_us)
                                 / static_cast<double>(denom);

        auto& agg = aggregates[std::make_tuple(solver, problem, mode)];
        agg.per_step_us.push_back(per_step_us);
        agg.solver_iters_seeds.push_back(iters);
        agg.accuracy_seeds.push_back(acc);
        agg.cv_seeds.push_back(cv);
        agg.status_seeds.emplace_back(row[*idx_status]);
        agg.cap_status_seeds.emplace_back(row[*idx_cap]);
    }
    return true;
}

[[nodiscard]] auto run_gate(const std::filesystem::path& summary_path,
                            const std::filesystem::path& baseline_path,
                            const gate_options&          opts) -> int
{
    std::map<cell_key, aggregate_row> aggregates;
    if(!load_summary_aggregates(summary_path, aggregates))
        return 1;

    csv_table baseline;
    std::string err;
    if(!read_csv(baseline_path, baseline, err))
    {
        std::cerr << "regression_check: " << err << '\n';
        return 1;
    }

    baseline_indices b{};
    if(!load_baseline_indices(baseline, baseline_path, b))
        return 1;

    const std::size_t max_idx = std::max({
        b.solver, b.problem, b.mode, b.max_us, b.min_acc, b.max_iters,
        b.max_cv, b.disposition, b.provenance_id, b.wall_gate_policy,
        b.exclusion_reason,
    });

    int breach_count = 0;
    int unexpected_pass_count = 0;
    int excluded_count = 0;
    int warn_missing_baseline = 0;

    auto emit_breach = [&](std::string_view solver, std::string_view problem,
                           std::string_view mode, std::string_view metric,
                           std::string_view measured, std::string_view bound)
    {
        std::cerr << "BREACH: solver=" << solver
                  << " problem=" << problem
                  << " mode=" << mode
                  << " metric=" << metric
                  << " measured=" << measured
                  << " bound=" << bound << '\n';
        ++breach_count;
    };

    auto emit_numeric_breach = [&](std::string_view solver, std::string_view problem,
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

    std::map<cell_key, bool> matched;
    for(const auto& kv : aggregates)
        matched[kv.first] = false;

    for(const auto& row : baseline.rows)
    {
        if(!row_has_columns(row, max_idx, baseline_path))
            return 1;

        const auto& solver = row[b.solver];
        const auto& problem = row[b.problem];
        const auto& mode = row[b.mode];
        const auto disposition = parse_disposition(row[b.disposition]);
        if(disposition == baseline_disposition::invalid)
        {
            std::cerr << "regression_check: invalid baseline disposition '"
                      << row[b.disposition] << "' for solver=" << solver
                      << " problem=" << problem << " mode=" << mode << '\n';
            return 1;
        }
        if(row[b.provenance_id].empty())
        {
            std::cerr << "regression_check: empty provenance_id for baseline"
                      << " solver=" << solver
                      << " problem=" << problem
                      << " mode=" << mode << '\n';
            return 1;
        }

        double max_us = 0.0;
        double min_acc_log10 = 0.0;
        double max_cv_log10 = 0.0;
        std::int64_t max_iters = 0;
        if(!parse_finite_double(row[b.max_us], max_us)
           || !parse_finite_double(row[b.min_acc], min_acc_log10)
           || !parse_int(row[b.max_iters], max_iters)
           || !parse_finite_double(row[b.max_cv], max_cv_log10))
        {
            std::cerr << "regression_check: invalid numeric baseline field"
                      << " for solver=" << solver
                      << " problem=" << problem
                      << " mode=" << mode << '\n';
            return 1;
        }

        if(disposition == baseline_disposition::excluded)
        {
            if(row[b.exclusion_reason].empty())
            {
                std::cerr << "regression_check: excluded baseline row lacks"
                          << " exclusion_reason for solver=" << solver
                          << " problem=" << problem
                          << " mode=" << mode << '\n';
                return 1;
            }
            std::cerr << "EXCLUDED: solver=" << solver
                      << " problem=" << problem
                      << " mode=" << mode
                      << " reason=" << row[b.exclusion_reason] << '\n';
            ++excluded_count;
            continue;
        }

        cell_key key{solver, problem, mode};
        auto it = aggregates.find(key);
        if(it == aggregates.end())
        {
            emit_breach(solver, problem, mode, "summary_cell",
                        "missing", "present");
            continue;
        }

        matched[key] = true;
        const auto& agg = it->second;
        if(agg.per_step_us.empty() || agg.solver_iters_seeds.empty()
           || agg.accuracy_seeds.empty() || agg.cv_seeds.empty())
        {
            emit_breach(solver, problem, mode, "summary_cell",
                        "empty", "nonempty");
            continue;
        }

        const double measured_per_step = median(agg.per_step_us);
        const auto measured_max_iters = max_int(agg.solver_iters_seeds);
        const double measured_max_acc = max_double(agg.accuracy_seeds);
        const double measured_acc_log10 = log10_or_neg_inf(measured_max_acc);
        const double measured_max_cv = max_double(agg.cv_seeds);
        const double measured_cv_log10 = log10_or_neg_inf(measured_max_cv);

        const bool pass_us = measured_per_step <= max_us;
        const bool pass_iters = measured_max_iters <= max_iters;
        const bool pass_acc = measured_acc_log10 <= min_acc_log10;
        const bool pass_cv = measured_cv_log10 <= max_cv_log10;
        const bool status_pass = correctness_status_passes(agg);
        const bool expected_fail_correctness_pass =
            status_pass
            && measured_max_acc > 0.0
            && measured_max_acc < opts.accuracy_cutoff
            && measured_max_cv < opts.cv_cutoff;

        if(disposition == baseline_disposition::pass)
        {
            if(!pass_us && !opts.disable_wall_gate)
                emit_numeric_breach(solver, problem, mode, "max_us_per_step",
                                    measured_per_step, max_us);
            if(!pass_iters)
                emit_numeric_breach(solver, problem, mode, "max_outer_iters",
                                    static_cast<double>(measured_max_iters),
                                    static_cast<double>(max_iters));
            if(!pass_acc)
                emit_numeric_breach(solver, problem, mode, "min_accuracy_log10",
                                    measured_acc_log10, min_acc_log10);
            if(!pass_cv)
                emit_numeric_breach(solver, problem, mode, "max_cv_log10",
                                    measured_cv_log10, max_cv_log10);
            if(!status_pass)
            {
                std::cerr << "CONVERGED_TO_STALLED: solver=" << solver
                          << " problem=" << problem
                          << " mode=" << mode
                          << " status=" << status_summary(agg)
                          << " cap_status=" << cap_summary(agg) << '\n';
                emit_breach(solver, problem, mode, "status",
                            "nonconverged-or-cap-exhausted",
                            "converged-without-cap");
            }
        }
        else if(disposition == baseline_disposition::expected_fail)
        {
            if(expected_fail_correctness_pass)
            {
                if(opts.promote_log_enabled)
                {
                    std::cerr << "PROMOTE: solver=" << solver
                              << " problem=" << problem
                              << " mode=" << mode
                              << " max_accuracy=" << measured_max_acc
                              << " max_cv=" << measured_max_cv << '\n';
                }
                std::cerr << "UNEXPECTED_PASS: solver=" << solver
                          << " problem=" << problem
                          << " mode=" << mode << '\n';
                ++unexpected_pass_count;
            }
        }
    }

    for(const auto& [key, was_matched] : matched)
    {
        if(was_matched) continue;
        const auto& [solver, problem, mode] = key;
        std::cerr << "WARN: cell not in baseline: solver=" << solver
                  << " problem=" << problem
                  << " mode=" << mode << '\n';
        ++warn_missing_baseline;
    }

    if(excluded_count > 0)
        std::cerr << "regression_check: " << excluded_count
                  << " excluded baseline cell(s)\n";
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
    if(warn_missing_baseline > 0)
        std::cerr << "regression_check: " << warn_missing_baseline
                  << " summary cell(s) not in baseline (informational)\n";
    return 0;
}

[[nodiscard]] auto write_text(const std::filesystem::path& path,
                              std::string_view             text) -> bool
{
    std::ofstream out(path);
    if(!out) return false;
    out << text;
    return static_cast<bool>(out);
}

[[nodiscard]] auto summary_fixture(std::string_view solver,
                                   std::string_view problem,
                                   std::string_view wall_us,
                                   std::string_view accuracy,
                                   std::string_view cv,
                                   std::string_view status,
                                   std::string_view row_disposition,
                                   std::string_view cap_status) -> std::string
{
    std::string out =
        "solver,library,problem,class,dimension,seed,mode,solver_iters,"
        "f_evals,g_evals,c_evals,J_evals,wall_time_us,final_objective,"
        "known_optimum,accuracy,constraint_violation,status,row_disposition,"
        "cap_status,exclusion_reason,solve_wall_time_us,end_to_end_wall_time_us,"
        "provenance_id\n";
    out += std::string{solver} + ",argmin," + std::string{problem}
         + ",inequality,2,42,publication,1,1,0,0,0,"
         + std::string{wall_us}
         + ",0.0,0.0," + std::string{accuracy}
         + "," + std::string{cv}
         + "," + std::string{status}
         + "," + std::string{row_disposition}
         + "," + std::string{cap_status}
         + ",,1," + std::string{wall_us} + ",selftest\n";
    return out;
}

[[nodiscard]] auto baseline_fixture(std::string_view solver,
                                    std::string_view problem,
                                    std::string_view max_us,
                                    std::string_view disposition,
                                    std::string_view reason = {}) -> std::string
{
    std::string out =
        "solver,problem,mode,max_us_per_step,min_accuracy_log10,"
        "max_outer_iters,max_cv_log10,disposition,provenance_id,"
        "wall_gate_policy,exclusion_reason\n";
    out += std::string{solver} + "," + std::string{problem}
         + ",default," + std::string{max_us}
         + ",-8.0,10,-8.0," + std::string{disposition}
         + ",selftest,enforced," + std::string{reason} + "\n";
    return out;
}

[[nodiscard]] auto run_self_test() -> int
{
    const auto tmp = std::filesystem::temp_directory_path();
    const auto summary_path = tmp / "argmin-regression-selftest-summary.csv";
    const auto baseline_path = tmp / "argmin-regression-selftest-baseline.csv";
    const gate_options opts{};

    if(!write_text(summary_path, summary_fixture("solver_a", "problem_a",
                                                 "100", "1e-13", "1e-10",
                                                 "converged", "included",
                                                 "none"))
       || !write_text(baseline_path, baseline_fixture("solver_a", "problem_a",
                                                      "1", "expected_fail")))
        return 1;
    if(run_gate(summary_path, baseline_path, opts) != 3)
    {
        std::cerr << "regression_check self-test: expected_fail correctness"
                  << " promotion did not exit 3\n";
        return 1;
    }

    if(!write_text(summary_path, summary_fixture("solver_inf", "problem_inf",
                                                 "10", "inf", "0.0",
                                                 "failed", "included",
                                                 "none"))
       || !write_text(baseline_path, baseline_fixture("solver_inf", "problem_inf",
                                                      "1", "expected_fail")))
        return 1;
    if(run_gate(summary_path, baseline_path, opts) != 0)
    {
        std::cerr << "regression_check self-test: expected_fail infinite"
                  << " accuracy row was not accepted as a failed cell\n";
        return 1;
    }

    if(!write_text(summary_path, summary_fixture("solver_b", "problem_b",
                                                 "10", "1e-4", "1e-4",
                                                 "failed", "excluded",
                                                 "none"))
       || !write_text(baseline_path, baseline_fixture("solver_b", "problem_b",
                                                      "0", "excluded",
                                                      "capability")))
        return 1;
    if(run_gate(summary_path, baseline_path, opts) != 0)
    {
        std::cerr << "regression_check self-test: excluded row was not"
                  << " reported as non-cohort\n";
        return 1;
    }

    if(!write_text(summary_path, summary_fixture("solver_c", "problem_c",
                                                 "1", "1e-10", "1e-10",
                                                 "stalled", "included",
                                                 "none"))
       || !write_text(baseline_path, baseline_fixture("solver_c", "problem_c",
                                                      "10", "pass")))
        return 1;
    if(run_gate(summary_path, baseline_path, opts) != 2)
    {
        std::cerr << "regression_check self-test: pass-to-stalled regression"
                  << " did not exit 2\n";
        return 1;
    }

    if(!write_text(baseline_path,
                   "solver,problem,mode,max_us_per_step\n"
                   "solver_d,problem_d,default,1\n"))
        return 1;
    if(validate_baseline_schema(baseline_path) != 1)
    {
        std::cerr << "regression_check self-test: malformed baseline header"
                  << " did not fail closed\n";
        return 1;
    }
    return 0;
}

[[nodiscard]] auto run_prohibited_solver_self_test() -> int
{
    const std::array<std::string_view, 14> emitted_solver_names{
        "bobyqa",
        "cmaes",
        "isres",
        "ipopt",
        "ipopt_monotone",
        "ipopt_sr1",
        "nlopt_slsqp",
        "kraft_slsqp_accurate",
        "nw_sqp_accurate",
        "filter_slsqp_accurate",
        "filter_nw_sqp_accurate",
        "tr_sqp_fast",
        "tr_sqp_accurate",
        "filter_trsqp_fast",
    };

    std::set<std::string_view> emitted_prohibited;
    for(auto solver : emitted_solver_names)
    {
        if(solver.size() >= 5 && solver.substr(0, 5) == "ipopt")
            emitted_prohibited.insert(solver);
    }

    std::set<std::string_view> configured{
        kProhibitedPerStepSolvers.begin(),
        kProhibitedPerStepSolvers.end(),
    };

    if(configured != emitted_prohibited)
    {
        std::cerr << "regression_check: PROHIBITED solver set mismatch\n";
        std::cerr << "configured:";
        for(auto solver : configured) std::cerr << ' ' << solver;
        std::cerr << "\nemitted:";
        for(auto solver : emitted_prohibited) std::cerr << ' ' << solver;
        std::cerr << '\n';
        return 1;
    }
    return 0;
}

void print_usage(std::ostream& os)
{
    os << "Usage: regression_check <publish_summary.csv> <regression_baseline.csv>\n"
          "                        [--accuracy-cutoff DOUBLE]\n"
          "                        [--cv-cutoff DOUBLE]\n"
          "                        [--self-test-baseline]\n"
          "\n"
          "Self-tests:\n"
          "  regression_check --self-test\n"
          "  regression_check --self-test-prohibited-solvers\n";
}

}

int main(int argc, char** argv)
{
    gate_options opts{};
    bool self_test = false;
    bool prohibited_self_test = false;
    bool baseline_self_test = false;
    std::vector<std::string_view> positional;

    const char* disable_wall_env = std::getenv("REGRESSION_CHECK_DISABLE_WALL_GATE");
    opts.disable_wall_gate =
        disable_wall_env != nullptr && std::string_view{disable_wall_env} == "1";
    if(opts.disable_wall_gate)
        std::cerr << "INFO: REGRESSION_CHECK_DISABLE_WALL_GATE=1;"
                  << " max_us_per_step gate disabled\n";

    const char* promote_log_env = std::getenv("ARGMIN_REGRESSION_CHECK_PROMOTE_LOG");
    opts.promote_log_enabled =
        promote_log_env != nullptr && std::string_view{promote_log_env} == "1";

    for(int i = 1; i < argc; ++i)
    {
        std::string_view arg = argv[i];
        if(arg == "--help" || arg == "-h")
        {
            print_usage(std::cout);
            return 0;
        }
        if(arg == "--self-test")
        {
            self_test = true;
            continue;
        }
        if(arg == "--self-test-prohibited-solvers")
        {
            prohibited_self_test = true;
            continue;
        }
        if(arg == "--self-test-baseline")
        {
            baseline_self_test = true;
            continue;
        }
        if(arg == "--accuracy-cutoff" && i + 1 < argc)
        {
            if(!parse_finite_double(argv[++i], opts.accuracy_cutoff))
            {
                std::cerr << "regression_check: invalid --accuracy-cutoff value\n";
                return 1;
            }
            continue;
        }
        if(arg == "--cv-cutoff" && i + 1 < argc)
        {
            if(!parse_finite_double(argv[++i], opts.cv_cutoff))
            {
                std::cerr << "regression_check: invalid --cv-cutoff value\n";
                return 1;
            }
            continue;
        }
        if(!arg.empty() && arg.front() == '-')
        {
            std::cerr << "regression_check: unknown or incomplete option '"
                      << arg << "'\n";
            print_usage(std::cerr);
            return 1;
        }
        positional.push_back(arg);
    }

    if(self_test)
        return run_self_test();
    if(prohibited_self_test)
        return run_prohibited_solver_self_test();

    if(positional.size() != 2)
    {
        print_usage(std::cerr);
        return 1;
    }

    std::filesystem::path summary_path = std::string{positional[0]};
    std::filesystem::path baseline_path = std::string{positional[1]};

    if(baseline_self_test)
        return validate_baseline_schema(baseline_path);

    return run_gate(summary_path, baseline_path, opts);
}
