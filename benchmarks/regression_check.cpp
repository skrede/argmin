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

// Libraries whose per-step timing is legitimate in the published summary.
// "argmin" is the gated per-iteration methodology (its per-step / instruction
// cost is the subject of the RT gate). The informative external comparators
// contribute per-step timing as reference data -- a black-box comparator's
// measured wall time over its reported iterations is a valid reference, not a
// methodology error. Any OTHER (unrecognized) library that injects per-step
// timing into the summary is rejected. This scopes the prohibition so it never
// trips the informative comparators, whose per-step timing is legitimate.
constexpr std::array<std::string_view, 7> kPerStepPermittedLibraries{
    "argmin",
    "nlopt",
    "ipopt",
    "dlib",
    "ceres",
    "libcmaes",
    "optim",
};

// Sentinel for an optional baseline column that is absent from the schema.
constexpr std::size_t kNoColumn = static_cast<std::size_t>(-1);

// Real-time-critical solver families: the SQP families ctrlpp drives for
// NMPC. They alone carry an absolute instructions/iter ceiling (an absolute
// bound needs a real per-iteration budget behind it). Population-based and
// derivative-free solvers get the ratio gate only, since their per-iteration
// cost varies by design (evaluations/iter scale with the population/simplex).
constexpr std::array<std::string_view, 4> kRealtimeCriticalFamilies{
    "nw_sqp",
    "filter_nw_sqp",
    "kraft_slsqp",
    "filter_slsqp",
};

// Upper-bound ratio factor for the per-iteration instruction-cost gate.
// A cell breaches when its measured instructions/iter exceeds the committed
// baseline instructions/iter times this factor. Calibrated empirically from
// the seed-to-seed spread of measured instructions/iter on the current tree
// (see the plan summary for the measured spread and the margin above it);
// instructions/iter is deterministic, so the spread is small and the factor
// is a tight upper bound rather than the wall-time gate's loose envelope.
constexpr double kInstrRegressionFactor = 1.50;

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
    // Optional path to the independent oracle verdict (oracle_verdict.csv).
    // When supplied, a pass cell's CORRECTNESS is witnessed by the oracle's
    // recomputed feasibility/KKT at the returned point (oracle_pass), not by
    // the solver's self-reported status/accuracy. When empty, the legacy
    // self-reported-status/accuracy correctness gate applies unchanged (the
    // prior-release baseline path, which ships no verdict).
    std::string oracle_verdict_path;
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
    std::string               library;
    std::vector<double>       per_step_us;
    std::vector<std::int64_t> solver_iters_seeds;
    std::vector<double>       accuracy_seeds;
    std::vector<double>       cv_seeds;
    std::vector<std::string>  status_seeds;
    std::vector<std::string>  cap_status_seeds;
    std::vector<std::int64_t> instructions_seeds;
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
    // Optional instruction-cost columns; kNoColumn when the baseline schema
    // predates the per-iteration cost gate (e.g. the legacy pinned baseline).
    std::size_t baseline_instr_per_iter{kNoColumn};
    std::size_t instr_ceiling{kNoColumn};
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

[[nodiscard]] auto solver_family(std::string_view solver) -> std::string
{
    constexpr std::string_view fast_sfx = "_fast";
    constexpr std::string_view acc_sfx  = "_accurate";
    if(solver.size() >= fast_sfx.size()
       && solver.substr(solver.size() - fast_sfx.size()) == fast_sfx)
        return std::string{solver.substr(0, solver.size() - fast_sfx.size())};
    if(solver.size() >= acc_sfx.size()
       && solver.substr(solver.size() - acc_sfx.size()) == acc_sfx)
        return std::string{solver.substr(0, solver.size() - acc_sfx.size())};
    return std::string{solver};
}

[[nodiscard]] auto is_realtime_critical_solver(std::string_view solver) -> bool
{
    const auto family = solver_family(solver);
    return std::find(kRealtimeCriticalFamilies.begin(),
                     kRealtimeCriticalFamilies.end(),
                     family)
           != kRealtimeCriticalFamilies.end();
}

// Access an optional baseline field, returning empty when the column is
// absent from the schema or from this particular (possibly shorter) row.
[[nodiscard]] auto optional_field(const std::vector<std::string>& row,
                                  std::size_t                     idx) -> std::string
{
    if(idx == kNoColumn || idx >= row.size())
        return {};
    return row[idx];
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

    // Optional per-iteration instruction-cost columns. Absent in the legacy
    // pinned baseline; when present they arm the ratio (all cells) and ceiling
    // (real-time-critical families) gate.
    auto instr_ipi = column_index(baseline.header, "baseline_instr_per_iter");
    auto instr_cap = column_index(baseline.header, "instr_ceiling");

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
        .baseline_instr_per_iter = instr_ipi.value_or(kNoColumn),
        .instr_ceiling = instr_cap.value_or(kNoColumn),
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
    auto idx_library = require_column(summary.header, "library", summary_path);
    auto idx_problem = require_column(summary.header, "problem", summary_path);
    auto idx_iters   = require_column(summary.header, "solver_iters", summary_path);
    auto idx_wall    = require_column(summary.header, "wall_time_us", summary_path);
    auto idx_acc     = require_column(summary.header, "accuracy", summary_path);
    auto idx_cv      = require_column(summary.header, "constraint_violation", summary_path);
    auto idx_status  = require_column(summary.header, "status", summary_path);
    auto idx_disp    = require_column(summary.header, "row_disposition", summary_path);
    auto idx_cap     = require_column(summary.header, "cap_status", summary_path);
    auto idx_instr   = require_column(summary.header, "instructions", summary_path);
    if(!idx_solver || !idx_library || !idx_problem || !idx_iters || !idx_wall
       || !idx_acc || !idx_cv || !idx_status || !idx_disp || !idx_cap || !idx_instr)
        return false;

    const std::size_t max_idx = std::max({
        *idx_solver, *idx_library, *idx_problem, *idx_iters, *idx_wall, *idx_acc,
        *idx_cv, *idx_status, *idx_disp, *idx_cap, *idx_instr,
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
        std::int64_t instructions = 0;
        double acc = 0.0;
        double cv = 0.0;
        // instructions may legitimately be the unavailable sentinel (-1); it is
        // parsed but NOT range-checked here so it reaches the gate, which fails
        // loud on a non-positive count for a cell the instruction gate covers.
        if(!parse_int(row[*idx_iters], iters)
           || !parse_int(row[*idx_wall], wall_us)
           || !parse_int(row[*idx_instr], instructions)
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
        if(agg.library.empty())
            agg.library = row[*idx_library];
        agg.per_step_us.push_back(per_step_us);
        agg.solver_iters_seeds.push_back(iters);
        agg.accuracy_seeds.push_back(acc);
        agg.cv_seeds.push_back(cv);
        agg.status_seeds.emplace_back(row[*idx_status]);
        agg.cap_status_seeds.emplace_back(row[*idx_cap]);
        agg.instructions_seeds.push_back(instructions);
    }
    return true;
}

[[nodiscard]] auto per_step_timing_permitted(std::string_view library) -> bool
{
    return std::find(kPerStepPermittedLibraries.begin(),
                     kPerStepPermittedLibraries.end(),
                     library)
           != kPerStepPermittedLibraries.end();
}

// Independent oracle verdict, keyed by (solver, problem). A cell is
// oracle-pass iff every seed's oracle_pass is "pass". Cells absent from the
// map have no verdict (the gate fails a pass cell closed on a missing verdict).
using oracle_verdict_map = std::map<std::pair<std::string, std::string>, bool>;

[[nodiscard]] auto load_oracle_verdict(const std::filesystem::path& path,
                                       oracle_verdict_map&          out) -> bool
{
    csv_table verdict;
    std::string err;
    if(!read_csv(path, verdict, err))
    {
        std::cerr << "regression_check: " << err << '\n';
        return false;
    }
    auto idx_solver  = require_column(verdict.header, "solver", path);
    auto idx_problem = require_column(verdict.header, "problem", path);
    auto idx_pass    = require_column(verdict.header, "oracle_pass", path);
    if(!idx_solver || !idx_problem || !idx_pass)
        return false;
    const std::size_t max_idx = std::max({*idx_solver, *idx_problem, *idx_pass});
    for(const auto& row : verdict.rows)
    {
        if(!row_has_columns(row, max_idx, path))
            return false;
        const std::string_view value = row[*idx_pass];
        if(value != "pass" && value != "fail")
        {
            std::cerr << "regression_check: invalid oracle_pass '" << value
                      << "' in " << path.string() << '\n';
            return false;
        }
        const bool seed_pass = (value == "pass");
        auto key = std::make_pair(row[*idx_solver], row[*idx_problem]);
        auto it = out.find(key);
        if(it == out.end())
            out.emplace(std::move(key), seed_pass);
        else
            it->second = it->second && seed_pass;  // pass iff every seed passes
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

    // Independent oracle verdict (optional). When supplied, it witnesses pass
    // cell correctness in place of the solver's self-reported status/accuracy.
    oracle_verdict_map oracle;
    const bool has_oracle = !opts.oracle_verdict_path.empty();
    if(has_oracle)
    {
        if(!load_oracle_verdict(opts.oracle_verdict_path, oracle))
            return 1;
        std::cerr << "regression_check: oracle-witnessed pass correctness"
                  << " (verdict " << opts.oracle_verdict_path << ")\n";
    }

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

    // Per-step timing in the published summary is permitted only from a
    // recognized source: argmin's own solvers (the gated per-iteration
    // methodology) and the informative external comparators (whose per-step
    // reference timing is legitimate). A per-step-timed row from any other
    // (unrecognized) library is rejected here as a methodology error. This is a
    // live check against the emitted data, not a constant compared to a copy of
    // itself, and it deliberately does not trip the informative comparators.
    for(const auto& [key, agg] : aggregates)
    {
        const auto& [solver, problem, mode] = key;
        if(!per_step_timing_permitted(agg.library) && !agg.per_step_us.empty())
            emit_breach(solver, problem, mode, "unsanctioned_per_step_library",
                        agg.library.empty() ? "unknown-library" : agg.library,
                        "argmin-or-informative-comparator");
    }

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

        // Per-iteration instruction-cost gate. Active for a cell only when the
        // baseline carries a committed instructions/iter reference (the legacy
        // pinned baseline omits the column). Ratio-to-baseline is an upper
        // bound applied to every gated cohort cell; the absolute ceiling is an
        // upper bound applied only to the real-time-critical SQP families. A
        // summary instruction count that could not arm (<= 0) fails loud here
        // rather than reading as a free pass -- the deterministic metric is
        // only trustworthy when the counter actually armed.
        const std::string baseline_ipi_field =
            optional_field(row, b.baseline_instr_per_iter);
        if(!baseline_ipi_field.empty())
        {
            double baseline_ipi = 0.0;
            if(!parse_finite_double(baseline_ipi_field, baseline_ipi)
               || baseline_ipi <= 0.0)
            {
                std::cerr << "regression_check: invalid baseline_instr_per_iter '"
                          << baseline_ipi_field << "' for solver=" << solver
                          << " problem=" << problem << " mode=" << mode << '\n';
                return 1;
            }

            bool any_unavailable = false;
            std::vector<double> ipi_seeds;
            for(std::size_t k = 0; k < agg.instructions_seeds.size(); ++k)
            {
                const std::int64_t instr = agg.instructions_seeds[k];
                if(instr <= 0)
                {
                    any_unavailable = true;
                    continue;
                }
                const std::int64_t iters_k = k < agg.solver_iters_seeds.size()
                                                 ? agg.solver_iters_seeds[k]
                                                 : 1;
                const std::int64_t denom = iters_k > 0 ? iters_k : 1;
                ipi_seeds.push_back(static_cast<double>(instr)
                                    / static_cast<double>(denom));
            }

            if(any_unavailable || ipi_seeds.empty())
            {
                emit_breach(solver, problem, mode, "instructions_unavailable",
                            "counter-not-armed-or-zero",
                            "positive-instruction-count");
            }
            else
            {
                const double measured_ipi = median(ipi_seeds);
                const double ratio_bound = baseline_ipi * kInstrRegressionFactor;
                if(measured_ipi > ratio_bound)
                    emit_numeric_breach(solver, problem, mode,
                                        "instr_per_iter_ratio",
                                        measured_ipi, ratio_bound);

                if(is_realtime_critical_solver(solver))
                {
                    const std::string ceiling_field =
                        optional_field(row, b.instr_ceiling);
                    double ceiling = 0.0;
                    if(ceiling_field.empty()
                       || !parse_finite_double(ceiling_field, ceiling)
                       || ceiling <= 0.0)
                    {
                        std::cerr << "regression_check: real-time-critical solver"
                                  << " requires a positive instr_ceiling; got '"
                                  << ceiling_field << "' for solver=" << solver
                                  << " problem=" << problem
                                  << " mode=" << mode << '\n';
                        return 1;
                    }
                    if(measured_ipi > ceiling)
                        emit_numeric_breach(solver, problem, mode,
                                            "instr_per_iter_ceiling",
                                            measured_ipi, ceiling);
                }
            }
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

        // Independent oracle verdict for this cell (joined by solver+problem;
        // a solver has a single dispatch mode, so the baseline mode is
        // determined by the solver name).
        const auto oracle_it = has_oracle
            ? oracle.find({solver, problem})
            : oracle.end();
        const bool cell_has_verdict = has_oracle && oracle_it != oracle.end();
        const bool oracle_pass = cell_has_verdict && oracle_it->second;

        if(disposition == baseline_disposition::pass)
        {
            // Envelope bounds apply regardless of the correctness witness.
            if(!pass_us && !opts.disable_wall_gate)
                emit_numeric_breach(solver, problem, mode, "max_us_per_step",
                                    measured_per_step, max_us);
            if(!pass_iters)
                emit_numeric_breach(solver, problem, mode, "max_outer_iters",
                                    static_cast<double>(measured_max_iters),
                                    static_cast<double>(max_iters));
            if(has_oracle)
            {
                // Correctness is witnessed by the independent oracle
                // (feasibility/KKT recomputed at the returned point), NOT the
                // solver's self-reported status or the accuracy column (which
                // is meaningless for the no-closed-form control cells). A pass
                // cell breaches if its oracle verdict is fail, or is missing
                // entirely (fail-closed).
                if(!cell_has_verdict)
                    emit_breach(solver, problem, mode, "oracle_verdict_missing",
                                "no-verdict", "oracle-pass-at-x*");
                else if(!oracle_pass)
                    emit_breach(solver, problem, mode, "oracle_correctness",
                                "oracle-fail-at-x*", "oracle-pass-at-x*");
            }
            else
            {
                // Legacy path (no verdict supplied, e.g. the prior-release
                // baseline): self-reported accuracy/cv/status correctness gate,
                // unchanged.
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
        }
        else if(disposition == baseline_disposition::expected_fail)
        {
            // An expected_fail cell that now clears the correctness witness is
            // a fix to surface. Under the oracle it is the oracle verdict; on
            // the legacy path it is the self-reported status/accuracy/cv.
            const bool correctness_pass = has_oracle
                ? oracle_pass
                : expected_fail_correctness_pass;
            if(correctness_pass)
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
                                   std::string_view cap_status,
                                   std::string_view instructions = "1000",
                                   std::string_view library = "argmin") -> std::string
{
    std::string out =
        "solver,library,problem,class,dimension,seed,mode,solver_iters,"
        "f_evals,g_evals,c_evals,J_evals,wall_time_us,final_objective,"
        "known_optimum,accuracy,constraint_violation,status,row_disposition,"
        "cap_status,exclusion_reason,solve_wall_time_us,end_to_end_wall_time_us,"
        "provenance_id,instructions\n";
    // solver_iters is fixed at 1, so instructions/iter equals the instructions
    // value verbatim -- the instruction-gate self-tests rely on that identity.
    out += std::string{solver} + "," + std::string{library} + ","
         + std::string{problem}
         + ",inequality,2,42,publication,1,1,0,0,0,"
         + std::string{wall_us}
         + ",0.0,0.0," + std::string{accuracy}
         + "," + std::string{cv}
         + "," + std::string{status}
         + "," + std::string{row_disposition}
         + "," + std::string{cap_status}
         + ",,1," + std::string{wall_us} + ",selftest,"
         + std::string{instructions} + "\n";
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

// Baseline fixture carrying the optional instruction-cost columns so the
// per-iteration gate is armed. instr_ceiling may be empty for a non-RT cell.
[[nodiscard]] auto instr_baseline_fixture(std::string_view solver,
                                          std::string_view problem,
                                          std::string_view disposition,
                                          std::string_view baseline_instr_per_iter,
                                          std::string_view instr_ceiling) -> std::string
{
    // The gate derives a summary cell's mode from the solver name, so the
    // baseline row must use the same dispatch mode or the cell will not match.
    const std::string mode = solver_dispatch_mode(solver);
    std::string out =
        "solver,problem,mode,max_us_per_step,min_accuracy_log10,"
        "max_outer_iters,max_cv_log10,disposition,provenance_id,"
        "wall_gate_policy,exclusion_reason,baseline_instr_per_iter,instr_ceiling\n";
    out += std::string{solver} + "," + std::string{problem}
         + "," + mode + ",1000000,-8.0,1000,-8.0," + std::string{disposition}
         + ",selftest,enforced,," + std::string{baseline_instr_per_iter}
         + "," + std::string{instr_ceiling} + "\n";
    return out;
}

// Exercise the per-iteration instruction-cost gate: ratio (all cells),
// absolute ceiling (real-time-critical families only), and the fail-loud
// path when the counter could not arm. Each case asserts both the breach and
// its clean counterpart, so the gate discriminates rather than merely
// confirms a green. The baseline correctness bounds are generous and the
// summary disposition is expected_fail with a failing accuracy, so the only
// gate that can fire is the instruction gate under test.
[[nodiscard]] auto run_instructions_self_test() -> int
{
    const auto tmp = std::filesystem::temp_directory_path();
    const auto summary_path = tmp / "argmin-regression-instr-summary.csv";
    const auto baseline_path = tmp / "argmin-regression-instr-baseline.csv";
    const gate_options opts{};

    // A non-real-time solver whose instructions/iter equals the committed
    // baseline (ratio 1.0) clears the ratio gate.
    if(!write_text(summary_path, summary_fixture("bobyqa", "problem_i",
                                                 "1", "1e-4", "1e-4",
                                                 "max_iterations", "included",
                                                 "none", "1000"))
       || !write_text(baseline_path, instr_baseline_fixture(
                          "bobyqa", "problem_i", "expected_fail", "1000", "")))
        return 1;
    if(run_gate(summary_path, baseline_path, opts) != 0)
    {
        std::cerr << "regression_check instructions self-test: at-baseline"
                  << " instructions/iter was not accepted\n";
        return 1;
    }

    // The same cell with instructions/iter far above baseline*factor breaches
    // the ratio gate (exit 2), independent of the exact calibrated factor.
    if(!write_text(summary_path, summary_fixture("bobyqa", "problem_i",
                                                 "1", "1e-4", "1e-4",
                                                 "max_iterations", "included",
                                                 "none", "100000")))
        return 1;
    if(run_gate(summary_path, baseline_path, opts) != 2)
    {
        std::cerr << "regression_check instructions self-test: ratio breach"
                  << " did not exit 2\n";
        return 1;
    }

    // An unavailable instruction count (the -1 sentinel) fails loud rather
    // than reading as a free pass.
    if(!write_text(summary_path, summary_fixture("bobyqa", "problem_i",
                                                 "1", "1e-4", "1e-4",
                                                 "max_iterations", "included",
                                                 "none", "-1")))
        return 1;
    if(run_gate(summary_path, baseline_path, opts) != 2)
    {
        std::cerr << "regression_check instructions self-test: unavailable"
                  << " counter (-1) did not fail loud\n";
        return 1;
    }

    // A real-time-critical family under its ratio bound but above its absolute
    // ceiling breaches the ceiling gate (exit 2).
    if(!write_text(summary_path, summary_fixture("nw_sqp_accurate", "problem_i",
                                                 "1", "1e-4", "1e-4",
                                                 "max_iterations", "included",
                                                 "none", "1000"))
       || !write_text(baseline_path, instr_baseline_fixture(
                          "nw_sqp_accurate", "problem_i", "expected_fail",
                          "100000", "500")))
        return 1;
    if(run_gate(summary_path, baseline_path, opts) != 2)
    {
        std::cerr << "regression_check instructions self-test: real-time-family"
                  << " ceiling breach did not exit 2\n";
        return 1;
    }

    // The same real-time cell under both its ratio bound and its ceiling is
    // clean.
    if(!write_text(baseline_path, instr_baseline_fixture(
                       "nw_sqp_accurate", "problem_i", "expected_fail",
                       "100000", "5000")))
        return 1;
    if(run_gate(summary_path, baseline_path, opts) != 0)
    {
        std::cerr << "regression_check instructions self-test: real-time cell"
                  << " within ratio and ceiling was not accepted\n";
        return 1;
    }

    // A real-time-critical family with an armed ratio column but no ceiling is
    // a configuration error (an absolute ceiling is mandatory for this family).
    if(!write_text(baseline_path, instr_baseline_fixture(
                       "nw_sqp_accurate", "problem_i", "expected_fail",
                       "100000", "")))
        return 1;
    if(run_gate(summary_path, baseline_path, opts) != 1)
    {
        std::cerr << "regression_check instructions self-test: real-time family"
                  << " missing its mandatory ceiling was not rejected\n";
        return 1;
    }

    return 0;
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
    const auto tmp = std::filesystem::temp_directory_path();
    const auto summary_path = tmp / "argmin-regression-prohibited-summary.csv";
    const auto baseline_path = tmp / "argmin-regression-prohibited-baseline.csv";
    const gate_options opts{};

    // A row from an unrecognized library that injects per-step timing must be
    // rejected by run_gate. The baseline bounds are deliberately generous so
    // the only breach is the per-step prohibition itself (gate exits 2).
    if(!write_text(summary_path, summary_fixture("rogue_solver", "problem_p",
                                                 "100", "1e-13", "1e-10",
                                                 "converged", "included",
                                                 "none", "1000",
                                                 "unregistered_lib"))
       || !write_text(baseline_path, baseline_fixture("rogue_solver", "problem_p",
                                                      "1000", "pass")))
        return 1;
    if(run_gate(summary_path, baseline_path, opts) != 2)
    {
        std::cerr << "regression_check prohibited self-test: per-step timing from"
                  << " an unrecognized library was not rejected by the gate\n";
        return 1;
    }

    // An informative external comparator (ipopt) emitting the identical
    // per-step-timed row is legitimate reference data and must NOT be tripped:
    // the check discriminates by library rather than merely confirming.
    if(!write_text(summary_path, summary_fixture("ipopt", "problem_p",
                                                 "100", "1e-13", "1e-10",
                                                 "converged", "included",
                                                 "none", "1000", "ipopt"))
       || !write_text(baseline_path, baseline_fixture("ipopt", "problem_p",
                                                      "1000", "pass")))
        return 1;
    if(run_gate(summary_path, baseline_path, opts) != 0)
    {
        std::cerr << "regression_check prohibited self-test: informative"
                  << " comparator per-step timing was wrongly rejected\n";
        return 1;
    }

    // An argmin solver's per-step timing (the gated methodology) is likewise
    // legitimate and must not be tripped.
    if(!write_text(summary_path, summary_fixture("bobyqa", "problem_p",
                                                 "100", "1e-13", "1e-10",
                                                 "converged", "included",
                                                 "none", "1000", "argmin"))
       || !write_text(baseline_path, baseline_fixture("bobyqa", "problem_p",
                                                      "1000", "pass")))
        return 1;
    if(run_gate(summary_path, baseline_path, opts) != 0)
    {
        std::cerr << "regression_check prohibited self-test: argmin per-step"
                  << " timing was wrongly rejected by the per-step prohibition\n";
        return 1;
    }
    return 0;
}

// Oracle-witnessed pass-correctness self-test (Option X). Proves the gate
// takes a pass cell's correctness from the independent oracle verdict, not the
// solver's self-reported status/accuracy, and falls back to the self-reported
// gate when no verdict is supplied.
[[nodiscard]] auto run_oracle_witness_self_test() -> int
{
    const auto tmp = std::filesystem::temp_directory_path();
    const auto summary_path = tmp / "argmin-regression-oracle-summary.csv";
    const auto baseline_path = tmp / "argmin-regression-oracle-baseline.csv";
    const auto verdict_path = tmp / "argmin-regression-oracle-verdict.csv";

    auto verdict_fixture = [](std::string_view oracle_pass) {
        return std::string{
            "solver,problem,mode,seed,oracle_pass,objective_distance,"
            "feasibility_residual,kkt_residual,witness\n"}
            + "filter_slsqp,problem_p,publication,42,"
            + std::string{oracle_pass}
            + ",1e-16,1e-15,,feasibility-at-x*\n";
    };
    // A pass cell whose solver self-reports STALLED but whose returned point is
    // optimal (oracle_pass=pass) must be CLEAN under the verdict -- the whole
    // point of Option X. The baseline marks it pass with generous bounds.
    if(!write_text(summary_path, summary_fixture("filter_slsqp",
                                                 "problem_p", "100", "1e-16",
                                                 "1e-15", "stalled", "included",
                                                 "none"))
       || !write_text(baseline_path,
                      baseline_fixture("filter_slsqp", "problem_p",
                                       "1000", "pass")))
        return 1;

    gate_options with_verdict{};
    with_verdict.oracle_verdict_path = verdict_path.string();

    if(!write_text(verdict_path, verdict_fixture("pass")))
        return 1;
    if(run_gate(summary_path, baseline_path, with_verdict) != 0)
    {
        std::cerr << "regression_check oracle self-test: stalled-but-optimal"
                  << " pass cell was not clean under an oracle_pass verdict\n";
        return 1;
    }

    // Same pass cell, oracle_pass=fail -> breach (exit 2): the independent
    // witness catches an infeasible/wrong returned point.
    if(!write_text(verdict_path, verdict_fixture("fail")))
        return 1;
    if(run_gate(summary_path, baseline_path, with_verdict) != 2)
    {
        std::cerr << "regression_check oracle self-test: oracle_pass=fail pass"
                  << " cell was not rejected\n";
        return 1;
    }

    // A pass cell missing from the verdict -> fail-closed breach (exit 2).
    if(!write_text(verdict_path,
                   "solver,problem,mode,seed,oracle_pass,objective_distance,"
                   "feasibility_residual,kkt_residual,witness\n"
                   "some_other_solver,other,publication,42,pass,0,0,,w\n"))
        return 1;
    if(run_gate(summary_path, baseline_path, with_verdict) != 2)
    {
        std::cerr << "regression_check oracle self-test: pass cell with a"
                  << " missing verdict did not fail closed\n";
        return 1;
    }

    // Legacy fallback (no verdict): the same stalled pass cell is gated on the
    // self-reported status and breaches (exit 2) -- proving the verdict is what
    // flips it to clean above, and the legacy path is unchanged.
    const gate_options legacy{};
    if(run_gate(summary_path, baseline_path, legacy) != 2)
    {
        std::cerr << "regression_check oracle self-test: legacy fallback did not"
                  << " gate the stalled pass cell on self-reported status\n";
        return 1;
    }
    return 0;
}

void print_usage(std::ostream& os)
{
    os << "Usage: regression_check <publish_summary.csv> <regression_baseline.csv>\n"
          "                        [--oracle-verdict oracle_verdict.csv]\n"
          "                        [--accuracy-cutoff DOUBLE]\n"
          "                        [--cv-cutoff DOUBLE]\n"
          "                        [--self-test-baseline]\n"
          "\n"
          "When --oracle-verdict is supplied, a pass cell's correctness is\n"
          "witnessed by the independent oracle (feasibility/KKT at the returned\n"
          "point); without it, the legacy self-reported-status gate applies.\n"
          "\n"
          "Self-tests:\n"
          "  regression_check --self-test\n"
          "  regression_check --self-test-prohibited-solvers\n"
          "  regression_check --self-test-instructions\n"
          "  regression_check --self-test-oracle-witness\n";
}

}

int main(int argc, char** argv)
{
    gate_options opts{};
    bool self_test = false;
    bool prohibited_self_test = false;
    bool instructions_self_test = false;
    bool baseline_self_test = false;
    bool oracle_witness_self_test = false;
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
        if(arg == "--self-test-instructions")
        {
            instructions_self_test = true;
            continue;
        }
        if(arg == "--self-test-baseline")
        {
            baseline_self_test = true;
            continue;
        }
        if(arg == "--self-test-oracle-witness")
        {
            oracle_witness_self_test = true;
            continue;
        }
        if(arg == "--oracle-verdict" && i + 1 < argc)
        {
            opts.oracle_verdict_path = argv[++i];
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
    if(instructions_self_test)
        return run_instructions_self_test();
    if(oracle_witness_self_test)
        return run_oracle_witness_self_test();

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
