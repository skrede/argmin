// Empirical sweep harness for the filter-acceptance trust-region SQP
// policy. Drives the policy across the Cartesian product of:
//
//   - sweep_kind:               {standard, extended, restoration}
//   - HS-suite cell:            {hs024, hs026, hs028, hs035, hs039,
//                                hs040, hs043, hs050, hs071, hs076}
//   - sqp_mode:                 {accurate, fast}
//   - filter envelope:          (gamma_f, gamma_h) -- asymmetric
//   - filter reject mode:       {tr_shrink, switching_condition}
//   - switching condition:      kappa, s
//   - initial trust radius:     delta0
//   - feasibility restoration:  restoration_max_iter (compile-time-guarded;
//                               the harness builds against the policy
//                               with or without the restoration knobs)
//   - seed:                     a traceability label only (HS cells are
//                               deterministic from initial_point()).
//
// argv-driven; the sweep_kind argv selects a semantic grid, the
// per-axis argv flags override it. Output: a single JSON document
// (single source of truth) plus an optional Markdown analysis. Internal
// wall budgets are enforced per (config, cell) and per cell; no
// external timeout dependency. The harness exits 0 on full completion,
// 1 on partial completion (any budget-exceeded entries), 2 on argv
// error.
//
// This harness is kept separate from filter_envelope_sweep.cpp:
//
//   - filter_envelope_sweep targets the filter_slsqp and filter_nw_sqp
//     line-search policies (different policy types -- separate
//     basic_solver instantiations than filter_trsqp_policy);
//   - filter_envelope_sweep's HS043-centric output schema is
//     incompatible with the multi-cell x multi-seed x multi-axis
//     Cartesian JSON consumed by the empirical-default-selection
//     workflow this harness feeds.
//
// Idioms are reused: argv parser, fmt_num, status_name, and
// write_json come from penalty_factor_sweep.cpp; the per-(cell, mode)
// strict-bar predicate composition mirrors soc_budget_sweep.cpp's
// within_strict_bar table. The within_strict_bar shapes match the
// tr_sqp_test.cpp and filter_trsqp_test.cpp acceptance bars verbatim
// for every cell in scope.
//
// Reference: Fletcher and Leyffer 2002 Math. Programming 91:239-269
//            Section 2.1 (filter dominance);
//            Fletcher, Leyffer, Toint 2002 SIAM J. Optim. 13(1):44-59
//            Section 3 (filter-TR convergence theory; tr_shrink reject);
//            Wachter and Biegler 2005 SIAM J. Optim. 16(1):1-31
//            Section 2.3 (switching condition; kappa, s);
//            Wachter and Biegler 2006 Math. Programming 106:25-57
//            Section 2.3 eq. 6 (filter envelope; gamma_f, gamma_h);
//            Nocedal and Wright 2e Section 18.5 Algorithm 18.4
//            (Byrd-Omojokun composite step);
//            Hock and Schittkowski 1981 Lecture Notes in Economics and
//            Mathematical Systems vol. 187, Springer (HS test suite).

#include "argmin/solver/basic_solver.h"
#include "argmin/solver/filter_trsqp_policy.h"
#include "argmin/solver/sqp_mode.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

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
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>
#include <algorithm>
#include <string_view>

namespace
{

using argmin::sqp_mode;
template <int N, sqp_mode Mode>
using policy_t = argmin::filter_trsqp_policy<N, Mode>;

// Filter-reject mode is nested inside the policy; alias for brevity.
template <sqp_mode Mode>
using reject_mode_t =
    typename policy_t<argmin::dynamic_dimension, Mode>::filter_reject_mode;

enum class sweep_kind_e : std::uint8_t
{
    standard,
    extended,
    restoration,
};

std::string_view sweep_kind_name(sweep_kind_e k)
{
    switch(k)
    {
        case sweep_kind_e::standard:    return "standard";
        case sweep_kind_e::extended:    return "extended";
        case sweep_kind_e::restoration: return "restoration";
    }
    return "unknown";
}

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

// Mode-string -> enum.
std::optional<sqp_mode> parse_mode(std::string_view s)
{
    if(s == "accurate") return sqp_mode::accurate;
    if(s == "fast")     return sqp_mode::fast;
    return std::nullopt;
}

std::string_view mode_name(sqp_mode m)
{
    return (m == sqp_mode::fast) ? "fast" : "accurate";
}

// Reject-mode string -> integer (the harness stores the reject mode as
// uint8 because the enum is nested per-Policy-instantiation; the cast
// happens at the call site).
std::optional<std::uint8_t> parse_reject_mode(std::string_view s)
{
    using rm_t = reject_mode_t<sqp_mode::accurate>;
    if(s == "tr_shrink")
        return static_cast<std::uint8_t>(rm_t::tr_shrink);
    if(s == "switching_condition")
        return static_cast<std::uint8_t>(rm_t::switching_condition);
    return std::nullopt;
}

std::string_view reject_mode_name(std::uint8_t v)
{
    using rm_t = reject_mode_t<sqp_mode::accurate>;
    if(v == static_cast<std::uint8_t>(rm_t::tr_shrink))
        return "tr_shrink";
    if(v == static_cast<std::uint8_t>(rm_t::switching_condition))
        return "switching_condition";
    return "unknown";
}

// Strict-bar predicate matching the live Catch2 REQUIRE/CHECK blocks
// in tests/unit/tr_sqp_test.cpp and tests/unit/filter_trsqp_test.cpp.
//
// Bar shape table (mirrors soc_budget_sweep.cpp::within_strict_bar):
//
//   - hs024 (f* = -1):     relative-against-|f*|, both modes.
//   - hs026 (f* = 0):      absolute, both modes
//                          (1e-2 / 5e-2 on |f| with cv < 1e-4 / 1e-2).
//   - hs028 (f* = 0):      absolute; accurate adds gradient_norm < 1e-4
//                          (mirrors filter_trsqp_test.cpp's CHECK).
//   - hs035 (f* = 1/9):    relative-against-|f*|, both modes.
//   - hs039 (f* = -1):     relative-against-|f*|, both modes.
//   - hs040 (f* = -0.25):  relative-against-|f*|, both modes.
//   - hs043 (f* = -44):    relative-against-|f*|, both modes.
//   - hs050 (f* = 0):      absolute; f* = 0 forces absolute form.
//   - hs071 (f* ~ 17.01):  relative-against-|f*|, both modes.
//   - hs076 (f* ~ -4.68):  relative-against-|f*|, both modes.
bool within_strict_bar(std::string_view name, sqp_mode mode,
                       double f, double f_star, double cv,
                       double grad_norm)
{
    if(name == "hs026")
    {
        const double bar    = (mode == sqp_mode::fast) ? 0.05 : 0.01;
        const double cv_bar = (mode == sqp_mode::fast) ? 1e-2 : 1e-4;
        return std::abs(f - 0.0) <= bar && cv < cv_bar;
    }
    if(name == "hs028")
    {
        if(mode == sqp_mode::fast)
            return std::abs(f - 0.0) <= 1e-2 && cv < 1e-2;
        return std::abs(f - 0.0) <= 1e-6 && cv < 1e-4 && grad_norm < 1e-4;
    }
    if(name == "hs050")
    {
        if(mode == sqp_mode::fast)
            return std::abs(f - 0.0) <= 1e-2 && cv < 1e-2;
        return std::abs(f - 0.0) <= 1e-6 && cv < 1e-4;
    }
    if(name == "hs024" || name == "hs035" || name == "hs039"
       || name == "hs040" || name == "hs043" || name == "hs071"
       || name == "hs076")
    {
        // Relative-against-|f*|.
        const double f_err = std::abs(f - f_star) / std::abs(f_star);
        if(mode == sqp_mode::fast)
            return f_err < 0.05 && cv < 1e-2;
        return f_err < 0.01 && cv < 1e-4;
    }
    return false;
}

// Sweep configuration: one row of the Cartesian product over the
// per-axis grids parsed from argv. reject_mode is stored as the
// underlying integer (the enum lives inside the policy type).
struct sweep_config
{
    double      gamma_f;
    double      gamma_h;
    std::uint8_t reject_mode;
    double      kappa;
    double      s;
    double      delta0;
    std::size_t restoration_max_iter;
    sqp_mode    mode;
};

// Per-(config, cell, seed) outcome record.
struct cell_result
{
    std::string  problem;
    std::uint32_t seed;
    std::string  status;
    double       f_final;
    double       cv_final;
    double       f_err;
    std::uint32_t outer_iters;
    double       wall_us;
    bool         within_strict_bar;
};

// One config-block carries one sweep_config and the list of
// per-(cell, seed) results produced under that config.
struct config_block
{
    sweep_config             config;
    std::vector<cell_result> results;
};

// Per-cell metadata: the optimal value used in the bar predicate and
// the cell-specific max_iterations cap (matching the unit tests).
struct cell_info
{
    std::string_view name;
    double           f_star;
    std::uint32_t    max_iterations;
};

// Per-cell f* and max_iterations table. f* values are queried from the
// problem instance at main(), so this table only sets max_iterations.
std::uint32_t default_max_iterations_for(std::string_view name)
{
    if(name == "hs028") return 500;
    return 200;
}

// Solver-options builder matching the existing unit tests' shape.
template <typename Policy>
argmin::solver_options<> make_solver_opts(std::uint32_t max_iterations)
{
    using policy_type = Policy;
    argmin::solver_options<> opts;
    opts.max_iterations = max_iterations;
    opts.set_gradient_threshold(policy_type::default_gradient_tolerance);
    opts.set_step_threshold(policy_type::default_step_tolerance_rel);
    opts.constraint_tolerance = policy_type::default_feasibility_tolerance;
    return opts;
}

// Drive one (config, cell, seed) tuple. The seed is recorded for
// traceability only; HS cells are deterministic from initial_point().
//
// restoration_max_iter, restoration_lambda_init, and
// restoration_feasibility_tolerance are written via a requires-clause
// guard so the harness builds against the pre-restoration policy AND
// against the post-restoration policy without source edits.
template <typename Policy, typename Problem>
cell_result run_one(const sweep_config& cfg,
                    std::string_view    problem_name,
                    std::uint32_t       seed,
                    const Problem&      problem,
                    std::uint32_t       max_iterations,
                    double              f_star)
{
    using policy_type   = Policy;
    using rm_t          = typename policy_type::filter_reject_mode;

    auto x0 = problem.initial_point();
    auto opts = make_solver_opts<policy_type>(max_iterations);

    policy_type policy;
    policy.options.gamma_f                = cfg.gamma_f;
    policy.options.gamma_h                = cfg.gamma_h;
    policy.options.reject_mode            = static_cast<rm_t>(cfg.reject_mode);
    policy.options.filter_switching_kappa = cfg.kappa;
    policy.options.filter_switching_s     = cfg.s;
    policy.options.initial_trust_radius   = cfg.delta0;

    // Compile-time-guarded restoration option writes. The policy
    // currently has no restoration_* fields; if a future revision adds
    // them, the requires-clause activates and the assignments take
    // effect. Without the fields, a non-zero restoration_max_iter
    // configuration is rejected with a runtime error.
    if constexpr(requires(policy_type p) { p.options.restoration_max_iter; })
    {
        policy.options.restoration_max_iter = cfg.restoration_max_iter;
    }
    else
    {
        if(cfg.restoration_max_iter != 0)
        {
            throw std::runtime_error(
                "restoration_max_iter sweep requires the restoration "
                "prototype; rebuild against the policy header carrying "
                "restoration_max_iter, restoration_lambda_init, and "
                "restoration_feasibility_tolerance.");
        }
    }

    argmin::basic_solver solver{policy, problem, x0, opts};

    const auto t0 = std::chrono::steady_clock::now();
    auto result = solver.solve(opts);
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();

    cell_result out;
    out.problem     = std::string{problem_name};
    out.seed        = seed;
    out.status      = std::string(status_name(result.status));
    out.f_final     = result.objective_value;
    out.cv_final    = solver.constraint_violation();
    if(std::abs(f_star) < 1e-300)
        out.f_err = std::abs(out.f_final - f_star);
    else
        out.f_err = std::abs(out.f_final - f_star) / std::abs(f_star);
    out.outer_iters = static_cast<std::uint32_t>(result.iterations);
    out.wall_us     = wall_us;
    out.within_strict_bar = within_strict_bar(
        problem_name, cfg.mode, out.f_final, f_star, out.cv_final,
        result.gradient_norm);
    return out;
}

// Per-cell dispatch on the problem name. Each branch instantiates the
// policy at the cell's compile-time problem_dimension under both
// sqp_mode variants and forwards to run_one().
cell_result dispatch_cell(const sweep_config& cfg,
                          std::string_view    problem_name,
                          std::uint32_t       seed,
                          std::uint32_t       max_iterations)
{
    using argmin::hs024;
    using argmin::hs026;
    using argmin::hs028;
    using argmin::hs035;
    using argmin::hs039;
    using argmin::hs040;
    using argmin::hs043;
    using argmin::hs050;
    using argmin::hs071;
    using argmin::hs076;

    auto run = [&](auto problem) -> cell_result {
        using problem_t = decltype(problem);
        constexpr int N = problem_t::problem_dimension;
        const double f_star = static_cast<double>(problem.optimal_value());
        if(cfg.mode == sqp_mode::fast)
        {
            return run_one<policy_t<N, sqp_mode::fast>>(
                cfg, problem_name, seed, problem, max_iterations, f_star);
        }
        return run_one<policy_t<N, sqp_mode::accurate>>(
            cfg, problem_name, seed, problem, max_iterations, f_star);
    };

    if(problem_name == "hs024") return run(hs024<>{});
    if(problem_name == "hs026") return run(hs026<>{});
    if(problem_name == "hs028") return run(hs028<>{});
    if(problem_name == "hs035") return run(hs035<>{});
    if(problem_name == "hs039") return run(hs039<>{});
    if(problem_name == "hs040") return run(hs040<>{});
    if(problem_name == "hs043") return run(hs043<>{});
    if(problem_name == "hs050") return run(hs050<>{});
    if(problem_name == "hs071") return run(hs071<>{});
    if(problem_name == "hs076") return run(hs076<>{});

    // Unknown cell: emit a placeholder row and let the caller see it.
    cell_result r;
    r.problem           = std::string{problem_name};
    r.seed              = seed;
    r.status            = "unknown_cell";
    r.f_final           = 0.0;
    r.cv_final          = 0.0;
    r.f_err             = 0.0;
    r.outer_iters       = 0;
    r.wall_us           = 0.0;
    r.within_strict_bar = false;
    return r;
}

// Comma-separated argv parser for grids of double, size_t, and string.
std::vector<double> parse_double_list(std::string_view s)
{
    std::vector<double> out;
    std::string buf;
    for(char c : s)
    {
        if(c == ',')
        {
            if(!buf.empty()) out.push_back(std::stod(buf));
            buf.clear();
        }
        else
        {
            buf.push_back(c);
        }
    }
    if(!buf.empty()) out.push_back(std::stod(buf));
    return out;
}

std::vector<std::size_t> parse_size_list(std::string_view s)
{
    std::vector<std::size_t> out;
    std::string buf;
    for(char c : s)
    {
        if(c == ',')
        {
            if(!buf.empty()) out.push_back(static_cast<std::size_t>(std::stoull(buf)));
            buf.clear();
        }
        else
        {
            buf.push_back(c);
        }
    }
    if(!buf.empty()) out.push_back(static_cast<std::size_t>(std::stoull(buf)));
    return out;
}

std::vector<std::uint32_t> parse_uint_list(std::string_view s)
{
    std::vector<std::uint32_t> out;
    std::string buf;
    for(char c : s)
    {
        if(c == ',')
        {
            if(!buf.empty()) out.push_back(static_cast<std::uint32_t>(std::stoul(buf)));
            buf.clear();
        }
        else
        {
            buf.push_back(c);
        }
    }
    if(!buf.empty()) out.push_back(static_cast<std::uint32_t>(std::stoul(buf)));
    return out;
}

std::vector<std::string> parse_str_list(std::string_view s)
{
    std::vector<std::string> out;
    std::string buf;
    for(char c : s)
    {
        if(c == ',')
        {
            if(!buf.empty()) out.push_back(buf);
            buf.clear();
        }
        else
        {
            buf.push_back(c);
        }
    }
    if(!buf.empty()) out.push_back(buf);
    return out;
}

// Argv state. The cells/modes/reject-modes/etc are validated at parse
// time and stored as their canonical forms (lowercase cell names,
// sqp_mode enum values, reject-mode integers).
struct sweep_invocation
{
    sweep_kind_e            sweep_kind{sweep_kind_e::standard};
    std::vector<std::string> cells;
    std::vector<sqp_mode>    modes;
    std::vector<double>      gamma_f_grid;
    std::vector<double>      gamma_h_grid;
    std::vector<std::uint8_t> reject_modes;
    std::vector<double>      kappa_grid;
    std::vector<double>      s_grid;
    std::vector<double>      delta0_grid;
    std::vector<std::size_t> restoration_iters_grid;
    std::vector<std::uint32_t> seeds;
    double                   per_config_wall_cap_sec{60.0};
    double                   per_cell_wall_budget_sec{5400.0};
    std::string              output_json{};
    std::string              output_md{};
    std::string              build_type{"unknown"};
    std::string              head_sha{"unknown"};
};

// Lowercase a cell name in-place ("HS024" -> "hs024"). Callers must
// keep the canonical lowercase form throughout the harness.
std::string lc(std::string_view s)
{
    std::string out;
    out.reserve(s.size());
    for(char c : s)
    {
        if(c >= 'A' && c <= 'Z') c = static_cast<char>(c + ('a' - 'A'));
        out.push_back(c);
    }
    return out;
}

void print_help()
{
    std::cout <<
        "filter_trsqp_sweep: empirical sweep harness for filter_trsqp_policy.\n"
        "\n"
        "Required:\n"
        "  --sweep-kind KIND          one of {standard, extended, restoration}\n"
        "  --output PATH              JSON output path (single source of truth)\n"
        "\n"
        "Grids (comma-separated; sweep_kind picks the defaults):\n"
        "  --cells LIST               default: HS024,HS026,HS028,HS035,HS039,HS040,\n"
        "                                      HS043,HS050,HS071,HS076\n"
        "  --modes LIST               default: accurate,fast\n"
        "  --gamma-grid LIST          default: 1e-3,1e-4,1e-5,1e-6 (used for both\n"
        "                             gamma_f and gamma_h; supply --gamma-f-grid /\n"
        "                             --gamma-h-grid to set them independently)\n"
        "  --gamma-f-grid LIST        per-axis override of --gamma-grid for gamma_f\n"
        "  --gamma-h-grid LIST        per-axis override of --gamma-grid for gamma_h\n"
        "  --reject-modes LIST        default: tr_shrink,switching_condition\n"
        "  --kappa-grid LIST          default: 1e-4 (Wachter-Biegler 2005 Table 1)\n"
        "  --s-grid LIST              default: 2.3   (Wachter-Biegler 2005 Table 1)\n"
        "  --delta0-grid LIST         default: 1.0   (Nocedal & Wright Section 4.1)\n"
        "  --restoration-iters-grid LIST  default: 0 (restoration disabled)\n"
        "  --seeds LIST               default: 42,43,44,45,46,47,48,49,50,51,52\n"
        "\n"
        "Internal wall budgets (no external timeout dependency):\n"
        "  --per-config-wall-cap-sec N    default: 60     (one (config, cell, seed)\n"
        "                                                  exceeding the cap is\n"
        "                                                  recorded as budget_exceeded)\n"
        "  --per-cell-wall-budget-sec N   default: 5400   (cumulative cap across all\n"
        "                                                  configs for one cell;\n"
        "                                                  un-run configs are\n"
        "                                                  recorded as\n"
        "                                                  not_run_budget_exceeded)\n"
        "\n"
        "Output:\n"
        "  --output PATH              JSON destination (required)\n"
        "  --output-md PATH           Markdown analysis (optional)\n"
        "  --build-type STRING        build label only\n"
        "  --head-sha STRING          HEAD SHA label only\n"
        "  --help                     this message\n";
}

} // namespace

int main(int argc, char** argv)
{
    sweep_invocation inv;

    // Defaults.
    inv.cells = {"hs024", "hs026", "hs028", "hs035", "hs039",
                 "hs040", "hs043", "hs050", "hs071", "hs076"};
    inv.modes = {sqp_mode::accurate, sqp_mode::fast};
    inv.gamma_f_grid = {1e-3, 1e-4, 1e-5, 1e-6};
    inv.gamma_h_grid = {1e-3, 1e-4, 1e-5, 1e-6};
    {
        using rm_t = reject_mode_t<sqp_mode::accurate>;
        inv.reject_modes = {static_cast<std::uint8_t>(rm_t::tr_shrink),
                            static_cast<std::uint8_t>(rm_t::switching_condition)};
    }
    inv.kappa_grid             = {1e-4};
    inv.s_grid                 = {2.3};
    inv.delta0_grid            = {1.0};
    inv.restoration_iters_grid = {0};
    inv.seeds                  = {42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52};

    bool sweep_kind_seen = false;
    std::optional<std::vector<double>> gamma_grid_override;

    for(int i = 1; i < argc; ++i)
    {
        std::string_view a{argv[i]};
        auto next = [&](std::string_view label) -> std::string_view {
            if(i + 1 >= argc)
            {
                std::cerr << "filter_trsqp_sweep: missing value for " << label
                          << "\n";
                std::exit(2);
            }
            return std::string_view{argv[++i]};
        };

        if(a == "--help" || a == "-h")
        {
            print_help();
            return 0;
        }
        else if(a == "--sweep-kind")
        {
            const auto v = next("--sweep-kind");
            if(v == "standard")
                inv.sweep_kind = sweep_kind_e::standard;
            else if(v == "extended")
                inv.sweep_kind = sweep_kind_e::extended;
            else if(v == "restoration")
                inv.sweep_kind = sweep_kind_e::restoration;
            else
            {
                std::cerr << "filter_trsqp_sweep: unknown sweep-kind " << v << "\n";
                return 2;
            }
            sweep_kind_seen = true;
        }
        else if(a == "--cells")
        {
            auto raw = parse_str_list(next("--cells"));
            inv.cells.clear();
            for(const auto& c : raw) inv.cells.push_back(lc(c));
        }
        else if(a == "--modes")
        {
            auto raw = parse_str_list(next("--modes"));
            inv.modes.clear();
            for(const auto& m : raw)
            {
                auto pm = parse_mode(m);
                if(!pm)
                {
                    std::cerr << "filter_trsqp_sweep: unknown mode " << m << "\n";
                    return 2;
                }
                inv.modes.push_back(*pm);
            }
        }
        else if(a == "--gamma-grid")
        {
            gamma_grid_override = parse_double_list(next("--gamma-grid"));
        }
        else if(a == "--gamma-f-grid")
        {
            inv.gamma_f_grid = parse_double_list(next("--gamma-f-grid"));
        }
        else if(a == "--gamma-h-grid")
        {
            inv.gamma_h_grid = parse_double_list(next("--gamma-h-grid"));
        }
        else if(a == "--reject-modes")
        {
            auto raw = parse_str_list(next("--reject-modes"));
            inv.reject_modes.clear();
            for(const auto& r : raw)
            {
                auto pr = parse_reject_mode(r);
                if(!pr)
                {
                    std::cerr << "filter_trsqp_sweep: unknown reject-mode " << r
                              << "\n";
                    return 2;
                }
                inv.reject_modes.push_back(*pr);
            }
        }
        else if(a == "--kappa-grid")
        {
            inv.kappa_grid = parse_double_list(next("--kappa-grid"));
        }
        else if(a == "--s-grid")
        {
            inv.s_grid = parse_double_list(next("--s-grid"));
        }
        else if(a == "--delta0-grid")
        {
            inv.delta0_grid = parse_double_list(next("--delta0-grid"));
        }
        else if(a == "--restoration-iters-grid")
        {
            inv.restoration_iters_grid =
                parse_size_list(next("--restoration-iters-grid"));
        }
        else if(a == "--seeds")
        {
            inv.seeds = parse_uint_list(next("--seeds"));
        }
        else if(a == "--per-config-wall-cap-sec")
        {
            inv.per_config_wall_cap_sec = std::stod(std::string{next("--per-config-wall-cap-sec")});
        }
        else if(a == "--per-cell-wall-budget-sec")
        {
            inv.per_cell_wall_budget_sec = std::stod(std::string{next("--per-cell-wall-budget-sec")});
        }
        else if(a == "--output")
        {
            inv.output_json = std::string{next("--output")};
        }
        else if(a == "--output-md")
        {
            inv.output_md = std::string{next("--output-md")};
        }
        else if(a == "--build-type")
        {
            inv.build_type = std::string{next("--build-type")};
        }
        else if(a == "--head-sha")
        {
            inv.head_sha = std::string{next("--head-sha")};
        }
        else
        {
            std::cerr << "filter_trsqp_sweep: unknown arg " << a << "\n";
            return 2;
        }
    }

    if(!sweep_kind_seen)
    {
        std::cerr << "filter_trsqp_sweep: --sweep-kind is required\n";
        return 2;
    }
    if(inv.output_json.empty())
    {
        std::cerr << "filter_trsqp_sweep: --output is required\n";
        return 2;
    }

    // --gamma-grid is a convenience overriding both gamma_f and
    // gamma_h. --gamma-f-grid / --gamma-h-grid take precedence at the
    // axis level; if --gamma-grid is supplied and the per-axis overrides
    // are NOT supplied separately, both axes share the grid.
    if(gamma_grid_override)
    {
        inv.gamma_f_grid = *gamma_grid_override;
        inv.gamma_h_grid = *gamma_grid_override;
        // Per-axis overrides come AFTER --gamma-grid in argv flip the
        // value above; this loop pass-through is just for clarity.
    }

    // Numeric grid sanity (mitigates argv tampering of the numeric
    // grids; reject infinite/NaN gamma values per the threat register).
    auto reject_nonfinite = [](const std::vector<double>& v,
                               std::string_view name) -> bool {
        for(double x : v)
        {
            if(!std::isfinite(x))
            {
                std::cerr << "filter_trsqp_sweep: non-finite value in "
                          << name << "-grid: " << x << "\n";
                return true;
            }
        }
        return false;
    };
    if(reject_nonfinite(inv.gamma_f_grid, "gamma-f")) return 2;
    if(reject_nonfinite(inv.gamma_h_grid, "gamma-h")) return 2;
    if(reject_nonfinite(inv.kappa_grid, "kappa"))     return 2;
    if(reject_nonfinite(inv.s_grid, "s"))             return 2;
    if(reject_nonfinite(inv.delta0_grid, "delta0"))   return 2;

    // Cartesian product over (gamma_f, gamma_h, reject_mode, kappa, s,
    // delta0, restoration_iter, sqp_mode). Cells and seeds form the
    // inner loop per-config.
    std::vector<sweep_config> configs;
    for(double gf : inv.gamma_f_grid)
    for(double gh : inv.gamma_h_grid)
    for(std::uint8_t rm : inv.reject_modes)
    for(double kp : inv.kappa_grid)
    for(double ss : inv.s_grid)
    for(double d0 : inv.delta0_grid)
    for(std::size_t rmi : inv.restoration_iters_grid)
    for(sqp_mode m : inv.modes)
    {
        configs.push_back(sweep_config{gf, gh, rm, kp, ss, d0, rmi, m});
    }

    // Solve loop with per-config and per-cell wall budgets. The
    // per-cell wall accumulator tracks cumulative wall across all
    // configs for one cell; once exceeded, the remaining configs that
    // would touch the cell record status="not_run_budget_exceeded"
    // without invoking the solver.
    std::vector<config_block> blocks;
    blocks.reserve(configs.size());

    const std::size_t total_runs = configs.size() * inv.cells.size()
                                 * inv.seeds.size();
    std::size_t run_idx = 0;
    bool any_budget_exceeded = false;

    // Cumulative wall per cell (microseconds; the budget is in seconds).
    std::vector<double> wall_us_by_cell(inv.cells.size(), 0.0);
    const double per_cell_wall_budget_us =
        inv.per_cell_wall_budget_sec * 1e6;
    const double per_config_wall_cap_us =
        inv.per_config_wall_cap_sec * 1e6;

    for(const auto& cfg : configs)
    {
        config_block block;
        block.config = cfg;
        block.results.reserve(inv.cells.size() * inv.seeds.size());

        const auto t_config_start = std::chrono::steady_clock::now();
        bool config_budget_hit = false;

        for(std::size_t ci = 0; ci < inv.cells.size(); ++ci)
        {
            const std::string& cell = inv.cells[ci];
            for(std::uint32_t seed : inv.seeds)
            {
                ++run_idx;

                // Per-cell wall budget: if cumulative wall on this
                // cell has already exceeded the budget, skip the
                // solve.
                if(wall_us_by_cell[ci] >= per_cell_wall_budget_us)
                {
                    cell_result r;
                    r.problem           = cell;
                    r.seed              = seed;
                    r.status            = "not_run_budget_exceeded";
                    r.f_final           = 0.0;
                    r.cv_final          = 0.0;
                    r.f_err             = 0.0;
                    r.outer_iters       = 0;
                    r.wall_us           = 0.0;
                    r.within_strict_bar = false;
                    block.results.push_back(std::move(r));
                    any_budget_exceeded = true;
                    continue;
                }

                // Per-config wall cap: if the config has already
                // burned its 60s cap on prior cells, mark the rest
                // budget_exceeded.
                if(config_budget_hit)
                {
                    cell_result r;
                    r.problem           = cell;
                    r.seed              = seed;
                    r.status            = "budget_exceeded";
                    r.f_final           = 0.0;
                    r.cv_final          = 0.0;
                    r.f_err             = 0.0;
                    r.outer_iters       = 0;
                    r.wall_us           = 0.0;
                    r.within_strict_bar = false;
                    block.results.push_back(std::move(r));
                    any_budget_exceeded = true;
                    continue;
                }

                const std::uint32_t mx = default_max_iterations_for(cell);
                cell_result r;
                try
                {
                    r = dispatch_cell(cfg, cell, seed, mx);
                }
                catch(const std::exception& ex)
                {
                    r.problem           = cell;
                    r.seed              = seed;
                    r.status            = "exception";
                    r.f_final           = 0.0;
                    r.cv_final          = 0.0;
                    r.f_err             = 0.0;
                    r.outer_iters       = 0;
                    r.wall_us           = 0.0;
                    r.within_strict_bar = false;
                    std::cerr << "filter_trsqp_sweep: exception on "
                              << cell << " (" << ex.what() << ")\n";
                }

                wall_us_by_cell[ci] += r.wall_us;

                // Progress line on stdout.
                std::cout << "sweep[" << run_idx << "/" << total_runs << "] "
                          << cell << "/" << mode_name(cfg.mode)
                          << " gamma_f=" << fmt_num(cfg.gamma_f)
                          << " gamma_h=" << fmt_num(cfg.gamma_h)
                          << " " << reject_mode_name(cfg.reject_mode)
                          << " kappa=" << fmt_num(cfg.kappa)
                          << " s=" << fmt_num(cfg.s)
                          << " delta0=" << fmt_num(cfg.delta0)
                          << " rmax=" << cfg.restoration_max_iter
                          << " seed=" << seed
                          << " -> " << r.status
                          << " f=" << r.f_final
                          << " cv=" << r.cv_final
                          << " " << r.outer_iters << "it"
                          << " " << static_cast<std::uint64_t>(r.wall_us) << "us"
                          << " bar=" << (r.within_strict_bar ? "Y" : "N")
                          << "\n";

                block.results.push_back(std::move(r));

                // Per-config wall cap check (post-solve; the next cell
                // in this config is skipped if the cap has been crossed).
                const auto t_config_now = std::chrono::steady_clock::now();
                const double wall_so_far_us =
                    std::chrono::duration<double, std::micro>(
                        t_config_now - t_config_start).count();
                if(wall_so_far_us >= per_config_wall_cap_us)
                {
                    config_budget_hit = true;
                }
            }
        }

        blocks.push_back(std::move(block));
    }

    // Write JSON (single source of truth).
    {
        std::ofstream out{inv.output_json};
        if(!out)
        {
            std::cerr << "filter_trsqp_sweep: cannot open " << inv.output_json
                      << " for writing\n";
            return 2;
        }

        out << "{\n";
        out << "  \"sweep_kind\": \"" << sweep_kind_name(inv.sweep_kind) << "\",\n";
        out << "  \"head_sha\": \"" << inv.head_sha << "\",\n";
        out << "  \"build_type\": \"" << inv.build_type << "\",\n";

        // Axes echo.
        out << "  \"axes\": {\n";
        out << "    \"cells\": [";
        for(std::size_t i = 0; i < inv.cells.size(); ++i)
        {
            if(i) out << ", ";
            out << "\"" << inv.cells[i] << "\"";
        }
        out << "],\n";
        out << "    \"modes\": [";
        for(std::size_t i = 0; i < inv.modes.size(); ++i)
        {
            if(i) out << ", ";
            out << "\"" << mode_name(inv.modes[i]) << "\"";
        }
        out << "],\n";
        auto echo_double = [&](std::string_view name,
                               const std::vector<double>& g) {
            out << "    \"" << name << "\": [";
            for(std::size_t i = 0; i < g.size(); ++i)
            {
                if(i) out << ", ";
                out << fmt_num(g[i]);
            }
            out << "],\n";
        };
        echo_double("gamma_f_grid", inv.gamma_f_grid);
        echo_double("gamma_h_grid", inv.gamma_h_grid);
        out << "    \"reject_modes\": [";
        for(std::size_t i = 0; i < inv.reject_modes.size(); ++i)
        {
            if(i) out << ", ";
            out << "\"" << reject_mode_name(inv.reject_modes[i]) << "\"";
        }
        out << "],\n";
        echo_double("kappa_grid",  inv.kappa_grid);
        echo_double("s_grid",      inv.s_grid);
        echo_double("delta0_grid", inv.delta0_grid);
        out << "    \"restoration_iters_grid\": [";
        for(std::size_t i = 0; i < inv.restoration_iters_grid.size(); ++i)
        {
            if(i) out << ", ";
            out << inv.restoration_iters_grid[i];
        }
        out << "],\n";
        out << "    \"seeds\": [";
        for(std::size_t i = 0; i < inv.seeds.size(); ++i)
        {
            if(i) out << ", ";
            out << inv.seeds[i];
        }
        out << "]\n";
        out << "  },\n";

        out << "  \"per_config_wall_cap_sec\": " << fmt_num(inv.per_config_wall_cap_sec) << ",\n";
        out << "  \"per_cell_wall_budget_sec\": " << fmt_num(inv.per_cell_wall_budget_sec) << ",\n";

        // Config blocks.
        out << "  \"configs\": [\n";
        for(std::size_t b = 0; b < blocks.size(); ++b)
        {
            const auto& blk = blocks[b];
            const auto& c   = blk.config;
            out << "    {\n";
            out << "      \"gamma_f\": "    << fmt_num(c.gamma_f) << ",\n";
            out << "      \"gamma_h\": "    << fmt_num(c.gamma_h) << ",\n";
            out << "      \"reject_mode\": \"" << reject_mode_name(c.reject_mode) << "\",\n";
            out << "      \"kappa\": "      << fmt_num(c.kappa) << ",\n";
            out << "      \"s\": "          << fmt_num(c.s) << ",\n";
            out << "      \"delta0\": "     << fmt_num(c.delta0) << ",\n";
            out << "      \"restoration_max_iter\": " << c.restoration_max_iter << ",\n";
            out << "      \"sqp_mode\": \"" << mode_name(c.mode) << "\",\n";
            out << "      \"results\": [\n";
            for(std::size_t i = 0; i < blk.results.size(); ++i)
            {
                const auto& r = blk.results[i];
                out << "        {"
                    << "\"problem\": \"" << r.problem << "\""
                    << ", \"seed\": " << r.seed
                    << ", \"status\": \"" << r.status << "\""
                    << ", \"f_final\": "  << fmt_num(r.f_final)
                    << ", \"cv_final\": " << fmt_num(r.cv_final)
                    << ", \"f_err\": "    << fmt_num(r.f_err)
                    << ", \"outer_iters\": " << r.outer_iters
                    << ", \"wall_us\": "  << fmt_num(r.wall_us)
                    << ", \"within_strict_bar\": "
                    << (r.within_strict_bar ? "true" : "false")
                    << "}";
                if(i + 1 < blk.results.size()) out << ",";
                out << "\n";
            }
            out << "      ]\n";
            out << "    }";
            if(b + 1 < blocks.size()) out << ",";
            out << "\n";
        }
        out << "  ]\n";
        out << "}\n";
    }
    std::cout << "\nJSON written to: " << inv.output_json << "\n";

    // Write optional Markdown analysis. One section per cell; one row
    // per (config, mode) ranked by (within_strict_bar count across
    // seeds DESC, median outer_iters ASC). The MD is human-readable;
    // the JSON is the single source of truth.
    if(!inv.output_md.empty())
    {
        std::ofstream md{inv.output_md};
        if(!md)
        {
            std::cerr << "filter_trsqp_sweep: cannot open " << inv.output_md
                      << " for writing\n";
            return 2;
        }
        md << "# filter_trsqp_sweep " << sweep_kind_name(inv.sweep_kind) << "\n\n";
        md << "build_type: `" << inv.build_type << "`  head_sha: `"
           << inv.head_sha << "`\n\n";
        md << "Total configs: " << blocks.size()
           << " ; per_config_wall_cap: " << inv.per_config_wall_cap_sec << " s"
           << " ; per_cell_wall_budget: " << inv.per_cell_wall_budget_sec
           << " s\n\n";

        struct row
        {
            const sweep_config* cfg;
            std::size_t         within_bar_count;
            double              median_iters;
            double              median_wall_us;
        };

        for(const auto& cell : inv.cells)
        {
            for(sqp_mode m : inv.modes)
            {
                md << "## " << cell << " (" << mode_name(m) << ")\n\n";
                md << "| gamma_f | gamma_h | reject_mode | kappa | s | delta0 | rmax | bar_count/seeds | median_iters | median_wall_us |\n";
                md << "|---------|---------|-------------|-------|---|--------|------|------------------|--------------|----------------|\n";

                std::vector<row> rows;
                for(const auto& blk : blocks)
                {
                    if(blk.config.mode != m) continue;
                    std::size_t bar_count = 0;
                    std::vector<double> iters_v;
                    std::vector<double> wall_v;
                    for(const auto& r : blk.results)
                    {
                        if(r.problem != cell) continue;
                        if(r.within_strict_bar) ++bar_count;
                        iters_v.push_back(static_cast<double>(r.outer_iters));
                        wall_v.push_back(r.wall_us);
                    }
                    auto median = [](std::vector<double>& v) -> double {
                        if(v.empty()) return 0.0;
                        std::sort(v.begin(), v.end());
                        return v[v.size() / 2];
                    };
                    rows.push_back(row{&blk.config, bar_count,
                                       median(iters_v), median(wall_v)});
                }
                std::sort(rows.begin(), rows.end(),
                          [](const row& a, const row& b) {
                              if(a.within_bar_count != b.within_bar_count)
                                  return a.within_bar_count > b.within_bar_count;
                              return a.median_iters < b.median_iters;
                          });
                for(const auto& rw : rows)
                {
                    md << "| " << fmt_num(rw.cfg->gamma_f)
                       << " | " << fmt_num(rw.cfg->gamma_h)
                       << " | " << reject_mode_name(rw.cfg->reject_mode)
                       << " | " << fmt_num(rw.cfg->kappa)
                       << " | " << fmt_num(rw.cfg->s)
                       << " | " << fmt_num(rw.cfg->delta0)
                       << " | " << rw.cfg->restoration_max_iter
                       << " | " << rw.within_bar_count
                       << " / " << inv.seeds.size()
                       << " | " << rw.median_iters
                       << " | " << rw.median_wall_us
                       << " |\n";
                }
                md << "\n";
            }
        }
        std::cout << "MD  written to: " << inv.output_md << "\n";
    }

    return any_budget_exceeded ? 1 : 0;
}
