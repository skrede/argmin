// Empirical fast-vs-accurate wall-budget sweep harness.
//
// Iterates the 15 (policy x problem) wall-comparison cells covered by
// the four SQP unit-test wall TEST_CASE blocks (kraft_slsqp: 3 problems,
// nw_sqp: 4 problems, filter_slsqp: 4 problems, filter_nw_sqp: 4 problems)
// and measures (t_fast, t_acc, ratio = t_fast / t_acc) across N
// repetitions per cell. Emits a single JSON document on stdout with:
//
//   - top-level metadata: git_sha (from popen "git rev-parse HEAD"),
//                         timestamp_utc (ISO-8601 UTC from std::chrono),
//                         host_info { hostname, cpu_model, cpu_freq_mhz, os }
//                         (all captured at runtime; never placeholders)
//   - records[]: per-record { policy, mode, problem, repetition,
//                             t_acc_s, t_fast_s, ratio }
//
// The driver mirrors the unit-test solve_wall_seconds<Policy, Problem>
// helper byte-for-byte semantically (same options, same x0, same max
// iterations as the wall TEST_CASE site for each cell).
//
// Reference: Hock, W. and Schittkowski, K. (1981) "Test Examples for
//            Nonlinear Programming Codes", Lecture Notes in Economics
//            and Mathematical Systems vol. 187, Springer.

#include "argmin/solver/basic_solver.h"
#include "argmin/solver/sqp_mode.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <sys/utsname.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace
{

constexpr int kRepetitions = 10;

// Per-cell wall measurement. Mirrors solve_wall_seconds in the four
// unit-test files: build solver_options from Policy's constexpr defaults,
// time a single solve() in steady_clock seconds.
template <typename Policy, typename Problem>
double solve_wall_seconds(const Problem& problem,
                          const Eigen::VectorXd& x0,
                          std::uint32_t max_iters)
{
    argmin::solver_options opts;
    opts.max_iterations = max_iters;
    opts.set_gradient_threshold(Policy::default_gradient_tolerance);
    opts.set_step_threshold(Policy::default_step_tolerance_rel);
    opts.constraint_tolerance = Policy::default_feasibility_tolerance;
    argmin::basic_solver solver{Policy{}, problem, x0, opts};
    const auto t0 = std::chrono::steady_clock::now();
    [[maybe_unused]] auto result = solver.solve(opts);
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

// Trim leading/trailing whitespace (including newlines) from str.
std::string trim(std::string s)
{
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

// Capture HEAD commit SHA via popen("git rev-parse HEAD"). If the
// invocation fails (driver run outside a git tree, or `git` missing),
// returns "unknown" — but the build environment guarantees a real
// argmin checkout, so this should always succeed.
std::string capture_git_sha()
{
    std::FILE* pipe = ::popen("git rev-parse HEAD 2>/dev/null", "r");
    if(pipe == nullptr)
    {
        return "unknown";
    }
    char buffer[128] = {0};
    std::string sha;
    if(std::fgets(buffer, sizeof(buffer), pipe) != nullptr)
    {
        sha = trim(buffer);
    }
    ::pclose(pipe);
    if(sha.empty())
    {
        return "unknown";
    }
    return sha;
}

// ISO-8601 UTC timestamp at startup. Uses std::chrono::system_clock so
// the value reflects driver start, not driver link time.
std::string capture_timestamp_utc()
{
    const auto now = std::chrono::system_clock::now();
    const auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_utc{};
    ::gmtime_r(&t, &tm_utc);
    std::ostringstream os;
    os << std::put_time(&tm_utc, "%Y-%m-%dT%H:%M:%SZ");
    return os.str();
}

std::string capture_hostname()
{
    char buf[256] = {0};
    if(::gethostname(buf, sizeof(buf) - 1) != 0)
    {
        return "unknown";
    }
    return std::string(buf);
}

// Read first "model name : ..." line from /proc/cpuinfo (Linux). On
// platforms without /proc/cpuinfo the field falls back to "unknown".
std::string capture_cpu_model()
{
    std::ifstream f("/proc/cpuinfo");
    if(!f.is_open())
    {
        return "unknown";
    }
    std::string line;
    while(std::getline(f, line))
    {
        constexpr std::string_view prefix = "model name";
        if(line.rfind(prefix, 0) == 0)
        {
            auto colon = line.find(':');
            if(colon != std::string::npos)
            {
                return trim(line.substr(colon + 1));
            }
        }
    }
    return "unknown";
}

// CPU frequency in MHz from /proc/cpuinfo first "cpu MHz" line. Returns
// negative value if unavailable; the JSON emitter writes null in that
// case.
double capture_cpu_freq_mhz()
{
    std::ifstream f("/proc/cpuinfo");
    if(!f.is_open())
    {
        return -1.0;
    }
    std::string line;
    while(std::getline(f, line))
    {
        constexpr std::string_view prefix = "cpu MHz";
        if(line.rfind(prefix, 0) == 0)
        {
            auto colon = line.find(':');
            if(colon != std::string::npos)
            {
                try
                {
                    return std::stod(line.substr(colon + 1));
                }
                catch(...)
                {
                    return -1.0;
                }
            }
        }
    }
    return -1.0;
}

std::string capture_os_string()
{
    ::utsname un{};
    if(::uname(&un) != 0)
    {
        return "unknown";
    }
    std::ostringstream os;
    os << un.sysname << ' ' << un.release << ' ' << un.machine;
    return os.str();
}

// Single record from one (policy, mode, problem, repetition) measurement.
struct record
{
    std::string policy;
    std::string mode;
    std::string problem;
    int repetition;
    double t_acc_s;
    double t_fast_s;
    double ratio;
};

// Run N reps of one cell (FastPolicy + AccuratePolicy) on Problem,
// appending to records. The policy aliases are passed as template
// parameters because the policy_accurate / policy_fast aliases on each
// SQP family take a problem_dimension NTTP and are not interchangeable.
template <typename FastPolicy, typename AccuratePolicy, typename Problem>
void run_cell(std::vector<record>& records,
              std::string_view policy_name,
              std::string_view problem_name,
              std::uint32_t max_iters)
{
    Problem problem;
    const Eigen::VectorXd x0 = problem.initial_point();
    for(int rep = 0; rep < kRepetitions; ++rep)
    {
        const double t_acc = solve_wall_seconds<AccuratePolicy>(
            problem, x0, max_iters);
        const double t_fast = solve_wall_seconds<FastPolicy>(
            problem, x0, max_iters);
        const double ratio = (t_acc > 0.0) ? (t_fast / t_acc) : 0.0;
        records.push_back({
            std::string(policy_name),
            "accurate-vs-fast",
            std::string(problem_name),
            rep,
            t_acc,
            t_fast,
            ratio,
        });
    }
}

// Escape a string for inclusion in a JSON string literal. The captured
// values (git SHA, hostname, CPU model, OS string) are well-behaved on
// Linux but the escape keeps the emitter robust to surprise.
std::string json_escape(std::string_view s)
{
    std::string out;
    out.reserve(s.size() + 2);
    for(char c : s)
    {
        switch(c)
        {
        case '"':  out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n";  break;
        case '\r': out += "\\r";  break;
        case '\t': out += "\\t";  break;
        default:
            if(static_cast<unsigned char>(c) < 0x20)
            {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x",
                              static_cast<unsigned char>(c));
                out += buf;
            }
            else
            {
                out += c;
            }
            break;
        }
    }
    return out;
}

// Emit the assembled JSON document to stdout. Pretty-printed for human
// readability; downstream analysis loads it via Python json.load which
// is whitespace-tolerant.
void emit_json(std::ostream& os,
               const std::string& git_sha,
               const std::string& timestamp_utc,
               const std::string& hostname,
               const std::string& cpu_model,
               double cpu_freq_mhz,
               const std::string& os_string,
               const std::vector<record>& records)
{
    os << "{\n";
    os << "  \"git_sha\": \"" << json_escape(git_sha) << "\",\n";
    os << "  \"timestamp_utc\": \"" << json_escape(timestamp_utc) << "\",\n";
    os << "  \"host_info\": {\n";
    os << "    \"hostname\": \"" << json_escape(hostname) << "\",\n";
    os << "    \"cpu_model\": \"" << json_escape(cpu_model) << "\",\n";
    if(cpu_freq_mhz > 0.0)
    {
        os << "    \"cpu_freq_mhz\": " << std::fixed << std::setprecision(2)
           << cpu_freq_mhz << ",\n";
    }
    else
    {
        os << "    \"cpu_freq_mhz\": null,\n";
    }
    os << "    \"os\": \"" << json_escape(os_string) << "\"\n";
    os << "  },\n";
    os << "  \"repetitions\": " << kRepetitions << ",\n";
    os << "  \"records\": [\n";
    for(std::size_t i = 0; i < records.size(); ++i)
    {
        const auto& r = records[i];
        os << "    {"
           << "\"policy\": \""    << json_escape(r.policy)  << "\", "
           << "\"mode\": \""      << json_escape(r.mode)    << "\", "
           << "\"problem\": \""   << json_escape(r.problem) << "\", "
           << "\"repetition\": "  << r.repetition           << ", "
           << "\"t_acc_s\": "     << std::scientific << std::setprecision(6)
                                  << r.t_acc_s              << ", "
           << "\"t_fast_s\": "    << std::scientific << std::setprecision(6)
                                  << r.t_fast_s             << ", "
           << "\"ratio\": "       << std::fixed << std::setprecision(6)
                                  << r.ratio
           << "}";
        if(i + 1 < records.size())
        {
            os << ",";
        }
        os << "\n";
    }
    os << "  ]\n";
    os << "}\n";
}

}

int main()
{
    using namespace argmin;

    // Policy alias short-hand. kraft + filter_slsqp have only 3 wall
    // cells; nw_sqp + filter_nw_sqp have 4 each (kraft_slsqp doesn't
    // exercise HS007; nw_sqp adds HS007; both filter variants add HS043).
    // The dim parameters mirror what the wall TEST_CASE blocks pick up
    // from `Problem<>::problem_dimension`.

    using kraft_acc_3   = kraft_slsqp_policy_accurate<3>;
    using kraft_fast_3  = kraft_slsqp_policy_fast<3>;
    using kraft_acc_4   = kraft_slsqp_policy_accurate<4>;
    using kraft_fast_4  = kraft_slsqp_policy_fast<4>;

    using nw_acc_2      = nw_sqp_policy_accurate<2>;
    using nw_fast_2     = nw_sqp_policy_fast<2>;
    using nw_acc_3      = nw_sqp_policy_accurate<3>;
    using nw_fast_3     = nw_sqp_policy_fast<3>;
    using nw_acc_4      = nw_sqp_policy_accurate<4>;
    using nw_fast_4     = nw_sqp_policy_fast<4>;

    using fslsqp_acc_3  = filter_slsqp_policy_accurate<3>;
    using fslsqp_fast_3 = filter_slsqp_policy_fast<3>;
    using fslsqp_acc_4  = filter_slsqp_policy_accurate<4>;
    using fslsqp_fast_4 = filter_slsqp_policy_fast<4>;

    using fnw_acc_3     = filter_nw_sqp_policy_accurate<3>;
    using fnw_fast_3    = filter_nw_sqp_policy_fast<3>;
    using fnw_acc_4     = filter_nw_sqp_policy_accurate<4>;
    using fnw_fast_4    = filter_nw_sqp_policy_fast<4>;

    std::vector<record> records;
    records.reserve(static_cast<std::size_t>(15) * kRepetitions);

    // kraft_slsqp: 3 cells (HS071, HS026, HS028).
    // max_iters mirrors the in-tree wall TEST_CASE block exactly.
    run_cell<kraft_fast_4, kraft_acc_4, hs071<>>(
        records, "kraft_slsqp", "HS071", 200);
    run_cell<kraft_fast_3, kraft_acc_3, hs026<>>(
        records, "kraft_slsqp", "HS026", 50);
    run_cell<kraft_fast_3, kraft_acc_3, hs028<>>(
        records, "kraft_slsqp", "HS028", 200);

    // nw_sqp: 4 cells (HS071, HS026, HS007, HS028).
    // HS007 is the 2-dim probe that exercises hand-rolled main LS.
    run_cell<nw_fast_4, nw_acc_4, hs071<>>(
        records, "nw_sqp", "HS071", 200);
    run_cell<nw_fast_3, nw_acc_3, hs026<>>(
        records, "nw_sqp", "HS026", 50);
    run_cell<nw_fast_2, nw_acc_2, hs007<>>(
        records, "nw_sqp", "HS007", 50);
    run_cell<nw_fast_3, nw_acc_3, hs028<>>(
        records, "nw_sqp", "HS028", 200);

    // filter_slsqp: 4 cells (HS071, HS043, HS026, HS028).
    run_cell<fslsqp_fast_4, fslsqp_acc_4, hs071<>>(
        records, "filter_slsqp", "HS071", 200);
    run_cell<fslsqp_fast_4, fslsqp_acc_4, hs043<>>(
        records, "filter_slsqp", "HS043", 200);
    run_cell<fslsqp_fast_3, fslsqp_acc_3, hs026<>>(
        records, "filter_slsqp", "HS026", 200);
    run_cell<fslsqp_fast_3, fslsqp_acc_3, hs028<>>(
        records, "filter_slsqp", "HS028", 200);

    // filter_nw_sqp: 4 cells (HS071, HS043, HS026, HS028).
    run_cell<fnw_fast_4, fnw_acc_4, hs071<>>(
        records, "filter_nw_sqp", "HS071", 500);
    run_cell<fnw_fast_4, fnw_acc_4, hs043<>>(
        records, "filter_nw_sqp", "HS043", 500);
    run_cell<fnw_fast_3, fnw_acc_3, hs026<>>(
        records, "filter_nw_sqp", "HS026", 200);
    run_cell<fnw_fast_3, fnw_acc_3, hs028<>>(
        records, "filter_nw_sqp", "HS028", 200);

    emit_json(std::cout,
              capture_git_sha(),
              capture_timestamp_utc(),
              capture_hostname(),
              capture_cpu_model(),
              capture_cpu_freq_mhz(),
              capture_os_string(),
              records);
    return 0;
}
