// Exact-bit golden pin for the real-time-path solver trajectories.
//
// This is a relative, same-target refactor guard: a code change must
// reproduce these exact IEEE-754 bit patterns on the SAME target the golden
// was captured on. It makes NO cross-compiler, cross-architecture, or
// cross-instantiation bit-identity claim -- a golden captured on one target is
// only ever enforced on a build whose target identity matches. On a
// non-matching target (or when no golden is present) the pin emits a visible
// "recapture needed" warning and does not assert, so it can neither pass
// silently as if it had asserted nor fail spuriously off its capturing target.
//
// Why exact bits and not a tolerance: a numerics-preserving refactor of the
// solver hot path must not move a single low bit of the trajectory. A
// near-equality comparison with any tolerance would wave through sub-ULP
// drift, which is precisely the failure this instrument exists to catch, so
// every comparison here is std::bit_cast<std::uint64_t> equality of the full
// double -- correct on signed zero and NaN payloads, which floating == would
// mishandle. Values are stored in the golden as hex uint64 so the committed
// artifact is exact and diffable; a decimal round-trip could not guarantee the
// bit exactness.
//
// Two modes share one translation unit:
//   * capture: set ARGMIN_BIT_IDENTITY_CAPTURE=<path> to run the trajectories
//     and write the golden (with this build's target identity) to <path>.
//   * assert (default): load the golden (from ARGMIN_BIT_IDENTITY_GOLDEN if
//     set, else oracles/solver_bit_identity_golden.csv), and on a matching
//     target assert every captured value bit-for-bit.
//
// Coverage: the four real-time-path solvers, each on a representative fixed-N
// fixture family -- nw_sqp on hs071 and sd012, filter_nw_sqp on hs071 and
// sd024, lm on a 2D Rosenbrock least-squares and sd_ls012 -- snapshotting the
// full iterate vector, the objective, and (where the state exposes them) the
// multipliers at three points spread across each trajectory.

#include "argmin/solver/options.h"
#include "argmin/solver/lm_policy.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/filter_nw_sqp_policy.h"

#include "argmin/test_functions/small_dense.h"
#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <map>
#include <string>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>

using namespace argmin;

namespace
{

using golden_map = std::map<std::string, std::uint64_t>;

struct loaded_golden
{
    std::string target;
    golden_map values;
};

// 2D Rosenbrock least-squares in fixed-N form (b = 5 per project convention),
// the real-time-path fixture for the lm hot loop. Mirrors the residual /
// jacobian interface the least-squares micro-probe drives, kept self-contained
// here so the pin depends on no benchmark translation unit.
struct rosenbrock_ls2
{
    static constexpr int problem_dimension = 2;

    [[nodiscard]] int dimension() const { return 2; }
    [[nodiscard]] int num_residuals() const { return 2; }

    [[nodiscard]] double value(const Eigen::Vector<double, 2>& x) const
    {
        const double r0 = 1.0 - x(0);
        const double r1 = std::sqrt(5.0) * (x(1) - x(0) * x(0));
        return 0.5 * (r0 * r0 + r1 * r1);
    }

    void residuals(const Eigen::Vector<double, 2>& x, Eigen::VectorXd& r) const
    {
        r(0) = 1.0 - x(0);
        r(1) = std::sqrt(5.0) * (x(1) - x(0) * x(0));
    }

    void jacobian(const Eigen::Vector<double, 2>& x, Eigen::MatrixXd& J) const
    {
        J(0, 0) = -1.0;
        J(0, 1) = 0.0;
        J(1, 0) = -2.0 * std::sqrt(5.0) * x(0);
        J(1, 1) = std::sqrt(5.0);
    }

    [[nodiscard]] Eigen::Vector<double, 2> initial_point() const
    {
        return Eigen::Vector<double, 2>{-1.0, 1.0};
    }
};

[[nodiscard]] std::uint64_t to_bits(double d)
{
    return std::bit_cast<std::uint64_t>(d);
}

[[nodiscard]] std::string to_hex(std::uint64_t v)
{
    std::ostringstream os;
    os << "0x" << std::hex << std::setw(16) << std::setfill('0') << v;
    return os.str();
}

// The running build's target identity: compiler id + major version,
// architecture, and build flavor, composed from predefined macros so capture
// and assert cannot disagree about what the running target is.
[[nodiscard]] std::string running_target()
{
    std::string compiler;
#if defined(__clang__)
    compiler = "clang-" + std::to_string(__clang_major__);
#elif defined(__GNUC__)
    compiler = "gcc-" + std::to_string(__GNUC__);
#elif defined(_MSC_VER)
    compiler = "msvc-" + std::to_string(_MSC_VER);
#else
    compiler = "unknown-compiler";
#endif

    std::string arch;
#if defined(__x86_64__) || defined(_M_X64)
    arch = "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    arch = "arm64";
#elif defined(__arm__) || defined(_M_ARM)
    arch = "arm32";
#else
    arch = "unknown-arch";
#endif

    std::string flavor;
#if defined(NDEBUG)
    flavor = "NDEBUG";
#else
    flavor = "DEBUG";
#endif

    return compiler + ";" + arch + ";" + flavor;
}

[[nodiscard]] std::string trim(const std::string& s)
{
    const auto b = s.find_first_not_of(" \t\r\n");
    if(b == std::string::npos) return {};
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

[[nodiscard]] std::string
key(const std::string& solver, const std::string& fixture,
    const std::string& tag, const std::string& quantity, long index)
{
    return solver + "|" + fixture + "|" + tag + "|" + quantity + "|" +
           std::to_string(index);
}

// Run a solver through a deterministic step trajectory and snapshot the full
// iterate, the objective, and (where the state exposes them) the multipliers
// at three points spread across the descent. The sequence -- three steps, on
// to ten, then the converged solve() continuation -- is identical run to run
// on a given binary, so the captured bits are a stable relative baseline.
template <typename Solver>
void capture_trajectory(golden_map& g, const std::string& solver_name,
                        const std::string& fixture, Solver& solver)
{
    auto snap = [&](const std::string& tag) {
        const auto& st = solver.state();
        const auto& x = st.x;
        for(long i = 0; i < static_cast<long>(x.size()); ++i)
            g[key(solver_name, fixture, tag, "x", i)] =
                to_bits(static_cast<double>(x[i]));
        g[key(solver_name, fixture, tag, "f", 0)] =
            to_bits(static_cast<double>(st.objective_value));
        // Capture multipliers only where the state exposes them as a vector
        // (the SQP families). The lm state's scalar `lambda` is the
        // Levenberg-Marquardt damping, not a multiplier estimate; it is
        // implicitly pinned through the iterate and objective already.
        if constexpr(requires { st.lambda.size(); st.lambda[0]; })
        {
            const auto& lam = st.lambda;
            for(long i = 0; i < static_cast<long>(lam.size()); ++i)
                g[key(solver_name, fixture, tag, "lambda", i)] =
                    to_bits(static_cast<double>(lam[i]));
        }
    };

    for(int i = 0; i < 3; ++i) solver.step();
    snap("step3");
    for(int i = 3; i < 10; ++i) solver.step();
    snap("step10");
    solver.solve();
    snap("final");
}

[[nodiscard]] solver_options<> sqp_opts()
{
    solver_options<> opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-8);
    opts.set_objective_threshold(1e-10);
    opts.set_step_threshold(1e-10);
    return opts;
}

[[nodiscard]] solver_options<> lm_opts()
{
    solver_options<> opts;
    opts.max_iterations = 200;
    opts.set_gradient_threshold(1e-12);
    opts.set_objective_threshold(1e-14);
    opts.set_step_threshold(1e-14);
    return opts;
}

template <typename Problem>
void run_nw(golden_map& g, const std::string& fixture)
{
    Problem problem;
    auto x0 = problem.initial_point();
    auto opts = sqp_opts();
    step_budget_solver solver{nw_sqp_policy<Problem::problem_dimension>{},
                              problem, x0, opts};
    capture_trajectory(g, "nw_sqp", fixture, solver);
}

template <typename Problem>
void run_filter_nw(golden_map& g, const std::string& fixture)
{
    Problem problem;
    auto x0 = problem.initial_point();
    auto opts = sqp_opts();
    step_budget_solver solver{filter_nw_sqp_policy<Problem::problem_dimension>{},
                              problem, x0, opts};
    capture_trajectory(g, "filter_nw_sqp", fixture, solver);
}

template <typename Problem>
void run_lm(golden_map& g, const std::string& fixture)
{
    Problem problem;
    auto x0 = problem.initial_point();
    auto opts = lm_opts();
    step_budget_solver solver{lm_policy<Problem::problem_dimension>{},
                              problem, x0, opts};
    capture_trajectory(g, "lm", fixture, solver);
}

[[nodiscard]] golden_map build_trajectories()
{
    golden_map g;
    run_nw<hs071<>>(g, "hs071");
    run_nw<sd012<>>(g, "sd012");
    run_filter_nw<hs071<>>(g, "hs071");
    run_filter_nw<sd024<>>(g, "sd024");
    run_lm<rosenbrock_ls2>(g, "rosenbrock2");
    run_lm<sd_ls012<>>(g, "sd_ls012");
    return g;
}

void write_golden(const std::string& path, const golden_map& g)
{
    std::ofstream out(path);
    out << "# argmin real-time-path solver trajectory exact-bit golden.\n";
    out << "#\n";
    out << "# Relative, same-target refactor guard: a code change must preserve\n";
    out << "# these IEEE-754 double bit patterns (hex uint64) on the SAME target\n";
    out << "# named below. No cross-compiler / cross-architecture /\n";
    out << "# cross-instantiation bit-identity is claimed or enforced.\n";
    out << "#\n";
    out << "# Authority: a locally-captured golden is the fast dev pre-check pin\n";
    out << "# used by the numerics-preserving plumbing work's per-site\n";
    out << "# re-assertion. The enforced golden of record is produced by running\n";
    out << "# this same capture mode on the authoritative build leg and\n";
    out << "# committing its output; the target-identity line then activates the\n";
    out << "# assertions only on the matching leg.\n";
    out << "#\n";
    out << "# captured (compile date " << __DATE__ << ")\n";
    out << "target_identity," << running_target() << "\n";
    for(const auto& [k, v] : g)
        out << k << ',' << to_hex(v) << '\n';
}

[[nodiscard]] loaded_golden load_golden(const std::string& path)
{
    loaded_golden lg;
    std::ifstream in(path);
    if(!in.is_open()) return lg;
    std::string line;
    while(std::getline(in, line))
    {
        if(line.empty() || line.front() == '#') continue;
        const auto comma = line.find(',');
        if(comma == std::string::npos) continue;
        const std::string k = trim(line.substr(0, comma));
        const std::string v = trim(line.substr(comma + 1));
        if(k == "target_identity")
        {
            lg.target = v;
            continue;
        }
        lg.values[k] = std::stoull(v, nullptr, 16);
    }
    return lg;
}

}

TEST_CASE("real-time-path solver trajectories are bit-identical to the "
          "same-target golden",
          "[bit-identity][refactor-guard]")
{
    const golden_map live = build_trajectories();
    const std::string target = running_target();

    // Capture mode: write the golden for this target and stop (no assertion).
    if(const char* cap = std::getenv("ARGMIN_BIT_IDENTITY_CAPTURE"))
    {
        write_golden(cap, live);
        WARN("Captured exact-bit golden for target '" << target << "' to '"
             << cap << "' (" << live.size() << " values).");
        SUCCEED("capture mode: golden written");
        return;
    }

    // Assert mode.
    std::string path = "oracles/solver_bit_identity_golden.csv";
    if(const char* p = std::getenv("ARGMIN_BIT_IDENTITY_GOLDEN")) path = p;

    const loaded_golden golden = load_golden(path);

    // No golden present yet: visible recapture prompt, no silent pass, no
    // spurious fail.
    if(golden.values.empty())
    {
        WARN("No bit-identity golden at '" << path << "' for target '" << target
             << "'. Run capture mode (ARGMIN_BIT_IDENTITY_CAPTURE=<path>) to "
                "create it; the pin is not enforced until it exists.");
        SUCCEED("no golden present: recapture needed on this target");
        return;
    }

    // Foreign target: the committed golden was captured elsewhere. Warn and do
    // not assert -- a local golden may legitimately differ from the
    // authoritative leg, so a mismatch here is not a refactor regression.
    if(golden.target != target)
    {
        WARN("Bit-identity golden was captured on target '" << golden.target
             << "' but this build is '" << target
             << "'. Skipping exact-bit assertions; recapture on this target to "
                "enforce the pin here.");
        SUCCEED("foreign target: golden not enforced on this leg");
        return;
    }

    // Matching target: assert every captured value bit-for-bit.
    for(const auto& [k, live_bits] : live)
    {
        const auto it = golden.values.find(k);
        INFO("live value has no golden entry: " << k);
        REQUIRE(it != golden.values.end());
        INFO("bit mismatch at " << k << " (golden " << to_hex(it->second)
             << " vs live " << to_hex(live_bits) << ")");
        CHECK(it->second == live_bits);
    }

    // The golden and the live trajectory must describe the same set of
    // quantities -- a shape change (dropped/added iterate, multiplier, or
    // snapshot) is itself a regression the pin should surface.
    CHECK(golden.values.size() == live.size());
}
