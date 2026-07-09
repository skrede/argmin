// Benchmark driver for argmin.
//
// Custom main() with CSV output, CLI argument parsing, and optional
// convergence trace export.  Google Benchmark is used for timing
// internals only -- not BENCHMARK_MAIN.
//
// Usage:
//   ./argmin_bench [-o results.csv] [--no-trace] [--compare-trace]
//                   [--trace-dir traces/]

#include "bench_config.h"
#include "bench_argmin.h"
#include "benchmark_result.h"
#include "problem_registry.h"
#include "trace_entry.h"

#ifdef ARGMIN_HAS_NLOPT
#include "bench_nlopt.h"

#include <nlopt.hpp>
#endif
#ifdef ARGMIN_HAS_CERES
#include "bench_ceres.h"
#endif
#ifdef ARGMIN_HAS_DLIB
#include "bench_dlib.h"
#endif
#ifdef ARGMIN_HAS_IPOPT
#include "bench_ipopt.h"
#endif
#ifdef ARGMIN_HAS_OPTIM
#include "bench_optim.h"
#endif

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include "bench_print.h"
#include <string>
#include <string_view>
#include <vector>

namespace
{

struct driver_config
{
    std::string output_file{"results.csv"};
    std::string trace_dir{"traces"};
    bool trace_enabled{true};
    bool compare_trace{false};
    std::uint64_t seed{42};
};

auto parse_args(int argc, char** argv) -> driver_config
{
    driver_config cfg;
    for(int i = 1; i < argc; ++i)
    {
        auto arg = std::string_view(argv[i]);
        if((arg == "--output" || arg == "-o") && i + 1 < argc)
            cfg.output_file = argv[++i];
        else if(arg == "--no-trace")
            cfg.trace_enabled = false;
        else if(arg == "--compare-trace")
            cfg.compare_trace = true;
        else if(arg == "--trace-dir" && i + 1 < argc)
            cfg.trace_dir = argv[++i];
        else if(arg == "--seed" && i + 1 < argc)
            cfg.seed = std::stoull(argv[++i]);
    }
    return cfg;
}

void write_trace_csv(const std::filesystem::path& path,
                     const std::vector<argmin::bench::trace_entry>& trace)
{
    std::ofstream out(path);
    out << "iter,f_evals,g_evals,c_evals,J_evals,wall_us,"
           "f_current,f_best,accuracy,cv,step_norm,kkt_residual\n";
    for(const auto& t : trace)
    {
        out << std::format("{},{},{},{},{},{},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e}\n",
                           t.iter, t.f_evals, t.g_evals, t.c_evals, t.J_evals,
                           t.wall_us,
                           t.f_current, t.f_best, t.accuracy, t.cv,
                           t.step_norm, t.kkt_residual);
    }
}

void print_summary(const std::vector<argmin::bench::benchmark_result>& results,
                   std::chrono::microseconds elapsed)
{
    auto secs = static_cast<double>(elapsed.count()) / 1e6;
    argmin::bench::println("--- Benchmark Summary ---");
    argmin::bench::println("Total solver/problem pairs: {}", results.size());
    argmin::bench::println("Elapsed: {:.3f} s", secs);
}

constexpr int max_iterations = 10000;

}

int main(int argc, char** argv)
{
    auto cfg = parse_args(argc, argv);

    // Library-defaults bench_config recovers existing argmin_bench
    // behavior on the pre-existing column set; new columns
    // (seed, mode, solver_iters, c_evals, J_evals) populate from this
    // config + adapter-native counters.
    auto config = argmin::bench::bench_config::library_defaults();
    config.seed = cfg.seed;

    std::vector<argmin::bench::benchmark_result> results;
    std::vector<std::vector<argmin::bench::trace_entry>> traces;

    auto t0 = std::chrono::steady_clock::now();

    // Run argmin solvers on all registered problems.
    argmin::bench::for_each_problem([&](std::string_view name, auto&& prob) {
        argmin::bench::run_all_argmin_solvers(
            name, prob, max_iterations, cfg.trace_enabled, results, traces,
            config, cfg.seed);
    });

    // Run comparison library benchmarks (if detected at build time).
    // Each adapter receives the same `traces` out-parameter; the four
    // trace-capable adapters (argmin / NLopt / IPOPT / Ceres) populate
    // per-iter rows under config.trace_enabled, and the two summary-only
    // adapters (dlib / kthohr_optim) push empty trace vectors per result
    // row to preserve the results[i] <-> traces[i] index invariant. Under
    // argmin_bench's library_defaults config, every adapter pushes empty
    // vectors — no per-iter trace files are ever produced from this driver.
#ifdef ARGMIN_HAS_NLOPT
    nlopt::srand(static_cast<unsigned long>(cfg.seed));
    argmin::bench::run_nlopt_benchmarks(results, traces, config);
#endif
#ifdef ARGMIN_HAS_CERES
    argmin::bench::run_ceres_benchmarks(results, traces, config);
#endif
#ifdef ARGMIN_HAS_DLIB
    argmin::bench::run_dlib_benchmarks(results, traces, config);
#endif
#ifdef ARGMIN_HAS_IPOPT
    argmin::bench::run_ipopt_benchmarks(results, traces, config);
#endif
#ifdef ARGMIN_HAS_OPTIM
    argmin::bench::run_optim_benchmarks(results, traces, config);
#endif

    auto t1 = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

    // Write main results CSV.
    {
        std::ofstream csv(cfg.output_file);
        csv << argmin::bench::csv_header() << '\n';
        for(const auto& r : results)
            csv << argmin::bench::csv_row(r) << '\n';
    }

    // Write per-solver/problem convergence trace CSVs.
    if(cfg.trace_enabled)
    {
        std::filesystem::create_directories(cfg.trace_dir);
        for(std::size_t i = 0; i < results.size() && i < traces.size(); ++i)
        {
            if(traces[i].empty())
                continue;
            auto filename = std::format("{}_{}.csv", results[i].solver,
                                        results[i].problem);
            write_trace_csv(std::filesystem::path(cfg.trace_dir) / filename,
                            traces[i]);
        }
    }

    // Compare-trace mode: rerun each pair without traces and report overhead.
    if(cfg.compare_trace)
    {
        argmin::bench::println("\n--- Trace Overhead Comparison ---");
        std::vector<argmin::bench::benchmark_result> no_trace_results;
        std::vector<std::vector<argmin::bench::trace_entry>> no_trace_traces;

        argmin::bench::for_each_problem([&](std::string_view name, auto&& prob) {
            argmin::bench::run_all_argmin_solvers(
                name, prob, max_iterations, false, no_trace_results, no_trace_traces,
                config, cfg.seed);
        });

        for(std::size_t i = 0; i < results.size() && i < no_trace_results.size(); ++i)
        {
            if(results[i].library != "argmin")
                continue;
            auto delta = results[i].wall_time_us - no_trace_results[i].wall_time_us;
            argmin::bench::println("  {}/{}: {:+d} us overhead",
                         results[i].solver, results[i].problem, delta);
        }

        // Append no-trace results to CSV as well.
        {
            std::ofstream csv(cfg.output_file, std::ios::app);
            for(const auto& r : no_trace_results)
                csv << argmin::bench::csv_row(r) << '\n';
        }
    }

    print_summary(results, elapsed);
    return EXIT_SUCCESS;
}
