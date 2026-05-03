// publish_bench driver: publication-grade Dolan-More performance-profile
// measurement run.
//
// Consumes bench_config::publication(seed) and iterates the cross product
// (seed, solver, problem) in-process. Emits a single summary CSV plus one
// per-iter trace file per (solver, problem, seed) triple under a
// caller-supplied output directory.
//
// Usage:
//   ./publish_bench --seed-start <N> --seed-count <N> --output-dir <path>

#include "trace_entry.h"
#include "bench_config.h"
#include "bench_argmin.h"
#include "benchmark_result.h"
#include "problem_registry.h"

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

#ifdef ARGMIN_HAS_LIBCMAES
#include "bench_libcmaes.h"
#endif

#include <chrono>
#include <print>
#include <format>
#include <string>
#include <vector>
#include <fstream>
#include <charconv>
#include <cstdint>
#include <iostream>
#include <filesystem>
#include <string_view>

namespace
{

struct cli_args
{
    int seed_start{42};
    int seed_count{11};
    std::string output_dir;
    bool have_output_dir{false};
};

constexpr std::string_view trace_csv_header =
    "iter,f_evals,g_evals,c_evals,J_evals,wall_us,"
    "f_current,f_best,accuracy,cv,step_norm,kkt_residual";

constexpr int max_iterations = 10000;

void print_usage(std::ostream& os)
{
    os << "Usage: publish_bench --output-dir <path> "
          "[--seed-start <N>] [--seed-count <N>]\n"
       << "\n"
       << "Options:\n"
       << "  --output-dir <path>    Directory to write summary CSV + traces/ subdir (required).\n"
       << "  --seed-start <N>       First seed in the contiguous sweep (default 42).\n"
       << "  --seed-count <N>       Number of consecutive seeds to run (default 11).\n"
       << "  --help                 Print this message and exit.\n";
}

auto parse_int(std::string_view s, int& out) -> bool
{
    auto* first = s.data();
    auto* last = s.data() + s.size();
    auto [ptr, ec] = std::from_chars(first, last, out);
    return ec == std::errc{} && ptr == last;
}

auto parse_args(int argc, char** argv, cli_args& cli) -> int
{
    for(int i = 1; i < argc; ++i)
    {
        auto arg = std::string_view(argv[i]);
        if(arg == "--help" || arg == "-h")
        {
            print_usage(std::cout);
            return 0;
        }
        else if(arg == "--seed-start" && i + 1 < argc)
        {
            if(!parse_int(argv[++i], cli.seed_start))
            {
                std::cerr << "publish_bench: invalid --seed-start value\n";
                return 1;
            }
        }
        else if(arg == "--seed-count" && i + 1 < argc)
        {
            if(!parse_int(argv[++i], cli.seed_count))
            {
                std::cerr << "publish_bench: invalid --seed-count value\n";
                return 1;
            }
        }
        else if(arg == "--output-dir" && i + 1 < argc)
        {
            cli.output_dir = argv[++i];
            cli.have_output_dir = true;
        }
        else
        {
            std::cerr << "publish_bench: unknown or incomplete argument '"
                      << arg << "'\n";
            print_usage(std::cerr);
            return 1;
        }
    }

    if(!cli.have_output_dir)
    {
        std::cerr << "publish_bench: --output-dir is required\n";
        print_usage(std::cerr);
        return 1;
    }
    if(cli.seed_count <= 0)
    {
        std::cerr << "publish_bench: --seed-count must be >= 1\n";
        return 1;
    }
    return -1;  // sentinel: continue to main work
}

void write_trace_file(const std::filesystem::path& path,
                      const std::vector<argmin::bench::trace_entry>& trace)
{
    std::ofstream trace_out(path);
    if(!trace_out)
    {
        std::cerr << "publish_bench: failed to open trace file '"
                  << path.string() << "' for writing; skipping\n";
        return;
    }
    trace_out << trace_csv_header << '\n';
    for(const auto& e : trace)
    {
        trace_out << std::format(
            "{},{},{},{},{},{},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e},{:.15e}\n",
            e.iter, e.f_evals, e.g_evals, e.c_evals, e.J_evals, e.wall_us,
            e.f_current, e.f_best, e.accuracy, e.cv, e.step_norm, e.kkt_residual);
    }
}

}

int main(int argc, char** argv)
{
    cli_args cli;
    int parse_rc = parse_args(argc, argv, cli);
    if(parse_rc >= 0)
        return parse_rc;

    std::filesystem::path out_dir    = cli.output_dir;
    std::filesystem::path traces_dir = out_dir / "traces";
    std::filesystem::create_directories(traces_dir);

    std::ofstream summary_out(out_dir / "publish_summary.csv");
    if(!summary_out)
    {
        std::cerr << "publish_bench: failed to open summary CSV at '"
                  << (out_dir / "publish_summary.csv").string() << "'\n";
        return 1;
    }
    summary_out << argmin::bench::csv_header() << '\n';

    for(int si = 0; si < cli.seed_count; ++si)
    {
        std::uint64_t seed = static_cast<std::uint64_t>(cli.seed_start + si);
        auto cfg = argmin::bench::bench_config::publication(seed);

        std::vector<argmin::bench::benchmark_result>         results;
        std::vector<std::vector<argmin::bench::trace_entry>> traces;
        // Invariant: results[i] corresponds to traces[i]. Adapters that
        // produce no per-iter trace push an empty vector to keep indexing
        // aligned across libraries.

        // argmin solvers — iterate every registered problem and dispatch
        // every applicable policy through the existing solver-selection
        // helper. Trace collection is on under publication mode.
        argmin::bench::for_each_problem([&](std::string_view name, auto&& prob) {
            argmin::bench::run_all_argmin_solvers(
                name, prob, max_iterations, cfg.trace_enabled, results, traces,
                cfg, seed);
        });

        #ifdef ARGMIN_HAS_NLOPT
        nlopt::srand(static_cast<unsigned long>(seed));
        argmin::bench::run_nlopt_benchmarks(results, traces, cfg);
        #endif
        #ifdef ARGMIN_HAS_IPOPT
        argmin::bench::run_ipopt_benchmarks(results, traces, cfg);
        #endif
        #ifdef ARGMIN_HAS_CERES
        argmin::bench::run_ceres_benchmarks(results, traces, cfg);
        #endif
        #ifdef ARGMIN_HAS_DLIB
        argmin::bench::run_dlib_benchmarks(results, traces, cfg);
        #endif
        #ifdef ARGMIN_HAS_OPTIM
        argmin::bench::run_optim_benchmarks(results, traces, cfg);
        #endif
        #ifdef ARGMIN_HAS_LIBCMAES
        argmin::bench::run_libcmaes_benchmarks(results, traces, cfg);
        #endif

        for(std::size_t i = 0; i < results.size(); ++i)
        {
            summary_out << argmin::bench::csv_row(results[i]) << '\n';

            if(i < traces.size() && !traces[i].empty())
            {
                auto trace_path = traces_dir / std::format(
                    "{}_{}_seed{}.csv",
                    results[i].solver, results[i].problem, seed);
                write_trace_file(trace_path, traces[i]);
            }
        }

        std::println("seed {}: emitted {} summary rows", seed, results.size());
    }

    summary_out.close();
    std::println("done: wrote {} + {}/",
                 (out_dir / "publish_summary.csv").string(),
                 traces_dir.string());
    return 0;
}
