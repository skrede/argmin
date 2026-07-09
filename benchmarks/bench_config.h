#ifndef HPP_GUARD_ARGMIN_BENCHMARKS_BENCH_CONFIG_H
#define HPP_GUARD_ARGMIN_BENCHMARKS_BENCH_CONFIG_H

// Shared run-mode configuration for the argmin benchmark adapters.
//
// `bench_config` carries tolerance / budget / trace / seed / repetition
// fields consumed by every comparator-library adapter (argmin, NLopt,
// IPOPT, Ceres, dlib, kthohr/optim). The default-constructed value
// (mode::library_defaults) reproduces the existing argmin_bench
// behavior on the pre-existing column set; the publication() factory
// returns the tightened tolerance / longer budget / per-iter trace
// configuration consumed by the second `publish_bench` binary.

#include <cmath>
#include <cstdint>

namespace argmin::bench
{

struct bench_config
{
    enum class mode : std::uint8_t { library_defaults, publication };

    mode the_mode{mode::library_defaults};
    double ftol_rel{1e-12};
    double xtol_rel{1e-12};
    double eps_feas{1e-8};
    int max_iter{10000};
    int max_f_evals{10000};
    double max_wall_time_s{600.0};
    bool trace_enabled{false};
    int trace_every_n{1};
    std::uint64_t seed{42};
    // NOTE: Dead in the publish_bench dispatch path. publish_driver.cpp,
    // bench_argmin.h, and bench_libcmaes.cpp do not consume this field
    // (verified 2026-05-01). Each (seed, problem, solver) tuple runs ONCE
    // per publish_bench invocation. Per-bench repetition is owned by the
    // micro_*.cpp binaries via their own local `reps` constants. The field
    // is retained for the bench_config API stability; do not assume it
    // means averaging in publish_summary.csv reports.
    int repetitions{1};

    [[nodiscard]] static auto library_defaults() -> bench_config
    {
        return bench_config{};
    }

    [[nodiscard]] static auto publication(std::uint64_t s = 42) -> bench_config
    {
        return bench_config{
            .the_mode = mode::publication,
            .ftol_rel = 1e-16,
            .xtol_rel = 1e-16,
            .eps_feas = 1e-8,
            .max_iter = 10000,
            .max_f_evals = 10000,
            .max_wall_time_s = 10.0,
            .trace_enabled = true,
            .trace_every_n = 1,
            .seed = s,
            .repetitions = 11,  // retained for API stability; not consumed by publish_bench (see field comment above)
        };
    }
};

[[nodiscard]] inline auto publication_feasible(double cv,
                                               const bench_config& config) -> bool
{
    return std::isfinite(cv) && cv <= config.eps_feas;
}

}

#endif
