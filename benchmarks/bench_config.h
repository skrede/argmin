#ifndef HPP_GUARD_NABLAPP_BENCHMARKS_BENCH_CONFIG_H
#define HPP_GUARD_NABLAPP_BENCHMARKS_BENCH_CONFIG_H

// Shared run-mode configuration for the nablapp benchmark adapters.
//
// `bench_config` carries tolerance / budget / trace / seed / repetition
// fields consumed by every comparator-library adapter (nablapp, NLopt,
// IPOPT, Ceres, dlib, kthohr/optim). The default-constructed value
// (mode::library_defaults) reproduces the existing nablapp_bench
// behavior on the pre-existing column set; the publication() factory
// returns the tightened tolerance / longer budget / per-iter trace
// configuration consumed by the second `publish_bench` binary.

#include <cstdint>

namespace nablapp::bench
{

struct bench_config
{
    enum class mode { library_defaults, publication };

    mode the_mode{mode::library_defaults};
    double ftol_rel{1e-12};
    double xtol_rel{1e-12};
    int max_iter{10000};
    int max_f_evals{10000};
    double max_wall_time_s{600.0};
    bool trace_enabled{false};
    int trace_every_n{1};
    std::uint64_t seed{42};
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
            .max_iter = 10000,
            .max_f_evals = 10000,
            .max_wall_time_s = 10.0,
            .trace_enabled = true,
            .trace_every_n = 1,
            .seed = s,
            .repetitions = 11,
        };
    }
};

}

#endif
