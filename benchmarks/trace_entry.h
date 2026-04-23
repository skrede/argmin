#ifndef HPP_GUARD_NABLAPP_BENCHMARKS_TRACE_ENTRY_H
#define HPP_GUARD_NABLAPP_BENCHMARKS_TRACE_ENTRY_H

#include <cstdint>

namespace nablapp::bench
{

struct trace_entry
{
    int iter{};
    int f_evals{};
    int g_evals{};
    int c_evals{};
    int J_evals{};
    std::int64_t wall_us{};
    double f_current{};
    double f_best{};
    double accuracy{};
    double cv{};
    double step_norm{};
    double kkt_residual{};
};

}

#endif
