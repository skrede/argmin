#ifndef HPP_GUARD_NABLAPP_BENCHMARKS_BENCH_LIBCMAES_H
#define HPP_GUARD_NABLAPP_BENCHMARKS_BENCH_LIBCMAES_H

#include "bench_config.h"
#include "trace_entry.h"
#include "benchmark_result.h"

#include <vector>

namespace nablapp::bench
{

void run_libcmaes_benchmarks(std::vector<benchmark_result>& results,
                             std::vector<std::vector<trace_entry>>& traces,
                             const bench_config& config);

}

#endif
