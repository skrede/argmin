#ifndef HPP_GUARD_NABLAPP_BENCHMARKS_BENCH_CERES_H
#define HPP_GUARD_NABLAPP_BENCHMARKS_BENCH_CERES_H

#include "bench_config.h"
#include "benchmark_result.h"

#include <vector>

namespace nablapp::bench
{

void run_ceres_benchmarks(std::vector<benchmark_result>& results, const bench_config& config);

}

#endif
