#ifndef HPP_GUARD_NABLAPP_BENCHMARKS_BENCH_NLOPT_H
#define HPP_GUARD_NABLAPP_BENCHMARKS_BENCH_NLOPT_H

#include "bench_config.h"
#include "benchmark_result.h"

#include <vector>

namespace nablapp::bench
{

void run_nlopt_benchmarks(std::vector<benchmark_result>& results, const bench_config& config);

}

#endif
