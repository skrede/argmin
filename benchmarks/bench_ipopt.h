#ifndef HPP_GUARD_NABLAPP_BENCHMARKS_BENCH_IPOPT_H
#define HPP_GUARD_NABLAPP_BENCHMARKS_BENCH_IPOPT_H

#include "bench_config.h"
#include "benchmark_result.h"

#include <vector>

namespace nablapp::bench
{

void run_ipopt_benchmarks(std::vector<benchmark_result>& results, const bench_config& config);

}

#endif
