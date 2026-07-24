#ifndef HPP_GUARD_ARGMIN_DETAIL_DIAGNOSTICS_ALLOC_COUNTER_H
#define HPP_GUARD_ARGMIN_DETAIL_DIAGNOSTICS_ALLOC_COUNTER_H

// Allocation-tracing gate for the micro-benchmark real-time hot loops.
//
// Under ARGMIN_BENCH_TRACE_ALLOC this header wires two independent
// allocation sensors:
//
//   1. An Eigen-native counting gate. When the consuming translation unit
//      is compiled with EIGEN_RUNTIME_NO_MALLOC, Eigen routes every internal
//      aligned_malloc through eigen_assert(is_malloc_allowed()). This header
//      defines eigen_assert as a report-and-continue hook that increments the
//      Eigen counter instead of aborting, so the gate stays live even in an
//      -O3 -march=native NDEBUG build (a user-defined eigen_assert is honored
//      regardless of NDEBUG). For the gate to observe a TU's solver
//      allocations this header must be included before any Eigen header in
//      that TU.
//
//   2. A C-allocator counter fed by the strong-symbol malloc / posix_memalign
//      overrides a linking consumer provides (tests/unit/alloc_sensor_counters.cpp
//      and alloc_sensor_b_linux.cpp on host, mcu/nucleo_h753zi/wrap_malloc.cpp on
//      target). That sensor is process-wide and
//      catches heap traffic the Eigen gate cannot see (std::vector free-sets,
//      decorator std::function storage, and any aligned_malloc that reaches
//      the C allocator without tripping the eigen_assert path).
//
// Without ARGMIN_BENCH_TRACE_ALLOC every entry point is an inline no-op
// returning zero, so the library and its consumers see no behavior change.

#ifdef ARGMIN_BENCH_TRACE_ALLOC

#include <atomic>
#include <cstddef>
#include <algorithm>

namespace argmin::detail::bench
{

// Storage is defined by exactly one linked consumer TU (tests/unit/alloc_sensor_counters.cpp
// on host, mcu/nucleo_h753zi/wrap_malloc.cpp on target) so a single definition
// backs every alloc-trace translation unit linked into the target.
extern std::atomic<std::size_t> g_eigen_malloc_count;
extern std::atomic<std::size_t> g_c_alloc_count;
extern std::atomic<bool> g_alloc_trace_armed;

// Report-and-continue hook invoked by the eigen_assert override below. It
// counts and returns; Eigen then proceeds with the allocation, which the
// C-allocator sensor also observes.
inline void on_eigen_malloc() noexcept
{
    g_eigen_malloc_count.fetch_add(1, std::memory_order_relaxed);
}

}

// Route Eigen's runtime malloc gate into on_eigen_malloc. Defined here so a
// consuming TU only needs to include this header (ahead of any Eigen header)
// to arm the Eigen-native sensor. A consumer that defines its own eigen_assert
// before its first Eigen include is respected; the #ifndef guard keeps the two
// definitions from clashing.
#ifndef eigen_assert
#define eigen_assert(x) \
    do { if(!(x)) ::argmin::detail::bench::on_eigen_malloc(); } while(0)
#endif

// Include order note: Eigen (third-party) is pulled in here, after the std
// headers and the eigen_assert definition, because the counting eigen_assert
// and on_eigen_malloc declaration must both precede any Eigen code.
#include <Eigen/Core>

namespace argmin::detail::bench
{

inline void arm_alloc_trace() noexcept
{
    g_alloc_trace_armed.store(true, std::memory_order_relaxed);
#ifdef EIGEN_RUNTIME_NO_MALLOC
    Eigen::internal::set_is_malloc_allowed(false);
#endif
}

inline void disarm_alloc_trace() noexcept
{
#ifdef EIGEN_RUNTIME_NO_MALLOC
    Eigen::internal::set_is_malloc_allowed(true);
#endif
    g_alloc_trace_armed.store(false, std::memory_order_relaxed);
}

inline void reset_alloc_count() noexcept
{
    g_eigen_malloc_count.store(0, std::memory_order_relaxed);
    g_c_alloc_count.store(0, std::memory_order_relaxed);
}

inline std::size_t read_eigen_malloc_count() noexcept
{
    return g_eigen_malloc_count.load(std::memory_order_relaxed);
}

inline std::size_t read_c_alloc_count() noexcept
{
    return g_c_alloc_count.load(std::memory_order_relaxed);
}

// Combined view: the larger of the two sensors. Either sensor observing
// traffic is sufficient evidence of a hot-loop allocation.
inline std::size_t read_alloc_count() noexcept
{
    return std::max(read_eigen_malloc_count(), read_c_alloc_count());
}

}

#else

#include <cstddef>

namespace argmin::detail::bench
{

inline void on_eigen_malloc() noexcept {}
inline void arm_alloc_trace() noexcept {}
inline void disarm_alloc_trace() noexcept {}
inline void reset_alloc_count() noexcept {}
inline std::size_t read_eigen_malloc_count() noexcept { return 0; }
inline std::size_t read_c_alloc_count() noexcept { return 0; }
inline std::size_t read_alloc_count() noexcept { return 0; }

}

#endif

#endif
