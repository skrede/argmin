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
//      overrides in alloc_trace_main.cpp. That sensor is process-wide and
//      catches heap traffic the Eigen gate cannot see (std::vector free-sets,
//      decorator std::function storage, and any aligned_malloc that reaches
//      the C allocator without tripping the eigen_assert path).
//
// Without ARGMIN_BENCH_TRACE_ALLOC every entry point is an inline no-op
// returning zero, so the library and its consumers see no behavior change.

#ifdef ARGMIN_BENCH_TRACE_ALLOC

#include <atomic>
#include <cstdio>
#include <cstddef>
#include <algorithm>

namespace argmin::detail::bench
{

// Storage lives in alloc_trace_main.cpp so a single definition backs every
// alloc-trace translation unit linked into the target.
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
// to arm the Eigen-native sensor. alloc_trace_main.cpp defines the same macro
// itself before its first Eigen include; the guard keeps the two definitions
// from clashing.
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

// Evaluate the gate for one policy probe and return a process exit code
// (0 pass, 1 fail).
//
// min_per_step encodes the reviewer-measured pre-hoist witness band: a policy
// expected to allocate must be seen to allocate at least this many times per
// armed step, proving the sensor is not blind. A value of 0 marks a policy
// expected to already be allocation-free, which is held to the
// zero-allocation gate immediately.
//
// Defining ARGMIN_ALLOC_GATE_EXPECT_ZERO flips every probe to the
// zero-allocation gate -- the post-hoist acceptance mode. The default leaves
// ARGMIN_ALLOC_GATE_EXPECT as the pre-fix witness, keeping the known
// allocating policies demonstrably observed while proving the instrument
// fires against the current code.
inline int evaluate_gate(const char* label, std::size_t armed_steps,
                         std::size_t min_per_step) noexcept
{
    const std::size_t eigen_c = read_eigen_malloc_count();
    const std::size_t c_alloc = read_c_alloc_count();
    const std::size_t observed = std::max(eigen_c, c_alloc);
    const double per_step = armed_steps
        ? static_cast<double>(observed) / static_cast<double>(armed_steps)
        : static_cast<double>(observed);

    // %lu with an explicit unsigned-long cast rather than %zu: some embedded
    // libcs (e.g. newlib-nano) do not implement the %z length modifier, and
    // this prints identically on a hosted libc.
    std::printf("  [alloc-gate] %-18s eigen_malloc=%lu c_alloc=%lu "
                "armed_steps=%lu per_step=%.2f\n",
                label, static_cast<unsigned long>(eigen_c),
                static_cast<unsigned long>(c_alloc),
                static_cast<unsigned long>(armed_steps), per_step);

#ifdef ARGMIN_ALLOC_GATE_EXPECT_ZERO
    if(observed != 0)
    {
        std::fprintf(stderr,
            "  [alloc-gate] %s FAIL: zero-alloc gate saw %lu allocations\n",
            label, static_cast<unsigned long>(observed));
        return 1;
    }
    std::printf("  [alloc-gate] %s PASS (zero-alloc gate)\n", label);
    return 0;
#else
    if(min_per_step == 0)
    {
        if(observed != 0)
        {
            std::fprintf(stderr,
                "  [alloc-gate] %s FAIL: policy expected allocation-free, "
                "saw %lu\n", label, static_cast<unsigned long>(observed));
            return 1;
        }
        std::printf("  [alloc-gate] %s PASS (allocation-free)\n", label);
        return 0;
    }
    if(per_step < static_cast<double>(min_per_step))
    {
        std::fprintf(stderr,
            "  [alloc-gate] %s FAIL: pre-fix witness expected >= %lu/step, "
            "saw %.2f/step -- gate may be blind\n",
            label, static_cast<unsigned long>(min_per_step), per_step);
        return 1;
    }
    std::printf("  [alloc-gate] %s PASS (pre-fix witness %.2f/step >= "
                "%lu/step)\n", label, per_step,
                static_cast<unsigned long>(min_per_step));
    return 0;
#endif
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

// No-op twin of the traced evaluate_gate, so measurement code (the on-device
// probe workload, the host micro drivers) compiles unchanged when built
// without ARGMIN_BENCH_TRACE_ALLOC. With no sensor there is nothing to
// evaluate, so the gate trivially passes.
inline int evaluate_gate(const char*, std::size_t, std::size_t) noexcept
{
    return 0;
}

}

#endif

#endif
