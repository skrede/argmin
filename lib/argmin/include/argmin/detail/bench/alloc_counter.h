#ifndef HPP_GUARD_ARGMIN_DETAIL_BENCH_ALLOC_COUNTER_H
#define HPP_GUARD_ARGMIN_DETAIL_BENCH_ALLOC_COUNTER_H

// Allocation-tracing API skeleton for the micro-benchmark hot-loop
// no-allocation gate.
//
// Under ARGMIN_BENCH_TRACE_ALLOC=1 this header declares an atomic
// allocation counter and the arm/disarm pair that toggles Eigen's
// runtime malloc-allowed gate. Without the macro, all entry points
// are inline no-ops that return zero, so library consumers see no
// runtime behavior change.
//
// The actual ::operator new override translation unit and the
// Eigen::internal::set_is_malloc_allowed(false/true) calls are
// landed by the bench-side wiring; the header is the consumed API.

#include <atomic>
#include <cstddef>

namespace argmin::detail::bench
{

// Adopted from: Eigen 3.4 internal::set_is_malloc_allowed
//               (Eigen/src/Core/util/Memory.h).
// Reference: Eigen runtime malloc gate (EIGEN_RUNTIME_NO_MALLOC).
//
// argmin variant: header skeleton + atomic counter; the actual
//                 ::operator new override TU lives in the bench-only
//                 translation unit guarded by ARGMIN_BENCH_TRACE_ALLOC=1.
//                 Hot-loop assertion site lives in the per-policy
//                 micro_*_sqp benches.
#ifdef ARGMIN_BENCH_TRACE_ALLOC

extern std::atomic<std::size_t> g_alloc_count;

void arm_alloc_trace() noexcept;
void disarm_alloc_trace() noexcept;

inline void reset_alloc_count() noexcept
{
    g_alloc_count.store(0);
}

inline std::size_t read_alloc_count() noexcept
{
    return g_alloc_count.load();
}

#else

inline void arm_alloc_trace() noexcept {}
inline void disarm_alloc_trace() noexcept {}

inline void reset_alloc_count() noexcept {}

inline std::size_t read_alloc_count() noexcept
{
    return 0;
}

#endif

}

#endif
