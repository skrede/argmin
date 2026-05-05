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

#ifdef ARGMIN_BENCH_TRACE_ALLOC
#include <Eigen/Core>
#endif

#include <atomic>
#include <cstddef>

namespace argmin::detail::bench
{

// Adopted from: Eigen 3.4 internal::set_is_malloc_allowed
//               (Eigen/src/Core/util/Memory.h).
// Reference: Eigen runtime malloc gate (EIGEN_RUNTIME_NO_MALLOC).
//
// argmin variant: header API + atomic counter declaration; the actual
//                 storage for g_alloc_count and the ::operator new
//                 override live in the per-policy micro_*_sqp.cpp
//                 translation units guarded by ARGMIN_BENCH_TRACE_ALLOC.
//                 arm/disarm toggle Eigen's runtime malloc-allowed gate
//                 (requires EIGEN_RUNTIME_NO_MALLOC at compile time for
//                 set_is_malloc_allowed to take effect).
#ifdef ARGMIN_BENCH_TRACE_ALLOC

extern std::atomic<std::size_t> g_alloc_count;

inline void arm_alloc_trace() noexcept
{
#ifdef EIGEN_RUNTIME_NO_MALLOC
    // Eigen's runtime malloc gate fires eigen_assert() the moment any
    // Eigen internal allocation happens. Combined with the bench-side
    // ::operator new override + counter, this catches both std::vector
    // / non-Eigen heap and Eigen's internal aligned-malloc surfaces.
    // Without EIGEN_RUNTIME_NO_MALLOC defined, the function set_is_malloc_allowed
    // is not declared by Eigen, so the call is gated out at compile time;
    // the bench-side counter alone is the active surface in that mode.
    Eigen::internal::set_is_malloc_allowed(false);
#endif
}

inline void disarm_alloc_trace() noexcept
{
#ifdef EIGEN_RUNTIME_NO_MALLOC
    Eigen::internal::set_is_malloc_allowed(true);
#endif
}

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
