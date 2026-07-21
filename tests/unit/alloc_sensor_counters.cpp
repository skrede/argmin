// Single definition of the allocation-sensor counters, always linked into the
// alloc-gate test executable. Sensor A (the Eigen-native counting eigen_assert
// in the diagnostics header) increments g_eigen_malloc_count through
// on_eigen_malloc on every platform. Sensor B (the glibc __libc_* overrides in
// alloc_sensor_b_linux.cpp) feeds g_c_alloc_count and is linked only on Linux;
// where it is absent g_c_alloc_count simply stays zero and the combined read
// degrades to Sensor A.

#include <atomic>
#include <cstddef>

namespace argmin::detail::bench
{

std::atomic<std::size_t> g_eigen_malloc_count{0};
std::atomic<std::size_t> g_c_alloc_count{0};
std::atomic<bool> g_alloc_trace_armed{false};

}
