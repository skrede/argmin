// Shared entry point for the allocation-gate micro-benchmark targets.
//
// Every alloc-trace target links this translation unit against exactly one
// micro_*.cpp driver. This TU owns three things the driver relies on:
//
//   1. The counting eigen_assert + EIGEN_RUNTIME_NO_MALLOC setup, defined
//      strictly before the first Eigen include so Eigen's runtime malloc gate
//      routes into the report-and-continue hook rather than aborting.
//   2. The single definition of the atomic counters, plus strong-symbol
//      overrides of the C allocator entry points (malloc / calloc / realloc /
//      free / posix_memalign / aligned_alloc / memalign). The overrides count
//      all heap traffic while the trace is armed -- including the Eigen
//      aligned_malloc traffic the old global-operator-new gate was blind to.
//   3. main(), which first runs a self-contained canary (an Eigen allocation
//      inside an armed region must be counted, not silently passed) and then
//      dispatches to the driver's argmin_alloc_trace_probe().
//
// The overrides forward to glibc's __libc_* implementations, which avoids the
// dlsym(RTLD_NEXT, ...) bootstrap recursion and keeps the allocator family
// mutually consistent (every allocation and free routes through the same
// arena). This is a Linux/glibc bench-only TU; it is never linked into the
// library INTERFACE target or a consumer.

// These two defines must precede any Eigen include (directly or transitively
// through alloc_counter.h). Kept here explicitly, ahead of the includes, so
// the counting gate is unambiguous in the shipping-config Release build.
#define EIGEN_RUNTIME_NO_MALLOC
#define eigen_assert(x) \
    do { if(!(x)) ::argmin::detail::bench::on_eigen_malloc(); } while(0)

#include "argmin/detail/bench/alloc_counter.h"

#include <Eigen/Dense>

#include <cerrno>
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <atomic>

// Driver-provided probe. Each micro_*.cpp compiled under ARGMIN_BENCH_TRACE_ALLOC
// defines exactly one of these; the linker binds the single instance.
int argmin_alloc_trace_probe();

namespace argmin::detail::bench
{

std::atomic<std::size_t> g_eigen_malloc_count{0};
std::atomic<std::size_t> g_c_alloc_count{0};
std::atomic<bool> g_alloc_trace_armed{false};

}

extern "C"
{
// glibc internal allocator entry points. Declared here because no public
// header exposes them; forwarding to these avoids the dlsym bootstrap
// recursion a RTLD_NEXT interposer would hit.
void* __libc_malloc(std::size_t);
void __libc_free(void*);
void* __libc_calloc(std::size_t, std::size_t);
void* __libc_realloc(void*, std::size_t);
void* __libc_memalign(std::size_t, std::size_t);
}

namespace
{

// Count one allocation iff the trace is currently armed. Constant-initialized
// atomics make this safe for allocations that occur before dynamic init.
inline void bump_c_alloc() noexcept
{
    if(argmin::detail::bench::g_alloc_trace_armed.load(std::memory_order_relaxed))
        argmin::detail::bench::g_c_alloc_count.fetch_add(
            1, std::memory_order_relaxed);
}

}

extern "C" void* malloc(std::size_t n)
{
    bump_c_alloc();
    return __libc_malloc(n);
}

extern "C" void* calloc(std::size_t nmemb, std::size_t size)
{
    bump_c_alloc();
    return __libc_calloc(nmemb, size);
}

extern "C" void* realloc(void* p, std::size_t n)
{
    bump_c_alloc();
    return __libc_realloc(p, n);
}

extern "C" void free(void* p)
{
    __libc_free(p);
}

extern "C" int posix_memalign(void** out, std::size_t alignment, std::size_t n)
{
    bump_c_alloc();
    void* p = __libc_memalign(alignment, n);
    if(!p)
        return ENOMEM;
    *out = p;
    return 0;
}

extern "C" void* aligned_alloc(std::size_t alignment, std::size_t n)
{
    bump_c_alloc();
    return __libc_memalign(alignment, n);
}

extern "C" void* memalign(std::size_t alignment, std::size_t n)
{
    bump_c_alloc();
    return __libc_memalign(alignment, n);
}

namespace
{

// Defeat dead-store elimination on the canary allocation.
inline void escape(void* p) noexcept
{
    asm volatile("" : : "g"(p) : "memory");
}

// S1 canary: an Eigen allocation inside an armed region must be counted by at
// least one sensor. This is the exact false-negative the old global-new gate
// let pass in Release.
int run_canary()
{
    argmin::detail::bench::reset_alloc_count();
    argmin::detail::bench::arm_alloc_trace();
    Eigen::VectorXd v(100);
    v.setConstant(1.0);
    escape(v.data());
    argmin::detail::bench::disarm_alloc_trace();

    const std::size_t eigen_c = argmin::detail::bench::read_eigen_malloc_count();
    const std::size_t c_alloc = argmin::detail::bench::read_c_alloc_count();
    std::printf("  [alloc-gate] canary VectorXd(100)   eigen_malloc=%zu "
                "c_alloc=%zu\n", eigen_c, c_alloc);
    if(eigen_c == 0 && c_alloc == 0)
    {
        std::fprintf(stderr,
            "  [alloc-gate] canary FAIL: armed Eigen allocation went "
            "uncounted -- gate is blind\n");
        return 1;
    }
    std::printf("  [alloc-gate] canary PASS (armed Eigen allocation observed)\n");
    return 0;
}

}

int main()
{
    if(run_canary() != 0)
        return 1;
    return argmin_alloc_trace_probe();
}
