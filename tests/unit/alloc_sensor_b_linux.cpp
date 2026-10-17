// Sensor B: strong-symbol overrides of the C allocator entry points, feeding
// the C-allocator counter while the trace is armed. This catches heap traffic
// the Eigen sensor cannot see (std::vector free-sets, decorator std::function
// storage, and any aligned_malloc that reaches the C allocator without tripping
// the eigen_assert path).
//
// The overrides forward to glibc's __libc_* implementations, which avoids the
// dlsym(RTLD_NEXT, ...) bootstrap recursion and keeps the allocator family
// mutually consistent (every allocation and free routes through the same
// arena). This is a glibc-only translation unit; it is compiled into the gate
// target only under the CMake Linux platform guard -- there is deliberately no
// platform preprocessor guard inside this source, the build system is the sole
// gate.

#include <atomic>
#include <cerrno>
#include <cstddef>

namespace argmin::detail::bench
{

extern std::atomic<std::size_t> g_c_alloc_count;
extern std::atomic<bool> g_alloc_trace_armed;

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
