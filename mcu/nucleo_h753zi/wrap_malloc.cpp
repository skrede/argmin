// On-target allocation sensor for the NUCLEO-H753ZI proof (MCU-03).
//
// Backs alloc_counter.h's arm/reset/read surface on bare-metal newlib the way
// benchmarks/alloc_trace_main.cpp backs it on glibc: it owns the single
// definition of the atomic counters and provides strong-symbol allocation
// observers. Two independent sensors, mirroring the host design:
//
//   1. malloc-family + operator new counters (PRIMARY per-step event count).
//      Every allocation event while the trace is armed increments
//      g_c_alloc_count -- this is the number evaluate_gate() reads as
//      allocs/step, matching the host semantics of "count every malloc, freed
//      or not." The malloc family is intercepted with the linker's --wrap
//      (see nucleo_h753zi/CMakeLists.txt), which catches Eigen's aligned
//      allocator and any C-library-internal path; operator new/delete are
//      overridden directly and forward to __real_malloc/__real_free so the C++
//      path is counted exactly once (never double-counted through __wrap_malloc).
//
//   2. _sbrk high-water (SECONDARY corroborator + hard bound). _sbrk is the
//      choke point every allocator path eventually crosses to grow the heap;
//      its byte high-water is the "one-time setup" number the report prints,
//      and its fixed _eheap ceiling makes an unexpected heap growth a bounded,
//      deterministic failure rather than a stack collision. It does NOT bump
//      the per-step event counter (that would double-count against the malloc
//      family), it tracks bytes and call count separately.
//
// The mandatory blindness canary (run_alloc_sensor_canary) proves the sensor
// is live: a deliberate allocation inside an armed window MUST register, else
// a reported zero is meaningless. This mirrors alloc_trace_main.cpp's canary.

#define EIGEN_RUNTIME_NO_MALLOC
#define eigen_assert(x) \
    do { if(!(x)) ::argmin::detail::bench::on_eigen_malloc(); } while(0)

#include "argmin/detail/bench/alloc_counter.h"

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cerrno>
#include <atomic>

// The linker script provides these; see stm32h753zi_flash.ld.
extern "C" char end;      // heap start
extern "C" char _eheap;   // fixed heap ceiling

namespace argmin::detail::bench
{
// Single definition of the counters alloc_counter.h declares extern, so one
// instance backs every alloc-trace TU linked into the image.
std::atomic<std::size_t> g_eigen_malloc_count{0};
std::atomic<std::size_t> g_c_alloc_count{0};
std::atomic<bool> g_alloc_trace_armed{false};
}

namespace
{
inline bool armed() noexcept
{
    return argmin::detail::bench::g_alloc_trace_armed.load(
        std::memory_order_relaxed);
}

inline void count_event() noexcept
{
    if(armed())
        argmin::detail::bench::g_c_alloc_count.fetch_add(
            1, std::memory_order_relaxed);
}

// _sbrk bookkeeping (secondary corroborator).
char* g_heap_end = nullptr;
std::size_t g_sbrk_high_water = 0;
std::size_t g_sbrk_calls = 0;
}

extern "C"
{
// --- malloc family, intercepted via -Wl,--wrap (PRIMARY event counter) ------
void* __real_malloc(std::size_t);
void  __real_free(void*);
void* __real_calloc(std::size_t, std::size_t);
void* __real_realloc(void*, std::size_t);

void* __wrap_malloc(std::size_t n)          { count_event(); return __real_malloc(n); }
void  __wrap_free(void* p)                  { __real_free(p); }
void* __wrap_calloc(std::size_t a, std::size_t b) { count_event(); return __real_calloc(a, b); }
void* __wrap_realloc(void* p, std::size_t n){ count_event(); return __real_realloc(p, n); }

// --- _sbrk high-water (SECONDARY, bounded by _eheap) ------------------------
void* _sbrk(std::ptrdiff_t incr)
{
    if(g_heap_end == nullptr)
        g_heap_end = &end;

    char* prev = g_heap_end;
    // Fixed-ceiling bound: deterministic, catches any unexpected growth
    // (stronger than a live-SP collision check for a zero-alloc proof).
    if(g_heap_end + incr > &_eheap)
    {
        errno = ENOMEM;
        return reinterpret_cast<void*>(-1);
    }
    g_heap_end += incr;
    ++g_sbrk_calls;
    const std::size_t used = static_cast<std::size_t>(g_heap_end - &end);
    if(used > g_sbrk_high_water)
        g_sbrk_high_water = used;
    return prev;
}
}

// __libc_init_array (called by the CMSIS Reset_Handler to run C++ static
// constructors) references _init/_fini, normally supplied by crti.o/crtn.o.
// Under -nostartfiles those CRT objects are not linked, so provide empty
// stubs -- the .init_array/.fini_array sections carry the actual ctor/dtor
// lists and are walked by __libc_init_array itself.
extern "C" void _init(void) {}
extern "C" void _fini(void) {}

// --- operator new/delete (PRIMARY C++ path, bypasses __wrap_malloc) ---------
// Forward to __real_malloc/__real_free (not the wrapped malloc) so the C++
// allocation is counted once here, never twice.
void* operator new(std::size_t n)            { count_event(); return __real_malloc(n); }
void* operator new[](std::size_t n)          { count_event(); return __real_malloc(n); }
void  operator delete(void* p) noexcept      { __real_free(p); }
void  operator delete[](void* p) noexcept    { __real_free(p); }
void  operator delete(void* p, std::size_t) noexcept   { __real_free(p); }
void  operator delete[](void* p, std::size_t) noexcept { __real_free(p); }

namespace argmin::mcu
{

// Secondary-sensor accessors for the report.
std::size_t sbrk_high_water_bytes() noexcept { return g_sbrk_high_water; }
std::size_t sbrk_call_count() noexcept       { return g_sbrk_calls; }

// Mandatory blindness canary: a deliberate allocation inside an armed window
// must be observed by the primary counter, else a reported zero is a blind
// sensor, not a real zero. Returns 0 on success (sensor lives), 1 on blindness.
int run_alloc_sensor_canary() noexcept
{
    argmin::detail::bench::reset_alloc_count();
    argmin::detail::bench::arm_alloc_trace();

    // Two deliberate allocations exercising both sensors: a C malloc (caught
    // by __wrap_malloc) and an Eigen heap vector (caught by operator new and
    // the EIGEN_RUNTIME_NO_MALLOC eigen_assert hook) -- the exact traffic a
    // real hot-loop leak would produce.
    void* raw = std::malloc(64);          // -> __wrap_malloc, counted
    Eigen::VectorXd v(37);                // -> operator new + eigen gate, counted
    v.setConstant(1.0);
    volatile double sink = v.sum();
    (void)sink;

    argmin::detail::bench::disarm_alloc_trace();
    const std::size_t observed = argmin::detail::bench::read_alloc_count();
    std::free(raw);                       // -> __wrap_free, disarmed (uncounted)

    return observed != 0 ? 0 : 1;
}

}
