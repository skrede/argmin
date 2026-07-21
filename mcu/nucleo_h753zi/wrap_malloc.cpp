// On-target allocation sensor for the NUCLEO-H753ZI proof.
//
// Backs alloc_counter.h's arm/reset/read surface on bare-metal newlib the way
// tests/unit/alloc_sensor_counters.cpp (with alloc_sensor_b_linux.cpp) backs it
// on glibc: it owns the single definition of the atomic counters and provides
// strong-symbol allocation observers. Two independent sensors, mirroring the host design:
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
// a reported zero is meaningless. This mirrors the host sensor's liveness
// canary (tests/unit/alloc_gate_test.cpp).

#define EIGEN_RUNTIME_NO_MALLOC
#define eigen_assert(x) \
    do { if(!(x)) ::argmin::detail::bench::on_eigen_malloc(); } while(0)

#include "argmin/detail/diagnostics/alloc_counter.h"

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

// Force ONE Eigen internal kernel temporary strictly above the pinned
// EIGEN_STACK_ALLOCATION_LIMIT (8192 B) and observe the alloca->heap fallback
// fire in an armed window, converting "did not overflow at these N" into "the
// guardrail demonstrably works". Sibling of run_alloc_sensor_canary:
// same arm -> deliberate op -> disarm -> read -> assert-nonzero shape.
//
// A 36x36 lower-triangular in-place solve routes its internal blocking
// temporary through Eigen's ei_declare_aligned_stack_constructed_variable
// ternary (Memory.h:755):
//   (sizeof(T)*SIZE <= EIGEN_STACK_ALLOCATION_LIMIT) ? alloca : aligned_malloc
// The temporary is a 36x36 double panel = 36*36*8 = 10368 B. Because
// 10368 > 8192, the ternary takes the aligned_malloc (heap) branch -- a real
// heap event the --wrap malloc + operator-new + eigen_assert sensors all
// register while armed. Host-verified with the EIGEN_ALLOCA recording shim:
// raw sizeof(T)*SIZE = 10368 B (the value the <= LIMIT test
// compares), comfortably over 8192 B, so the fallback is deterministic. On this
// arm64 host and the arm-none-eabi target EIGEN_CPUID is undefined, so Eigen's
// cache-blocking uses the same non-x86 defaults -> the host sizing is
// target-faithful.
//
// The operand matrices live in static (BSS) storage mapped in place, so the
// ONLY heap traffic inside the armed window is the ~10.4 KiB fallback transient
// -- well under the 24 KiB _Min_Heap_Size (stm32h753zi_flash.ld) and freed back
// before run_all_rt_policies (the limit is NOT raised). The probe runs as its
// own arm/reset/disarm window, fully outside the four RT windows'
// arm/reset/disarm regions (nucleo_main.cpp calls it between the canary and
// run_all_rt_policies), so it cannot perturb their 0.00/step counts.
//
// Returns 0 iff the guardrail fired (nonzero alloc observed), 1 otherwise (a
// zero here would mean the over-limit temporary was NOT sent to the heap --
// i.e. the guardrail is broken). observed_allocs receives the count.
namespace
{
constexpr int kOverflowN   = 36;   // 36*36*sizeof(double) = 10368 B > 8192 B
constexpr int kOverflowRhs = 8;
double g_overflow_a[kOverflowN * kOverflowN];
double g_overflow_b[kOverflowN * kOverflowRhs];
}

std::size_t overflow_probe_forced_bytes() noexcept
{
    return static_cast<std::size_t>(kOverflowN) * kOverflowN * sizeof(double);
}

int run_overflow_guardrail_probe(std::size_t& observed_allocs) noexcept
{
    Eigen::Map<Eigen::MatrixXd> A(g_overflow_a, kOverflowN, kOverflowN);
    Eigen::Map<Eigen::MatrixXd> B(g_overflow_b, kOverflowN, kOverflowRhs);

    // Setup OUTSIDE the armed window: a well-conditioned lower-triangular system
    // (diagonal shifted strictly positive) so the solve is finite. Writing into
    // the mapped BSS buffers allocates nothing.
    A.setRandom();
    A.diagonal().array() += static_cast<double>(kOverflowN);
    B.setRandom();

    argmin::detail::bench::reset_alloc_count();
    argmin::detail::bench::arm_alloc_trace();

    // The single deliberate over-limit op: its 10368 B kernel temporary must
    // fall back to the heap (> 8192 B limit) and be observed by the sensor.
    A.triangularView<Eigen::Lower>().solveInPlace(B);
    volatile double sink = B(0, 0);
    (void)sink;

    argmin::detail::bench::disarm_alloc_trace();
    observed_allocs = argmin::detail::bench::read_alloc_count();

    return observed_allocs != 0 ? 0 : 1;
}

}
