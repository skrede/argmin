// NUCLEO-H753ZI on-device allocation proof driver (MCU-04).
//
// Bare-metal, no RTOS. CMSIS startup runs SystemInit + C++ static ctors, then
// calls main(). This TU:
//   1. brings up ARM semihosting so printf reaches the attached debugger
//      console (--specs=rdimon.specs),
//   2. runs the mandatory blindness canary (wrap_malloc.cpp) -- a deliberate
//      allocation MUST be observed, else the reported zeros are a blind sensor,
//   3. drives the four RT-claimed policies through their fixed-N steady-state
//      windows (rt_probe_workload.h), each of which prints its allocs/step via
//      evaluate_gate, and
//   4. reports the _sbrk one-time-setup high-water bytes.
//
// The on-device allocs/step number this prints is captured by the operator
// (see README.md); the expected result is 0.00/step on every RT window,
// reproducing the Phase-60 host contract, with the canary proving the sensor
// is live.
//
// The two macros below must precede any Eigen include (transitively through
// rt_probe_workload.h -> alloc_counter.h) so the Eigen-native counting sensor
// is armed, exactly as benchmarks/alloc_trace_main.cpp sets them on the host.

#define EIGEN_RUNTIME_NO_MALLOC
#define eigen_assert(x) \
    do { if(!(x)) ::argmin::detail::bench::on_eigen_malloc(); } while(0)

#include "probe/rt_probe_workload.h"

#include <cstdio>
#include <cstddef>

namespace argmin::mcu
{
void usart3_console_init() noexcept;
int run_alloc_sensor_canary() noexcept;
int run_overflow_guardrail_probe(std::size_t& observed_allocs) noexcept;
std::size_t overflow_probe_forced_bytes() noexcept;
std::size_t sbrk_high_water_bytes() noexcept;
std::size_t sbrk_call_count() noexcept;
}

int main()
{
    argmin::mcu::usart3_console_init();

    std::printf("[argmin] NUCLEO-H753ZI on-device allocation proof\n");

    if(argmin::mcu::run_alloc_sensor_canary() != 0)
    {
        std::fprintf(stderr,
            "[argmin] BLINDNESS CANARY FAILED: sensor did not observe a "
            "deliberate allocation -- reported zeros are meaningless\n");
        for(;;) { /* halt: an unreliable sensor must not report a green proof */ }
    }
    std::printf("[argmin] blindness canary PASS (deliberate allocation observed)\n");

    // Over-limit guardrail probe: its own armed window, fully OUTSIDE the
    // four RT windows below, forcing an Eigen kernel temporary above the pinned
    // 8192 B EIGEN_STACK_ALLOCATION_LIMIT so the alloca->heap fallback fires.
    // This proves the guardrail works (a nonzero alloc in THIS window), while
    // the RT windows must remain 0.00/step (the probe does not perturb them).
    std::size_t overflow_allocs = 0;
    if(argmin::mcu::run_overflow_guardrail_probe(overflow_allocs) != 0)
    {
        std::fprintf(stderr,
            "[overflow-probe] FAILED: forced %lu B temporary > 8192 B limit but "
            "NO heap fallback observed -- guardrail did not fire\n",
            static_cast<unsigned long>(
                argmin::mcu::overflow_probe_forced_bytes()));
        for(;;) { /* halt: a non-firing guardrail must not report a green proof */ }
    }
    std::printf("[overflow-probe] forced %lu B temporary > 8192 B limit: heap "
                "fallback observed (%lu allocs) PASS\n",
                static_cast<unsigned long>(
                    argmin::mcu::overflow_probe_forced_bytes()),
                static_cast<unsigned long>(overflow_allocs));

    const int rc = argmin::mcu::run_all_rt_policies();

    std::printf("[argmin] _sbrk one-time setup high-water = %lu bytes "
                "(%lu calls)\n",
                static_cast<unsigned long>(argmin::mcu::sbrk_high_water_bytes()),
                static_cast<unsigned long>(argmin::mcu::sbrk_call_count()));
    std::printf("[argmin] RT-window gate result: %s\n",
                rc == 0 ? "PASS (0 allocs/step on all windows)"
                        : "FAIL (see per-window lines above)");

    for(;;) { /* proof complete; halt for the operator to read the console */ }
    return rc;
}
