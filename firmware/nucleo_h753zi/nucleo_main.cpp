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

extern "C" void initialise_monitor_handles(void);

namespace argmin::firmware
{
int run_alloc_sensor_canary() noexcept;
std::size_t sbrk_high_water_bytes() noexcept;
std::size_t sbrk_call_count() noexcept;
}

int main()
{
    initialise_monitor_handles();

    std::printf("[argmin] NUCLEO-H753ZI on-device allocation proof\n");

    if(argmin::firmware::run_alloc_sensor_canary() != 0)
    {
        std::fprintf(stderr,
            "[argmin] BLINDNESS CANARY FAILED: sensor did not observe a "
            "deliberate allocation -- reported zeros are meaningless\n");
        for(;;) { /* halt: an unreliable sensor must not report a green proof */ }
    }
    std::printf("[argmin] blindness canary PASS (deliberate allocation observed)\n");

    const int rc = argmin::firmware::run_all_rt_policies();

    std::printf("[argmin] _sbrk one-time setup high-water = %zu bytes "
                "(%zu calls)\n",
                argmin::firmware::sbrk_high_water_bytes(),
                argmin::firmware::sbrk_call_count());
    std::printf("[argmin] RT-window gate result: %s\n",
                rc == 0 ? "PASS (0 allocs/step on all windows)"
                        : "FAIL (see per-window lines above)");

    for(;;) { /* proof complete; halt for the operator to read the console */ }
    return rc;
}
