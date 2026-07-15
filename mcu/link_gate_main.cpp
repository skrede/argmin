// arm-none-eabi + newlib-nano -fno-exceptions -fno-rtti cross-compile-link
// gate (build-only, no hardware, no board-specific code).
//
// This TU is compiled and LINKED with -fno-exceptions -fno-rtti under
// --specs=nano.specs --specs=nosys.specs for Cortex-M7 hard-float. It forces
// the four real-time-claimed solver policies through step_budget_solver's
// init/step/reset/solve template bodies (via rt_probe_workload.h). A single
// surviving throw in any instantiated path, or an unresolved newlib-nano C++
// runtime symbol (__cxa_* / _Unwind_*) under this exact multilib, fails the
// link -- which is precisely the newlib-nano -fno-exceptions link behavior
// MCU-02 gates, isolated from every firmware-image concern (no linker script,
// no startup, no CMSIS; nosys.specs supplies the weak _sbrk/_write stubs).
//
// ARGMIN_BENCH_TRACE_ALLOC is intentionally NOT defined here: alloc_counter.h's
// arm/reset/read/gate entry points are inline no-ops, so this binary is a pure
// instantiation + execution surface. Convergence is not checked; the exit
// status is the workload rc truncated to a byte so the calls are not elided.

#include "probe/rt_probe_workload.h"

int main()
{
    return argmin::mcu::run_all_rt_policies() & 0xff;
}
