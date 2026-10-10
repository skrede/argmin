#ifndef HPP_GUARD_ARGMIN_MCU_NUCLEO_H753ZI_DWT_CYCLE_CLOCK_H
#define HPP_GUARD_ARGMIN_MCU_NUCLEO_H753ZI_DWT_CYCLE_CLOCK_H

#include "stm32h7xx.h"

#include <cstdint>

namespace argmin::mcu
{

// Core-clock cycle counter: DWT CYCCNT, enabled through DEMCR.TRCENA and
// DWT_CTRL.CYCCNTENA (ARMv7-M Architecture Reference Manual, C1.8 "The Data
// Watchpoint and Trace unit"). The 32-bit counter wraps after 2^32 cycles; the
// unsigned difference in stop() is exact for any window shorter than that.
struct dwt_cycle_clock
{
    static constexpr bool enabled = true;

    dwt_cycle_clock()
        : start_{0}
    {}

    void enable()
    {
        DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
        // The Cortex-M7 DWT implements the CoreSight software lock: without a debugger
        // session, writes to DWT_CTRL are ignored until the lock access key is written.
        DWT->LAR = 0xC5ACCE55u;
        DWT->CYCCNT = 0;
        DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    }

    void start() { start_ = DWT->CYCCNT; }

    std::uint32_t stop() const { return DWT->CYCCNT - start_; }

private:
    std::uint32_t start_;
};

}

#endif
