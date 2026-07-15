// Minimal USART3 console for headless capture over the ST-Link VCP.
//
// The NUCLEO-144 ST-Link virtual COM port is wired to USART3 (PD8=TX, PD9=RX)
// by the on-board solder bridges, so routing printf out USART3 makes the probe
// report appear on the host's /dev/ttyACM0 with no debugger/semihosting/OpenOCD
// -- the UART fallback mcu/README.md anticipates. This replaces the
// --specs=rdimon.specs semihosting path (which needs an attached debugger).
//
// No HAL: direct register writes only. After the CMSIS SystemInit (which does
// not configure the PLL), the H753 runs on HSI = 64 MHz, so PCLK1 = 64 MHz and
// USART3's default kernel clock (rcc_pclk1) = 64 MHz; BRR = 64e6 / 115200 ~= 556
// for oversampling-by-16. A strong _write override sends stdout/stderr bytes
// out USART3, so std::printf (and evaluate_gate's output) reach the VCP.

#include "stm32h7xx.h"

#include <cstddef>

namespace argmin::mcu
{

void usart3_console_init() noexcept
{
    // Clocks: GPIOD on AHB4, USART3 on APB1. Read back to ensure the enable
    // has taken effect before touching the peripherals.
    RCC->AHB4ENR |= RCC_AHB4ENR_GPIODEN;
    (void)RCC->AHB4ENR;
    RCC->APB1LENR |= RCC_APB1LENR_USART3EN;
    (void)RCC->APB1LENR;

    // PD8 = USART3_TX, alternate function 7. (RX/PD9 not needed for output.)
    GPIOD->MODER = (GPIOD->MODER & ~(0x3u << (8 * 2))) | (0x2u << (8 * 2)); // AF mode
    GPIOD->AFR[1] = (GPIOD->AFR[1] & ~(0xFu << ((8 - 8) * 4)))
                  | (0x7u << ((8 - 8) * 4));                                // AF7

    USART3->CR1 = 0;                       // disable while configuring
    USART3->BRR = 556;                     // 64 MHz / 115200, OVER16
    USART3->CR1 = USART_CR1_TE | USART_CR1_UE;
}

inline void usart3_putc(char c) noexcept
{
    while(!(USART3->ISR & USART_ISR_TXE_TXFNF)) { }
    USART3->TDR = static_cast<uint8_t>(c);
}

}

// Strong override of newlib's _write: route all stdout/stderr through USART3.
// nosys.specs supplies a weak _write stub this replaces. '\n' is expanded to
// CR+LF so a plain terminal renders the lines cleanly.
extern "C" int _write(int, const char* ptr, int len)
{
    for(int i = 0; i < len; ++i)
    {
        if(ptr[i] == '\n')
            argmin::mcu::usart3_putc('\r');
        argmin::mcu::usart3_putc(ptr[i]);
    }
    return len;
}
