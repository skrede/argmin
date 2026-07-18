// ESP32 accessible-proof driver.
//
// Runs the shared fixed-N four-policy RT probe (probe/rt_probe_workload.h)
// under ESP-IDF / FreeRTOS and measures allocations two independent ways,
// mirroring the host two-sensor design:
//
//   1. The portable Eigen-native sensor (EIGEN_RUNTIME_NO_MALLOC + the counting
//      eigen_assert defined below, exactly as benchmarks/alloc_trace_main.cpp
//      wires it): counts Eigen allocation attempts inside each armed steady
//      window. This backs the per-step evaluate_gate result the workload
//      prints -- the authoritative "0 allocs/step" number.
//
//   2. ESP-IDF heap_trace (standalone): a whole-heap, process-wide cross-check
//      and the mandatory blindness canary. heap_trace_start(HEAP_TRACE_ALL)
//      arms+resets, heap_trace_summary().total_allocations reads, stop disarms.
//
// This is the *accessible* claim: FreeRTOS is present and IDF's allocator is in
// the loop, so it is a softer zero-heap statement than the bare-metal NUCLEO
// proof. The on-device numbers are captured by the operator (no board attached
// on the build host); this file is verified to cross-compile + link to a
// flashable .bin.

#define EIGEN_RUNTIME_NO_MALLOC
#define eigen_assert(x) \
    do { if(!(x)) ::argmin::detail::bench::on_eigen_malloc(); } while(0)

#include "probe/rt_probe_workload.h"

#include "esp_heap_trace.h"
#include "esp_heap_caps.h"
#include "driver/uart.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <cstdio>
#include <cstring>
#include <cstddef>
#include <atomic>

// Counters backing alloc_counter.h's Eigen-native sensor, one definition for
// the image (mirrors benchmarks/alloc_trace_main.cpp on host).
namespace argmin::detail::bench
{
std::atomic<std::size_t> g_eigen_malloc_count{0};
std::atomic<std::size_t> g_c_alloc_count{0};
std::atomic<bool> g_alloc_trace_armed{false};
}

namespace
{

constexpr uart_port_t kTelemetryUart = UART_NUM_1;
constexpr int kTelemetryTx = 17;   // GPIO17 (WROOM: free; WROVER: PSRAM CS -- check board)
constexpr int kTelemetryRx = 16;   // GPIO16 (WROOM: free; WROVER: PSRAM CLK -- check board)

// heap_trace record buffer, internal RAM.
heap_trace_record_t g_trace_buf[256];

void telemetry_init()
{
    uart_config_t cfg = {};
    cfg.baud_rate = 115200;
    cfg.data_bits = UART_DATA_8_BITS;
    cfg.parity    = UART_PARITY_DISABLE;
    cfg.stop_bits = UART_STOP_BITS_1;
    cfg.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;
    uart_param_config(kTelemetryUart, &cfg);
    uart_set_pin(kTelemetryUart, kTelemetryTx, kTelemetryRx,
                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    // tx_buffer_size 0 => uart_write_bytes blocks until sent (fine for low-rate
    // telemetry, no TX ring buffer needed).
    uart_driver_install(kTelemetryUart, 256, 0, 0, nullptr, 0);
}

void telemetry_line(const char* s)
{
    uart_write_bytes(kTelemetryUart, s, std::strlen(s));
}

// Mandatory blindness canary: a deliberate allocation inside an armed
// heap_trace window MUST be observed, else a reported zero is a blind sensor.
int heap_sensor_canary()
{
    heap_trace_start(HEAP_TRACE_ALL);
    void* p = heap_caps_malloc(64, MALLOC_CAP_8BIT);
    heap_trace_stop();

    heap_trace_summary_t s;
    heap_trace_summary(&s);
    heap_caps_free(p);
    return (p != nullptr && s.total_allocations >= 1) ? 0 : 1;
}

void probe_task(void*)
{
    // Warm up first-use allocators (first stdio/float printf allocates a libc
    // lock that would otherwise show as spurious trace noise).
    std::printf("[argmin] ESP32 accessible allocation proof (RTOS-present)\n");
    std::printf("[argmin] warmup float %.2f\n", 0.0);

    if(heap_sensor_canary() != 0)
    {
        std::printf("[argmin] BLINDNESS CANARY FAILED: heap sensor blind -- "
                    "reported zeros are meaningless\n");
        telemetry_line("argmin ESP32: CANARY FAIL\r\n");
        vTaskDelete(nullptr);
        return;
    }
    std::printf("[argmin] blindness canary PASS (deliberate allocation observed)\n");

    // Whole-heap cross-check bracket around the full four-policy run: the
    // per-step 0.00 result comes from the Eigen-native sensor inside the
    // workload (the evaluate_gate lines); this heap_trace total additionally
    // shows the one-time setup allocations and proves the heap sensor is live.
    heap_trace_start(HEAP_TRACE_ALL);
    const int rc = argmin::mcu::run_all_rt_policies();
    heap_trace_stop();

    heap_trace_summary_t s;
    heap_trace_summary(&s);

    char msg[160];
    std::snprintf(msg, sizeof(msg),
        "argmin ESP32: RT-window gate %s; whole-run heap total_allocations=%u "
        "(one-time setup, informational)\r\n",
        rc == 0 ? "PASS(0/step)" : "FAIL",
        static_cast<unsigned>(s.total_allocations));
    std::printf("[argmin] %s", msg);
    telemetry_line(msg);

    vTaskDelete(nullptr);
}

}

extern "C" void app_main()
{
    telemetry_init();
    heap_trace_init_standalone(g_trace_buf,
                               sizeof(g_trace_buf) / sizeof(g_trace_buf[0]));

    // The fixed-N solve needs far more stack than app_main's default (3584 B);
    // IDF FreeRTOS stack sizes are in BYTES. 40 KiB gives Eigen temporaries
    // ample headroom (verify with uxTaskGetStackHighWaterMark on hardware).
    xTaskCreatePinnedToCore(probe_task, "argmin_probe", 40 * 1024,
                            nullptr, 5, nullptr, 0);
}
