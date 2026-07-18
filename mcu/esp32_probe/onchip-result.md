# ESP32 on-chip allocation result (operator bench artifact)

Captured 2026-07-15 on a physical classic ESP32 (CP2102 bridge, dual-core
Xtensa LX6 @ 160 MHz, 2 MB flash), ESP-IDF v6.0.2, app commit `719dbf8`,
flashed over USB (`/dev/ttyUSB1`). Console on UART0 (CP2102), telemetry on
UART1 → GPIO17 → FT232 (`/dev/ttyUSB0`). No SD card / external programmer —
USB only.

This is the *accessible* proof: FreeRTOS present, IDF allocator in the
loop — a softer zero-heap statement than the bare-metal NUCLEO proof.

## Console (UART0)

```
[argmin] ESP32 accessible allocation proof (RTOS-present)
[argmin] blindness canary PASS (deliberate allocation observed)
  [alloc-gate] kraft_slsqp hs071   eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] kraft_slsqp hs071 PASS (zero-alloc gate)
  [alloc-gate] nw_sqp hs071        eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] nw_sqp hs071 PASS (zero-alloc gate)
  [alloc-gate] filter_nw_sqp hs071 eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] filter_nw_sqp hs071 PASS (zero-alloc gate)
  [alloc-gate] lm rosenbrock_ls    eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] lm rosenbrock_ls PASS (zero-alloc gate)
[argmin] argmin ESP32: RT-window gate PASS(0/step); whole-run heap total_allocations=274 (one-time setup, informational)
```

## Telemetry (UART1 / GPIO17 / FT232)

```
argmin ESP32: RT-window gate PASS(0/step); whole-run heap total_allocations=274 (one-time setup, informational)
```

## Result

- **0.00 allocs/step** on all four RT-claimed policies (`kraft_slsqp`,
  `nw_sqp`, `filter_nw_sqp`, `lm`) in the armed steady-state windows —
  reproduces the host contract on real hardware.
- **Blindness canary PASS**: a deliberate `heap_caps_malloc(64)` in an armed
  window was observed, so the reported zeros are meaningful (not a blind
  sensor). Both sensors agree: the Eigen-native per-step counter
  (`eigen_malloc=0`) and the ESP-IDF `heap_trace` whole-heap cross-check.
- Whole-run `heap_trace total_allocations = 274` — one-time construction/setup
  across the four policies, informational (the RT claim is per-step, not
  zero-after-construction). The second UART telemetry path works as wired.
