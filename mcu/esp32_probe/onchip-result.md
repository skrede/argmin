# ESP32 on-chip allocation result (operator bench artifact)

Re-certified 2026-07-18 on a physical classic ESP32 against the current tree
(app commit `fc79603`, ESP-IDF `v6.1-dev-6485-g055ba9d3f9c`, xtensa-esp-elf
GCC 16.1.0). Flashed over USB via the CP2102 bridge (`/dev/cu.usbserial-0001`);
console on UART0 (CP2102), telemetry on UART1 → GPIO17 → FT232R
(`/dev/cu.usbserial-AI0483EF`). Boot-once capture taken across a physical
reset with both readers attached. No SD card / external programmer — USB only.

This is the *accessible* proof: FreeRTOS present, IDF allocator in the loop —
a softer zero-heap statement than the bare-metal NUCLEO proof.

## Console (UART0)

```
[argmin] ESP32 accessible allocation proof (RTOS-present)
[argmin] warmup float 0.00
[argmin] blindness canary PASS (deliberate allocation observed)
  [alloc-gate] kraft_slsqp hs071  eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] kraft_slsqp hs071 PASS (zero-alloc gate)
  [alloc-gate] nw_sqp hs071       eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] nw_sqp hs071 PASS (zero-alloc gate)
  [alloc-gate] filter_nw_sqp hs071 eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] filter_nw_sqp hs071 PASS (zero-alloc gate)
  [alloc-gate] lm rosenbrock_ls   eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] lm rosenbrock_ls PASS (zero-alloc gate)
[argmin] argmin ESP32: RT-window gate PASS(0/step); whole-run heap total_allocations=274 (one-time setup, informational)
```

## Telemetry (UART1 / GPIO17 / FT232R)

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

## Re-certification against the `EIGEN_ALLOCA` flag change

The current tree defines `EIGEN_ALLOCA=__builtin_alloca` and
`EIGEN_STACK_ALLOCATION_LIMIT=8192` on the probe component; the earlier green
capture (commit `719dbf8`, ESP-IDF v6.0.2) predates those defines. Two arms of
the component were built under the current toolchain to isolate the defines'
effect — arm A with the defines, arm B with both removed:

| arm | config | `.flash.text` | image sha256 |
|---|---|---|---|
| A | defines present (current tree) | 505706 B | `8762a545…9212ec` |
| B | defines removed (predecessor config) | 505710 B | `1c1ce5d4…4df44b2` |

The two images are **not** byte-identical, and the difference is real code, not
build metadata: `.flash.text` differs by 4 bytes and `.rodata`/`.data` sizes are
equal. A second build of arm A reproduces 505706 B exactly, so the build is
deterministic and the 4-byte delta is attributable to the defines (Eigen's
internal stack-temporary path), not to nondeterminism. Removing the defines is
therefore **not** image-neutral on this target, so the predecessor capture does
not transitively certify the current image.

Certification is by the fresh on-device capture above, taken from the exact
current image (arm A, the bytes flashed and run). All four RT-claimed policies
report 0.00 allocs/step with the defines in place, so the flag change preserves
the zero-per-step contract on real hardware.
