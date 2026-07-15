# firmware/ — on-device embeddability proofs

A standalone sibling project (its own `project()`, configured with a cross
toolchain) that proves the header-only argmin solver is embeddable and
allocation-free on real hardware. It consumes argmin **header-only** via an
include path and is never added to the library's own include path.

Two honest, non-interchangeable proofs on two differently-shaped targets:

| Target | Claim | Environment | Instrument | Report | Status |
|---|---|---|---|---|---|
| **NUCLEO-H753ZI** | *rigorous* | bare-metal Cortex-M7, **no RTOS**, newlib-nano | `_sbrk` high-water + `--wrap` malloc + `operator new` | semihosting | build + link + instrument here; **allocs/step = operator capture** |
| **ESP32 / ESP-IDF** | *accessible* | FreeRTOS present, IDF allocator in the loop | `heap_trace` / `heap_caps` | second UART | see `esp32/` |

The ESP32 proof runs under FreeRTOS with IDF's allocator in the loop — a real
embeddability result but a **softer** zero-heap statement than the bare-metal
NUCLEO proof. It is never presented as the bare-metal claim.

Both probes exercise the same four RT-claimed policies (`kraft_slsqp`,
`nw_sqp`, `filter_nw_sqp`, `lm`) on fixed-N problems (`probe/rt_probe_workload.h`)
and must reproduce the host contract of **0.00 allocs/step** on the RT windows
(`docs/rt-safety-matrix.md`), witnessed by a mandatory blindness canary.

## 1. `-fno-exceptions` cross-compile-link gate (CI, no hardware)

The per-commit gate: cross-compile-and-link the four policies under
newlib-nano `-fno-exceptions -fno-rtti`, no board code. Run by the
`arm-no-exceptions-link` CI job; locally:

```sh
cmake -S firmware -B build-arm-gate \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/firmware/cmake/toolchain-arm-none-eabi.cmake" \
  -DFIRMWARE_LINK_GATE_ONLY=ON
cmake --build build-arm-gate
arm-none-eabi-nm --undefined-only build-arm-gate/argmin_arm_link_gate.elf   # must be empty
```

## 2. NUCLEO-H753ZI full image (build here, flash + measure = operator step)

Requires an `arm-none-eabi` GCC (14.x in CI; 16.1.0 verified locally). Fetches
CMSIS-Core 6 + `cmsis-device-h7` (no HAL/CubeMX/RTOS).

```sh
cmake -S firmware -B build-arm-nucleo \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/firmware/cmake/toolchain-arm-none-eabi.cmake"
cmake --build build-arm-nucleo            # -> nucleo_h753zi/nucleo_h753zi_probe.elf (+ .map)
```

Verified at build time (no hardware): valid ELF, **zero undefined symbols**,
exactly one `_sbrk` (our strong override, not the nosys stub), fits 2 MB flash
and 128 KB DTCM (`.data`+`.bss` ≈ 18 KB; 64 KB stack reserve; 1 KB bounded
heap). `EIGEN_STACK_ALLOCATION_LIMIT` swept and pinned at 8192 (largest single
Eigen fixed temporary ≤ 4096 B at these fixed N).

### Operator capture (board in hand)

The allocs/step number is an operator step — no NUCLEO hardware / QEMU on the
build host. With a NUCLEO-H753ZI attached over the on-board ST-Link:

```sh
# terminal 1 — semihosting-enabled debug server
openocd -f board/st_nucleo_h753zi.cfg \
  -c "init; arm semihosting enable; reset run"

# terminal 2 — flash + run (or use the gdb 'load' path)
arm-none-eabi-gdb build-arm-nucleo/nucleo_h753zi/nucleo_h753zi_probe.elf \
  -ex "target extended-remote :3333" -ex "load" -ex "continue"
```

Expected console output (semihosting):

```
[argmin] NUCLEO-H753ZI on-device allocation proof
[argmin] blindness canary PASS (deliberate allocation observed)
  [alloc-gate] kraft_slsqp hs071   ... per_step=0.00 ... PASS (zero-alloc gate)
  [alloc-gate] nw_sqp hs071        ... per_step=0.00 ... PASS (zero-alloc gate)
  [alloc-gate] filter_nw_sqp hs071 ... per_step=0.00 ... PASS (zero-alloc gate)
  [alloc-gate] lm rosenbrock_ls    ... per_step=0.00 ... PASS (zero-alloc gate)
[argmin] _sbrk one-time setup high-water = <bytes> bytes (<n> calls)
[argmin] RT-window gate result: PASS (0 allocs/step on all windows)
```

The blindness canary line is load-bearing: a reported `per_step=0.00` is only
meaningful because the canary proved the sensor observes a deliberate
allocation. Record the console transcript as the operator bench artifact.

If semihosting hangs at the first `printf` (a reported `nano.specs`+`rdimon`
init-ordering issue), the UART fallback is a small change: replace
`--specs=rdimon.specs` with a hand-written `_write` pushing bytes out a USART
data register (no HAL), keeping `nano.specs`/`nosys.specs` for the rest.
