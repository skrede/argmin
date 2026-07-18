# mcu/ — on-device embeddability proofs

A standalone sibling project (its own `project()`, configured with a cross
toolchain) that proves the header-only argmin solver is embeddable and
allocation-free on real hardware. It consumes argmin **header-only** via an
include path and is never added to the library's own include path.

Two honest, non-interchangeable proofs on two differently-shaped targets:

| Target | Claim | Environment | Instrument | Report | Status |
|---|---|---|---|---|---|
| **NUCLEO-H753ZI** | *rigorous* | bare-metal Cortex-M7, **no RTOS**, newlib-nano | `_sbrk` high-water + `--wrap` malloc + `operator new` | USART3 → ST-Link VCP | **measured on hardware: 0.00 allocs/step on all four policies, canary PASS** (`nucleo_h753zi/onchip-result.md`) |
| **ESP32 / ESP-IDF** | *accessible* | FreeRTOS present, IDF allocator in the loop | `heap_trace` / `heap_caps` | second UART | **measured on hardware: 0.00 allocs/step, canary PASS** (`esp32_probe/onchip-result.md`) |

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
cmake -S mcu -B build-arm-gate \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/mcu/cmake/toolchain-arm-none-eabi.cmake" \
  -DMCU_LINK_GATE_ONLY=ON
cmake --build build-arm-gate
arm-none-eabi-nm --undefined-only build-arm-gate/argmin_arm_link_gate.elf   # must be empty
```

## 2. NUCLEO-H753ZI full image (build here, flash + measure = operator step)

Requires an `arm-none-eabi` GCC (14.x in CI; 16.1.0 verified locally). Fetches
CMSIS-Core 6 + `cmsis-device-h7` (no HAL/CubeMX/RTOS).

```sh
cmake -S mcu -B build-arm-nucleo \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/mcu/cmake/toolchain-arm-none-eabi.cmake"
cmake --build build-arm-nucleo            # -> nucleo_h753zi/nucleo_h753zi_probe.elf (+ .map)
```

Verified at build time (no hardware): valid ELF, **zero undefined symbols**,
exactly one `_sbrk` (our strong override, not the nosys stub), fits 2 MB flash
(≈ 375 KB text+rodata) and 128 KB DTCM: `.data`+`.bss` = 832 B plus the
`._user_heap_stack` reserve of 88 KB (24 KB bounded heap + 64 KB stack) =
88.8 KB used, ≈ 39 KB headroom. `EIGEN_STACK_ALLOCATION_LIMIT` pinned at 8192,
justified by the measured maxima (not a tight optimum). The largest single
*dynamic* (alloca) temporary was instrumented on host across **all four** RT
policies: `kraft_slsqp` **223 B** (LSEI triangular solve, 79 B + 223 B per inner
QP iteration at `N = 4`), and `nw_sqp` / `filter_nw_sqp` / `lm` **0 B** (no
alloca temporary at their fixed N). The limit also gates fixed-size Eigen stack
objects at compile time, whose largest here is a **512 B** panel buffer — a hard
compile floor. 8192 clears both (16× the 512 B floor, ~36.7× the 223 B dynamic
max) and was **swept while live** on arm-gcc 15.2.Rel1
([nucleo_h753zi/sweep-result.md](nucleo_h753zi/sweep-result.md): 512/1024/8192
build to distinct binaries, 256 fails the limit static_assert). Eigen's stack
path requires `EIGEN_ALLOCA` to be defined explicitly on this target — see
[docs/embedded.md](../docs/embedded.md).

### Operator capture (board in hand)

The allocs/step number is an operator step — no NUCLEO hardware / QEMU on the
build host. The image reports over **USART3 → the on-board ST-Link VCP**
(`/dev/ttyACM0`, 115200 8N1); no semihosting, OpenOCD, or gdb is involved
(`nosys.specs` + `usart3_console.cpp`). Flashing is drag-drop to the ST-Link
mass-storage drive, which programs the target and resets it.

The report is printed **once at boot** and the firmware then idles: opening the
serial port does not re-trigger it (verified — a port open with no flash yields
no output). So the capture must already be running when the flash-induced reset
happens:

```sh
arm-none-eabi-objcopy -O binary \
  build-arm-nucleo/nucleo_h753zi/nucleo_h753zi_probe.elf /tmp/probe.bin

stty -F /dev/ttyACM0 115200 raw -echo -echoe -echok
timeout 90 cat /dev/ttyACM0 > /tmp/nucleo_capture.log &   # start BEFORE flashing
sleep 1
cp /tmp/probe.bin /run/media/$USER/NOD_H753ZI/ && sync    # programs + resets
```

The report lands within ~5 s of the copy. Expected console output:

```
[argmin] NUCLEO-H753ZI on-device allocation proof
[argmin] blindness canary PASS (deliberate allocation observed)
  [alloc-gate] kraft_slsqp hs071  ... per_step=0.00 ... PASS (zero-alloc gate)
  [alloc-gate] nw_sqp hs071       ... per_step=0.00 ... PASS (zero-alloc gate)
  [alloc-gate] filter_nw_sqp hs071 ... per_step=0.00 ... PASS (zero-alloc gate)
  [alloc-gate] lm rosenbrock_ls   ... per_step=0.00 ... PASS (zero-alloc gate)
[argmin] _sbrk one-time setup high-water = <bytes> bytes (<n> calls)
[argmin] RT-window gate result: PASS (0 allocs/step on all windows)
```

The blindness canary line is load-bearing: a reported `per_step=0.00` is only
meaningful because the canary proved the sensor observes a deliberate
allocation. Record the console transcript as the operator bench artifact.

Because the report is boot-once, a capture that opens the port *after* the
reset reads an empty stream, and a capture taken without a confirmed flash can
show a *stale* resident image's report. When a capture is meant to prove a
change, flash both arms and check that the console actually differs (the
`_sbrk` high-water line is a convenient image fingerprint) — the
`EIGEN_ALLOCA` A/B in `nucleo_h753zi/onchip-result.md` is the worked example.

## 3. ESP32 accessible proof (`esp32_probe/`, build here, flash + measure = operator step)

The *accessible* claim: the same four-policy fixed-N probe under ESP-IDF /
FreeRTOS with IDF's allocator in the loop — a real embeddability result but a
**softer** zero-heap statement than the bare-metal NUCLEO proof. Requires
ESP-IDF v6 (present at `/opt/esp-idf`).

```sh
. /opt/esp-idf/export.sh
idf.py -C mcu/esp32_probe set-target esp32
idf.py -C mcu/esp32_probe build            # -> build/argmin_esp32_probe.bin
```

Verified at build time (no hardware): cross-compiles + links to a flashable
`.bin` for `xtensa` with C++ exceptions and RTTI off (ESP-IDF v6 defaults).
Two independent sensors, mirroring the host design: the portable Eigen-native
counter (`EIGEN_RUNTIME_NO_MALLOC` + counting `eigen_assert`) gives the
per-step `0.00`/step result inside each armed window; ESP-IDF `heap_trace`
(standalone) is the whole-heap cross-check and blindness canary.

### Operator capture (board in hand)

```sh
idf.py -C mcu/esp32_probe -p /dev/ttyUSB0 flash monitor
```

Console (UART0) shows the canary + per-window `per_step=0.00` lines; the
telemetry summary also goes out **UART1** (TX=GPIO17, RX=GPIO16 at 115200).

**Board check before wiring:** GPIO16/17 are free on **ESP32-WROOM** but are
the PSRAM CLK/CS lines on **ESP32-WROVER** — confirm the module (silkscreen or
the boot-log PSRAM line); on a WROVER pick another free GPIO pair
(`kTelemetryTx`/`kTelemetryRx` in `main/probe_main.cpp`, a config-constant
change). The `double` solvers are soft-float on the classic ESP32 (single-
precision HW FPU) — that affects wall-time only, not the allocation count.

