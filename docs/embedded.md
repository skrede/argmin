# Embedded Cross-Compiling: Eigen Stack Temporaries and the Zero-Allocation Contract

The [Real-Time Safety Matrix](rt-safety-matrix.md) certifies the RT-claimed
policies at 0.00 allocations per steady-state step on host. Carrying that
contract onto a cross-compiled bare-metal target requires one non-obvious
Eigen configuration flag. This page documents the mechanism, the measured
consequence of getting it wrong, and the stack-budgeting rule that comes with
getting it right.

## TL;DR — required flags for bare-metal targets

```
-DEIGEN_ALLOCA=__builtin_alloca
-DEIGEN_STACK_ALLOCATION_LIMIT=<see budgeting rule below>
```

alongside the usual scalar-MCU set (`EIGEN_DONT_VECTORIZE`,
`EIGEN_MAX_ALIGN_BYTES=8`). Define `EIGEN_ALLOCA` **explicitly** — do not rely
on the C++ dialect flag to imply it (see below).

## The mechanism

Eigen's dense kernels (blocked triangular solves, Householder-sequence
applies, product blocking) create small internal temporaries through
`ei_declare_aligned_stack_constructed_variable` (Eigen 3.4.0,
`Eigen/src/Core/util/Memory.h`). That macro has two compile-time forms:

- **`EIGEN_ALLOCA` defined**: the temporary is an `alloca` on the caller's
  stack frame when its size is at or below `EIGEN_STACK_ALLOCATION_LIMIT`,
  and a heap `aligned_malloc` above it (a runtime check).
- **`EIGEN_ALLOCA` undefined**: the temporary is an unconditional heap
  `aligned_malloc`. `EIGEN_STACK_ALLOCATION_LIMIT` is not consulted at all —
  sweeping it changes nothing in the emitted binary.

`EIGEN_ALLOCA` is auto-detected as:

```c
#if EIGEN_OS_LINUX || EIGEN_OS_MAC || (defined alloca)
  #define EIGEN_ALLOCA alloca
```

On a hosted Linux/macOS build the OS branch always wins, so the stack path is
unconditionally on. On a bare-metal newlib target neither OS macro exists and
everything hinges on whether the `alloca` **macro** happens to be visible when
`Memory.h` is preprocessed — and newlib only exposes it in the GNU dialect:

| Build | Dialect | `EIGEN_ALLOCA` | Eigen internal temporaries |
|---|---|---|---|
| host Linux | any | defined (OS branch) | stack |
| arm-none-eabi / xtensa newlib | `-std=gnu++20` | defined (newlib `alloca` macro) | stack |
| arm-none-eabi / xtensa newlib | `-std=c++20` | **undefined** (`__STRICT_ANSI__` hides the macro) | **heap, every call** |

So a strict-conformance dialect choice (`CMAKE_CXX_EXTENSIONS OFF`) silently
converts every internal Eigen temporary into a per-call heap allocation on
newlib targets — with no warning, no functional change, and no effect from
`EIGEN_STACK_ALLOCATION_LIMIT`. Defining `EIGEN_ALLOCA=__builtin_alloca`
(a documented Eigen override) makes the stack path unconditional on GCC/Clang
regardless of dialect and libc header visibility.

## The measured consequence (NUCLEO-H753ZI, 2026-07-15)

The on-device proof first captured `kraft_slsqp` at **2.80 allocations/step**
on the bare-metal NUCLEO-H753ZI while the identical workload measured 0.00/step
on host and on the ESP32 (`mcu/nucleo_h753zi/onchip-result.md`,
`mcu/esp32_probe/onchip-result.md`). The three-way split is exactly the table
above: the NUCLEO image built at strict `-std=c++20`; the ESP32 probe
component forces `-std=gnu++20`; the host is Linux.

A controlled on-device A/B confirms the attribution — two images differing only
in the alloca configuration, flashed to the same board minutes apart, blindness
canary PASS in both:

| Image | `kraft_slsqp` | `_sbrk` high-water | Gate |
|---|---|---|---|
| `-UEIGEN_ALLOCA` | 28 events, **2.80/step** FAIL | 13136 B (97 calls) | FAIL |
| `EIGEN_ALLOCA=__builtin_alloca` | 0 events, **0.00/step** PASS | 12672 B (95 calls) | PASS |

The allocations were attributed by instrumenting `EIGEN_ALLOCA` on host:
a single call site — the transformed-constraint triangular solve in
`argmin/detail/lsi.h` (`R^T G_t^T = (G P)^T`, a matrix-right-hand-side solve)
— whose `triangular_solve_matrix` kernel requests **two** blocking buffers per
call, **79 B + 223 B** at `N = 4`, once per inner QP iteration. The host
reproduction under the instrumented macro shows the same alternating pair at
2.60/step (26 events / 10 steps) against the device's 2.80 (28 events; one
extra inner QP iteration on-device, consistent with libm rounding
differences). On host/ESP32 these land on the stack; on the strict-dialect
NUCLEO build each one is a heap malloc.

Note what this is **not**: it is not an arm code-generation effect, not a
newlib allocator effect, and not an `EIGEN_STACK_ALLOCATION_LIMIT` sizing
effect (the limit is dead code in the no-alloca configuration, which is why
sweeping it left the NUCLEO binary byte-identical).

## Why the stack path is the right call for RT (and not the heap)

Keeping the per-step heap allocations is strictly worse on every RT axis:

- **Determinism**: newlib's `malloc` has data-dependent latency and, on a
  long-running control loop, fragmentation exposure. An `alloca` is a frame
  pointer bump — constant time, freed deterministically at scope exit.
- **Contract coherence**: the zero-alloc claim would hold for `nw_sqp`,
  `filter_nw_sqp`, and `lm` but fail for `kraft_slsqp` on one target for a
  build-flag reason, not an algorithmic one.
- **Footprint**: the measured cost at the RT-claimed fixed-`N` regime is a few
  hundred bytes of transient stack (302 B peak at `N = 4`), against a stack
  reserve that was already provisioned for exactly this
  (`mcu/nucleo_h753zi/stm32h753zi_flash.ld` reserves 64 KiB with the Eigen
  stack temporary named in the comment).

The one honest counterargument is failure visibility: heap exhaustion is
detectable (a failed `malloc` returns null; the allocation gate counts it),
while a bare-metal stack overflow is silent corruption. That risk is what
`EIGEN_STACK_ALLOCATION_LIMIT` bounds, so it must be budgeted, not maximized.

## The stack-budgeting rule

`EIGEN_STACK_ALLOCATION_LIMIT` is a **per-temporary cap**, not a total: any
single temporary above it falls back to the heap at runtime (where the
allocation gate catches it as an honest red — a detectable failure, not a
crash). Choose it so the worst case fits:

```
stack reserve  >=  static call-chain high-water (-fstack-usage)
                 + k x EIGEN_STACK_ALLOCATION_LIMIT   (k = temporaries live at once; 2 for the
                                                       triangular-solve kernel pair above)
                 + ISR / margin
```

Do **not** set the limit to the size of the whole reserve (a 64 KiB limit
inside a 64 KiB stack budgets for overflow). At the fixed-`N` regime the
RT claims live in (`N < 49`, see the scope section of the
[Real-Time Safety Matrix](rt-safety-matrix.md)), the temporaries measured so far
are small — 79 B and 223 B per inner QP iteration at `N = 4` — so the 8192 pin
on the NUCLEO image leaves a wide margin against a 64 KiB reserve.

Two honest limits on that statement: those sizes come from instrumenting one
call path (the Kraft-LSEI triangular solve), not all four RT-claimed policies,
and the pin itself has not been swept while the alloca path was actually live —
the earlier sweep ran with `EIGEN_ALLOCA` undefined, where the limit is dead
code and no sweep can move anything. Treat 8192 as a budgeted default, not a
measured optimum, and re-sweep if you change `N`, the policy set, or the target.

## Status

The fix is applied and verified on hardware: both probe builds define
`EIGEN_ALLOCA=__builtin_alloca` and `EIGEN_STACK_ALLOCATION_LIMIT=8192`
explicitly (`mcu/CMakeLists.txt` via `ARGMIN_EIGEN_MCU_DEFS`;
`mcu/esp32_probe/main/CMakeLists.txt`), so neither target's behavior hinges on
dialect or libc header visibility. The NUCLEO-H753ZI now measures **0.00
allocs/step on all four RT-claimed policies with the blindness canary PASS**
(`mcu/nucleo_h753zi/onchip-result.md`, 2026-07-15). The ESP32 capture (green)
predates the explicit flags but they change nothing there — its `-std=gnu++20`
dialect already implied the same alloca configuration.
