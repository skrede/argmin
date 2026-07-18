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
[Real-Time Safety Matrix](rt-safety-matrix.md)), the measured temporaries are
small. Instrumenting `EIGEN_ALLOCA` on host across **all four** RT-claimed
policies (`mcu/probe/measure_temporaries_main.cpp`) gives the largest single
*dynamic* (alloca) temporary per policy:

| Policy (fixed N) | Largest single dynamic temporary |
|---|---|
| `kraft_slsqp` (hs071, N=4) | 223 B — LSEI triangular solve, `argmin/detail/lsi.h` (79 B + 223 B live as a pair per inner QP iteration) |
| `nw_sqp` (hs071, N=4) | 0 B — no alloca temporary at this N |
| `filter_nw_sqp` (hs071, N=4) | 0 B |
| `lm` (rosenbrock_ls, N=2) | 0 B |

So the dynamic `L_max` is **223 B**. The limit *also* gates **fixed-size** Eigen
stack objects at compile time — `static_assert(Size*sizeof(T) <=
EIGEN_STACK_ALLOCATION_LIMIT)` in `DenseStorage.h` — and the largest here is a
**512 B** 8×8 `TriangularMatrixMatrix` panel buffer. That sets a hard compile
floor: a limit below 512 B fails to build. The 8192 pin clears both binding
constraints — 16× the 512 B compile floor and ~36.7× the 223 B dynamic max.

The pin was **swept while live** (`EIGEN_ALLOCA` defined) on arm-gcc 15.2.Rel1
and recorded in [`mcu/nucleo_h753zi/sweep-result.md`](../mcu/nucleo_h753zi/sweep-result.md):
the three buildable rungs (512 / 1024 / 8192) produce **distinct** binaries — the
limit appears as the immediate in ~77 runtime `cmp r0, #LIMIT` alloca-vs-heap
decisions — while a rung below the 512 B floor (256) fails to compile with a
limit-derived `static_assert`. That two-axis result is the falsifiable positive
control that the knob is live; it replaces the earlier **vacuous** sweep, which
ran with `EIGEN_ALLOCA` undefined (limit dead code, byte-identical binary, no
sweep able to move anything).

### Composed stack budget for this image

The `-fstack-usage` `.su` data (arm-gcc 15.2, pin 8192) gives a deepest single
frame of **19,968 B** (`measure_steady_window` instantiated for `kraft_slsqp`).
The four RT windows run **sequentially with no nesting**, so their frames do not
sum across windows. `-fstack-usage` is per-frame and does **not** account for
`alloca`, so it is a floor — add the measured alloca live-set and a margin:

```
budget_floor = 19,968 B   deepest .su frame (kraft measure_steady_window)
             +    302 B   k x L_max_alloca  (k = 2 temporaries live at once:
                          kraft's 79 B + 223 B triangular-solve pair)
             +  ISR / margin
            ~= 20.3 KiB  <<  65,536 B  (_Min_Stack_Size, stm32h753zi_flash.ld:39)
```

That leaves ~45 KiB (≈ 69 %) of the 64 KiB reserve free — an order of magnitude
of headroom. Note the alloca contribution is `k x L_max_alloca` (the *measured*
live-set), **not** `k x EIGEN_STACK_ALLOCATION_LIMIT`: the pin is a cap the
temporaries never approach. For a rigorous root-to-leaf worst case (instead of a
single-frame floor), aggregate the `.su` frames along the deepest call path with
GCC 15's `-fcallgraph-info=su,da` or a community aggregator (`avstack.pl`,
`puncover`) and add the same `k x L_max_alloca` + margin; the sequential-window
structure keeps the result dominated by one window's path, still far under
65,536 B.

Treat 8192 as a budgeted pin justified by the measured maxima above (not a tight
optimum); re-sweep if you change `N`, the policy set, or the target.

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
