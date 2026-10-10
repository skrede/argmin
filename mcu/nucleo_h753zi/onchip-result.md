# NUCLEO-H753ZI on-chip allocation result (operator bench artifact)

Re-captured 2026-07-20 on a physical NUCLEO-H753ZI (bare-metal Cortex-M7 @ 64 MHz
HSI, no RTOS, newlib-nano), with the over-limit guardrail probe added. The pinned image
(`EIGEN_STACK_ALLOCATION_LIMIT=8192`, `EIGEN_ALLOCA=__builtin_alloca`) was flashed
over the on-board ST-Link by drag-drop to the `NOD_H753ZI` mass-storage drive,
which programs the flash and resets the target so the boot-once report streams.
The report is streamed over USART3 → the ST-Link VCP; on this macOS host the VCP
enumerates as `/dev/cu.usbmodem112203` at **115200 baud** (BSD `stty -f`) — no
semihosting/OpenOCD.

**Toolchain:** Arm GNU Toolchain **15.2.Rel1** (`arm-none-eabi-gcc`
15.2.1 20251203). The capture certifies this toolchain's codegen.

**Capture-method note.** The firmware report is boot-once. Mass-storage (DND)
programming momentarily re-enumerates the ST-Link CDC, which invalidates a live
reader's fd mid-stream, so a single `cat` opened before the copy is torn down
during programming. The headless clean method is to **re-attach the reader after
the CDC comes back**: start a reader before the copy (best effort), copy the
`.bin` (which programs and resets the target), then re-open `cat` on the
re-enumerated VCP as soon as it reappears — the target boots and streams the
report ~5 s after the copy, so the freshly-attached reader catches a pristine
boot-once stream. (A `st-flash --connect-under-reset reset` connects but leaves
the core halted with no output; the physical **NRST (B2)** button is the
equivalent manual reset when an operator is present.)

## Console (USART3 / ST-Link VCP, physical NRST, 115200 baud)

```
[argmin] NUCLEO-H753ZI on-device allocation proof
[argmin] blindness canary PASS (deliberate allocation observed)
[overflow-probe] forced 10368 B temporary > 8192 B limit: heap fallback observed (1 allocs) PASS
  [alloc-gate] kraft_slsqp hs071  eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] kraft_slsqp hs071 PASS (zero-alloc gate)
  [alloc-gate] nw_sqp hs071       eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] nw_sqp hs071 PASS (zero-alloc gate)
  [alloc-gate] filter_nw_sqp hs071 eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] filter_nw_sqp hs071 PASS (zero-alloc gate)
  [alloc-gate] lm rosenbrock_ls   eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] lm rosenbrock_ls PASS (zero-alloc gate)
[argmin] _sbrk one-time setup high-water = 12688 bytes (15 calls)
[argmin] RT-window gate result: PASS (0 allocs/step on all windows)
```

## Result

- **All four RT-claimed policies (`kraft_slsqp`, `nw_sqp`, `filter_nw_sqp`,
  `lm`): 0.00 allocs/step** in the armed steady-state windows — the RT claim
  holds on real bare-metal hardware with the pinned 8192 B limit, reproducing the
  host contract. Both sensors agree (Eigen-native + `_sbrk`/`--wrap`).
- **Blindness canary PASS**: a deliberate `malloc(64)` + `Eigen::VectorXd(37)`
  in an armed window was observed, so the zeros are meaningful, not blind.
- **Over-limit guardrail probe PASS**: a single armed probe window
  (`run_overflow_guardrail_probe`, run between the canary and the four RT windows,
  fully outside their arm/reset/disarm regions) forced a 36×36 lower-triangular
  solve whose internal kernel temporary is **10368 B (36·36·8) > 8192 B**. Eigen's
  `ei_declare_aligned_stack_constructed_variable` ternary therefore took the
  `aligned_malloc` (heap) branch — **1 heap alloc observed** in that window. This
  converts "did not overflow at these N" into "the guardrail demonstrably fires"
  when a temporary exceeds the pinned limit. The four RT windows remained **0.00**
  — the probe did **not** perturb them (its operands live in static BSS via `Map`,
  so the only armed-window heap traffic is the ~10.4 KiB fallback transient, freed
  back before the RT windows; the limit was not raised).

### Fresh-image fingerprint (proves this is the newly-flashed image)

`_sbrk` one-time setup high-water = **12,688 bytes / 15 calls** for the image with
the overflow probe. This is a distinct, self-consistent fingerprint of the
freshly-flashed image — **not** the stale pre-probe baseline (12,672 B / 95 calls)
and **not** the pre-`EIGEN_ALLOCA`-fix broken image (13,136 B / 97 calls):

- **Bytes (12,688 vs 12,672, +16 B):** the byte high-water is still set by the RT
  policies' one-time setup peak (which frees back); the probe's 10,368 B
  transient is smaller than that peak, so it barely moves the high-water.
- **Calls (15 vs 95):** the overflow probe runs first and grows the newlib heap arena
  to ~10–12 KiB in a handful of `_sbrk` extensions, then frees the transient back
  to the free list. The subsequent RT-policy one-time allocations are then
  satisfied from that already-grown arena **without new `_sbrk` calls**, collapsing
  the call count from 95 to 15. The byte high-water is unchanged because the true
  simultaneous peak is still the RT setup. This is the expected, reasoned
  consequence of adding the probe window ahead of the RT windows.

## History: `kraft_slsqp` at 2.80/step, root-caused to a missing `EIGEN_ALLOCA`

The first capture on this board (2026-07-15, before the flag fix) read:

```
  [alloc-gate] kraft_slsqp hs071   eigen_malloc=28 c_alloc=28 armed_steps=10 per_step=2.80
  [alloc-gate] kraft_slsqp hs071 FAIL: zero-alloc gate saw 28 allocations
[argmin] _sbrk one-time setup high-water = 13136 bytes (97 calls)
[argmin] RT-window gate result: FAIL (see per-window lines above)
```

with the other three policies already at 0.00/step. The initial hypothesis —
an **arm-none-eabi/newlib-target-specific** allocation under arm codegen,
since the host measured 0.00/step under nominally identical flags — was
**wrong**.

### Root cause: `EIGEN_ALLOCA` undefined at strict `-std=c++20`

The allocations are Eigen internal kernel temporaries that are supposed to be
`alloca`s. Eigen enables its stack path only when
`EIGEN_OS_LINUX || EIGEN_OS_MAC || (defined alloca)` holds
(`Memory.h:587-592`). This image builds at strict `-std=c++20`
(`CMAKE_CXX_EXTENSIONS OFF`), where `__STRICT_ANSI__` hides newlib's `alloca`
macro, so `EIGEN_ALLOCA` was undefined and every such temporary became an
unconditional heap `aligned_malloc`. In that configuration
`EIGEN_STACK_ALLOCATION_LIMIT` is never consulted — which is exactly why
sweeping it 8192 → 65536 had left the binary byte-identical: the sweep ruled
out the limit's *value*, not the alloca mechanism. The host never reproduced
it because `__linux__` short-circuits the check; the ESP32 did not because its
probe component forces `-std=gnu++20`, where newlib exposes the macro.

Attribution (instrumented `EIGEN_ALLOCA` on host, same probe workload): one
call site — the transformed-constraint triangular solve in
`argmin/detail/lsi.h` (`R^T G_t^T = (G P)^T`, matrix-RHS), whose
`triangular_solve_matrix` kernel takes two blocking buffers per call
(79 B + 223 B at N = 4), once per inner QP iteration — 26 events / 10 steps
on host (2.60/step) vs the 28 (2.80/step) measured on-device (one extra inner
iteration on-device, consistent with libm rounding differences).

### Controlled on-device A/B (2026-07-15)

Both images flashed to this board minutes apart, same capture path, blindness
canary PASS in both. The two builds differ only in the alloca configuration
(`-UEIGEN_ALLOCA` reproducing the pre-fix compile, and its now-dead stack
limit):

| Image | `kraft_slsqp` | `_sbrk` high-water | Gate |
|---|---|---|---|
| `-UEIGEN_ALLOCA` (pre-fix config) | 28 events, **2.80/step** FAIL | 13136 B (97 calls) | FAIL |
| `EIGEN_ALLOCA=__builtin_alloca` (fix) | 0 events, **0.00/step** PASS | 12672 B (95 calls) | PASS |

The pre-fix build reproduces the original capture exactly — 28 events,
2.80/step, 13136 bytes, 97 calls — so the flag, not the target or the
toolchain, is the causal variable. The other three policies read 0.00/step in
both arms.

Fix: `EIGEN_ALLOCA=__builtin_alloca` is defined explicitly in
`ARGMIN_EIGEN_MCU_DEFS`, with `EIGEN_STACK_ALLOCATION_LIMIT` re-pinned to 8192
as a budgeted per-temporary cap against the 64 KiB stack reserve in
`stm32h753zi_flash.ld` (which was already provisioned for these temporaries).
The 2026-07-18 re-capture above re-certifies 0.00/step on all four RT
windows at the pinned limit and additionally proves the guardrail fires when a
temporary exceeds it.

## Float capture (2026-10-10)

The probe image now runs every real-time window twice, once per scalar
(`double`, then `float`, labeled with a trailing ` float`), and brackets each
window with the DWT cycle counter. Both captures below were taken on the same
physical NUCLEO-H753ZI (Cortex-M7, `SystemCoreClock` = 64 MHz, the HSI
default; the clock tree is not reconfigured) on a Linux host, where the ST-Link
VCP enumerates as `/dev/ttyACM0` at 115200 baud. The reader was started before
the `.bin` was copied to the ST-Link drive, and each log holds a complete
boot-once report. The DWT cycle counter counts without a debugger attached.

**Toolchain:** `arm-none-eabi-gcc` 16.2.0 (Arch Repository build), same
`ARGMIN_EIGEN_MCU_DEFS` (`EIGEN_ALLOCA=__builtin_alloca`,
`EIGEN_STACK_ALLOCATION_LIMIT=8192`).

### Capture 1: commit `f83a218`, FAIL

Image `probe.bin` sha256
`308ff50efaeedadf2336b6db3ee5e46f2d2aba3177effaa9e1486643d99ec2e4`; ELF text
764,268 B, data 464 B, bss 103,160 B.

```
[argmin] NUCLEO-H753ZI on-device allocation proof
[argmin] blindness canary PASS (deliberate allocation observed)
[overflow-probe] forced 10368 B temporary > 8192 B limit: heap fallback observed (1 allocs) PASS
[argmin] SystemCoreClock = 64000000 Hz
  [timing] kraft_slsqp hs071 cycles=1887658 armed_steps=10 per_step=188765
  [alloc-gate] kraft_slsqp hs071  eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] kraft_slsqp hs071 PASS (zero-alloc gate)
  [timing] nw_sqp hs071 cycles=2663083 armed_steps=10 per_step=266308
  [alloc-gate] nw_sqp hs071       eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] nw_sqp hs071 PASS (zero-alloc gate)
  [timing] filter_nw_sqp hs071 cycles=2664932 armed_steps=10 per_step=266493
  [alloc-gate] filter_nw_sqp hs071 eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] filter_nw_sqp hs071 PASS (zero-alloc gate)
  [timing] lm rosenbrock_ls cycles=45410 armed_steps=10 per_step=4541
  [alloc-gate] lm rosenbrock_ls   eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] lm rosenbrock_ls PASS (zero-alloc gate)
  [timing] kraft_slsqp hs071 float cycles=1699461 armed_steps=10 per_step=169946
  [alloc-gate] kraft_slsqp hs071 float eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] kraft_slsqp hs071 float PASS (zero-alloc gate)
  [timing] nw_sqp hs071 float cycles=2049872 armed_steps=10 per_step=204987
  [alloc-gate] nw_sqp hs071 float eigen_malloc=2 c_alloc=2 armed_steps=10 per_step=0.20
  [alloc-gate] nw_sqp hs071 float FAIL: zero-alloc gate saw 2 allocations
  [timing] filter_nw_sqp hs071 float cycles=2078956 armed_steps=10 per_step=207895
  [alloc-gate] filter_nw_sqp hs071 float eigen_malloc=2 c_alloc=2 armed_steps=10 per_step=0.20
  [alloc-gate] filter_nw_sqp hs071 float FAIL: zero-alloc gate saw 2 allocations
  [timing] lm rosenbrock_ls float cycles=37588 armed_steps=10 per_step=3758
  [alloc-gate] lm rosenbrock_ls float eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] lm rosenbrock_ls float PASS (zero-alloc gate)
[argmin] _sbrk one-time setup high-water = 12688 bytes (15 calls)
[argmin] RT-window gate result: FAIL (see per-window lines above)
```

`float` `nw_sqp` and `filter_nw_sqp` allocated at **0.20/step** (2 allocations
in 10 armed steps). The other six windows were 0.00/step.

**Root cause: the null-step path at the solution allocated.** In `float` the
hs071 solve converges at step 7, inside the armed window (steps 3-12), so the
window measured steps taken at the solution. The `double` solve had not
converged by step 12, so the `double` windows never reached that path. A host
reproduction with the MCU Eigen macro set found two sources:

- `extract_qp_multipliers` took its multipliers as `const VectorX&`, so the
  bounded QP's fixed-capacity multiplier vector was copied into a heap-allocated
  `VectorX` on every null step (`nw_sqp`, `filter_nw_sqp`).
- `filter_nw_sqp`'s Lagrangian-gradient norm built a stacked
  `Matrix<S, Dynamic, N>` Jacobian per call.

The defect was not specific to `float`: on host, a `double` `filter_nw_sqp` run
that keeps stepping at the solution allocated 17 times over steps 3-42. The
earlier zero-allocation record held only because its windows ended before
convergence.

**Fix: commit `6d862a7`.** `extract_qp_multipliers` reads the multipliers
through `Eigen::MatrixBase<Derived>`, so no copy is made. `filter_nw_sqp` builds
the stacked Jacobian in its preallocated state buffer. The host allocation gate
gained an at-solution window: after a converged solve it arms the counter over
20 further steps. That window runs for `kraft_slsqp`, `nw_sqp` and
`filter_nw_sqp` in both scalars. Without the fix 2 of the gate's 15 cases fail;
with it they pass. `double` trajectories are bit-identical.

### Capture 2: commit `6d862a7`, PASS

Image `probe.bin` sha256
`81d5fab44824838c45aab50a29a7d7011218d6f17e1a0820f707d67d60c782cb`; ELF text
765,484 B, data 464 B, bss 103,160 B.

```
[argmin] NUCLEO-H753ZI on-device allocation proof
[argmin] blindness canary PASS (deliberate allocation observed)
[overflow-probe] forced 10368 B temporary > 8192 B limit: heap fallback observed (1 allocs) PASS
[argmin] SystemCoreClock = 64000000 Hz
  [timing] kraft_slsqp hs071 cycles=1878122 armed_steps=10 per_step=187812
  [alloc-gate] kraft_slsqp hs071  eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] kraft_slsqp hs071 PASS (zero-alloc gate)
  [timing] nw_sqp hs071 cycles=2707784 armed_steps=10 per_step=270778
  [alloc-gate] nw_sqp hs071       eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] nw_sqp hs071 PASS (zero-alloc gate)
  [timing] filter_nw_sqp hs071 cycles=2713989 armed_steps=10 per_step=271398
  [alloc-gate] filter_nw_sqp hs071 eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] filter_nw_sqp hs071 PASS (zero-alloc gate)
  [timing] lm rosenbrock_ls cycles=44497 armed_steps=10 per_step=4449
  [alloc-gate] lm rosenbrock_ls   eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] lm rosenbrock_ls PASS (zero-alloc gate)
  [timing] kraft_slsqp hs071 float cycles=1709559 armed_steps=10 per_step=170955
  [alloc-gate] kraft_slsqp hs071 float eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] kraft_slsqp hs071 float PASS (zero-alloc gate)
  [timing] nw_sqp hs071 float cycles=2071380 armed_steps=10 per_step=207138
  [alloc-gate] nw_sqp hs071 float eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] nw_sqp hs071 float PASS (zero-alloc gate)
  [timing] filter_nw_sqp hs071 float cycles=2100960 armed_steps=10 per_step=210096
  [alloc-gate] filter_nw_sqp hs071 float eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] filter_nw_sqp hs071 float PASS (zero-alloc gate)
  [timing] lm rosenbrock_ls float cycles=36672 armed_steps=10 per_step=3667
  [alloc-gate] lm rosenbrock_ls float eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] lm rosenbrock_ls float PASS (zero-alloc gate)
[argmin] _sbrk one-time setup high-water = 12688 bytes (15 calls)
[argmin] RT-window gate result: PASS (0 allocs/step on all windows)
```

### Reading

- **Allocations: 0.00/step on all eight windows.** That covers `kraft_slsqp`,
  `nw_sqp`, `filter_nw_sqp` and `lm`, each in `double` and in `float`. The
  blindness canary and the over-limit guardrail probe both PASS, and the
  `_sbrk` one-time setup high-water (12,688 B, 15 calls) is unchanged from the
  `double`-only image.
- **Cycles per armed step at 64 MHz, recorded only:**

  | window | `double` | `float` |
  |---|---|---|
  | `kraft_slsqp` hs071 | 187,812 | 170,955 |
  | `nw_sqp` hs071 | 270,778 | 207,138 |
  | `filter_nw_sqp` hs071 | 271,398 | 210,096 |
  | `lm` rosenbrock_ls | 4,449 | 3,667 |

  These figures are not a timing claim, and the two columns do not compare
  like with like. Each window covers steps 3-12 of a cold solve. The `float`
  `nw_sqp` and `filter_nw_sqp` solves converge at step 7 and spend the rest of
  the window at the solution, while the `double` solves are still descending. The core also runs
  at the HSI default rather than the part's rated clock.
