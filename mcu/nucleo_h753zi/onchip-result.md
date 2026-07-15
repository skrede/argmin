# NUCLEO-H753ZI on-chip allocation result (operator bench artifact)

Captured 2026-07-15 on a physical NUCLEO-H753ZI (bare-metal Cortex-M7 @ 64 MHz
HSI, no RTOS, newlib-nano), app flashed over the on-board ST-Link by drag-drop
to the `NOD_H753ZI` mass-storage drive. Report streamed over USART3 → the
ST-Link VCP (`/dev/ttyACM0`) — no semihosting/OpenOCD. This is the D-02
*rigorous* bare-metal proof.

## Console (USART3 / ST-Link VCP)

```
[argmin] NUCLEO-H753ZI on-device allocation proof
[argmin] blindness canary PASS (deliberate allocation observed)
  [alloc-gate] kraft_slsqp hs071  eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] kraft_slsqp hs071 PASS (zero-alloc gate)
  [alloc-gate] nw_sqp hs071       eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] nw_sqp hs071 PASS (zero-alloc gate)
  [alloc-gate] filter_nw_sqp hs071 eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] filter_nw_sqp hs071 PASS (zero-alloc gate)
  [alloc-gate] lm rosenbrock_ls   eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] lm rosenbrock_ls PASS (zero-alloc gate)
[argmin] _sbrk one-time setup high-water = 12672 bytes (95 calls)
[argmin] RT-window gate result: PASS (0 allocs/step on all windows)
```

## Result

- **All four RT-claimed policies (`kraft_slsqp`, `nw_sqp`, `filter_nw_sqp`,
  `lm`): 0.00 allocs/step** in the armed steady-state windows — the RT claim
  holds on real bare-metal hardware, reproducing the Phase 60 host contract.
  Both sensors agree (Eigen-native + `_sbrk`/`--wrap`).
- **Blindness canary PASS**: a deliberate `malloc(64)` + `Eigen::VectorXd(37)`
  in an armed window was observed, so the zeros are meaningful, not blind.
- `_sbrk` one-time setup high-water = **12,672 bytes** (95 calls) across the
  four policies (informational; the RT claim is zero-*per-step*, not
  zero-total-heap).

This result required `EIGEN_ALLOCA` to be defined explicitly for the
cross-build; the history below records the failure it fixed and the controlled
experiment that attributes it. Flags and the stack-budgeting rule:
`docs/embedded.md`.

## History: `kraft_slsqp` at 2.80/step, root-caused to a missing `EIGEN_ALLOCA`

The first capture on this board (same day, before the flag fix) read:

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
