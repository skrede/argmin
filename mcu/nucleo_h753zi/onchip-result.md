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
  [alloc-gate] kraft_slsqp hs071   eigen_malloc=28 c_alloc=28 armed_steps=10 per_step=2.80
  [alloc-gate] kraft_slsqp hs071 FAIL: zero-alloc gate saw 28 allocations
  [alloc-gate] nw_sqp hs071        eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] nw_sqp hs071 PASS (zero-alloc gate)
  [alloc-gate] filter_nw_sqp hs071 eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] filter_nw_sqp hs071 PASS (zero-alloc gate)
  [alloc-gate] lm rosenbrock_ls    eigen_malloc=0 c_alloc=0 armed_steps=10 per_step=0.00
  [alloc-gate] lm rosenbrock_ls PASS (zero-alloc gate)
[argmin] _sbrk one-time setup high-water = 13136 bytes (97 calls)
[argmin] RT-window gate result: FAIL (see per-window lines above)
```

## Result

- **nw_sqp, filter_nw_sqp, lm: 0.00 allocs/step** in the armed steady-state
  windows — the RT claim holds on real bare-metal hardware, reproducing the
  Phase 60 host contract. Both sensors agree (Eigen-native + `_sbrk`/`--wrap`).
- **Blindness canary PASS**: a deliberate `malloc(64)` + `Eigen::VectorXd(37)`
  in an armed window was observed, so the zeros are meaningful, not blind.
- `_sbrk` one-time setup high-water = **13,136 bytes** across the four policies
  (informational; the RT claim is zero-*per-step*, not zero-total-heap).

## Finding: `kraft_slsqp` allocates 2.80/step on arm bare-metal (deferred)

`kraft_slsqp` heap-allocates **2.80×/step** on the NUCLEO (28 events / 10
steps, both sensors agreeing) — while it is **0.00/step on the host, even when
the host is compiled with the identical flags** (`-fno-exceptions -fno-rtti`,
`EIGEN_MAX_ALIGN_BYTES=8`, `EIGEN_DONT_VECTORIZE`, and the exact same probe
workload). It is therefore an **arm-none-eabi/newlib-target-specific**
allocation, not a config artifact.

This was a **pre-flagged risk, now measured** — exactly what SEED-041 scoped
("the actual on-device ARM/newlib proof needs a different hook … the
NUCLEO-H753ZI proof worth starting" — and it names `kraft_slsqp`). History:
the round-1 embeddability review measured kraft at 14 mallocs/step on host
(the old gate was blind to Eigen allocations); Phase 55 hoisted kraft's
workspaces to 0.00/step **on host** by retyping the LSEI/LSI Householder-Q
storage to fixed-N-max-bounded form. That host-only fix does not fully hold
under arm codegen: some Eigen/QP-substrate temporary that `alloca`s on x86
spills to the heap on arm.

**Ruled out** (measured): `EIGEN_STACK_ALLOCATION_LIMIT` (sweeping 8192 →
65536 left the binary byte-identical and kraft still allocated); the probe
warmup pattern (identical on host = 0.00); the individual Eigen macros.
**Follow-up**: narrow which call in the `lsei.h`/`lsi.h`/`ldp.h` Kraft-LSEI
path allocates under arm-none-eabi codegen (candidates from the round-1 review:
Householder-Q materializations `lsei.h:170-171,232`, the deliberate fixed→
dynamic buffer copy `lsi.h:104-124`, the LDP dynamic `r(n+1)` `ldp.h:112`).
The gate is left honestly RED for kraft — not force-greened.
