# RT-safety matrix

This page states, per shipped solver and driver, what real-time (RT) guarantees
argmin makes and — for every "yes" — names the test or CI artifact that proves it.
The discipline is deliberate: a "yes" without a proving artifact would export a false
guarantee to a downstream worst-case-execution-time budget. Cells that are honest "no"
are what make the rest of the table trustworthy.

argmin is a passive RT kernel: it owns no thread, scheduler, timer, or loop. The caller
owns scheduling. Every property below is a *local* property argmin achieves on its own,
with no dependency on any consumer. The stack-wide contract this table feeds lives in
the ctrlpp/plexus real-time coordination charter (`STACK-RT-SAFETY-CONTRACT.md`); publish
argmin rows against the same frozen schema.

## What each column means

- **allocation-free?** — no heap allocation on the warm-started steady-state `step()` at a
  fixed compile-time dimension `N`, as measured by the un-blinded allocation-counting gate
  (a global `operator new`/`delete` counter plus `malloc` interposition, armed over a
  warm-reset window after a warm-up solve). A "yes" here is certified by a gate registered
  in **zero mode** (fails the build if a single allocation fires on an armed step); a "no"
  citing a **witness-mode** gate reports the measured steady-state count without asserting
  zero. Allocation-freedom is a property of the *policy at fixed `N`*, not of the driver.
- **bounded-iterations?** — the solve terminates in a finite, statically bounded iteration
  count. The outer loop honors `max_iterations`; every inner leaf loop carries a documented
  cap (line search ≤ 40 evaluations, NNLS ≤ 3n, Steihaug-CG ≤ 2·(n+m), feasibility
  restoration ≤ 10 steps).
- **wall-clock-free?** — the `step()` path reads no wall clock. This falls out of the
  **driver type**: `step_budget_solver` and `stepper` never include `<chrono>`, so a
  translation unit that budgets purely by iterations cannot transitively acquire the clock.
  The clock-bearing `time_budget_solver` / `step_and_time_budget_solver` are the only drivers
  that read a clock, and only there.
- **exceptions-off-clean?** — compiles *and links* under `-fno-exceptions -fno-rtti` at the
  C++20 floor: no `throw`, no `dynamic_cast`/`typeid`. The library has zero throw sites; the
  instantiation probe additionally links-and-runs the RT-claimed policies so a surviving
  throw fails the build rather than passing a header parse.
- **deterministic(seeded)?** — identical inputs (and, for the stochastic policies, an
  identical seed) reproduce the same trajectory bit-for-bit. The SQP / L-BFGS-B / LM /
  Gauss-Newton / MMA families carry no RNG; the CMA-ES and ISRES families take an injectable,
  seed-deterministic RNG.
- **evidence** — the named artifact proving each "yes". Alloc gates run under the ctest label
  `alloc-gate` (CI job *labeled instruments (alloc-gate, oracle-pin)*). The exceptions-off
  probe is `argmin_no_exceptions_probe` (CI job *exceptions-off (instantiation probe)*). The
  standards matrix builds and tests at C++20 and C++23 (CI jobs *c++20 / c++23 (build + ctest)*).

## Drivers

Driver rows carry the taxonomy properties; the `allocation-free?` and `deterministic?`
cells are inherited from whichever policy is plugged in (see the policy rows). All four
satisfy the behavioral solving concepts, so solver groups and benchmark adapters stay
uniform across driver choices.

| module | allocation-free? | bounded-iterations? | wall-clock-free? | exceptions-off-clean? | deterministic(seeded)? | evidence |
|---|---|---|---|---|---|---|
| `step_budget_solver` | per policy | yes | **yes (by construction)** | yes | per policy | `rt.h` re-exports it and pins that no transitively-included header pulls `<chrono>`; the concept probe asserts it satisfies the loop-owning solving concept |
| `stepper` | per policy | yes (caller owns the budget; one bounded step per call, no internal loop) | **yes (by construction)** | yes | per policy | `rt.h` concept probe asserts `stepper` satisfies the passive steppable concept but *not* the loop-owning refinement; no `<chrono>` in its include closure |
| `time_budget_solver` | per policy | yes | **no** (adds a `steady_clock` deadline; owns `solve_result` wall-time reporting) | yes | no (termination point is wall-clock-dependent) | the only driver pulling `<chrono>`; excluded from the `rt.h` umbrella |
| `step_and_time_budget_solver` | per policy | yes | **no** (both caps; whichever fires first) | yes | no (termination point is wall-clock-dependent) | excluded from the `rt.h` umbrella |

## Policies

Every "yes" in the `allocation-free?` column below names a zero-mode gate. The SQP zero-mode
gates certify the **warm-started pre-convergence RT operating regime** (bounded per-tick
steps, never idling); see the footnotes for the two characterized off-hot-loop residuals, and
"Scope of the allocation-free claim" below for the dense fixed-`N` regime these gates hold in.

| module | allocation-free? | bounded-iterations? | wall-clock-free? | exceptions-off-clean? | deterministic(seeded)? | evidence |
|---|---|---|---|---|---|---|
| `kraft_slsqp` | **yes** | yes | yes | yes | yes | `sqp_alloc_gate_kraft` (zero mode, 0.00/step); `argmin_no_exceptions_probe`; no RNG |
| `filter_slsqp` | **yes** [^restore] | yes | yes | yes | yes | `sqp_alloc_gate_filter_slsqp` (zero mode, 0.00/step); `argmin_no_exceptions_probe`; no RNG |
| `nw_sqp` | **yes** | yes | yes | yes | yes | `sqp_alloc_gate_nw` (zero mode, 0.00/step, fixed-`N` steady state); `argmin_no_exceptions_probe`; no RNG |
| `filter_nw_sqp` | **yes** [^restore] | yes | yes | yes | yes | `sqp_alloc_gate_filter_nw` (zero mode, 0.00/step, fixed-`N` steady state); `argmin_no_exceptions_probe`; no RNG |
| `tr_sqp` | **yes** [^nullspace] | yes | yes | yes | yes | `sqp_alloc_gate_tr_sqp` (zero mode, 0.00/step, on an equality-constrained fixture); `argmin_no_exceptions_probe`; no RNG |
| `filter_trsqp` | **yes** [^restore] [^nullspace] | yes | yes | yes | yes | `sqp_alloc_gate_filter_trsqp` (zero mode, 0.00/step, on an equality-constrained fixture); `argmin_no_exceptions_probe`; no RNG |
| `lbfgsb` | **yes** | yes | yes | yes | yes | `alloc_gate_lbfgsb` (zero mode, incl. the bound-active generalized-Cauchy-point/subspace branch); `argmin_no_exceptions_probe`; no RNG |
| `byrd_lbfgsb` | **yes** | yes | yes | yes | yes | `alloc_gate_byrd_lbfgsb` (zero mode, bound-active path); `argmin_no_exceptions_probe`; no RNG |
| `lm` | **yes** | yes | yes | yes | yes | `alloc_gate_lm` (zero mode, 0.00/step, fixed-`N` steady state); `argmin_no_exceptions_probe`; no RNG |
| `projected_gn` | **yes** | yes | yes | yes | yes | `alloc_gate_projected_gn` (zero mode, active-bound fixture); `argmin_no_exceptions_probe`; no RNG |
| `projected_gradient_gn` | **yes** | yes | yes | yes* | yes | `alloc_gate_projected_gradient_gn` (zero mode, active-bound fixture); throw-free library-wide (not individually instantiated by the probe); no RNG |
| `augmented_lagrangian` | **yes** | yes | yes | yes | yes | `alloc_gate_augmented_lagrangian` (zero mode; bounded resumable inner solve, mu-change + warm-reset armed); `argmin_no_exceptions_probe`; no RNG |
| `bobyqa` | **no** | yes | yes | yes* | yes | no alloc gate (derivative-free quadratic-model rebuild allocates; not RT-claimed); throw-free library-wide; no RNG |
| `cobyla` | **no** | yes | yes | yes* | yes | no alloc gate (not RT-claimed); throw-free library-wide; no RNG |
| `cmaes` | **no** | yes | yes | yes | yes (seeded) | no alloc gate (stochastic population sampling allocates; not RT-claimed); `argmin_no_exceptions_probe` (throw sites converted to status returns); seed determinism: `cmaes` generation-pin + sampler bit-identity tests |
| `isres` | **no** | yes | yes | yes* | yes (seeded) | no alloc gate (stochastic; not RT-claimed); throw-free library-wide; seeded stochastic-ranking ES tests |
| `mma` | **no** | yes | yes | yes* | yes | no alloc gate (separable-approximation subproblem allocates; not RT-claimed); throw-free library-wide; no RNG |
| `gcmma` | **no** | yes | yes | yes* | yes | no alloc gate (not RT-claimed); throw-free library-wide; no RNG |
| `ccsa_quadratic` | **no** | yes | yes | yes* | yes | no alloc gate (not RT-claimed); throw-free library-wide; no RNG |

\* `exceptions-off-clean?` marked `yes*` is a library-wide property (the library has zero
`throw` sites and no RTTI, and the standards-matrix CI compiles the policy at C++20 and C++23):
these policies are throw-free but are not among the set the `-fno-exceptions -fno-rtti`
instantiation probe links and runs. The probe covers the RT-claimed set (`lbfgsb`,
`byrd_lbfgsb`, `cmaes`, `kraft_slsqp`, `filter_slsqp`, `nw_sqp`, `filter_nw_sqp`, `tr_sqp`,
`filter_trsqp`, `augmented_lagrangian`, `lm`, `projected_gn`).

## Scope of the allocation-free claim

The zero-mode `allocation-free?` guarantees above hold in the **dense unblocked-Householder
regime**: a fixed compile-time dimension `N < 49`. Eigen's `HouseholderSequence` switches from
its unblocked per-reflector apply (which reuses caller-supplied storage and allocates nothing)
to a blocked apply that allocates a block-reflector workspace once the sequence length exceeds
its block size of 48 — `enum { BlockSize = 48 }` in `Eigen/src/Householder/HouseholderSequence.h`,
gating the blocked branch on `m_length > BlockSize`; at `N = 49` the blocked path is the first
to allocate. This is exactly the regime where a small-dense fixed-`N` solver is the right tool:
dense O(N³) work at `N ≥ 49` belongs on a sparse/dynamic path, so **no allocation-free claim is
made or advertised for `N ≥ 49`**.

Per-algo level reached (measured at the pinned Release config; the alloc-gate evidence certifies
it): the three separately-gated real-time-path solvers — `nw_sqp`, `filter_nw_sqp`, and `lm` —
each reach **steady-state 0.00 allocations/step, gated red-on-regression** (four, three, and two
armed windows respectively); restoration is covered through the `filter_nw` gate. None reaches
the stricter **zero-after-construction**: a one-time setup cost survives at construction plus the
first step — **75** allocations for `nw_sqp`, **76** for `filter_nw_sqp`, **5** for `lm` (the
QP-solver workspace built at construction plus per-state buffer resizes, warmed by the harness
before any armed steady window). These one-time allocations are documented and deferred, not
gated. The off-hot-loop post-convergence restoration idle window improved from 56 to 3
allocations per 10 idle steps as a side effect of the shared active-set QP substrate plumbing;
it remains informational and ungated (see the restoration footnote).

## Footnotes on the allocation-free cells

[^restore]: **Post-convergence feasibility restoration (filter families).** The filter zero-mode
    gates certify the *pre-convergence* RT regime. The shared `detail::restore_l1` feasibility
    restoration allocates its local work vectors, but on a fixed-`N` fixture this fires only on
    *post-convergence idle steps* — a zero-step QP at a converged iterate whose constraint
    violation sits marginally above the restoration trigger (an `-O3` FMA/vectorization rounding
    artifact; an `-O2` build never restores). This is outside the real-time operating regime
    (warm-started, bounded per-tick steps, never idling). It is a characterized, documented
    residual pending a future allocation-free rewrite of the shared restoration helper, left
    byte-for-byte unchanged by choice. `kraft_slsqp` has no restoration path, so its window is
    incidentally allocation-free post-convergence too.

[^nullspace]: **Box/inequality free-set-restart projection (trust-region families).** The
    trust-region gates are certified on an equality-constrained fixture, which never engages the
    Lin-More free-set restart. On a bound- or inequality-constrained iterate whose pinned set
    exhausts the reduced free dimension, the augmented normal equations go rank-deficient and the
    shared `detail::null_space_project` takes a pivoted-QR fallback branch that heap-allocates per
    projection. Removing this in place would require a numerics-changing ridge (it flips
    box-active suite cells), so it is a characterized shared-subsystem residual pending an
    allocation-free rewrite of the shared projection helper.
