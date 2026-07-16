# RT-safety matrix

This page states, per shipped solver and driver, what real-time (RT) guarantees
argmin makes and — for every "yes" — either names the test or CI artifact that proves it or
marks the claim as argued from reasoning with nothing standing behind it. The discipline is
deliberate: a "yes" whose strength a reader cannot see would export a false guarantee to a
downstream worst-case-execution-time budget. The honest "no" cells, and the "yes" cells
marked argued, are what make the rest of the table trustworthy.

argmin is a passive RT kernel: it owns no thread, scheduler, timer, or loop. The caller
owns scheduling. Every property below is a *local* property argmin achieves on its own,
with no dependency on any consumer. The stack-wide contract this table feeds lives in
the ctrlpp/plexus real-time coordination charter (`STACK-RT-SAFETY-CONTRACT.md`); publish
argmin rows against the same ratified column set. That contract is still a living document
and has not frozen its schema, so the column set is agreed rather than final.

## What each column means

- **allocation-free?** — no heap allocation on the warm-started steady-state `step()` at a
  fixed compile-time dimension `N`, as measured by the un-blinded allocation-counting gate
  (a global `operator new`/`delete` counter plus `malloc` interposition, armed over a
  warm-reset window after a warm-up solve). A "yes" here is certified by a gate registered
  in **zero mode** (fails the build if a single allocation fires on an armed step); a "no"
  citing a **witness-mode** gate reports the measured steady-state count without asserting
  zero. Allocation-freedom is a property of the *policy at fixed `N`*, not of the driver.
- **bounded-iterations?** — the solve terminates in a finite iteration count: every loop,
  outer and inner, carries a cap, so no path can spin unboundedly. The outer loop honors
  `max_iterations`; each inner leaf loop carries a documented cap (line search ≤ 40
  evaluations, NNLS ≤ 3n, Steihaug-CG ≤ 2·(n+m), feasibility restoration ≤ 10 steps). Those
  bounds are *configured*, not static: the NNLS `3n` cap is hard-coded, but the other three
  are option defaults a caller may raise, and a caller that raises one raises its own
  worst-case bound with it. This is still a usable real-time property — a real-time
  integrator pins its options at the configuration it ships, and at that configuration each
  default cap is a real bound. It is a property of the configuration rather than of the type,
  and no test asserts that the caps bind, which is why the column renders as argued
  throughout rather than gated.
- **wall-clock-free?** — the `step()` path reads no wall clock. This falls out of the
  **driver type**: `step_budget_solver` and `stepper` never include `<chrono>`, so a
  translation unit that budgets purely by iterations cannot transitively acquire the clock.
  The clock-bearing `time_budget_solver` / `step_and_time_budget_solver` are the only drivers
  that read a clock, and only there. Two probe translation units — one including the two
  drivers, one including every policy — are scanned with the compiler's own dependency dump,
  which fails if `<chrono>` appears anywhere in the transitive include graph. The scan reads
  the real include set rather than a guard macro, and it walks the graph rather than the
  literal include lines, so a header that reaches a clock indirectly is caught. The probes are
  separate so that each claim fails on its own evidence rather than on the other's.
- **exceptions-off-clean?** — compiles *and links* under `-fno-exceptions -fno-rtti` at the
  C++20 floor: no `throw`, no `dynamic_cast`/`typeid`. The library has zero throw sites; the
  instantiation probe additionally links-and-runs the RT-claimed policies so a surviving
  throw fails the build rather than passing a header parse.
- **deterministic(seeded)?** — identical inputs (and, for the stochastic policies, an
  identical seed) reproduce the same trajectory bit-for-bit *on a fixed target*. The SQP /
  L-BFGS-B / LM / Gauss-Newton / MMA families carry no RNG; the CMA-ES and ISRES families take
  an injectable, seed-deterministic RNG. Where the claim rests on a policy carrying no RNG, a
  probe translation unit including those policies is scanned for one: no `<random>` in the
  transitive include graph, and no RNG facility named in any argmin header that graph reaches.
  The second half is not redundant — `<cstdlib>` is reachable in every build, so `rand()` is
  always callable, and a hand-rolled generator needs no header at all. `cobyla` is the
  exception and is excluded from that scan: it jitters degenerate geometry steps with a
  self-contained linear congruential generator (inherited from the NLopt/SGJ line), seeded
  from the problem dimensions. It reproduces, but it reproduces *because of how it is seeded*
  rather than because there is no generator in it, so its cell stays argued and says so. This
  column is a per-policy property; what is and is not guaranteed once you cross an
  architecture or an instantiation boundary is a different question, and it is answered by the
  tiered claim in [Determinism](determinism.md).
- **evidence** — for a gated cell, the named artifact proving the "yes"; for an argued cell,
  the reasoning it rests on, published next to the claim rather than left implicit. Alloc
  gates run under the ctest label `alloc-gate` (CI job *labeled instruments (alloc-gate,
  oracle-pin)*). The exceptions-off probe is `argmin_no_exceptions_probe`, which also runs
  under the standards matrix; that matrix builds and tests at C++20 and C++23 as two separate
  legs (CI jobs *c++20 (build + ctest)* and *c++23 (build + ctest)*).

### Gated and argued

Every "yes" in the tables below carries its evidence class, and the distinction is the point
of the table rather than a caveat on it:

- **yes** *(gated)* — a named check stands behind the claim and turns red if the claim stops
  being true. The artifact is named in the evidence column, it exists, and a named CI job
  runs it.
- yes *(argued)* — the claim rests on reasoning with no artifact behind it. It is a claim the
  maintainers believe and have stated their grounds for, but nothing fails if it silently
  becomes false.

An argued cell is an honest claim, not a hidden one, and it is not a "no". But a reader
budgeting worst-case execution time should treat the two differently: a gated cell is a
property this repository defends on every commit, while an argued cell is a property you may
wish to re-verify against your own configuration. About a third of the "yes" cells below are
argued — most visibly the entire `bounded-iterations?` column, and the
`exceptions-off-clean?` cells for the policies the instantiation probe does not link. Those
cells never had artifacts; naming that is what the class is for.

## Drivers

Driver rows carry the taxonomy properties; the `allocation-free?` and `deterministic?`
cells are inherited from whichever policy is plugged in (see the policy rows). All four
satisfy the behavioral solving concepts, so solver groups and benchmark adapters stay
uniform across driver choices.

Both tables below are rendered from the checked-in cell data file
`scripts/rt_matrix_cells.json` by `scripts/rt_matrix.py`. A hand edit between the generated
markers is overwritten by the next render, so a correction to a claim, its class, or its
evidence belongs in the data file. Everything outside the markers — the column legend, the
scope argument below, and the footnotes — is hand-written analysis and is never generated.

<!-- BEGIN GENERATED: drivers -->
| module | allocation-free? | bounded-iterations? | wall-clock-free? | exceptions-off-clean? | deterministic(seeded)? | evidence |
|---|---|---|---|---|---|---|
| `step_budget_solver` | per policy | yes *(argued)* | **yes** *(gated)* | **yes** *(gated)* | per policy | the outer loop honors max_iterations, an option default rather than a static limit; runner_chrono_freedom_scan (GCC/Clang only); runner_chrono_freedom_compiles; c++20 (build + ctest); c++23 (build + ctest); argmin_no_exceptions_probe (the probe drives every policy through this driver) |
| `stepper` | per policy | yes *(argued)* | **yes** *(gated)* | yes *(argued)* | per policy | the caller owns the budget; one bounded step per call, with no internal loop; runner_chrono_freedom_scan (GCC/Clang only); runner_chrono_freedom_compiles; c++20 (build + ctest); c++23 (build + ctest); throw-free and RTTI-free library-wide, but the instantiation probe does not link this driver |
| `time_budget_solver` | per policy | yes *(argued)* | no (adds a steady_clock deadline; owns the wall-time reporting) | yes *(argued)* | no (the termination point is wall-clock-dependent) | the stored max_iterations remains a hard safety cap; the deadline can only terminate earlier; throw-free and RTTI-free library-wide, but the instantiation probe does not link this driver |
| `step_and_time_budget_solver` | per policy | yes *(argued)* | no (both caps; whichever fires first) | yes *(argued)* | no (the termination point is wall-clock-dependent) | the stored max_iterations remains a hard safety cap; the deadline can only terminate earlier; throw-free and RTTI-free library-wide, but the instantiation probe does not link this driver |
<!-- END GENERATED: drivers -->

## Policies

Every "yes" in the `allocation-free?` column below names a zero-mode gate. The SQP zero-mode
gates certify the **warm-started pre-convergence RT operating regime** (bounded per-tick
steps, never idling); see the footnotes for the two characterized off-hot-loop residuals, and
"Scope of the allocation-free claim" below for the dense fixed-`N` regime these gates hold in.

<!-- BEGIN GENERATED: policies -->
| module | allocation-free? | bounded-iterations? | wall-clock-free? | exceptions-off-clean? | deterministic(seeded)? | evidence |
|---|---|---|---|---|---|---|
| `kraft_slsqp` | **yes** *(gated)* | yes *(argued)* | **yes** *(gated)* | **yes** *(gated)* | **yes** *(gated)* | sqp_alloc_gate_kraft (zero mode, 0.00/step); labeled instruments (alloc-gate, oracle-pin); every loop is iteration-capped, but the binding caps are option defaults rather than static limits; policy_chrono_freedom_scan (GCC/Clang only); policy_chrono_freedom_compiles; c++20 (build + ctest); c++23 (build + ctest); argmin_no_exceptions_probe; policy_rng_freedom_scan (GCC/Clang only); policy_rng_freedom_compiles |
| `filter_slsqp` | **yes** *(gated)* [^restore] | yes *(argued)* | **yes** *(gated)* | **yes** *(gated)* | **yes** *(gated)* | sqp_alloc_gate_filter_slsqp (zero mode, 0.00/step); labeled instruments (alloc-gate, oracle-pin); every loop is iteration-capped, but the binding caps are option defaults rather than static limits; policy_chrono_freedom_scan (GCC/Clang only); policy_chrono_freedom_compiles; c++20 (build + ctest); c++23 (build + ctest); argmin_no_exceptions_probe; policy_rng_freedom_scan (GCC/Clang only); policy_rng_freedom_compiles |
| `nw_sqp` | **yes** *(gated)* | yes *(argued)* | **yes** *(gated)* | **yes** *(gated)* | **yes** *(gated)* | sqp_alloc_gate_nw (zero mode, 0.00/step, fixed-N steady state); labeled instruments (alloc-gate, oracle-pin); every loop is iteration-capped, but the binding caps are option defaults rather than static limits; policy_chrono_freedom_scan (GCC/Clang only); policy_chrono_freedom_compiles; c++20 (build + ctest); c++23 (build + ctest); argmin_no_exceptions_probe; policy_rng_freedom_scan (GCC/Clang only); policy_rng_freedom_compiles |
| `filter_nw_sqp` | **yes** *(gated)* [^restore] | yes *(argued)* | **yes** *(gated)* | **yes** *(gated)* | **yes** *(gated)* | sqp_alloc_gate_filter_nw (zero mode, 0.00/step, fixed-N steady state); labeled instruments (alloc-gate, oracle-pin); every loop is iteration-capped, but the binding caps are option defaults rather than static limits; policy_chrono_freedom_scan (GCC/Clang only); policy_chrono_freedom_compiles; c++20 (build + ctest); c++23 (build + ctest); argmin_no_exceptions_probe; policy_rng_freedom_scan (GCC/Clang only); policy_rng_freedom_compiles |
| `tr_sqp` | **yes** *(gated)* [^nullspace] | yes *(argued)* | **yes** *(gated)* | **yes** *(gated)* | **yes** *(gated)* | sqp_alloc_gate_tr_sqp (zero mode, 0.00/step, on an equality-constrained fixture); labeled instruments (alloc-gate, oracle-pin); every loop is iteration-capped, but the binding caps are option defaults rather than static limits; policy_chrono_freedom_scan (GCC/Clang only); policy_chrono_freedom_compiles; c++20 (build + ctest); c++23 (build + ctest); argmin_no_exceptions_probe; policy_rng_freedom_scan (GCC/Clang only); policy_rng_freedom_compiles |
| `filter_trsqp` | **yes** *(gated)* [^restore] [^nullspace] | yes *(argued)* | **yes** *(gated)* | **yes** *(gated)* | **yes** *(gated)* | sqp_alloc_gate_filter_trsqp (zero mode, 0.00/step, on an equality-constrained fixture); labeled instruments (alloc-gate, oracle-pin); every loop is iteration-capped, but the binding caps are option defaults rather than static limits; policy_chrono_freedom_scan (GCC/Clang only); policy_chrono_freedom_compiles; c++20 (build + ctest); c++23 (build + ctest); argmin_no_exceptions_probe; policy_rng_freedom_scan (GCC/Clang only); policy_rng_freedom_compiles |
| `lbfgsb` | **yes** *(gated)* | yes *(argued)* | **yes** *(gated)* | **yes** *(gated)* | **yes** *(gated)* | alloc_gate_lbfgsb (zero mode, including the bound-active generalized-Cauchy-point/subspace branch); labeled instruments (alloc-gate, oracle-pin); every loop is iteration-capped, but the binding caps are option defaults rather than static limits; policy_chrono_freedom_scan (GCC/Clang only); policy_chrono_freedom_compiles; c++20 (build + ctest); c++23 (build + ctest); argmin_no_exceptions_probe; policy_rng_freedom_scan (GCC/Clang only); policy_rng_freedom_compiles |
| `byrd_lbfgsb` | **yes** *(gated)* | yes *(argued)* | **yes** *(gated)* | **yes** *(gated)* | **yes** *(gated)* | alloc_gate_byrd_lbfgsb (zero mode, bound-active path); labeled instruments (alloc-gate, oracle-pin); every loop is iteration-capped, but the binding caps are option defaults rather than static limits; policy_chrono_freedom_scan (GCC/Clang only); policy_chrono_freedom_compiles; c++20 (build + ctest); c++23 (build + ctest); argmin_no_exceptions_probe; policy_rng_freedom_scan (GCC/Clang only); policy_rng_freedom_compiles |
| `lm` | **yes** *(gated)* | yes *(argued)* | **yes** *(gated)* | **yes** *(gated)* | **yes** *(gated)* | alloc_gate_lm (zero mode, 0.00/step, fixed-N steady state); labeled instruments (alloc-gate, oracle-pin); every loop is iteration-capped, but the binding caps are option defaults rather than static limits; policy_chrono_freedom_scan (GCC/Clang only); policy_chrono_freedom_compiles; c++20 (build + ctest); c++23 (build + ctest); argmin_no_exceptions_probe; policy_rng_freedom_scan (GCC/Clang only); policy_rng_freedom_compiles |
| `projected_gn` | **yes** *(gated)* | yes *(argued)* | **yes** *(gated)* | **yes** *(gated)* | **yes** *(gated)* | alloc_gate_projected_gn (zero mode, active-bound fixture); labeled instruments (alloc-gate, oracle-pin); every loop is iteration-capped, but the binding caps are option defaults rather than static limits; policy_chrono_freedom_scan (GCC/Clang only); policy_chrono_freedom_compiles; c++20 (build + ctest); c++23 (build + ctest); argmin_no_exceptions_probe; projected_gn_policy: trajectory is run-to-run deterministic on a fixed target (run-to-run and post-reset bit-exactness) |
| `projected_gradient_gn` | **yes** *(gated)* | yes *(argued)* | **yes** *(gated)* | yes *(argued)* | **yes** *(gated)* | alloc_gate_projected_gradient_gn (zero mode, active-bound fixture); labeled instruments (alloc-gate, oracle-pin); every loop is iteration-capped, but the binding caps are option defaults rather than static limits; policy_chrono_freedom_scan (GCC/Clang only); policy_chrono_freedom_compiles; c++20 (build + ctest); c++23 (build + ctest); throw-free and RTTI-free library-wide, but the instantiation probe does not link this policy; policy_rng_freedom_scan (GCC/Clang only); policy_rng_freedom_compiles |
| `augmented_lagrangian` | **yes** *(gated)* | yes *(argued)* | **yes** *(gated)* | **yes** *(gated)* | **yes** *(gated)* | alloc_gate_augmented_lagrangian (zero mode; bounded resumable inner solve, mu-change and warm-reset armed); labeled instruments (alloc-gate, oracle-pin); every loop is iteration-capped, but the binding caps are option defaults rather than static limits; policy_chrono_freedom_scan (GCC/Clang only); policy_chrono_freedom_compiles; c++20 (build + ctest); c++23 (build + ctest); argmin_no_exceptions_probe; policy_rng_freedom_scan (GCC/Clang only); policy_rng_freedom_compiles |
| `bobyqa` | not RT-claimed (no alloc gate; the derivative-free quadratic-model rebuild allocates) | yes *(argued)* | **yes** *(gated)* | yes *(argued)* | **yes** *(gated)* | every loop is iteration-capped, but the binding caps are option defaults rather than static limits; policy_chrono_freedom_scan (GCC/Clang only); policy_chrono_freedom_compiles; c++20 (build + ctest); c++23 (build + ctest); throw-free and RTTI-free library-wide, but the instantiation probe does not link this policy; policy_rng_freedom_scan (GCC/Clang only); policy_rng_freedom_compiles |
| `cobyla` | not RT-claimed (no alloc gate) | yes *(argued)* | **yes** *(gated)* | yes *(argued)* | yes *(argued)* | every loop is iteration-capped, but the binding caps are option defaults rather than static limits; policy_chrono_freedom_scan (GCC/Clang only); policy_chrono_freedom_compiles; c++20 (build + ctest); c++23 (build + ctest); throw-free and RTTI-free library-wide, but the instantiation probe does not link this policy; the policy jitters degenerate geometry steps with a self-contained generator seeded from the problem dimensions, so identical inputs do reproduce -- but the claim rests on that seeding rather than on the absence of an RNG, which is why the no-RNG scan excludes it |
| `cmaes` | not RT-claimed (no alloc gate; stochastic population sampling allocates) | yes *(argued)* | **yes** *(gated)* | **yes** *(gated)* | **yes** *(gated)* | every loop is iteration-capped, but the binding caps are option defaults rather than static limits; policy_chrono_freedom_scan (GCC/Clang only); policy_chrono_freedom_compiles; c++20 (build + ctest); c++23 (build + ctest); argmin_no_exceptions_probe (throw sites converted to status returns); cmaes: one hand-derived generation pins mean, p_sigma, p_c, and sigma; cmaes: one hand-derived generation pins the rank-mu covariance C; cmaes_sampling: marsaglia_normal is bit-identical across identically-seeded odd-consumption sequences |
| `isres` | not RT-claimed (no alloc gate; stochastic) | yes *(argued)* | **yes** *(gated)* | yes *(argued)* | **yes** *(gated)* | every loop is iteration-capped, but the binding caps are option defaults rather than static limits; policy_chrono_freedom_scan (GCC/Clang only); policy_chrono_freedom_compiles; c++20 (build + ctest); c++23 (build + ctest); throw-free and RTTI-free library-wide, but the instantiation probe does not link this policy; isres_policy: a fixed seed is bit-for-bit deterministic within one process (same process, same seed); isres_policy: determinism is seed-specific, not accidental; isres_policy: the reset path is deterministic from the same seed |
| `mma` | not RT-claimed (no alloc gate; the separable-approximation subproblem allocates) | yes *(argued)* | **yes** *(gated)* | yes *(argued)* | **yes** *(gated)* | every loop is iteration-capped, but the binding caps are option defaults rather than static limits; policy_chrono_freedom_scan (GCC/Clang only); policy_chrono_freedom_compiles; c++20 (build + ctest); c++23 (build + ctest); throw-free and RTTI-free library-wide, but the instantiation probe does not link this policy; policy_rng_freedom_scan (GCC/Clang only); policy_rng_freedom_compiles |
| `gcmma` | not RT-claimed (no alloc gate) | yes *(argued)* | **yes** *(gated)* | yes *(argued)* | **yes** *(gated)* | every loop is iteration-capped, but the binding caps are option defaults rather than static limits; policy_chrono_freedom_scan (GCC/Clang only); policy_chrono_freedom_compiles; c++20 (build + ctest); c++23 (build + ctest); throw-free and RTTI-free library-wide, but the instantiation probe does not link this policy; policy_rng_freedom_scan (GCC/Clang only); policy_rng_freedom_compiles |
| `ccsa_quadratic` | not RT-claimed (no alloc gate) | yes *(argued)* | **yes** *(gated)* | yes *(argued)* | **yes** *(gated)* | every loop is iteration-capped, but the binding caps are option defaults rather than static limits; policy_chrono_freedom_scan (GCC/Clang only); policy_chrono_freedom_compiles; c++20 (build + ctest); c++23 (build + ctest); throw-free and RTTI-free library-wide, but the instantiation probe does not link this policy; policy_rng_freedom_scan (GCC/Clang only); policy_rng_freedom_compiles |
<!-- END GENERATED: policies -->

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

The matrix is certified on host. Carrying the zero-alloc claim onto a
cross-compiled bare-metal target additionally requires Eigen's stack-temporary
path to be enabled explicitly (`-DEIGEN_ALLOCA=__builtin_alloca`): on newlib
targets built at strict `-std=c++20`, Eigen silently routes its internal
kernel temporaries to the heap instead of the stack, which is measurable as
per-step allocations (`kraft_slsqp` measured 2.80/step on the bare-metal
NUCLEO-H753ZI without the flag, against 0.00/step with it — on-device A/B, and
0.00/step on host and ESP32 throughout). With the flag, all four RT-claimed
policies measure 0.00 allocs/step on real bare-metal hardware
(`mcu/nucleo_h753zi/onchip-result.md`). Mechanism, attribution, and the
stack-budgeting rule: [Embedded Cross-Compiling](embedded.md).

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
