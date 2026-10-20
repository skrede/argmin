# QP solver head-to-head benchmark

This benchmark compares argmin's three quadratic-program solvers (dense ADMM, dense active-set and
sparse ADMM) against OSQP, qpOASES, DAQP, ProxQP, HPIPM and TinyMPC on the same model-predictive
control problems, measured through the ctrlpp benchmark harness,
`benchmarks/comparison/qp_head_to_head/`. It answers three questions: whether block-structured
(Riccati-class) storage would pay off for argmin's sparse solver, which argmin dense solver should
be the documented default for each regime and accuracy tier, and whether argmin's sparse ADMM
solver can replace OSQP in ctrlpp's linear MPC. The decision rules below were committed before any
measurement was recorded, and the verdicts are computed from them mechanically by
`benchmarks/scripts/qp_head_to_head_verdicts.py`.

## Headline result

- **Block-structured storage: NO-GO.** On the nominal variant argmin's sparse ADMM solver takes
  0.845 to 1.48 times HPIPM's median tick on the four long-horizon cells, below the threshold of 3.
  On the tight variant the ratio is 3.45 to 4.35 and meets the threshold, but it does not grow with
  the horizon in the nx8_nu3 family (4.35 at N = 50, 4.32 at N = 100); the nx4_nu2 family grows only
  from 3.45 to 3.49.
- **Dense default:** a single default in three of the six regime and tier combinations, a portfolio
  in the other three.

  | Regime | mpc tier | high tier |
  |---|---|---|
  | fixed | portfolio: `argmin_dense_admm` wins 20 cell-variants, `argmin_dense_active_set` wins 1 | portfolio: `argmin_dense_admm` wins 16, `argmin_dense_active_set` wins 5 |
  | closed_loop | default: `argmin_dense_admm` | portfolio: `argmin_dense_admm` wins 19, `argmin_dense_active_set` wins 1 |
  | churning | default: `argmin_dense_admm` | default: `argmin_dense_active_set`, the only candidate passing every tick |

- **Replacing OSQP in ctrlpp's linear MPC: keep.** argmin's sparse ADMM solver passes every tick on
  every in-scope cell-variant and takes 0.459 to 0.608 times OSQP's median tick on the nominal
  variant, but on the tight variant it exceeds 5/4 of OSQP's median tick on six cells: nx4_nu2_N10
  and nx4_nu2_N20 (1.56), nx4_nu2_N50 and nx4_nu2_N100 (2.11), nx8_nu3_N50 (1.77) and nx8_nu3_N100
  (1.78).

These outcomes come from run `20261009T045427Z-706d5262e9`, measured on one host with its clock
fixed (boost and SMT off, 3.375 GHz observed) under the rules below, which were committed before
the run.

## Pre-registered decision rules

<!-- preregistration:begin -->
These rules, thresholds, cells, variants, arm settings, build policy and pins were fixed before any
recorded run existed. Once a recorded run exists, nothing between these markers is edited; a
changed rule requires a fresh recorded run. Every recorded run's manifest cites the commit that
froze this block, and the tables at the end of the block are the verbatim output of
`qp_head_to_head_verdicts.py --print-preregistration`, which a test keeps identical to the
constants the verdict command uses.

All three rules read only the per-group pass rate and the median `tick_ns` over passing ticks
(see Statistics). A group is one (cohort, cell, variant, regime, tier, arm) combination; a
cell-variant is one cell in one variant. Ratios and medians are exact rationals compared with no
tolerance.

### Rule A: block-structured storage

Scope: the sparse cohort, the `closed_loop` regime and the `mpc` tier, on the long-horizon cells
(N >= 50) in the nominal variant and in the tight variant unless the binding gate refused that
family's tight variant.

For each in-scope cell-variant, the structure-exploiting reference `hpipm` qualifies if its pass
rate is exactly 1. The ratio is `median tick(argmin_sparse_admm) / median tick(hpipm)`. It is
defined only when the reference qualifies and `argmin_sparse_admm` has a pass rate of exactly 1; an
undefined ratio fails that cell-variant.

The outcome is GO if and only if all of the following hold:

- every in-scope cell-variant has a defined ratio of at least 3 (a ratio of exactly 3 meets it);
- for every (nx, nu, variant) family with both N = 50 and N = 100 in scope, the ratio at N = 100 is
  strictly greater than the ratio at N = 50 (an equal ratio is not growth, and an undefined ratio
  at either horizon fails growth);
- at least one such complete family exists.

Otherwise the outcome is NO-GO, listing every failing condition: per cell-variant, no qualifying
reference, the argmin pass rate, or the ratio below 3; per family, no growth; or growth not
testable. Either outcome is recorded as a follow-up investigation of argmin's sparse solver; this
benchmark implements nothing.

### Rule B: dense default per regime and tier

One outcome for each of the six regime and tier combinations (`fixed`, `closed_loop`, `churning`
times `mpc`, `high`). The candidates are `argmin_dense_admm` and `argmin_dense_active_set`. The
scope is every dense-cohort cell-variant of that regime, plus `dense_mpc_shaped` in the `fixed`
regime, excluding every refused tight variant. A candidate qualifies if and only if its pass rate
is exactly 1 on every in-scope cell-variant.

- Neither candidate qualifies: `no-default`, with the failing pass rates and cell-variants listed.
- Exactly one qualifies: it is the `default`.
- Both qualify: their exact median ticks are compared per cell-variant. The lower median wins the
  cell-variant; equal medians are a tie that neither wins.
  - One candidate wins at least one cell-variant and the other wins none: that candidate is the
    `default`, with the ties listed.
  - Each wins at least one cell-variant: `portfolio`, stating the cell-variants each wins and the
    ties.
  - Every cell-variant is tied: `portfolio`, marked indistinguishable.

The outcome is documentation only; no default in argmin's code changes because of it.

### Rule C: replacing OSQP in ctrlpp's linear MPC

Scope: the sparse cohort, the `closed_loop` regime and the `mpc` tier, on every cell in the
nominal variant and in the tight variant unless the binding gate refused that family's tight
variant.

A cell-variant is satisfied if and only if both conditions hold:

- `argmin_sparse_admm` has a pass rate of exactly 1;
- `median tick(argmin_sparse_admm) / median tick(osqp)` is defined and at most 5/4 (exactly 5/4 is
  within), or `osqp` has no passing tick on that cell-variant, in which case the ratio condition is
  met and the cell-variant is marked `osqp: no passing tick`.

The outcome is `replace` if and only if every in-scope cell-variant is satisfied; otherwise it is
`keep`, naming each failing cell-variant and its failing condition (pass rate, ratio, or both; an
undefined ratio is a ratio failure). The outcome is advice to ctrlpp; making the swap is ctrlpp's
work.

### Statistics

The unit of observation is one tick, one row of the record. For each group, the pass rate is the
number of passing ticks over all ticks, as an exact fraction. The median `tick_ns` is taken over
passing ticks only, as an exact rational (the mean of the two middle values for an even count); a
group with no passing tick has no median and is shown as `no passing tick`. A group's 300 ticks
(5 repetitions of 60) are pooled into one sample, so a median or quartile mixes within-run and
between-run spread rather than summarizing per-repetition values.

Reported but never read by a rule: the 25th and 75th percentiles of `tick_ns` over passing ticks
(linear interpolation at position (n - 1) q of the sorted values); the setup (first tick of each
sequence), update, solve and condense medians and the instruction-count medians, all over passing
ticks with the measurement available; and, over every tick, the iteration minimum, median and
maximum, the worst KKT residual, the count per fail class, the allocation maximum, the active-set
fallback count and the count of residuals within one ulp of the tier threshold. Every displayed
number is rounded to three significant figures, half to even; display strings never feed a
decision.

### Integrity

- The published run is the first recorded run whose data passes validation. Every recorded run
  attempted is appended to a run ledger kept beside the archived data and is counted in this
  document; a later run is published only when every earlier recorded run failed validation, with
  each failure reason stated.
- A run is recorded only when its clock state is `fixed` (see Host state). The verdict command's
  `--advisory` mode accepts an unfixed clock and stamps its output as not publishable; that output
  is never published.
- The verdict command consumes exactly one run and refuses it, writing nothing, on any of: a
  missing or invalid manifest; a manifest whose pins, argmin commit, build policy, constants or arm
  settings differ from this pre-registration, whose expressibility matrix differs from the arm
  registry, that lacks a passed convention probe within tolerance for a rule-deciding native-dual
  arm or a terminal-weight hash for a long-horizon cell, or whose binding-gate outcomes disagree
  with its own figures; a dirty harness working tree; a run id not matching the pre-registered
  pattern or differing between rows; a CSV whose SHA-256 differs from the manifest's; unknown,
  missing or reordered columns; a value outside its domain; a duplicate (cohort, arm, cell, variant,
  regime, tier, repetition, step) row; a pass flag inconsistent with `kkt_max` and the tier
  threshold; a fail class inconsistent with the pass flag; a fallback on any arm other than
  `argmin_dense_active_set`; a row for a combination that is not expressible or was refused; and a
  missing row for any pre-registered combination.
- The raw CSVs and the manifest stay in ctrlpp's archive beside the harness that produced them.
  This document carries the summary tables, every commit SHA, the host state and the reproduction
  recipe, and cites the ctrlpp harness commit.

### Amendments

Three amendments were made before the freeze, before any recorded run existed:

1. Rule A requires `argmin_sparse_admm` to have a pass rate of exactly 1 on a cell-variant for its
   ratio to be defined; it previously required only one passing tick. Reason: the references
   already had to pass every tick, and rules B and C already require it of argmin; a median over a
   partially passing sample would let the ticks argmin happened to solve decide the verdict.
2. The convention-probe requirement covers every rule-deciding native-dual arm; the three polished
   report-only arms are exempt. Reason: the earlier wording said every native-dual arm, which
   disagreed with the verdict command's manifest check; the command's scope is the intended one.
3. The dense-only problem's dimensions moved from the terminal-weight column of the problem-set
   table into a table of their own. Reason: they were in the wrong column; no value changed.

Three more were made after the freeze, on the evidence of the harness's smoke runs, before any
recorded run existed:

4. The convention probe solves ProxQP at `eps_abs = eps_rel = 1e-9`; its tolerance stays 1e-6 and
   every timed setting is unchanged. Reason: at its high-tier `1e-6` ProxQP returned multipliers
   with the right signs but errors of 2.45e-6 (dense) and 3.75e-6 (sparse), which is its solve
   tolerance, not a convention error; a probe tolerance equal to the solve tolerance left no margin.
5. TinyMPC is report-only and `hpipm` is Rule A's only reference. Reason: at its pin, `tiny_setup`
   stores `Q + rho I` and `R + rho I` as the stage weights its ADMM fixed point minimizes, so it
   solves a different QP from every other arm; on the tight variant it would still pass every tick
   and could be Rule A's best reference on that different problem.
6. In the judge's complementarity term the multiplier scale enters as `1 + ||y||_inf` instead of
   `||y||_inf`. Reason: the old term is invariant to the scale of `y`, so on a tick where no row
   binds any residual multiplier on a slack row scored its slack over the problem scale (0.655 for
   `argmin_dense_admm` and 0.159 for OSQP on one such smoke tick), failing operator-splitting arms
   through the metric rather than the solution. Where `||y||_inf >= 1` the term changes by at most
   a factor of 2.

### Constants

| Quantity | Value | Constant |
|---|---|---|
| Block-structure ratio threshold (at least) | 3 | `BLOCK_STRUCTURE_RATIO_THRESHOLD` |
| OSQP replacement ratio limit (at most) | 5/4 | `OSQP_REPLACEMENT_RATIO_LIMIT` |
| Long-horizon minimum N | 50 | `LONG_HORIZON_MIN_N` |
| Pass threshold, mpc tier (KKT, inclusive) | 0.001 | `TIERS` |
| Pass threshold, high tier (KKT, inclusive) | 1e-06 | `TIERS` |
| Repetitions | 5 | `REPETITIONS` |
| Timed ticks per sequence | 60 | `STEPS_PER_SEQUENCE` |
| Fixed-regime untimed warm resolves | 20 | `FIXED_WARMUP_RESOLVES` |
| Fixed-regime timed resolves | 60 | `FIXED_TIMED_RESOLVES` |
| Dual convention probe tolerance | 1e-06 | `CONVENTION_PROBE_TOLERANCE` |
| Reference solution KKT maximum | 1e-09 | `REFERENCE_SOLUTION_KKT_MAX` |
| Binding row tolerance | 1e-06 | `BINDING_ROW_TOLERANCE` |
| Binding gate: median binding fraction (at least) | 0.1 | `BINDING_MEDIAN_FRACTION_MIN` |
| Binding gate: state-row binding step fraction (at least) | 0.5 | `BINDING_STATE_ROW_STEP_FRACTION_MIN` |
| OSQP adaptive rho interval (iterations) | 25 | `OSQP_ADAPTIVE_RHO_INTERVAL` |
| Legacy terminal weight | 10.0 I | `LEGACY_TERMINAL_WEIGHT` |
| Long-horizon terminal weight | P_inf(Q + rho I, R + rho I), `rho 1.0`, `max_iterations 1000`, `gain_tolerance 1e-05` | `TERMINAL_RICCATI` |
| Display significant figures (round half to even) | 3 | `DISPLAY_SIGNIFICANT_FIGURES` |
| Fail classes of a non-passing tick | inaccurate, max_iter, infeasible_claim, error | `FAIL_CLASSES` |
| Arm allowed an in-tick feasible-start fallback | argmin_dense_active_set | `FALLBACK_ARM` |
| Run id pattern | ^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{10}$ | `RUN_ID_PATTERN` |
| Manifest schema | qp-head-to-head-manifest/1 | `MANIFEST_SCHEMA` |
| Verdicts schema | qp-head-to-head-verdicts/1 | `VERDICTS_SCHEMA` |

### Problem set

| Cell | nx | nu | N | Terminal weight | Cohorts | Regimes |
|---|---|---|---|---|---|---|
| nx2_nu1_N10 | 2 | 1 | 10 | 10.0 I | dense, sparse | fixed, closed_loop, churning |
| nx4_nu2_N10 | 4 | 2 | 10 | 10.0 I | dense, sparse | fixed, closed_loop, churning |
| nx4_nu2_N20 | 4 | 2 | 20 | 10.0 I | dense, sparse | fixed, closed_loop, churning |
| nx8_nu3_N20 | 8 | 3 | 20 | 10.0 I | dense, sparse | fixed, closed_loop, churning |
| nx8_nu3_N30 | 8 | 3 | 30 | 10.0 I | dense, sparse | fixed, closed_loop, churning |
| nx12_nu4_N30 | 12 | 4 | 30 | 10.0 I | dense, sparse | fixed, closed_loop, churning |
| nx4_nu2_N50 | 4 | 2 | 50 | P_inf | dense, sparse | fixed, closed_loop, churning |
| nx4_nu2_N100 | 4 | 2 | 100 | P_inf | dense, sparse | fixed, closed_loop, churning |
| nx8_nu3_N50 | 8 | 3 | 50 | P_inf | dense, sparse | fixed, closed_loop, churning |
| nx8_nu3_N100 | 8 | 3 | 100 | P_inf | dense, sparse | fixed, closed_loop, churning |

| Problem | n | m | Cohorts | Regimes |
|---|---|---|---|---|
| dense_mpc_shaped | 10 | 5 | dense | fixed |

| Variant | Parameters |
|---|---|
| nominal | `input_limit 0.5`, `state_limit none`, `tracking_target 0.0`, `x0_amplitude 5.0` |
| tight | `input_limit 0.2`, `state_limit 1.0`, `tracking_target 2.0`, `x0_amplitude 1.0` |

### Arms

| Cohort | Arm | Role | Dual kind | Tiers | Expressible on |
|---|---|---|---|---|---|
| dense | `argmin_dense_admm` | candidate | native | mpc, high | every cell |
| dense | `argmin_dense_active_set` | candidate | native | mpc, high | every cell |
| dense | `qpoases` | competitor | native | mpc, high | every cell |
| dense | `daqp` | competitor | native | mpc, high | every cell |
| dense | `proxqp_dense` | competitor | native | mpc, high | every cell |
| dense | `osqp` | competitor | native | mpc, high | every cell |
| dense | `argmin_dense_admm_polished` | report_only | native | mpc | every cell |
| sparse | `argmin_sparse_admm` | candidate | native | mpc, high | every cell |
| sparse | `osqp` | competitor | native | mpc, high | every cell |
| sparse | `proxqp_sparse` | competitor | native | mpc, high | every cell |
| sparse | `hpipm` | reference | native | mpc, high | every cell |
| sparse | `tinympc` | report_only | recovered | mpc, high | long-horizon cells only |
| sparse | `argmin_sparse_admm_polished` | report_only | native | mpc | every cell |
| sparse | `osqp_polished` | report_only | native | mpc | every cell |

| Arm | Tier | Overridden settings |
|---|---|---|
| `argmin_dense_admm` | mpc | `eps_abs 0.001`, `eps_rel 0.001`, `max_iterations 4000`, `warm_start true`, `polish false` |
| `argmin_dense_admm` | high | `eps_abs 1e-06`, `eps_rel 1e-06`, `max_iterations 4000`, `warm_start true`, `polish true` |
| `argmin_dense_active_set` | mpc | `tolerance 0.001`, `max_iterations 4000` |
| `argmin_dense_active_set` | high | `tolerance 1e-06`, `max_iterations 4000` |
| `qpoases` | mpc | `preset MPC`, `print_level NONE`, `nwsr_cap 4000` |
| `qpoases` | high | `preset RELIABLE`, `print_level NONE`, `nwsr_cap 4000` |
| `daqp` | mpc | `primal_tol 0.001`, `iter_limit 4000` |
| `daqp` | high | `primal_tol 1e-06`, `iter_limit 4000` |
| `proxqp_dense` | mpc | `eps_abs 0.001`, `eps_rel 0.001`, `max_iter 4000`, `initial_guess WARM_START_WITH_PREVIOUS_RESULT` |
| `proxqp_dense` | high | `eps_abs 1e-06`, `eps_rel 1e-06`, `max_iter 4000`, `initial_guess WARM_START_WITH_PREVIOUS_RESULT` |
| `osqp` | mpc | `eps_abs 0.001`, `eps_rel 0.001`, `max_iter 4000`, `warm_starting true`, `polishing false`, `adaptive_rho true`, `adaptive_rho_interval 25`, `verbose false` |
| `osqp` | high | `eps_abs 1e-06`, `eps_rel 1e-06`, `max_iter 4000`, `warm_starting true`, `polishing true`, `adaptive_rho true`, `adaptive_rho_interval 25`, `verbose false` |
| `argmin_dense_admm_polished` | mpc | `eps_abs 0.001`, `eps_rel 0.001`, `max_iterations 4000`, `warm_start true`, `polish true` |
| `argmin_sparse_admm` | mpc | `eps_abs 0.001`, `eps_rel 0.001`, `max_iterations 4000`, `warm_start true`, `polish false` |
| `argmin_sparse_admm` | high | `eps_abs 1e-06`, `eps_rel 1e-06`, `max_iterations 4000`, `warm_start true`, `polish true` |
| `proxqp_sparse` | mpc | `eps_abs 0.001`, `eps_rel 0.001`, `max_iter 4000`, `initial_guess WARM_START_WITH_PREVIOUS_RESULT`, `backend SparseCholesky` |
| `proxqp_sparse` | high | `eps_abs 1e-06`, `eps_rel 1e-06`, `max_iter 4000`, `initial_guess WARM_START_WITH_PREVIOUS_RESULT`, `backend SparseCholesky` |
| `hpipm` | mpc | `mode SPEED`, `tol_stat 0.001`, `tol_eq 0.001`, `tol_ineq 0.001`, `tol_comp 0.001`, `iter_max 50`, `warm_start 1` |
| `hpipm` | high | `mode BALANCE`, `tol_stat 1e-06`, `tol_eq 1e-06`, `tol_ineq 1e-06`, `tol_comp 1e-06`, `iter_max 100`, `warm_start 1` |
| `tinympc` | mpc | `abs_pri_tol 0.001`, `abs_dua_tol 0.001`, `max_iter 4000`, `rho 1.0`, `adaptive_rho false` |
| `tinympc` | high | `abs_pri_tol 1e-06`, `abs_dua_tol 1e-06`, `max_iter 4000`, `rho 1.0`, `adaptive_rho false` |
| `argmin_sparse_admm_polished` | mpc | `eps_abs 0.001`, `eps_rel 0.001`, `max_iterations 4000`, `warm_start true`, `polish true` |
| `osqp_polished` | mpc | `eps_abs 0.001`, `eps_rel 0.001`, `max_iter 4000`, `warm_starting true`, `polishing true`, `adaptive_rho true`, `adaptive_rho_interval 25`, `verbose false` |

### Pins and build policy

| Component | Commit |
|---|---|
| argmin | `0a638cb87b6e959f7cb7b49f0afa34f137b93b69` |
| osqp | `236713ce9a56c182ac3230d52108f952afce1523` |
| qpoases | `680f18c8ef0018a120e1604b769f056e8368df97` |
| daqp | `29a8987d8b9a596c1ab6acf84c17d38b068b73bc` |
| hpipm | `16a8db59f595bda642f460aa0f62e308f4b76db3` |
| blasfeo | `9628393214623b29e5dd2a8784f7423edbc93e04` |
| tinympc | `023f36bd5b27267e1a96ed47a774d120b5707f13` |
| proxsuite | `b93d7778ffc3299d84b5cb0851022a29bf24a596` |

| Build setting | Value |
|---|---|
| compiler | `gcc` |
| build_type | `Release` |
| march | `-march=x86-64-v3` |
| release_flags | `-O3 -DNDEBUG` |
| blasfeo_target | `X64_INTEL_HASWELL` |
| blasfeo_la | `HIGH_PERFORMANCE` |
| blasfeo_mf | `PANELMAJ` |
| hpipm_target | `AVX` |
| daqp_faster_math | `false` |
| daqp_native_optimization | `false` |

### Record format

`run_id, cohort, cell, variant, nx, nu, N, n, m, regime, tier, arm, polish, rep, seed, step, setup_ns, update_ns, solve_ns, tick_ns, condense_ns, instr_setup, instr_update, instr_solve, instr_tick, iterations, kkt_stat, kkt_prim, kkt_comp, kkt_max, native_status, pass, fail_class, allocs_tick, dual_kind, fallback, binding_frac, churn_frac`
<!-- preregistration:end -->

## Methodology

### Cohorts and arms

The arms are split into two cohorts by the problem form they solve. The dense cohort solves the
condensed problem over the input sequence `U` and holds `argmin_dense_admm`,
`argmin_dense_active_set`, qpOASES, DAQP, ProxQP's dense backend and OSQP. The sparse cohort solves
the non-condensed problem over states and inputs and holds `argmin_sparse_admm`, OSQP, ProxQP's
sparse backend, HPIPM and TinyMPC. OSQP appears in both cohorts, on the condensed data in one and
on the sparse data in the other, because it is the incumbent in ctrlpp's linear MPC and a common
yardstick for both forms; the OSQP arm that decides rule C is OSQP's C API built from source under
the uniform build policy.

HPIPM (an interior-point method on the stage-wise OCP form) is the structure-exploiting
reference: its gap to the generic sparse arms is what block-structured storage could buy, so it
decides rule A only. TinyMPC (ADMM with a cached Riccati solve) is report-only: at its pin it adds
`rho I` to the stage weights its fixed point minimizes, so it solves a different QP. The polished sub-arms
(`argmin_dense_admm_polished`, `argmin_sparse_admm_polished`, `osqp_polished`) run at the MPC tier,
are reported, and are excluded from every rule.

An arm runs only where it can express the instance. TinyMPC cannot take the legacy cells' terminal
weight, so it is not expressible on them. A not-expressible combination produces no rows, is
recorded in the run manifest, and is never scored as a failure.

### Problem set

Every cell uses the plant and stage cost of ctrlpp's sparse MPC sweep: `A = 0.9 I + 0.1` on the
superdiagonal, `B(i, i) = 1` for `i < nu`, `Q = I`, `R = 0.1 I`. A cell is named
`nx{nx}_nu{nu}_N{N}`, and a family is the cells sharing (nx, nu). The six legacy cells keep the
terminal weight `Q_N = 10 I`. The four long-horizon cells (N = 50 and N = 100 at nx = 4 and nx = 8)
use `Q_N = P_inf(Q + rho I, R + rho I)` with `rho = 1.0`, computed by TinyMPC's own Riccati
iteration (at most 1000 iterations, stopping when the largest change in the gain is below 1e-5),
and every arm uses that same terminal weight so TinyMPC can take part. `dense_mpc_shaped` (n = 10,
m = 5) is a fixed dense problem from ctrlpp's QP problem set; it runs in the dense cohort and the
fixed regime only, since it has no initial state to drive, and is labeled with the nominal
variant.

Every cell runs in two variants. The nominal variant bounds the inputs by `|u_k| <= 0.5`, has no
state box, tracks the origin, and uses an initial-state amplitude of 5. The tight variant bounds
the inputs by `|u_k| <= 0.2`, boxes the actuated states (`i < nu`) to `[-1, 1]` on stages 1 to N,
tracks a target of 2.0 on the actuated states (0 on the others) at every stage including the
terminal one, and uses an initial-state amplitude of 1. The tight variant is feasible by
construction: every row of `A` has infinity-norm 1, so from `|x0| <= 1` the zero-input response
stays inside `[-1, 1]`, and `U = 0` is feasible for every initial state the drivers produce, which
also gives the active-set solver a feasible starting point. It is active by construction: the
target lies outside the box, so the optimum drives the actuated states onto the upper bound and
holds them there, which needs an input of `0.1 - 0.1 x_{k,i+1}`, inside `[0, 0.2]`.

Rows are ordered the same way in every arm. Sparse: the initial-state rows (nx), the dynamics
(N nx), the input boxes (N nu), then in the tight variant the state boxes (N nu), so
`n = (N + 1) nx + N nu`. Condensed: the input boxes (N nu), then in the tight variant the state
boxes (N nu), over `U`, so `n = N nu`. The initial-state and dynamics rows are equalities in every
arm.

Before any timed run, a binding gate checks that the tight variant really binds, from reference
solutions only. Each instance's reference solution is the first of qpOASES (reliable preset,
condensed), HPIPM (BALANCE mode, tolerance 1e-10) and argmin's dense active-set solver (tolerance
1e-10) whose answer has a KKT residual of at most 1e-9 under the common judge. An inequality row
binds when `min(u_i - (Ax)_i, (Ax)_i - l_i) <= 1e-6 (1 + max(|l_i|, |u_i|))`. A family passes when,
on every one of its cells, the median over the closed-loop steps of the fraction of inequality
rows that bind is at least 0.10, and at least one state-box row binds on at least half of the
closed-loop steps. A family that fails has its tight variant refused: it is not measured, it is
listed with its gate figures, and it is excluded from every rule. The gate figures are recorded in
the run manifest.

### Regimes

The initial-state drivers are identical for every arm (state index `i`, step `s`, amplitude `a` of
the variant):

- `fixed`: `x0 = a cos(0.7 i)`. After setup, 20 untimed warm resolves are followed by 60 timed
  resolves on identical data, each re-supplying the same vectors.
- `closed_loop`: `x0(s) = a cos(0.8 s + 0.7 i)` for `s = 0..59`, a prescribed sequence replayed to
  every arm, so that every arm solves the same problems; a trajectory simulated per arm would hand
  different arms different problems.
- `churning`: `x0(seed, s) = a (2 f - 1)`, where `f` comes from the churn hash of ctrlpp's
  existing OSQP churning benchmark extended with a seed term (seed 0 reproduces the existing
  sequence), for `s = 0..59`; the seed is the repetition index.

In `closed_loop` and `churning` each of the 60 timed ticks updates the problem to `x0(s)` and then
solves. Every arm warm-starts from its own previous solution through its documented mechanism.

There are 5 repetitions, each a separate process per cohort, and within each the arm order is
rotated by the repetition index. Every expressible combination therefore has exactly 300 rows.

### Quality rule

A tick passes if and only if its `kkt_max` is at most the tier threshold, inclusive: 1e-3 at the
`mpc` tier and 1e-6 at the `high` tier. `kkt_max` is the relative KKT residual of ctrlpp's QP
accuracy judge, evaluated by the harness on each arm's primal and dual solution mapped to OSQP's
sign convention over the instance's own row order. Its complementarity term is
`max_i |y_i (bound_i - row_i)| / ((1 + ||y||_inf) (||Ax|| + ||l|| + ||u||))`, so a multiplier left
on a row that does not bind scores its own size. A solver's own status never decides a pass. A
tick that does not pass is classified as `inaccurate` (the solver claimed success and the judge
failed it), `max_iter`, `infeasible_claim` or `error`.

Each arm runs with its own documented settings for the tier (the arm-settings table above). Every
rule-deciding native-dual arm must pass a convention probe at setup: a small QP with an active
upper bound, an active lower bound and an equality row, solved exactly by enumeration, whose mapped
multipliers must match to 1e-6, or the run aborts naming the arm. The probe is untimed and runs
each arm at its high-tier settings, except ProxQP, which it solves at `eps_abs = eps_rel = 1e-9`.
TinyMPC returns no multipliers;
they are recovered by least squares on the rows active within `threshold (1 + |bound_i|)` and then
judged like any other arm's, and its rows are marked `dual_kind = recovered`.

When argmin's dense active-set solver reports an infeasible start, the arm resets and resolves from
`U = 0` inside the same tick; that row is marked `fallback = 1`, the time stays in the tick, and the
count is reported per cell-variant.

### Timing and counting

Times are `std::chrono::steady_clock` integer nanoseconds. `tick_ns` spans the update and the solve
(including any fallback) and is the only time a rule reads. `update_ns` times the arm's own
data-update call where one exists; for arms whose solve call consumes the new vectors directly
(argmin's resolve call, qpOASES's hotstart) it times the staging of the vectors into the arm, and
the manifest records each arm's update interface as `separate` or `fused`. `setup_ns` is recorded on
the first timed row of each sequence; for argmin arms it includes the first solve that argmin's
pose call performs. `condense_ns`, in the dense cohort, is the harness's recomputation of the
condensed vectors, identical for every dense arm; it is excluded from the tick and reported beside
it. Retired user-space instructions are counted per phase, and heap allocations per tick come from
a process-wide sensor armed around the tick; either is `-1` where unavailable. Wall times are
directional for this host; instruction counts are the more portable record.

### Build policy and pins

Every arm is built in one configure, Release, with gcc (version recorded), `-O3 -DNDEBUG` and
`-march=x86-64-v3`, single-threaded. argmin is pinned through ctrlpp's argmin tag override at the
commit in the pins table; ctrlpp's own default argmin pin is unchanged. Each competitor is pinned
at a full commit SHA: OSQP v1.0.0 (built from source), qpOASES 3.2.2, DAQP v0.10.3 (with
`FASTER_MATH` and `NATIVE_OPTIMIZATION` off), HPIPM with target `AVX`, BLASFEO with target
`X64_INTEL_HASWELL`, `HIGH_PERFORMANCE` linear algebra and the panel-major format, TinyMPC, and
proxsuite v0.7.3 from the system package (header-only, compiled under the same flags). Every pin is
recorded in the run manifest.

### Host state

A run is recorded only with a fixed clock: CPU boost off globally and on every frequency policy,
the `performance` governor, equal minimum and maximum scaling frequency on the pinned CPU, SMT off,
affinity to a single CPU other than CPU 0, and the instruction counter available. Any other state
is unfixed. The harness verifies and records the host state with every run. These numbers are a
fresh baseline and are not comparable with the earlier publication set, which was measured with
boost enabled.

### Known deviations from deployed configurations

- OSQP's `adaptive_rho_interval` is pinned to 25 iterations, where ctrlpp's deployed wrapper leaves
  it at OSQP's time-based automatic default; a time-based interval would make the arm's iterates
  depend on timing.
- The tight variant uses hard state boxes, where ctrlpp deploys slack-softened state bounds.

## Results

The tables in this section and the next are the verbatim output of the verdict command for run
`20261009T045427Z-706d5262e9`, one table per cohort, regime and tier. Every arm has 300 ticks per
row (5 repetitions of 60), a pass rate of `1/1` means all 300 passed, and every median and quartile
is taken over passing ticks only. Wall times are directional: they hold for this host at this fixed
clock. The instruction counts are user-space instructions retired and do not depend on the clock
frequency, so they are the column to compare across hosts. Iteration counts are each solver's own
and measure different work per iteration, so they are not comparable across methods. The
allocation column is the largest number of heap allocations in any one tick, counted process-wide
while the tick runs, so it includes the arm's harness adapter. In the dense
tables the last column is the number of ticks on which `argmin_dense_active_set` fell back to the
feasible start `U = 0`.

### dense cohort, fixed regime, mpc tier

The dense cohort on the condensed problem, each arm re-solving identical data 60 times after 20
untimed warm resolves, judged at the mpc tier's KKT threshold of 1e-3.

| Cell | Variant | Arm | Pass rate | Median tick (us) | Tick p25 - p75 (us) | Median setup / update / solve (us) | Median condense (us) | Median instructions setup / update / solve / tick | Iterations min / median / max | Worst KKT residual | Failures | Allocations max | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nx2_nu1_N10 | nominal | `argmin_dense_admm` | 1/1 | 9.85 | 9.79 - 9.95 | 19.6 / 0.0300 / 9.82 | 0.150 | 161000 / 392 / 84600 / 85000 | 25 / 25 / 25 | 3.18e-15 | none | 0 | 0 |
| nx2_nu1_N10 | nominal | `argmin_dense_active_set` | 1/1 | 10.1 | 9.98 - 10.2 | 57.3 / 0.0300 / 10.0 | 0.160 | 311000 / 368 / 95700 / 96000 | 0 / 0 / 0 | 3.48e-17 | none | 8 | 0 |
| nx2_nu1_N10 | nominal | `qpoases` | 1/1 | 1.23 | 1.21 - 1.26 | 28.7 / 0.0300 / 1.20 | 0.150 | 144000 / 411 / 10600 / 11000 | 0 / 0 / 0 | 1.18e-13 | none | 19 | 0 |
| nx2_nu1_N10 | nominal | `daqp` | 1/1 | 0.640 | 0.640 - 0.660 | 4.48 / 0.210 / 0.430 | 0.150 | 13300 / 2320 / 4220 / 6540 | 1 / 1 / 1 | 8.14e-16 | none | 0 | 0 |
| nx2_nu1_N10 | nominal | `proxqp_dense` | 1/1 | 1.14 | 1.13 - 1.15 | 65.6 / 0.550 / 0.590 | 0.150 | 343000 / 5240 / 5450 / 10700 | 0 / 0 / 0 | 2.34e-4 | none | 7 | 0 |
| nx2_nu1_N10 | nominal | `osqp` | 1/1 | 8.26 | 8.22 - 8.31 | 19.2 / 0.190 / 8.07 | 0.150 | 104000 / 1360 / 75300 / 76700 | 25 / 25 / 25 | 3.26e-15 | none | 0 | 0 |
| nx2_nu1_N10 | nominal | `argmin_dense_admm_polished` | 1/1 | 14.0 | 14.0 - 14.1 | 40.1 / 0.0300 / 14.0 | 0.150 | 209000 / 369 / 133000 / 133000 | 25 / 25 / 25 | 7.14e-17 | none | 0 | 0 |
| nx2_nu1_N10 | tight | `argmin_dense_admm` | 1/1 | 9.85 | 9.81 - 9.91 | 29.3 / 0.0300 / 9.81 | 0.190 | 242000 / 370 / 81600 / 82000 | 25 / 25 / 25 | 3.18e-16 | none | 0 | 0 |
| nx2_nu1_N10 | tight | `argmin_dense_active_set` | 1/1 | 16.0 | 15.9 - 16.1 | 70.3 / 0.0300 / 16.0 | 0.200 | 457000 / 369 / 161000 / 162000 | 0 / 0 / 0 | 1.31e-16 | none | 13 | 0 |
| nx2_nu1_N10 | tight | `qpoases` | 1/1 | 1.57 | 1.55 - 1.60 | 40.2 / 0.0400 / 1.53 | 0.190 | 253000 / 494 / 13800 / 14300 | 0 / 0 / 0 | 3.17e-13 | none | 19 | 0 |
| nx2_nu1_N10 | tight | `daqp` | 1/1 | 0.830 | 0.810 - 0.850 | 4.92 / 0.290 / 0.530 | 0.190 | 18900 / 3220 / 5580 / 8800 | 1 / 1 / 1 | 2.58e-16 | none | 0 | 0 |
| nx2_nu1_N10 | tight | `proxqp_dense` | 1/1 | 1.31 | 1.30 - 1.32 | 54.0 / 0.670 / 0.640 | 0.190 | 245000 / 6250 / 5980 / 12200 | 0 / 0 / 0 | 2.68e-4 | none | 7 | 0 |
| nx2_nu1_N10 | tight | `osqp` | 1/1 | 11.6 | 11.5 - 11.6 | 26.6 / 0.210 / 11.4 | 0.190 | 145000 / 1610 / 110000 / 111000 | 25 / 25 / 25 | 5.64e-16 | none | 0 | 0 |
| nx2_nu1_N10 | tight | `argmin_dense_admm_polished` | 1/1 | 17.3 | 17.2 - 17.4 | 57.0 / 0.0300 / 17.3 | 0.200 | 330000 / 370 / 169000 / 169000 | 25 / 25 / 25 | 6.55e-17 | none | 0 | 0 |
| nx4_nu2_N10 | nominal | `argmin_dense_admm` | 1/1 | 16.8 | 16.7 - 16.9 | 34.8 / 0.0300 / 16.7 | 0.230 | 301000 / 295 / 130000 / 130000 | 25 / 25 / 25 | 2.96e-15 | none | 0 | 0 |
| nx4_nu2_N10 | nominal | `argmin_dense_active_set` | 1/1 | 25.4 | 25.2 - 25.5 | 127 / 0.0300 / 25.4 | 0.250 | 977000 / 296 / 264000 / 264000 | 0 / 0 / 0 | 1.59e-16 | none | 8 | 0 |
| nx4_nu2_N10 | nominal | `qpoases` | 1/1 | 1.82 | 1.81 - 1.83 | 53.6 / 0.0400 / 1.78 | 0.230 | 357000 / 334 / 16700 / 17100 | 0 / 0 / 0 | 3.88e-13 | none | 19 | 0 |
| nx4_nu2_N10 | nominal | `daqp` | 1/1 | 1.05 | 1.04 - 1.06 | 7.86 / 0.390 / 0.660 | 0.220 | 40000 / 4390 / 7300 / 11700 | 1 / 1 / 1 | 4.08e-15 | none | 0 | 0 |
| nx4_nu2_N10 | nominal | `proxqp_dense` | 1/1 | 1.50 | 1.49 - 1.52 | 84.0 / 0.750 / 0.750 | 0.230 | 547000 / 7680 / 7700 / 15400 | 0 / 0 / 0 | 1.38e-4 | none | 7 | 0 |
| nx4_nu2_N10 | nominal | `osqp` | 1/1 | 16.1 | 16.0 - 16.1 | 35.5 / 0.220 / 15.8 | 0.240 | 226000 / 1550 / 152000 / 154000 | 25 / 25 / 25 | 4.59e-15 | none | 0 | 0 |
| nx4_nu2_N10 | nominal | `argmin_dense_admm_polished` | 1/1 | 27.5 | 27.3 - 27.7 | 59.1 / 0.0300 / 27.4 | 0.250 | 431000 / 298 / 259000 / 259000 | 25 / 25 / 25 | 2.29e-15 | none | 0 | 0 |
| nx4_nu2_N10 | tight | `argmin_dense_admm` | 1/1 | 18.4 | 18.3 - 18.5 | 59.3 / 0.0400 / 18.3 | 0.290 | 506000 / 345 / 144000 / 144000 | 25 / 25 / 25 | 2.85e-16 | none | 0 | 0 |
| nx4_nu2_N10 | tight | `argmin_dense_active_set` | 1/1 | 70.2 | 69.9 - 70.5 | 268 / 0.0400 / 70.2 | 0.300 | 2390000 / 349 / 794000 / 795000 | 0 / 0 / 0 | 4.40e-16 | none | 23 | 0 |
| nx4_nu2_N10 | tight | `qpoases` | 1/1 | 2.49 | 2.46 - 2.51 | 117 / 0.0400 / 2.45 | 0.290 | 1030000 / 417 / 24700 / 25100 | 0 / 0 / 0 | 1.59e-13 | none | 19 | 0 |
| nx4_nu2_N10 | tight | `daqp` | 1/1 | 1.68 | 1.66 - 1.69 | 10.6 / 0.660 / 1.02 | 0.280 | 63600 / 6640 / 12400 / 19000 | 1 / 1 / 1 | 2.20e-16 | none | 0 | 0 |
| nx4_nu2_N10 | tight | `proxqp_dense` | 1/1 | 1.83 | 1.82 - 1.87 | 128 / 0.960 / 0.870 | 0.290 | 796000 / 10100 / 8880 / 19000 | 0 / 0 / 0 | 2.62e-4 | none | 7 | 0 |
| nx4_nu2_N10 | tight | `osqp` | 1/1 | 23.4 | 23.3 - 23.6 | 55.7 / 0.270 / 23.2 | 0.290 | 355000 / 2190 / 242000 / 244000 | 25 / 25 / 25 | 7.28e-16 | none | 0 | 0 |
| nx4_nu2_N10 | tight | `argmin_dense_admm_polished` | 1/1 | 40.2 | 40.1 - 40.3 | 101 / 0.0400 / 40.1 | 0.340 | 776000 / 345 / 412000 / 413000 | 25 / 25 / 25 | 1.47e-16 | none | 0 | 0 |
| nx4_nu2_N20 | nominal | `argmin_dense_admm` | 1/1 | 34.2 | 34.1 - 34.4 | 91.5 / 0.0500 / 34.2 | 0.500 | 820000 / 370 / 252000 / 252000 | 25 / 25 / 25 | 3.48e-15 | none | 0 | 0 |
| nx4_nu2_N20 | nominal | `argmin_dense_active_set` | 1/1 | 70.9 | 70.7 - 71.2 | 459 / 0.0400 / 70.9 | 0.590 | 3470000 / 369 / 748000 / 748000 | 0 / 0 / 0 | 1.29e-16 | none | 8 | 0 |
| nx4_nu2_N20 | nominal | `qpoases` | 1/1 | 3.93 | 3.91 - 3.96 | 122 / 0.0500 / 3.88 | 0.505 | 1050000 / 409 / 37600 / 38000 | 0 / 0 / 0 | 3.93e-13 | none | 19 | 0 |
| nx4_nu2_N20 | nominal | `daqp` | 1/1 | 2.13 | 2.12 - 2.15 | 19.3 / 0.910 / 1.22 | 0.500 | 157000 / 10300 / 14900 / 25200 | 1 / 1 / 1 | 3.11e-15 | none | 0 | 0 |
| nx4_nu2_N20 | nominal | `proxqp_dense` | 1/1 | 3.32 | 3.29 - 3.36 | 172 / 1.80 / 1.51 | 0.520 | 1220000 / 18900 / 15600 / 34500 | 0 / 0 / 0 | 1.97e-4 | none | 7 | 0 |
| nx4_nu2_N20 | nominal | `osqp` | 1/1 | 45.4 | 45.2 - 45.6 | 88.7 / 0.280 / 45.1 | 0.500 | 672000 / 2270 / 430000 / 432000 | 25 / 25 / 25 | 3.70e-15 | none | 0 | 0 |
| nx4_nu2_N20 | nominal | `argmin_dense_admm_polished` | 1/1 | 73.3 | 73.0 - 74.1 | 199 / 0.0500 / 73.3 | 0.580 | 1290000 / 370 / 714000 / 714000 | 25 / 25 / 25 | 1.92e-15 | none | 0 | 0 |
| nx4_nu2_N20 | tight | `argmin_dense_admm` | 1/1 | 41.2 | 40.9 - 41.2 | 145 / 0.0550 / 41.1 | 0.610 | 1390000 / 470 / 298000 / 298000 | 25 / 25 / 25 | 7.24e-16 | none | 0 | 0 |
| nx4_nu2_N20 | tight | `argmin_dense_active_set` | 1/1 | 474 | 471 - 481 | 1720 / 0.0600 / 474 | 0.690 | 16700000 / 469 / 5290000 / 5290000 | 0 / 0 / 0 | 1.13e-16 | none | 43 | 0 |
| nx4_nu2_N20 | tight | `qpoases` | 1/1 | 5.20 | 5.14 - 5.32 | 499 / 0.0600 / 5.14 | 0.590 | 5810000 / 540 / 57000 / 57500 | 0 / 0 / 0 | 2.53e-13 | none | 19 | 0 |
| nx4_nu2_N20 | tight | `daqp` | 1/1 | 4.48 | 4.46 - 4.49 | 34.5 / 1.68 / 2.79 | 0.570 | 282000 / 17200 / 35100 / 52300 | 1 / 1 / 1 | 2.78e-16 | none | 0 | 0 |
| nx4_nu2_N20 | tight | `proxqp_dense` | 1/1 | 4.56 | 4.48 - 4.73 | 306 / 2.50 / 2.06 | 0.590 | 2070000 / 25900 / 18600 / 44500 | 0 / 0 / 0 | 2.25e-4 | none | 7 | 0 |
| nx4_nu2_N20 | tight | `osqp` | 1/1 | 71.4 | 70.7 - 73.0 | 151 / 0.410 / 71.0 | 0.590 | 1200000 / 3580 / 728000 / 731000 | 25 / 25 / 25 | 1.45e-15 | none | 0 | 0 |
| nx4_nu2_N20 | tight | `argmin_dense_admm_polished` | 1/1 | 136 | 136 - 137 | 351 / 0.0600 / 136 | 0.730 | 2510000 / 470 / 1420000 / 1420000 | 25 / 25 / 25 | 2.26e-16 | none | 0 | 0 |
| nx8_nu3_N20 | nominal | `argmin_dense_admm` | 1/1 | 60.9 | 60.7 - 61.2 | 160 / 0.0600 / 60.9 | 1.41 | 1650000 / 445 / 423000 / 424000 | 25 / 25 / 25 | 3.77e-15 | none | 0 | 0 |
| nx8_nu3_N20 | nominal | `argmin_dense_active_set` | 1/1 | 233 | 233 - 236 | 1090 / 0.0700 / 233 | 1.72 | 9570000 / 444 / 2400000 / 2400000 | 0 / 0 / 0 | 1.73e-16 | none | 16 | 0 |
| nx8_nu3_N20 | nominal | `qpoases` | 1/1 | 7.48 | 7.42 - 7.59 | 241 / 0.0600 / 7.41 | 1.33 | 2340000 / 484 / 70300 / 70800 | 0 / 0 / 0 | 2.39e-13 | none | 19 | 0 |
| nx8_nu3_N20 | nominal | `daqp` | 1/1 | 4.04 | 4.02 - 4.09 | 38.8 / 1.76 / 2.27 | 1.24 | 386000 / 18200 / 26700 / 44900 | 1 / 1 / 1 | 3.40e-15 | none | 0 | 0 |
| nx8_nu3_N20 | nominal | `proxqp_dense` | 1/1 | 6.50 | 6.41 - 6.58 | 256 / 3.44 / 3.04 | 1.38 | 2230000 / 35800 / 26800 / 62600 | 0 / 0 / 0 | 1.91e-4 | none | 7 | 0 |
| nx8_nu3_N20 | nominal | `osqp` | 1/1 | 91.5 | 91.3 - 91.8 | 175 / 0.380 / 91.1 | 1.35 | 1390000 / 2990 / 840000 / 843000 | 25 / 25 / 25 | 3.48e-15 | none | 0 | 0 |
| nx8_nu3_N20 | nominal | `argmin_dense_admm_polished` | 1/1 | 156 | 155 - 157 | 349 / 0.0700 / 156 | 1.68 | 2760000 / 445 / 1530000 / 1530000 | 25 / 25 / 25 | 1.63e-15 | none | 0 | 0 |
| nx8_nu3_N20 | tight | `argmin_dense_admm` | 1/1 | 81.0 | 80.8 - 81.1 | 298 / 0.0700 / 80.9 | 1.69 | 2800000 / 595 / 522000 / 523000 | 25 / 25 / 25 | 1.63e-15 | none | 0 | 0 |
| nx8_nu3_N20 | tight | `argmin_dense_active_set` | 1/1 | 1710 | 1670 - 1730 | 6310 / 0.100 / 1710 | 1.92 | 59400000 / 594 / 17300000 / 17300000 | 0 / 0 / 0 | 1.29e-16 | none | 63 | 0 |
| nx8_nu3_N20 | tight | `qpoases` | 1/1 | 9.56 | 9.47 - 9.66 | 1680 / 0.0800 / 9.47 | 1.58 | 20800000 / 665 / 106000 / 107000 | 0 / 0 / 0 | 1.75e-13 | none | 19 | 0 |
| nx8_nu3_N20 | tight | `daqp` | 1/1 | 9.86 | 9.77 - 10.0 | 86.3 / 3.90 / 5.90 | 1.49 | 748000 / 32100 / 67100 / 99300 | 1 / 1 / 1 | 4.31e-16 | none | 0 | 0 |
| nx8_nu3_N20 | tight | `proxqp_dense` | 1/1 | 8.93 | 8.86 - 9.01 | 825 / 4.92 / 3.99 | 1.66 | 6270000 / 50000 / 32200 / 82200 | 0 / 0 / 0 | 2.30e-4 | none | 7 | 0 |
| nx8_nu3_N20 | tight | `osqp` | 1/1 | 140 | 139 - 141 | 315 / 0.560 / 139 | 1.66 | 2570000 / 4910 / 1390000 / 1400000 | 25 / 25 / 25 | 2.88e-15 | none | 0 | 0 |
| nx8_nu3_N20 | tight | `argmin_dense_admm_polished` | 1/1 | 322 | 321 - 323 | 709 / 0.0800 / 322 | 1.89 | 5500000 / 595 / 3240000 / 3240000 | 25 / 25 / 25 | 1.04e-16 | none | 0 | 0 |
| nx8_nu3_N30 | nominal | `argmin_dense_admm` | 1/1 | 118 | 118 - 119 | 362 / 0.0900 / 118 | 3.37 | 3670000 / 598 / 773000 / 774000 | 25 / 25 / 25 | 3.03e-15 | none | 0 | 0 |
| nx8_nu3_N30 | nominal | `argmin_dense_active_set` | 1/1 | 506 | 505 - 508 | 2690 / 0.100 / 506 | 3.58 | 24200000 / 597 / 5200000 / 5200000 | 0 / 0 / 0 | 3.81e-16 | none | 16 | 0 |
| nx8_nu3_N30 | nominal | `qpoases` | 1/1 | 15.6 | 15.5 - 15.7 | 511 / 0.0900 / 15.5 | 3.19 | 5140000 / 637 / 140000 / 140000 | 0 / 0 / 0 | 2.34e-13 | none | 19 | 0 |
| nx8_nu3_N30 | nominal | `daqp` | 1/1 | 8.66 | 8.62 - 8.70 | 88.7 / 3.82 / 4.82 | 2.97 | 1010000 / 33900 / 51600 / 85400 | 1 / 1 / 1 | 1.46e-15 | none | 0 | 0 |
| nx8_nu3_N30 | nominal | `proxqp_dense` | 1/1 | 14.4 | 14.3 - 14.5 | 504 / 7.92 / 6.48 | 3.33 | 4390000 / 78300 / 50800 / 129000 | 0 / 0 / 0 | 2.01e-4 | none | 7 | 0 |
| nx8_nu3_N30 | nominal | `osqp` | 1/1 | 195 | 194 - 195 | 396 / 0.560 / 194 | 3.27 | 3230000 / 4150 / 1760000 / 1770000 | 25 / 25 / 25 | 2.89e-15 | none | 0 | 0 |
| nx8_nu3_N30 | nominal | `argmin_dense_admm_polished` | 1/1 | 357 | 355 - 359 | 656 / 0.100 / 356 | 3.52 | 6370000 / 598 / 3440000 / 3450000 | 25 / 25 / 25 | 1.92e-15 | none | 0 | 0 |
| nx8_nu3_N30 | tight | `argmin_dense_admm` | 1/1 | 156 | 155 - 156 | 639 / 0.120 / 155 | 3.72 | 6060000 / 796 / 939000 / 940000 | 25 / 25 / 25 | 2.12e-15 | none | 0 | 0 |
| nx8_nu3_N30 | tight | `argmin_dense_active_set` | 1/1 | 7440 | 7400 - 7460 | 24500 / 0.130 / 7440 | 3.84 | 225000000 / 795 / 69000000 / 69000000 | 0 / 0 / 0 | 1.28e-16 | none | 93 | 0 |
| nx8_nu3_N30 | tight | `qpoases` | 1/1 | 19.2 | 19.1 - 19.3 | 5010 / 0.150 / 19.0 | 3.60 | 62400000 / 920 / 207000 / 208000 | 0 / 0 / 0 | 1.71e-13 | none | 19 | 0 |
| nx8_nu3_N30 | tight | `daqp` | 1/1 | 21.5 | 21.5 - 21.6 | 247 / 8.41 / 13.1 | 3.47 | 2140000 / 63800 / 139000 / 203000 | 1 / 1 / 1 | 1.04e-15 | none | 0 | 0 |
| nx8_nu3_N30 | tight | `proxqp_dense` | 1/1 | 20.2 | 20.1 - 20.3 | 1700 / 11.6 / 8.60 | 3.69 | 13500000 / 112000 / 64700 / 177000 | 0 / 0 / 0 | 2.03e-4 | none | 7 | 0 |
| nx8_nu3_N30 | tight | `osqp` | 1/1 | 303 | 299 - 308 | 709 / 0.800 / 302 | 3.62 | 6000000 / 7000 / 2960000 / 2970000 | 25 / 25 / 25 | 5.65e-15 | none | 0 | 0 |
| nx8_nu3_N30 | tight | `argmin_dense_admm_polished` | 1/1 | 806 | 805 - 810 | 1670 / 0.130 / 806 | 3.78 | 13100000 / 796 / 7970000 / 7970000 | 25 / 25 / 25 | 1.74e-16 | none | 0 | 0 |
| nx12_nu4_N30 | nominal | `argmin_dense_admm` | 1/1 | 175 | 172 - 176 | 586 / 0.120 / 175 | 5.86 | 6100000 / 674 / 1060000 / 1060000 | 25 / 25 / 25 | 6.59e-15 | none | 0 | 0 |
| nx12_nu4_N30 | nominal | `argmin_dense_active_set` | 1/1 | 1160 | 1150 - 1160 | 8940 / 0.120 / 1160 | 5.93 | 84200000 / 669 / 11900000 / 11900000 | 0 / 0 / 0 | 3.22e-16 | none | 22 | 0 |
| nx12_nu4_N30 | nominal | `qpoases` | 1/1 | 24.0 | 23.9 - 24.1 | 1450 / 0.130 / 23.8 | 5.74 | 15200000 / 718 / 220000 / 221000 | 0 / 0 / 0 | 1.55e-13 | none | 19 | 0 |
| nx12_nu4_N30 | nominal | `daqp` | 1/1 | 15.2 | 15.2 - 15.3 | 172 / 6.63 / 8.60 | 5.61 | 2060000 / 54000 / 88100 / 142000 | 1 / 1 / 1 | 2.70e-15 | none | 0 | 0 |
| nx12_nu4_N30 | nominal | `proxqp_dense` | 1/1 | 23.4 | 23.4 - 23.5 | 765 / 13.1 / 10.4 | 5.83 | 7650000 / 121000 / 77600 / 199000 | 0 / 0 / 0 | 2.92e-5 | none | 7 | 0 |
| nx12_nu4_N30 | nominal | `osqp` | 1/1 | 335 | 334 - 336 | 730 / 0.750 / 334 | 5.83 | 5970000 / 5150 / 2990000 / 3000000 | 25 / 25 / 25 | 6.22e-15 | none | 0 | 0 |
| nx12_nu4_N30 | nominal | `argmin_dense_admm_polished` | 1/1 | 663 | 661 - 666 | 1200 / 0.120 / 663 | 5.85 | 11400000 / 670 / 6410000 / 6410000 | 25 / 25 / 25 | 3.36e-16 | none | 0 | 0 |
| nx12_nu4_N30 | tight | `argmin_dense_admm` | 1/1 | 258 | 255 - 260 | 1330 / 0.150 / 258 | 6.28 | 10400000 / 970 / 1390000 / 1390000 | 25 / 25 / 25 | 3.12e-15 | none | 0 | 0 |
| nx12_nu4_N30 | tight | `argmin_dense_active_set` | 1/1 | 19500 | 19400 - 19500 | 64300 / 0.180 / 19500 | 6.38 | 562000000 / 969 / 172000000 / 172000000 | 0 / 0 / 0 | 1.65e-16 | none | 123 | 0 |
| nx12_nu4_N30 | tight | `qpoases` | 1/1 | 32.1 | 32.0 - 32.4 | 11900 / 0.190 / 31.9 | 6.17 | 145000000 / 1040 / 345000 / 346000 | 0 / 0 / 0 | 1.56e-13 | none | 19 | 0 |
| nx12_nu4_N30 | tight | `daqp` | 1/1 | 37.7 | 37.4 - 38.1 | 542 / 14.7 / 23.0 | 6.16 | 4550000 / 103000 / 231000 / 334000 | 1 / 1 / 1 | 1.09e-15 | none | 0 | 0 |
| nx12_nu4_N30 | tight | `proxqp_dense` | 1/1 | 32.6 | 32.4 - 33.0 | 3080 / 18.9 / 13.7 | 6.18 | 24800000 / 174000 / 97600 / 271000 | 0 / 0 / 0 | 4.97e-5 | none | 7 | 0 |
| nx12_nu4_N30 | tight | `osqp` | 1/1 | 511 | 510 - 516 | 1210 / 1.03 / 510 | 6.21 | 10200000 / 9020 / 4930000 / 4940000 | 25 / 25 / 25 | 5.71e-15 | none | 0 | 0 |
| nx12_nu4_N30 | tight | `argmin_dense_admm_polished` | 1/1 | 1570 | 1570 - 1580 | 3090 / 0.160 / 1570 | 6.27 | 24500000 / 970 / 15500000 / 15500000 | 25 / 25 / 25 | 1.33e-16 | none | 0 | 0 |
| nx4_nu2_N50 | nominal | `argmin_dense_admm` | 1/1 | 132 | 132 - 135 | 412 / 0.0900 / 132 | 2.80 | 4250000 / 595 / 823000 / 824000 | 25 / 25 / 25 | 5.03e-15 | none | 0 | 0 |
| nx4_nu2_N50 | nominal | `argmin_dense_active_set` | 1/1 | 608 | 607 - 610 | 2960 / 0.100 / 608 | 2.87 | 28600000 / 594 / 6230000 / 6230000 | 0 / 0 / 0 | 3.18e-16 | none | 22 | 0 |
| nx4_nu2_N50 | nominal | `qpoases` | 1/1 | 18.8 | 18.7 - 18.9 | 592 / 0.100 / 18.7 | 2.61 | 5970000 / 634 / 169000 / 170000 | 0 / 0 / 0 | 3.83e-13 | none | 19 | 0 |
| nx4_nu2_N50 | nominal | `daqp` | 1/1 | 10.7 | 10.6 - 10.8 | 114 / 4.66 / 5.96 | 2.42 | 1310000 / 40000 / 61600 / 102000 | 1 / 1 / 1 | 2.56e-15 | none | 0 | 0 |
| nx4_nu2_N50 | nominal | `proxqp_dense` | 1/1 | 16.5 | 16.4 - 16.6 | 472 / 9.09 / 7.38 | 2.73 | 5020000 / 87100 / 57600 / 145000 | 0 / 0 / 0 | 2.09e-4 | none | 7 | 0 |
| nx4_nu2_N50 | nominal | `osqp` | 1/1 | 245 | 245 - 246 | 509 / 0.580 / 245 | 2.67 | 4140000 / 4460 / 2190000 / 2190000 | 25 / 25 / 25 | 3.63e-15 | none | 0 | 0 |
| nx4_nu2_N50 | nominal | `argmin_dense_admm_polished` | 1/1 | 443 | 441 - 446 | 745 / 0.100 / 443 | 2.85 | 7680000 / 595 / 4250000 / 4250000 | 25 / 25 / 25 | 1.93e-16 | none | 0 | 0 |
| nx4_nu2_N50 | tight | `argmin_dense_admm` | 1/1 | 184 | 183 - 185 | 594 / 0.130 / 183 | 3.00 | 6200000 / 845 / 1070000 / 1070000 | 25 / 25 / 25 | 1.90e-15 | none | 0 | 0 |
| nx4_nu2_N50 | tight | `argmin_dense_active_set` | 1/1 | 10800 | 10800 - 10900 | 33800 / 0.160 / 10800 | 3.09 | 306000000 / 844 / 96600000 / 96600000 | 0 / 0 / 0 | 1.08e-16 | none | 103 | 0 |
| nx4_nu2_N50 | tight | `qpoases` | 1/1 | 23.6 | 23.5 - 24.0 | 6080 / 0.140 / 23.5 | 2.93 | 74700000 / 915 / 246000 / 247000 | 0 / 0 / 0 | 2.20e-13 | none | 19 | 0 |
| nx4_nu2_N50 | tight | `daqp` | 1/1 | 26.9 | 26.7 - 27.1 | 326 / 10.2 / 16.7 | 2.87 | 2790000 / 75200 / 170000 / 246000 | 1 / 1 / 1 | 7.59e-16 | none | 0 | 0 |
| nx4_nu2_N50 | tight | `proxqp_dense` | 1/1 | 23.0 | 22.8 - 23.4 | 1440 / 13.4 / 9.57 | 2.99 | 12500000 / 124000 / 72000 / 196000 | 0 / 0 / 0 | 3.92e-4 | none | 7 | 0 |
| nx4_nu2_N50 | tight | `osqp` | 1/1 | 400 | 399 - 402 | 915 / 0.890 / 399 | 2.96 | 7880000 / 7660 / 3900000 / 3910000 | 25 / 25 / 25 | 3.99e-15 | none | 0 | 0 |
| nx4_nu2_N50 | tight | `argmin_dense_admm_polished` | 1/1 | 1010 | 1010 - 1020 | 1580 / 0.130 / 1010 | 3.01 | 15100000 / 845 / 10000000 / 10000000 | 25 / 25 / 25 | 1.90e-16 | none | 0 | 0 |
| nx4_nu2_N100 | nominal | `argmin_dense_admm` | 1/1 | 435 | 432 - 438 | 2500 / 0.180 / 435 | 9.73 | 17800000 / 1000 / 2400000 / 2400000 | 25 / 25 / 25 | 4.51e-15 | none | 0 | 0 |
| nx4_nu2_N100 | nominal | `argmin_dense_active_set` | 1/1 | 2910 | 2900 - 2910 | 18700 / 0.200 / 2910 | 9.93 | 160000000 / 998 / 29000000 / 29000000 | 0 / 0 / 0 | 1.14e-16 | none | 38 | 0 |
| nx4_nu2_N100 | nominal | `qpoases` | 1/1 | 83.2 | 82.9 - 83.3 | 2880 / 0.190 / 83.0 | 9.51 | 25200000 / 1040 / 623000 / 624000 | 0 / 0 / 0 | 3.72e-13 | none | 19 | 0 |
| nx4_nu2_N100 | nominal | `daqp` | 1/1 | 44.7 | 44.7 - 44.8 | 652 / 18.9 / 25.8 | 9.52 | 7800000 / 130000 / 219000 / 349000 | 1 / 1 / 1 | 2.52e-15 | none | 0 | 0 |
| nx4_nu2_N100 | nominal | `proxqp_dense` | 1/1 | 64.0 | 63.9 - 64.2 | 2210 / 34.8 / 29.3 | 9.62 | 17400000 / 319000 / 190000 / 509000 | 0 / 0 / 0 | 2.09e-4 | none | 7 | 0 |
| nx4_nu2_N100 | nominal | `osqp` | 1/1 | 931 | 927 - 937 | 2770 / 1.01 / 930 | 9.69 | 20300000 / 8090 / 8130000 / 8140000 | 25 / 25 / 25 | 3.40e-15 | none | 0 | 0 |
| nx4_nu2_N100 | nominal | `argmin_dense_admm_polished` | 1/1 | 2130 | 2120 - 2130 | 5660 / 0.190 / 2130 | 9.77 | 36000000 / 998 / 20600000 / 20600000 | 25 / 25 / 25 | 1.82e-16 | none | 2 | 0 |
| nx4_nu2_N100 | tight | `argmin_dense_admm` | 1/1 | 694 | 691 - 700 | 2830 / 0.230 / 694 | 10.2 | 27400000 / 1490 / 3400000 / 3400000 | 25 / 25 / 25 | 2.49e-15 | none | 0 | 0 |
| nx4_nu2_N100 | tight | `argmin_dense_active_set` | 1/1 | 133000 | 131000 - 134000 | 381000 / 0.260 / 133000 | 10.6 | 3300000000 / 1490 / 1110000000 / 1110000000 | 0 / 0 / 0 | 1.63e-16 | none | 203 | 0 |
| nx4_nu2_N100 | tight | `qpoases` | 1/1 | 90.6 | 90.5 - 91.2 | 49000 / 0.250 / 90.4 | 9.98 | 564000000 / 1560 / 876000 / 877000 | 0 / 0 / 0 | 2.21e-13 | none | 19 | 0 |
| nx4_nu2_N100 | tight | `daqp` | 1/1 | 116 | 116 - 118 | 2320 / 46.2 / 69.9 | 10.0 | 18600000 / 260000 / 621000 / 881000 | 1 / 1 / 1 | 9.76e-16 | none | 0 | 0 |
| nx4_nu2_N100 | tight | `proxqp_dense` | 1/1 | 89.1 | 88.7 - 89.5 | 6380 / 49.7 / 39.5 | 10.1 | 51300000 / 451000 / 234000 / 685000 | 0 / 0 / 0 | 3.93e-4 | none | 7 | 0 |
| nx4_nu2_N100 | tight | `osqp` | 1/1 | 1500 | 1500 - 1510 | 4160 / 1.51 / 1500 | 10.2 | 34100000 / 14500 / 14200000 / 14200000 | 25 / 25 / 25 | 6.78e-15 | none | 0 | 0 |
| nx4_nu2_N100 | tight | `argmin_dense_admm_polished` | 1/1 | 5510 | 5500 - 5530 | 10100 / 0.250 / 5510 | 10.3 | 79800000 / 1500 / 55700000 / 55700000 | 25 / 25 / 25 | 1.75e-16 | none | 9 | 0 |
| nx8_nu3_N50 | nominal | `argmin_dense_admm` | 1/1 | 265 | 263 - 271 | 990 / 0.150 / 265 | 8.13 | 9980000 / 852 / 1670000 / 1670000 | 25 / 25 / 25 | 2.52e-15 | none | 0 | 0 |
| nx8_nu3_N50 | nominal | `argmin_dense_active_set` | 1/1 | 1550 | 1540 - 1550 | 8610 / 0.160 / 1550 | 8.25 | 82700000 / 822 / 15500000 / 15500000 | 0 / 0 / 0 | 1.08e-16 | none | 30 | 0 |
| nx8_nu3_N50 | nominal | `qpoases` | 1/1 | 43.4 | 43.0 - 43.6 | 1500 / 0.160 / 43.2 | 8.11 | 14600000 / 871 / 362000 / 363000 | 0 / 0 / 0 | 2.39e-13 | none | 19 | 0 |
| nx8_nu3_N50 | nominal | `daqp` | 1/1 | 24.1 | 24.1 - 24.1 | 305 / 10.3 / 13.8 | 8.05 | 3660000 / 78600 / 128000 / 207000 | 1 / 1 / 1 | 2.07e-15 | none | 0 | 0 |
| nx8_nu3_N50 | nominal | `proxqp_dense` | 1/1 | 38.2 | 38.1 - 38.4 | 1060 / 21.2 / 17.0 | 8.13 | 10600000 / 196000 / 119000 / 315000 | 0 / 0 / 0 | 2.03e-4 | none | 7 | 0 |
| nx8_nu3_N50 | nominal | `osqp` | 1/1 | 529 | 528 - 530 | 1250 / 0.870 / 528 | 8.09 | 10300000 / 6350 / 4690000 / 4700000 | 25 / 25 / 25 | 3.03e-15 | none | 0 | 0 |
| nx8_nu3_N50 | nominal | `argmin_dense_admm_polished` | 1/1 | 1090 | 1080 - 1100 | 1830 / 0.150 / 1090 | 8.20 | 18900000 / 851 / 10600000 / 10600000 | 25 / 25 / 25 | 1.62e-16 | none | 0 | 0 |
| nx8_nu3_N50 | tight | `argmin_dense_admm` | 1/1 | 399 | 397 - 401 | 1870 / 0.180 / 399 | 8.69 | 16900000 / 1200 / 2170000 / 2170000 | 25 / 25 / 25 | 3.59e-15 | none | 0 | 0 |
| nx8_nu3_N50 | tight | `argmin_dense_active_set` | 1/1 | 45300 | 45200 - 45300 | 139000 / 0.210 / 45300 | 8.88 | 1230000000 / 1170 / 404000000 / 404000000 | 0 / 0 / 0 | 2.99e-16 | none | 153 | 0 |
| nx8_nu3_N50 | tight | `qpoases` | 1/1 | 52.5 | 52.4 - 52.9 | 22000 / 0.220 / 52.3 | 8.60 | 262000000 / 1300 / 520000 / 521000 | 0 / 0 / 0 | 1.75e-13 | none | 19 | 0 |
| nx8_nu3_N50 | tight | `daqp` | 1/1 | 61.8 | 61.4 - 61.9 | 1020 / 24.1 / 37.6 | 8.55 | 8420000 / 155000 / 358000 / 513000 | 1 / 1 / 1 | 8.54e-16 | none | 0 | 0 |
| nx8_nu3_N50 | tight | `proxqp_dense` | 1/1 | 53.4 | 53.2 - 53.6 | 3900 / 30.3 / 23.1 | 8.58 | 33900000 / 277000 / 148000 / 425000 | 0 / 0 / 0 | 2.00e-4 | none | 7 | 0 |
| nx8_nu3_N50 | tight | `osqp` | 1/1 | 842 | 838 - 844 | 2050 / 1.23 / 840 | 8.55 | 17500000 / 11100 / 8010000 / 8020000 | 25 / 25 / 25 | 6.33e-15 | none | 0 | 0 |
| nx8_nu3_N50 | tight | `argmin_dense_admm_polished` | 1/1 | 2750 | 2740 - 2760 | 4270 / 0.190 / 2750 | 8.82 | 42000000 / 1190 / 27300000 / 27300000 | 25 / 25 / 25 | 1.33e-16 | none | 4 | 0 |
| nx8_nu3_N100 | nominal | `argmin_dense_admm` | 1/1 | 974 | 966 - 979 | 4750 / 0.260 / 973 | 29.2 | 44400000 / 1380 / 4710000 / 4710000 | 25 / 25 / 25 | 1.63e-15 | none | 0 | 0 |
| nx8_nu3_N100 | nominal | `argmin_dense_active_set` | 1/1 | 8180 | 8160 - 8200 | 53700 / 0.280 / 8180 | 29.6 | 515000000 / 1370 / 81200000 / 81200000 | 0 / 0 / 0 | 2.45e-16 | none | 56 | 0 |
| nx8_nu3_N100 | nominal | `qpoases` | 1/1 | 189 | 189 - 190 | 8070 / 0.270 / 189 | 29.2 | 64300000 / 1410 / 1360000 / 1360000 | 0 / 0 / 0 | 2.46e-13 | none | 19 | 0 |
| nx8_nu3_N100 | nominal | `daqp` | 1/1 | 110 | 110 - 111 | 2060 / 46.7 / 63.6 | 29.1 | 23500000 / 269000 / 477000 / 746000 | 1 / 1 / 1 | 2.22e-15 | none | 0 | 0 |
| nx8_nu3_N100 | nominal | `proxqp_dense` | 1/1 | 148 | 148 - 148 | 5030 / 79.1 / 68.9 | 29.0 | 37900000 / 694000 / 398000 / 1090000 | 0 / 0 / 0 | 2.03e-4 | none | 7 | 0 |
| nx8_nu3_N100 | nominal | `osqp` | 1/1 | 2060 | 2060 - 2070 | 6660 / 1.33 / 2060 | 29.1 | 53800000 / 11700 / 17900000 / 17900000 | 25 / 25 / 25 | 3.11e-15 | none | 0 | 0 |
| nx8_nu3_N100 | nominal | `argmin_dense_admm_polished` | 1/1 | 5750 | 5740 - 5780 | 10500 / 0.280 / 5750 | 29.5 | 96400000 / 1370 / 56700000 / 56700000 | 25 / 25 / 25 | 1.96e-16 | none | 9 | 0 |
| nx8_nu3_N100 | tight | `argmin_dense_admm` | 1/1 | 1510 | 1510 - 1520 | 8490 / 0.330 / 1510 | 29.8 | 76700000 / 2150 / 6960000 / 6970000 | 25 / 25 / 25 | 5.30e-15 | none | 0 | 0 |
| nx8_nu3_N100 | tight | `argmin_dense_active_set` | 1/1 | 669000 | 669000 - 678000 | 1690000 / 0.370 / 669000 | 33.7 | 14400000000 / 2130 / 4960000000 / 4960000000 | 0 / 0 / 0 | 2.56e-16 | none | 303 | 0 |
| nx8_nu3_N100 | tight | `qpoases` | 1/1 | 204 | 204 - 206 | 167000 / 0.360 / 204 | 29.7 | 1940000000 / 2190 / 1890000 / 1890000 | 0 / 0 / 0 | 1.74e-13 | none | 19 | 0 |
| nx8_nu3_N100 | tight | `daqp` | 1/1 | 283 | 281 - 284 | 8180 / 115 / 168 | 29.9 | 58700000 / 555000 / 1340000 / 1900000 | 1 / 1 / 1 | 1.67e-15 | none | 0 | 0 |
| nx8_nu3_N100 | tight | `proxqp_dense` | 1/1 | 197 | 195 - 199 | 18100 / 106 / 89.6 | 29.9 | 142000000 / 981000 / 489000 / 1470000 | 0 / 0 / 0 | 1.96e-4 | none | 7 | 0 |
| nx8_nu3_N100 | tight | `osqp` | 1/1 | 3230 | 3220 - 3240 | 10300 / 2.12 / 3230 | 30.0 | 84800000 / 21300 / 29700000 / 29700000 | 25 / 25 / 25 | 1.52e-14 | none | 0 | 0 |
| nx8_nu3_N100 | tight | `argmin_dense_admm_polished` | 1/1 | 15600 | 15600 - 15700 | 25300 / 0.350 / 15600 | 30.2 | 234000000 / 2130 / 164000000 / 164000000 | 25 / 25 / 25 | 2.15e-16 | none | 17 | 0 |
| dense_mpc_shaped | nominal | `argmin_dense_admm` | 1/1 | 9.33 | 9.27 - 9.43 | 27.3 / 0.0300 / 9.30 | 0.00 | 218000 / 332 / 78400 / 78700 | 25 / 25 / 25 | 1.72e-16 | none | 0 | 0 |
| dense_mpc_shaped | nominal | `argmin_dense_active_set` | 1/1 | 6.37 | 6.27 - 6.44 | 35.6 / 0.0300 / 6.33 | 0.00 | 115000 / 331 / 56400 / 56700 | 0 / 0 / 0 | 4.57e-17 | none | 9 | 0 |
| dense_mpc_shaped | nominal | `qpoases` | 1/1 | 1.66 | 1.64 - 1.68 | 19.0 / 0.0300 / 1.63 | 0.00 | 78200 / 334 / 14700 / 15000 | 0 / 0 / 0 | 1.18e-13 | none | 19 | 0 |
| dense_mpc_shaped | nominal | `daqp` | 1/1 | 0.460 | 0.450 - 0.470 | 6.41 / 0.190 / 0.270 | 0.00 | 14600 / 2000 / 2460 / 4460 | 1 / 1 / 1 | 8.57e-17 | none | 0 | 0 |
| dense_mpc_shaped | nominal | `proxqp_dense` | 1/1 | 1.05 | 1.04 - 1.07 | 47.3 / 0.510 / 0.540 | 0.00 | 132000 / 4610 / 5040 / 9650 | 0 / 0 / 0 | 1.58e-4 | none | 7 | 0 |
| dense_mpc_shaped | nominal | `osqp` | 1/1 | 7.13 | 7.07 - 7.19 | 19.7 / 0.170 / 6.95 | 0.00 | 72000 / 1150 / 54800 / 55900 | 25 / 25 / 25 | 2.74e-16 | none | 0 | 0 |
| dense_mpc_shaped | nominal | `argmin_dense_admm_polished` | 1/1 | 12.2 | 12.1 - 12.3 | 49.1 / 0.0300 / 12.2 | 0.00 | 253000 / 315 / 113000 / 113000 | 25 / 25 / 25 | 3.43e-17 | none | 0 | 0 |

### dense cohort, fixed regime, high tier

The dense cohort in the fixed regime, judged at the high tier's KKT threshold of 1e-6.

| Cell | Variant | Arm | Pass rate | Median tick (us) | Tick p25 - p75 (us) | Median setup / update / solve (us) | Median condense (us) | Median instructions setup / update / solve / tick | Iterations min / median / max | Worst KKT residual | Failures | Allocations max | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nx2_nu1_N10 | nominal | `argmin_dense_admm` | 1/1 | 14.1 | 14.0 - 14.2 | 49.7 / 0.0300 / 14.1 | 0.150 | 290000 / 365 / 133000 / 133000 | 25 / 25 / 25 | 7.14e-17 | none | 0 | 0 |
| nx2_nu1_N10 | nominal | `argmin_dense_active_set` | 1/1 | 10.0 | 9.97 - 10.1 | 49.9 / 0.0300 / 9.99 | 0.160 | 309000 / 368 / 95700 / 96000 | 0 / 0 / 0 | 3.48e-17 | none | 8 | 0 |
| nx2_nu1_N10 | nominal | `qpoases` | 1/1 | 1.28 | 1.27 - 1.30 | 14.5 / 0.0400 / 1.24 | 0.150 | 43900 / 428 / 11300 / 11700 | 0 / 0 / 0 | 6.12e-17 | none | 19 | 0 |
| nx2_nu1_N10 | nominal | `daqp` | 1/1 | 0.640 | 0.640 - 0.660 | 3.92 / 0.215 / 0.430 | 0.150 | 13000 / 2320 / 4220 / 6540 | 1 / 1 / 1 | 8.14e-16 | none | 0 | 0 |
| nx2_nu1_N10 | nominal | `proxqp_dense` | 1/1 | 1.14 | 1.14 - 1.15 | 63.7 / 0.560 / 0.590 | 0.150 | 388000 / 5240 / 5450 / 10700 | 0 / 0 / 0 | 2.00e-7 | none | 7 | 0 |
| nx2_nu1_N10 | nominal | `osqp` | 1/1 | 13.8 | 13.7 - 14.0 | 18.1 / 0.190 / 13.6 | 0.160 | 98600 / 1340 / 124000 / 125000 | 25 / 25 / 25 | 7.14e-17 | none | 60 | 0 |
| nx2_nu1_N10 | tight | `argmin_dense_admm` | 1/1 | 17.3 | 17.2 - 17.5 | 63.2 / 0.0300 / 17.3 | 0.200 | 409000 / 387 / 169000 / 169000 | 25 / 25 / 25 | 6.55e-17 | none | 0 | 0 |
| nx2_nu1_N10 | tight | `argmin_dense_active_set` | 1/1 | 16.0 | 15.9 - 16.1 | 67.2 / 0.0300 / 15.9 | 0.200 | 455000 / 386 / 161000 / 162000 | 0 / 0 / 0 | 1.31e-16 | none | 13 | 0 |
| nx2_nu1_N10 | tight | `qpoases` | 1/1 | 2.00 | 1.96 - 2.07 | 104 / 0.0400 / 1.95 | 0.200 | 796000 / 494 / 19400 / 19900 | 0 / 0 / 0 | 1.97e-16 | none | 19 | 0 |
| nx2_nu1_N10 | tight | `daqp` | 1/1 | 0.830 | 0.820 - 0.842 | 4.66 / 0.290 / 0.530 | 0.190 | 18900 / 3200 / 5580 / 8780 | 1 / 1 / 1 | 2.58e-16 | none | 0 | 0 |
| nx2_nu1_N10 | tight | `proxqp_dense` | 1/1 | 1.30 | 1.29 - 1.32 | 60.7 / 0.670 / 0.630 | 0.190 | 324000 / 6250 / 5980 / 12200 | 0 / 0 / 0 | 2.07e-7 | none | 7 | 0 |
| nx2_nu1_N10 | tight | `osqp` | 1/1 | 19.3 | 19.2 - 19.5 | 25.3 / 0.220 / 19.1 | 0.200 | 139000 / 1600 / 179000 / 181000 | 25 / 25 / 25 | 1.31e-16 | none | 60 | 0 |
| nx4_nu2_N10 | nominal | `argmin_dense_admm` | 1/1 | 27.4 | 27.2 - 27.6 | 107 / 0.0300 / 27.4 | 0.240 | 809000 / 352 / 259000 / 259000 | 25 / 25 / 25 | 2.29e-15 | none | 0 | 0 |
| nx4_nu2_N10 | nominal | `argmin_dense_active_set` | 1/1 | 25.3 | 25.1 - 25.4 | 127 / 0.0300 / 25.2 | 0.250 | 977000 / 318 / 264000 / 264000 | 0 / 0 / 0 | 1.59e-16 | none | 8 | 0 |
| nx4_nu2_N10 | nominal | `qpoases` | 1/1 | 2.05 | 2.04 - 2.07 | 48.9 / 0.0400 / 2.02 | 0.230 | 351000 / 334 / 19900 / 20300 | 0 / 0 / 0 | 1.59e-16 | none | 19 | 0 |
| nx4_nu2_N10 | nominal | `daqp` | 1/1 | 1.05 | 1.03 - 1.08 | 7.40 / 0.390 / 0.660 | 0.220 | 40300 / 4390 / 7300 / 11700 | 1 / 1 / 1 | 4.08e-15 | none | 0 | 0 |
| nx4_nu2_N10 | nominal | `proxqp_dense` | 1/1 | 1.51 | 1.50 - 1.52 | 87.1 / 0.750 / 0.750 | 0.235 | 607000 / 7690 / 7700 / 15400 | 0 / 0 / 0 | 7.73e-8 | none | 7 | 0 |
| nx4_nu2_N10 | nominal | `osqp` | 1/1 | 29.1 | 28.9 - 29.3 | 34.5 / 0.220 / 28.9 | 0.240 | 222000 / 1560 / 266000 / 268000 | 25 / 25 / 25 | 1.59e-16 | none | 60 | 0 |
| nx4_nu2_N10 | tight | `argmin_dense_admm` | 1/1 | 40.2 | 40.1 - 40.5 | 113 / 0.0400 / 40.2 | 0.320 | 915000 / 345 / 412000 / 413000 | 25 / 25 / 25 | 1.47e-16 | none | 0 | 0 |
| nx4_nu2_N10 | tight | `argmin_dense_active_set` | 1/1 | 70.0 | 69.8 - 70.2 | 273 / 0.0400 / 69.9 | 0.310 | 2380000 / 356 / 794000 / 795000 | 0 / 0 / 0 | 4.40e-16 | none | 23 | 0 |
| nx4_nu2_N10 | tight | `qpoases` | 1/1 | 3.68 | 3.66 - 3.71 | 373 / 0.0400 / 3.64 | 0.290 | 4030000 / 415 / 43300 / 43700 | 0 / 0 / 0 | 1.83e-16 | none | 19 | 0 |
| nx4_nu2_N10 | tight | `daqp` | 1/1 | 1.68 | 1.65 - 1.69 | 10.4 / 0.660 / 1.02 | 0.280 | 63400 / 6660 / 12400 / 19100 | 1 / 1 / 1 | 2.20e-16 | none | 0 | 0 |
| nx4_nu2_N10 | tight | `proxqp_dense` | 1/1 | 1.83 | 1.81 - 1.90 | 136 / 0.970 / 0.870 | 0.290 | 911000 / 10100 / 8870 / 19000 | 0 / 0 / 0 | 2.00e-7 | none | 7 | 0 |
| nx4_nu2_N10 | tight | `osqp` | 1/1 | 44.7 | 44.4 - 45.5 | 54.8 / 0.290 / 44.4 | 0.300 | 353000 / 2210 / 446000 / 448000 | 25 / 25 / 25 | 2.56e-16 | none | 60 | 0 |
| nx4_nu2_N20 | nominal | `argmin_dense_admm` | 1/1 | 73.1 | 72.9 - 73.5 | 273 / 0.0500 / 73.0 | 0.590 | 1780000 / 370 / 714000 / 714000 | 25 / 25 / 25 | 1.92e-15 | none | 0 | 0 |
| nx4_nu2_N20 | nominal | `argmin_dense_active_set` | 1/1 | 71.2 | 70.9 - 71.4 | 451 / 0.0500 / 71.1 | 0.610 | 3470000 / 369 / 748000 / 748000 | 0 / 0 / 0 | 1.29e-16 | none | 8 | 0 |
| nx4_nu2_N20 | nominal | `qpoases` | 1/1 | 5.01 | 4.98 - 5.04 | 324 / 0.0500 / 4.96 | 0.500 | 3530000 / 409 / 55400 / 55800 | 0 / 0 / 0 | 2.16e-16 | none | 19 | 0 |
| nx4_nu2_N20 | nominal | `daqp` | 1/1 | 2.16 | 2.12 - 2.18 | 19.1 / 0.910 / 1.24 | 0.490 | 157000 / 10300 / 14900 / 25200 | 1 / 1 / 1 | 3.11e-15 | none | 0 | 0 |
| nx4_nu2_N20 | nominal | `proxqp_dense` | 1/1 | 3.31 | 3.27 - 3.36 | 182 / 1.80 / 1.51 | 0.510 | 1330000 / 18900 / 15600 / 34500 | 0 / 0 / 0 | 1.47e-7 | none | 7 | 0 |
| nx4_nu2_N20 | nominal | `osqp` | 1/1 | 87.4 | 87.2 - 87.9 | 87.9 / 0.310 / 87.1 | 0.540 | 669000 / 2270 / 800000 / 802000 | 25 / 25 / 25 | 2.16e-16 | none | 60 | 0 |
| nx4_nu2_N20 | tight | `argmin_dense_admm` | 1/1 | 137 | 136 - 137 | 299 / 0.0600 / 136 | 0.740 | 2800000 / 470 / 1420000 / 1420000 | 25 / 25 / 25 | 2.26e-16 | none | 0 | 0 |
| nx4_nu2_N20 | tight | `argmin_dense_active_set` | 1/1 | 469 | 464 - 471 | 1710 / 0.0600 / 469 | 0.700 | 16700000 / 469 / 5290000 / 5290000 | 0 / 0 / 0 | 1.13e-16 | none | 43 | 0 |
| nx4_nu2_N20 | tight | `qpoases` | 1/1 | 9.35 | 9.29 - 9.45 | 2300 / 0.0600 / 9.29 | 0.590 | 30700000 / 540 / 126000 / 127000 | 0 / 0 / 0 | 3.39e-16 | none | 19 | 0 |
| nx4_nu2_N20 | tight | `daqp` | 1/1 | 4.50 | 4.47 - 4.57 | 34.5 / 1.68 / 2.81 | 0.580 | 282000 / 17200 / 35100 / 52300 | 1 / 1 / 1 | 2.78e-16 | none | 0 | 0 |
| nx4_nu2_N20 | tight | `proxqp_dense` | 1/1 | 4.54 | 4.48 - 4.64 | 297 / 2.48 / 2.04 | 0.590 | 2310000 / 25900 / 18600 / 44500 | 0 / 0 / 0 | 1.72e-7 | none | 7 | 0 |
| nx4_nu2_N20 | tight | `osqp` | 1/1 | 150 | 150 - 151 | 150 / 0.430 / 150 | 0.660 | 1190000 / 3560 / 1510000 / 1510000 | 25 / 25 / 25 | 5.65e-16 | none | 60 | 0 |
| nx8_nu3_N20 | nominal | `argmin_dense_admm` | 1/1 | 157 | 156 - 158 | 395 / 0.0700 / 156 | 1.68 | 3590000 / 445 / 1530000 / 1530000 | 25 / 25 / 25 | 1.63e-15 | none | 0 | 0 |
| nx8_nu3_N20 | nominal | `argmin_dense_active_set` | 1/1 | 234 | 233 - 234 | 1030 / 0.0800 / 234 | 1.70 | 9570000 / 444 / 2400000 / 2400000 | 0 / 0 / 0 | 1.73e-16 | none | 16 | 0 |
| nx8_nu3_N20 | nominal | `qpoases` | 1/1 | 10.2 | 10.2 - 10.3 | 1190 / 0.0600 / 10.1 | 1.37 | 13900000 / 484 / 113000 / 113000 | 0 / 0 / 0 | 2.49e-16 | none | 19 | 0 |
| nx8_nu3_N20 | nominal | `daqp` | 1/1 | 4.03 | 4.02 - 4.07 | 38.5 / 1.76 / 2.27 | 1.26 | 386000 / 18200 / 26700 / 44900 | 1 / 1 / 1 | 3.40e-15 | none | 0 | 0 |
| nx8_nu3_N20 | nominal | `proxqp_dense` | 1/1 | 6.47 | 6.39 - 6.60 | 263 / 3.45 / 3.02 | 1.39 | 2410000 / 35800 / 26800 / 62600 | 0 / 0 / 0 | 1.41e-7 | none | 7 | 0 |
| nx8_nu3_N20 | nominal | `osqp` | 1/1 | 187 | 186 - 188 | 174 / 0.410 / 187 | 1.51 | 1390000 / 2990 / 1670000 / 1670000 | 25 / 25 / 25 | 3.46e-16 | none | 60 | 0 |
| nx8_nu3_N20 | tight | `argmin_dense_admm` | 1/1 | 322 | 320 - 323 | 875 / 0.0900 / 322 | 1.89 | 7540000 / 595 / 3240000 / 3240000 | 25 / 25 / 25 | 1.04e-16 | none | 0 | 0 |
| nx8_nu3_N20 | tight | `argmin_dense_active_set` | 1/1 | 1700 | 1670 - 1770 | 6200 / 0.100 / 1700 | 1.90 | 59400000 / 594 / 17300000 / 17300000 | 0 / 0 / 0 | 1.29e-16 | none | 63 | 0 |
| nx8_nu3_N20 | tight | `qpoases` | 1/1 | 17.1 | 16.9 - 17.2 | 8620 / 0.0800 / 17.0 | 1.57 | 117000000 / 665 / 221000 / 221000 | 0 / 0 / 0 | 2.78e-16 | none | 19 | 0 |
| nx8_nu3_N20 | tight | `daqp` | 1/1 | 9.78 | 9.71 - 9.92 | 86.2 / 3.86 / 5.91 | 1.48 | 748000 / 32100 / 67100 / 99300 | 1 / 1 / 1 | 4.31e-16 | none | 0 | 0 |
| nx8_nu3_N20 | tight | `proxqp_dense` | 1/1 | 8.90 | 8.81 - 8.96 | 800 / 4.92 / 3.98 | 1.63 | 6540000 / 50000 / 32200 / 82200 | 0 / 0 / 0 | 4.19e-8 | none | 7 | 0 |
| nx8_nu3_N20 | tight | `osqp` | 1/1 | 319 | 318 - 322 | 315 / 0.610 / 319 | 1.80 | 2570000 / 4910 / 3090000 / 3100000 | 25 / 25 / 25 | 8.67e-16 | none | 60 | 0 |
| nx8_nu3_N30 | nominal | `argmin_dense_admm` | 1/1 | 356 | 355 - 359 | 966 / 0.100 / 356 | 3.53 | 8620000 / 598 / 3440000 / 3450000 | 25 / 25 / 25 | 1.92e-15 | none | 0 | 0 |
| nx8_nu3_N30 | nominal | `argmin_dense_active_set` | 1/1 | 506 | 505 - 507 | 2560 / 0.100 / 505 | 3.57 | 24200000 / 597 / 5200000 / 5200000 | 0 / 0 / 0 | 3.81e-16 | none | 16 | 0 |
| nx8_nu3_N30 | nominal | `qpoases` | 1/1 | 22.4 | 22.2 - 22.5 | 5180 / 0.100 / 22.2 | 3.27 | 58500000 / 637 / 242000 / 242000 | 0 / 0 / 0 | 2.54e-16 | none | 19 | 0 |
| nx8_nu3_N30 | nominal | `daqp` | 1/1 | 8.66 | 8.63 - 8.71 | 88.5 / 3.82 / 4.83 | 2.96 | 1010000 / 33900 / 51600 / 85400 | 1 / 1 / 1 | 1.46e-15 | none | 0 | 0 |
| nx8_nu3_N30 | nominal | `proxqp_dense` | 1/1 | 14.4 | 14.3 - 14.5 | 466 / 7.92 / 6.47 | 3.33 | 4690000 / 78300 / 50800 / 129000 | 0 / 0 / 0 | 1.55e-7 | none | 7 | 0 |
| nx8_nu3_N30 | nominal | `osqp` | 1/1 | 432 | 429 - 435 | 395 / 0.580 / 432 | 3.47 | 3230000 / 4180 / 3840000 / 3840000 | 25 / 25 / 25 | 3.18e-16 | none | 60 | 0 |
| nx8_nu3_N30 | tight | `argmin_dense_admm` | 1/1 | 808 | 806 - 818 | 1920 / 0.130 / 807 | 3.78 | 16700000 / 796 / 7970000 / 7970000 | 25 / 25 / 25 | 1.74e-16 | none | 0 | 0 |
| nx8_nu3_N30 | tight | `argmin_dense_active_set` | 1/1 | 7420 | 7410 - 7460 | 24300 / 0.130 / 7420 | 3.84 | 224000000 / 795 / 69000000 / 69000000 | 0 / 0 / 0 | 1.28e-16 | none | 93 | 0 |
| nx8_nu3_N30 | tight | `qpoases` | 1/1 | 37.9 | 37.8 - 38.0 | 35900 / 0.150 / 37.8 | 3.64 | 475000000 / 920 / 489000 / 490000 | 0 / 0 / 0 | 3.48e-16 | none | 19 | 0 |
| nx8_nu3_N30 | tight | `daqp` | 1/1 | 21.6 | 21.6 - 21.7 | 248 / 8.42 / 13.2 | 3.48 | 2140000 / 63800 / 139000 / 203000 | 1 / 1 / 1 | 1.04e-15 | none | 0 | 0 |
| nx8_nu3_N30 | tight | `proxqp_dense` | 1/1 | 20.2 | 20.0 - 20.3 | 1630 / 11.6 / 8.59 | 3.69 | 13700000 / 112000 / 64700 / 176000 | 0 / 0 / 0 | 1.74e-7 | none | 7 | 0 |
| nx8_nu3_N30 | tight | `osqp` | 1/1 | 720 | 716 - 723 | 712 / 0.870 / 719 | 3.77 | 5990000 / 7000 / 6860000 / 6870000 | 25 / 25 / 25 | 1.22e-15 | none | 60 | 0 |
| nx12_nu4_N30 | nominal | `argmin_dense_admm` | 1/1 | 664 | 662 - 666 | 1770 / 0.120 / 663 | 5.86 | 15600000 / 696 / 6410000 / 6410000 | 25 / 25 / 25 | 3.36e-16 | none | 0 | 0 |
| nx12_nu4_N30 | nominal | `argmin_dense_active_set` | 1/1 | 1160 | 1150 - 1160 | 8720 / 0.120 / 1160 | 5.93 | 84200000 / 669 / 11900000 / 11900000 | 0 / 0 / 0 | 3.22e-16 | none | 22 | 0 |
| nx12_nu4_N30 | nominal | `qpoases` | 1/1 | 34.8 | 34.7 - 34.8 | 17100 / 0.130 / 34.6 | 5.78 | 177000000 / 738 / 386000 / 387000 | 0 / 0 / 0 | 6.21e-16 | none | 19 | 0 |
| nx12_nu4_N30 | nominal | `daqp` | 1/1 | 15.2 | 15.2 - 15.3 | 172 / 6.62 / 8.58 | 5.59 | 2060000 / 54000 / 88100 / 142000 | 1 / 1 / 1 | 2.70e-15 | none | 0 | 0 |
| nx12_nu4_N30 | nominal | `proxqp_dense` | 1/1 | 23.5 | 23.4 - 23.6 | 786 / 13.1 / 10.4 | 5.84 | 7870000 / 121000 / 77600 / 199000 | 0 / 0 / 0 | 8.11e-7 | none | 7 | 0 |
| nx12_nu4_N30 | nominal | `osqp` | 1/1 | 795 | 790 - 798 | 731 / 0.750 / 794 | 5.92 | 5970000 / 5150 / 7120000 / 7120000 | 25 / 25 / 25 | 5.06e-16 | none | 60 | 0 |
| nx12_nu4_N30 | tight | `argmin_dense_admm` | 1/1 | 1570 | 1570 - 1580 | 5080 / 0.160 / 1570 | 6.27 | 31200000 / 970 / 15500000 / 15500000 | 25 / 25 / 25 | 1.33e-16 | none | 0 | 0 |
| nx12_nu4_N30 | tight | `argmin_dense_active_set` | 1/1 | 19600 | 19400 - 19600 | 64500 / 0.180 / 19600 | 6.46 | 562000000 / 969 / 172000000 / 172000000 | 0 / 0 / 0 | 1.65e-16 | none | 123 | 0 |
| nx12_nu4_N30 | tight | `qpoases` | 1/1 | 59.8 | 59.8 - 60.0 | 116000 / 0.190 / 59.6 | 6.20 | 1470000000 / 1040 / 802000 / 803000 | 0 / 0 / 0 | 4.86e-16 | none | 19 | 0 |
| nx12_nu4_N30 | tight | `daqp` | 1/1 | 37.4 | 37.3 - 37.6 | 540 / 14.7 / 22.7 | 6.14 | 4550000 / 103000 / 231000 / 334000 | 1 / 1 / 1 | 1.09e-15 | none | 0 | 0 |
| nx12_nu4_N30 | tight | `proxqp_dense` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 0 / 0 / 0 | 4.97e-5 | inaccurate 300 | 7 | 0 |
| nx12_nu4_N30 | tight | `osqp` | 1/1 | 1250 | 1250 - 1260 | 1440 / 1.08 / 1250 | 6.33 | 10200000 / 9020 / 11900000 / 11900000 | 25 / 25 / 25 | 1.66e-15 | none | 60 | 0 |
| nx4_nu2_N50 | nominal | `argmin_dense_admm` | 1/1 | 444 | 442 - 447 | 1000 / 0.100 / 443 | 2.83 | 9290000 / 595 / 4250000 / 4250000 | 25 / 25 / 25 | 1.93e-16 | none | 0 | 0 |
| nx4_nu2_N50 | nominal | `argmin_dense_active_set` | 1/1 | 607 | 606 - 610 | 2960 / 0.100 / 607 | 2.88 | 28600000 / 594 / 6230000 / 6230000 | 0 / 0 / 0 | 3.18e-16 | none | 22 | 0 |
| nx4_nu2_N50 | nominal | `qpoases` | 1/1 | 27.2 | 27.0 - 27.4 | 6990 / 0.100 / 27.1 | 2.68 | 77000000 / 634 / 298000 / 298000 | 0 / 0 / 0 | 3.64e-16 | none | 19 | 0 |
| nx4_nu2_N50 | nominal | `daqp` | 1/1 | 10.6 | 10.6 - 10.8 | 114 / 4.64 / 5.99 | 2.44 | 1310000 / 40000 / 61600 / 102000 | 1 / 1 / 1 | 2.56e-15 | none | 0 | 0 |
| nx4_nu2_N50 | nominal | `proxqp_dense` | 1/1 | 16.5 | 16.4 - 16.6 | 516 / 9.12 / 7.38 | 2.76 | 5340000 / 87100 / 57600 / 145000 | 0 / 0 / 0 | 1.64e-7 | none | 7 | 0 |
| nx4_nu2_N50 | nominal | `osqp` | 1/1 | 551 | 548 - 557 | 508 / 0.630 / 550 | 2.83 | 4150000 / 4430 / 4920000 / 4930000 | 25 / 25 / 25 | 2.61e-16 | none | 60 | 0 |
| nx4_nu2_N50 | tight | `argmin_dense_admm` | 1/1 | 1010 | 1010 - 1010 | 1800 / 0.130 / 1010 | 2.99 | 17200000 / 845 / 10000000 / 10000000 | 25 / 25 / 25 | 1.90e-16 | none | 0 | 0 |
| nx4_nu2_N50 | tight | `argmin_dense_active_set` | 1/1 | 10800 | 10800 - 10900 | 33700 / 0.180 / 10800 | 3.09 | 306000000 / 844 / 96600000 / 96600000 | 0 / 0 / 0 | 1.08e-16 | none | 103 | 0 |
| nx4_nu2_N50 | tight | `qpoases` | 1/1 | 52.1 | 51.9 - 52.3 | 48100 / 0.150 / 51.9 | 2.95 | 639000000 / 915 / 665000 / 665000 | 0 / 0 / 0 | 4.75e-16 | none | 19 | 0 |
| nx4_nu2_N50 | tight | `daqp` | 1/1 | 26.8 | 26.7 - 27.2 | 326 / 10.2 / 16.6 | 2.86 | 2790000 / 75200 / 170000 / 246000 | 1 / 1 / 1 | 7.59e-16 | none | 0 | 0 |
| nx4_nu2_N50 | tight | `proxqp_dense` | 1/1 | 22.9 | 22.8 - 23.0 | 1390 / 13.3 / 9.54 | 2.99 | 13500000 / 124000 / 72000 / 196000 | 0 / 0 / 0 | 2.47e-9 | none | 7 | 0 |
| nx4_nu2_N50 | tight | `osqp` | 1/1 | 972 | 967 - 976 | 919 / 0.940 / 971 | 3.02 | 7880000 / 7660 / 9380000 / 9380000 | 25 / 25 / 25 | 9.03e-16 | none | 60 | 0 |
| nx4_nu2_N100 | nominal | `argmin_dense_admm` | 1/1 | 2120 | 2120 - 2130 | 6350 / 0.180 / 2120 | 9.80 | 40600000 / 998 / 20600000 / 20600000 | 25 / 25 / 25 | 1.82e-16 | none | 2 | 0 |
| nx4_nu2_N100 | nominal | `argmin_dense_active_set` | 1/1 | 2910 | 2900 - 2920 | 18600 / 0.200 / 2910 | 9.94 | 160000000 / 998 / 29000000 / 29000000 | 0 / 0 / 0 | 1.14e-16 | none | 38 | 0 |
| nx4_nu2_N100 | nominal | `qpoases` | 1/1 | 119 | 119 - 120 | 91100 / 0.190 / 119 | 9.56 | 812000000 / 1040 / 1160000 / 1160000 | 0 / 0 / 0 | 2.28e-16 | none | 19 | 0 |
| nx4_nu2_N100 | nominal | `daqp` | 1/1 | 44.8 | 44.7 - 44.8 | 647 / 18.9 / 25.8 | 9.55 | 7800000 / 130000 / 219000 / 349000 | 1 / 1 / 1 | 2.52e-15 | none | 0 | 0 |
| nx4_nu2_N100 | nominal | `proxqp_dense` | 1/1 | 64.0 | 63.8 - 64.1 | 2270 / 34.7 / 29.2 | 9.58 | 18300000 / 319000 / 190000 / 509000 | 0 / 0 / 0 | 1.64e-7 | none | 7 | 0 |
| nx4_nu2_N100 | nominal | `osqp` | 1/1 | 2480 | 2470 - 2500 | 2770 / 1.01 / 2480 | 9.72 | 20300000 / 8080 / 23100000 / 23100000 | 25 / 25 / 25 | 1.82e-16 | none | 60 | 0 |
| nx4_nu2_N100 | tight | `argmin_dense_admm` | 1/1 | 5500 | 5480 - 5510 | 11800 / 0.250 / 5500 | 10.3 | 86300000 / 1470 / 55700000 / 55700000 | 25 / 25 / 25 | 1.75e-16 | none | 9 | 0 |
| nx4_nu2_N100 | tight | `argmin_dense_active_set` | 1/1 | 132000 | 131000 - 133000 | 378000 / 0.260 / 132000 | 10.7 | 3300000000 / 1500 / 1110000000 / 1110000000 | 0 / 0 / 0 | 1.63e-16 | none | 203 | 0 |
| nx4_nu2_N100 | tight | `qpoases` | 1/1 | 204 | 203 - 205 | 876000 / 0.250 / 204 | 10.0 | 8510000000 / 1540 / 2530000 / 2530000 | 0 / 0 / 0 | 7.00e-16 | none | 19 | 0 |
| nx4_nu2_N100 | tight | `daqp` | 1/1 | 117 | 116 - 118 | 2330 / 46.2 / 70.8 | 9.94 | 18600000 / 260000 / 621000 / 881000 | 1 / 1 / 1 | 9.76e-16 | none | 0 | 0 |
| nx4_nu2_N100 | tight | `proxqp_dense` | 1/1 | 89.1 | 88.9 - 89.5 | 8160 / 49.6 / 39.4 | 10.1 | 54000000 / 450000 / 234000 / 684000 | 0 / 0 / 0 | 1.93e-9 | none | 7 | 0 |
| nx4_nu2_N100 | tight | `osqp` | 1/1 | 4170 | 4150 - 4200 | 5210 / 1.54 / 4160 | 10.2 | 34100000 / 14500 / 40300000 / 40300000 | 25 / 25 / 25 | 2.84e-15 | none | 60 | 0 |
| nx8_nu3_N50 | nominal | `argmin_dense_admm` | 1/1 | 1090 | 1080 - 1100 | 2590 / 0.150 / 1090 | 8.20 | 23700000 / 848 / 10600000 / 10600000 | 25 / 25 / 25 | 1.62e-16 | none | 0 | 0 |
| nx8_nu3_N50 | nominal | `argmin_dense_active_set` | 1/1 | 1550 | 1550 - 1560 | 8590 / 0.170 / 1550 | 8.24 | 82700000 / 822 / 15500000 / 15500000 | 0 / 0 / 0 | 1.08e-16 | none | 30 | 0 |
| nx8_nu3_N50 | nominal | `qpoases` | 1/1 | 63.2 | 62.9 - 63.6 | 32000 / 0.160 / 63.0 | 8.12 | 318000000 / 891 / 657000 / 658000 | 0 / 0 / 0 | 2.16e-16 | none | 19 | 0 |
| nx8_nu3_N50 | nominal | `daqp` | 1/1 | 24.1 | 24.1 - 24.2 | 305 / 10.3 / 13.8 | 8.05 | 3660000 / 78600 / 128000 / 207000 | 1 / 1 / 1 | 2.07e-15 | none | 0 | 0 |
| nx8_nu3_N50 | nominal | `proxqp_dense` | 1/1 | 38.2 | 38.1 - 38.4 | 1120 / 21.3 / 16.9 | 8.13 | 11200000 / 196000 / 119000 / 315000 | 0 / 0 / 0 | 1.57e-7 | none | 7 | 0 |
| nx8_nu3_N50 | nominal | `osqp` | 1/1 | 1290 | 1290 - 1300 | 1260 / 0.890 / 1290 | 8.11 | 10300000 / 6340 / 11900000 / 11900000 | 25 / 25 / 25 | 2.71e-16 | none | 60 | 0 |
| nx8_nu3_N50 | tight | `argmin_dense_admm` | 1/1 | 2750 | 2740 - 2760 | 5770 / 0.200 / 2750 | 8.80 | 50400000 / 1170 / 27300000 / 27300000 | 25 / 25 / 25 | 1.33e-16 | none | 4 | 0 |
| nx8_nu3_N50 | tight | `argmin_dense_active_set` | 1/1 | 45200 | 45200 - 45300 | 139000 / 0.220 / 45200 | 8.87 | 1230000000 / 1190 / 404000000 / 404000000 | 0 / 0 / 0 | 2.99e-16 | none | 153 | 0 |
| nx8_nu3_N50 | tight | `qpoases` | 1/1 | 108 | 108 - 109 | 258000 / 0.220 / 108 | 8.59 | 3090000000 / 1300 / 1360000 / 1360000 | 0 / 0 / 0 | 4.78e-16 | none | 19 | 0 |
| nx8_nu3_N50 | tight | `daqp` | 1/1 | 61.5 | 61.4 - 61.7 | 1020 / 24.1 / 37.4 | 8.57 | 8420000 / 155000 / 358000 / 513000 | 1 / 1 / 1 | 8.54e-16 | none | 0 | 0 |
| nx8_nu3_N50 | tight | `proxqp_dense` | 1/1 | 53.4 | 53.2 - 53.6 | 3940 / 30.3 / 23.0 | 8.61 | 34400000 / 277000 / 148000 / 425000 | 0 / 0 / 0 | 1.67e-7 | none | 7 | 0 |
| nx8_nu3_N50 | tight | `osqp` | 1/1 | 2150 | 2140 - 2160 | 2060 / 1.28 / 2150 | 8.63 | 17500000 / 11100 / 20600000 / 20700000 | 25 / 25 / 25 | 2.83e-15 | none | 60 | 0 |
| nx8_nu3_N100 | nominal | `argmin_dense_admm` | 1/1 | 5760 | 5740 - 5790 | 13300 / 0.280 / 5760 | 29.5 | 115000000 / 1370 / 56700000 / 56700000 | 25 / 25 / 25 | 1.96e-16 | none | 9 | 0 |
| nx8_nu3_N100 | nominal | `argmin_dense_active_set` | 1/1 | 8170 | 8150 - 8200 | 51700 / 0.280 / 8170 | 29.5 | 515000000 / 1370 / 81200000 / 81200000 | 0 / 0 / 0 | 2.45e-16 | none | 56 | 0 |
| nx8_nu3_N100 | nominal | `qpoases` | 1/1 | 279 | 278 - 280 | 417000 / 0.270 / 279 | 29.1 | 3200000000 / 1410 / 2580000 / 2580000 | 0 / 0 / 0 | 4.90e-16 | none | 19 | 0 |
| nx8_nu3_N100 | nominal | `daqp` | 1/1 | 110 | 110 - 111 | 2060 / 46.7 / 63.7 | 29.0 | 23500000 / 269000 / 477000 / 746000 | 1 / 1 / 1 | 2.22e-15 | none | 0 | 0 |
| nx8_nu3_N100 | nominal | `proxqp_dense` | 1/1 | 148 | 148 - 149 | 4150 / 79.2 / 69.0 | 29.0 | 39700000 / 694000 / 398000 / 1090000 | 0 / 0 / 0 | 1.57e-7 | none | 7 | 0 |
| nx8_nu3_N100 | nominal | `osqp` | 1/1 | 6700 | 6660 - 6730 | 6630 / 1.35 / 6700 | 29.2 | 53800000 / 11700 / 61900000 / 61900000 | 25 / 25 / 25 | 3.92e-16 | none | 60 | 0 |
| nx8_nu3_N100 | tight | `argmin_dense_admm` | 1/1 | 15600 | 15600 - 15600 | 30200 / 0.350 / 15600 | 30.2 | 268000000 / 2130 / 164000000 / 164000000 | 25 / 25 / 25 | 2.15e-16 | none | 17 | 0 |
| nx8_nu3_N100 | tight | `argmin_dense_active_set` | 1/1 | 675000 | 647000 - 678000 | 1690000 / 0.370 / 675000 | 34.3 | 14400000000 / 2130 / 4960000000 / 4960000000 | 0 / 0 / 0 | 2.56e-16 | none | 303 | 0 |
| nx8_nu3_N100 | tight | `qpoases` | 1/1 | 442 | 441 - 443 | 4630000 / 0.360 / 441 | 29.9 | 38600000000 / 2190 / 5410000 / 5420000 | 0 / 0 / 0 | 1.24e-15 | none | 19 | 0 |
| nx8_nu3_N100 | tight | `daqp` | 1/1 | 283 | 281 - 284 | 8170 / 115 / 168 | 29.8 | 58700000 / 555000 / 1340000 / 1900000 | 1 / 1 / 1 | 1.67e-15 | none | 0 | 0 |
| nx8_nu3_N100 | tight | `proxqp_dense` | 1/1 | 197 | 195 - 199 | 18100 / 106 / 89.5 | 29.9 | 143000000 / 981000 / 489000 / 1470000 | 0 / 0 / 0 | 2.17e-7 | none | 7 | 0 |
| nx8_nu3_N100 | tight | `osqp` | 1/1 | 11000 | 10800 - 11100 | 12400 / 2.12 / 11000 | 30.1 | 84800000 / 21300 / 102000000 / 102000000 | 25 / 25 / 25 | 2.97e-15 | none | 60 | 0 |
| dense_mpc_shaped | nominal | `argmin_dense_admm` | 1/1 | 12.2 | 12.1 - 12.6 | 49.2 / 0.0300 / 12.2 | 0.00 | 335000 / 331 / 113000 / 113000 | 25 / 25 / 25 | 3.43e-17 | none | 0 | 0 |
| dense_mpc_shaped | nominal | `argmin_dense_active_set` | 1/1 | 6.33 | 6.29 - 6.43 | 23.0 / 0.0300 / 6.30 | 0.00 | 114000 / 331 / 56400 / 56700 | 0 / 0 / 0 | 4.57e-17 | none | 9 | 0 |
| dense_mpc_shaped | nominal | `qpoases` | 1/1 | 1.88 | 1.87 - 1.93 | 68.4 / 0.0300 / 1.85 | 0.00 | 459000 / 334 / 17700 / 18000 | 0 / 0 / 0 | 4.57e-17 | none | 19 | 0 |
| dense_mpc_shaped | nominal | `daqp` | 1/1 | 0.460 | 0.450 - 0.470 | 3.87 / 0.190 / 0.270 | 0.00 | 14400 / 2000 / 2460 / 4460 | 1 / 1 / 1 | 8.57e-17 | none | 0 | 0 |
| dense_mpc_shaped | nominal | `proxqp_dense` | 1/1 | 1.04 | 1.02 - 1.05 | 31.4 / 0.500 / 0.540 | 0.00 | 167000 / 4610 / 5040 / 9650 | 0 / 0 / 0 | 9.49e-7 | none | 7 | 0 |
| dense_mpc_shaped | nominal | `osqp` | 1/1 | 10.7 | 10.6 - 10.8 | 13.3 / 0.180 / 10.5 | 0.00 | 69200 / 1150 / 84200 / 85300 | 25 / 25 / 25 | 6.85e-17 | none | 60 | 0 |

### dense cohort, closed_loop regime, mpc tier

The dense cohort following the same prescribed 60-step initial-state sequence in every arm,
updating and re-solving at each tick, judged at the mpc tier's threshold of 1e-3.

| Cell | Variant | Arm | Pass rate | Median tick (us) | Tick p25 - p75 (us) | Median setup / update / solve (us) | Median condense (us) | Median instructions setup / update / solve / tick | Iterations min / median / max | Worst KKT residual | Failures | Allocations max | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nx2_nu1_N10 | nominal | `argmin_dense_admm` | 1/1 | 9.84 | 9.78 - 9.98 | 19.6 / 0.0300 / 9.81 | 0.150 | 161000 / 389 / 84600 / 85000 | 25 / 25 / 50 | 4.72e-4 | none | 0 | 0 |
| nx2_nu1_N10 | nominal | `argmin_dense_active_set` | 1/1 | 37.3 | 19.9 - 65.5 | 50.2 / 0.0300 / 37.2 | 0.170 | 310000 / 368 / 317000 / 318000 | 0 / 10 / 34 | 2.36e-16 | none | 178 | 0 |
| nx2_nu1_N10 | nominal | `qpoases` | 1/1 | 8.40 | 4.66 - 11.6 | 26.5 / 0.0300 / 8.37 | 0.150 | 141000 / 411 / 70600 / 71000 | 0 / 5 / 9 | 1.18e-13 | none | 88 | 0 |
| nx2_nu1_N10 | nominal | `daqp` | 1/1 | 1.16 | 1.00 - 1.80 | 3.93 / 0.220 / 0.940 | 0.150 | 13100 / 2320 / 7040 / 9340 | 1 / 6 / 10 | 2.89e-15 | none | 0 | 0 |
| nx2_nu1_N10 | nominal | `proxqp_dense` | 7/10 | 17.8 | 8.66 - 22.6 | 57.9 / 0.610 / 17.2 | 0.170 | 341000 / 5240 / 137000 / 143000 | 0 / 6 / 11 | 2.02e-1 | inaccurate 90 | 7 | 0 |
| nx2_nu1_N10 | nominal | `osqp` | 1/1 | 8.31 | 8.25 - 10.9 | 18.9 / 0.190 / 8.12 | 0.160 | 100000 / 1350 / 75300 / 76700 | 25 / 25 / 50 | 4.72e-4 | none | 0 | 0 |
| nx2_nu1_N10 | nominal | `argmin_dense_admm_polished` | 1/1 | 14.1 | 14.0 - 14.3 | 34.4 / 0.0300 / 14.1 | 0.160 | 209000 / 389 / 132000 / 132000 | 25 / 25 / 50 | 2.38e-4 | none | 0 | 0 |
| nx2_nu1_N10 | tight | `argmin_dense_admm` | 1/1 | 19.7 | 19.2 - 28.7 | 29.2 / 0.0300 / 19.7 | 0.190 | 242000 / 387 / 160000 / 160000 | 25 / 50 / 125 | 5.68e-4 | none | 0 | 0 |
| nx2_nu1_N10 | tight | `argmin_dense_active_set` | 1/1 | 40.0 | 15.2 - 45.5 | 68.8 / 0.0400 / 40.0 | 0.210 | 455000 / 369 / 377000 / 377000 | 0 / 10 / 10 | 4.05e-16 | none | 132 | 120 |
| nx2_nu1_N10 | tight | `qpoases` | 1/1 | 7.20 | 1.94 - 11.6 | 40.9 / 0.0400 / 7.16 | 0.200 | 253000 / 494 / 64300 / 64800 | 0 / 2 / 7 | 3.17e-13 | none | 117 | 0 |
| nx2_nu1_N10 | tight | `daqp` | 1/1 | 2.29 | 0.955 - 3.42 | 4.75 / 0.290 / 2.00 | 0.190 | 18900 / 3210 / 20200 / 23500 | 1 / 5 / 15 | 3.19e-15 | none | 0 | 0 |
| nx2_nu1_N10 | tight | `proxqp_dense` | 7/12 | 25.3 | 4.32 - 37.6 | 51.3 / 0.720 / 24.6 | 0.220 | 243000 / 6250 / 188000 / 194000 | 0 / 7 / 23 | 1.00e+0 | inaccurate 45, infeasible_claim 80 | 7 | 0 |
| nx2_nu1_N10 | tight | `osqp` | 1/1 | 21.6 | 11.9 - 31.1 | 26.4 / 0.220 / 21.3 | 0.190 | 142000 / 1630 / 215000 / 216000 | 25 / 50 / 125 | 5.16e-4 | none | 0 | 0 |
| nx2_nu1_N10 | tight | `argmin_dense_admm_polished` | 1/1 | 27.5 | 26.5 - 36.4 | 51.4 / 0.0300 / 27.5 | 0.200 | 331000 / 387 / 247000 / 248000 | 25 / 50 / 125 | 2.39e-4 | none | 0 | 0 |
| nx4_nu2_N10 | nominal | `argmin_dense_admm` | 1/1 | 16.8 | 16.7 - 16.8 | 35.0 / 0.0300 / 16.8 | 0.230 | 301000 / 295 / 130000 / 130000 | 25 / 25 / 50 | 9.87e-4 | none | 0 | 0 |
| nx4_nu2_N10 | nominal | `argmin_dense_active_set` | 1/1 | 212 | 103 - 298 | 121 / 0.0400 / 212 | 0.270 | 977000 / 294 / 2040000 / 2040000 | 0 / 29.5 / 50 | 3.47e-5 | none | 294 | 0 |
| nx4_nu2_N10 | nominal | `qpoases` | 1/1 | 17.4 | 13.4 - 30.7 | 54.9 / 0.0400 / 17.4 | 0.250 | 359000 / 334 / 162000 / 162000 | 0 / 8 / 17 | 3.88e-13 | none | 159 | 0 |
| nx4_nu2_N10 | nominal | `daqp` | 1/1 | 4.14 | 3.02 - 5.07 | 7.82 / 0.400 / 3.74 | 0.220 | 40300 / 4420 / 33100 / 37600 | 5 / 9 / 18 | 1.36e-4 | none | 0 | 0 |
| nx4_nu2_N10 | nominal | `proxqp_dense` | 41/60 | 44.2 | 37.6 - 48.1 | 83.2 / 0.840 / 43.3 | 0.260 | 552000 / 7680 / 338000 / 346000 | 0 / 9 / 15 | 2.74e-1 | inaccurate 95 | 7 | 0 |
| nx4_nu2_N10 | nominal | `osqp` | 1/1 | 16.2 | 16.1 - 16.2 | 35.5 / 0.220 / 15.9 | 0.240 | 224000 / 1560 / 152000 / 154000 | 25 / 25 / 75 | 9.82e-4 | none | 0 | 0 |
| nx4_nu2_N10 | nominal | `argmin_dense_admm_polished` | 1/1 | 27.8 | 27.5 - 28.0 | 59.2 / 0.0300 / 27.7 | 0.250 | 431000 / 295 / 260000 / 260000 | 25 / 25 / 50 | 7.40e-5 | none | 0 | 0 |
| nx4_nu2_N10 | tight | `argmin_dense_admm` | 1/1 | 35.9 | 35.8 - 53.3 | 57.0 / 0.0400 / 35.9 | 0.290 | 506000 / 347 / 282000 / 283000 | 25 / 50 / 225 | 5.35e-4 | none | 0 | 0 |
| nx4_nu2_N10 | tight | `argmin_dense_active_set` | 1/1 | 202 | 112 - 233 | 269 / 0.0400 / 202 | 0.370 | 2380000 / 344 / 1990000 / 1990000 | 0 / 20 / 20 | 5.54e-16 | none | 462 | 165 |
| nx4_nu2_N10 | tight | `qpoases` | 1/1 | 21.7 | 13.9 - 29.9 | 117 / 0.0400 / 21.7 | 0.300 | 1030000 / 415 / 207000 / 208000 | 0 / 5 / 11 | 1.59e-13 | none | 173 | 0 |
| nx4_nu2_N10 | tight | `daqp` | 1/1 | 8.14 | 5.40 - 10.4 | 10.6 / 0.625 / 7.49 | 0.290 | 63400 / 6660 / 97000 / 104000 | 1 / 11 / 23 | 4.76e-15 | none | 0 | 0 |
| nx4_nu2_N10 | tight | `proxqp_dense` | 31/60 | 83.0 | 30.8 - 167 | 120 / 1.09 / 81.9 | 0.390 | 788000 / 10100 / 610000 / 620000 | 0 / 14 / 52 | 1.00e+0 | inaccurate 40, infeasible_claim 105 | 8 | 0 |
| nx4_nu2_N10 | tight | `osqp` | 1/1 | 65.6 | 45.1 - 66.6 | 55.1 / 0.270 / 65.3 | 0.290 | 352000 / 2190 / 703000 / 705000 | 25 / 75 / 125 | 5.12e-4 | none | 0 | 0 |
| nx4_nu2_N10 | tight | `argmin_dense_admm_polished` | 59/60 | 75.5 | 58.0 - 93.1 | 95.8 / 0.0400 / 75.5 | 0.340 | 776000 / 354 / 690000 / 690000 | 25 / 75 / 225 | 4.50e-1 | inaccurate 5 | 0 | 0 |
| nx4_nu2_N20 | nominal | `argmin_dense_admm` | 1/1 | 34.4 | 34.2 - 34.6 | 91.2 / 0.0500 / 34.3 | 0.510 | 821000 / 370 / 252000 / 252000 | 25 / 25 / 25 | 2.11e-4 | none | 0 | 0 |
| nx4_nu2_N20 | nominal | `argmin_dense_active_set` | 1/1 | 894 | 528 - 1280 | 442 / 0.0500 / 894 | 0.650 | 3470000 / 369 / 8660000 / 8660000 | 0 / 36.5 / 74 | 3.48e-5 | none | 380 | 0 |
| nx4_nu2_N20 | nominal | `qpoases` | 1/1 | 46.7 | 35.9 - 83.7 | 122 / 0.0500 / 46.7 | 0.520 | 1050000 / 409 / 501000 / 502000 | 0 / 8 / 19 | 3.93e-13 | none | 166 | 0 |
| nx4_nu2_N20 | nominal | `daqp` | 1/1 | 6.86 | 4.21 - 9.70 | 19.2 / 0.920 / 5.94 | 0.500 | 157000 / 10300 / 57100 / 67400 | 6 / 9 / 20 | 8.36e-15 | none | 0 | 0 |
| nx4_nu2_N20 | nominal | `proxqp_dense` | 11/15 | 78.7 | 67.5 - 92.3 | 172 / 1.94 / 76.7 | 0.600 | 1220000 / 18900 / 622000 / 641000 | 0 / 9 / 16 | 1.74e-2 | inaccurate 80 | 8 | 0 |
| nx4_nu2_N20 | nominal | `osqp` | 1/1 | 45.7 | 45.3 - 45.9 | 88.2 / 0.290 / 45.4 | 0.500 | 670000 / 2270 / 430000 / 432000 | 25 / 25 / 25 | 2.11e-4 | none | 0 | 0 |
| nx4_nu2_N20 | nominal | `argmin_dense_admm_polished` | 1/1 | 73.9 | 73.5 - 75.2 | 187 / 0.0500 / 73.9 | 0.590 | 1290000 / 370 / 720000 / 720000 | 25 / 25 / 25 | 2.64e-5 | none | 0 | 0 |
| nx4_nu2_N20 | tight | `argmin_dense_admm` | 1/1 | 120 | 80.9 - 200 | 144 / 0.0600 / 120 | 0.620 | 1390000 / 470 / 875000 / 876000 | 25 / 75 / 175 | 4.89e-4 | none | 0 | 0 |
| nx4_nu2_N20 | tight | `argmin_dense_active_set` | 1/1 | 1560 | 1410 - 1650 | 1700 / 0.0900 / 1560 | 0.800 | 16700000 / 469 / 15300000 / 15300000 | 0 / 40 / 40 | 2.97e-16 | none | 1723 | 210 |
| nx4_nu2_N20 | tight | `qpoases` | 1/1 | 72.6 | 47.5 - 85.9 | 501 / 0.0600 / 72.6 | 0.630 | 5820000 / 540 / 847000 / 848000 | 0 / 6 / 15 | 2.89e-13 | none | 229 | 0 |
| nx4_nu2_N20 | tight | `daqp` | 1/1 | 25.8 | 19.4 - 31.9 | 34.3 / 1.69 / 23.6 | 0.580 | 281000 / 17200 / 308000 / 325000 | 5 / 13 / 41 | 3.51e-15 | none | 0 | 0 |
| nx4_nu2_N20 | tight | `proxqp_dense` | 11/20 | 413 | 272 - 464 | 248 / 2.88 / 410 | 0.820 | 2060000 / 25900 / 3020000 / 3050000 | 0 / 24.5 / 66 | 1.00e+0 | inaccurate 25, infeasible_claim 110 | 8 | 0 |
| nx4_nu2_N20 | tight | `osqp` | 1/1 | 139 | 137 - 203 | 149 / 0.400 / 139 | 0.600 | 1190000 / 3550 / 1430000 / 1430000 | 25 / 50 / 175 | 5.50e-4 | none | 0 | 0 |
| nx4_nu2_N20 | tight | `argmin_dense_admm_polished` | 11/12 | 216 | 177 - 225 | 260 / 0.0600 / 216 | 0.740 | 2510000 / 470 / 1990000 / 2000000 | 25 / 75 / 175 | 6.17e-1 | inaccurate 25 | 0 | 0 |
| nx8_nu3_N20 | nominal | `argmin_dense_admm` | 1/1 | 61.0 | 60.8 - 61.3 | 160 / 0.0600 / 61.0 | 1.43 | 1650000 / 445 / 423000 / 424000 | 25 / 25 / 25 | 1.99e-4 | none | 0 | 0 |
| nx8_nu3_N20 | nominal | `argmin_dense_active_set` | 1/1 | 3240 | 2540 - 4190 | 1090 / 0.0800 / 3240 | 1.79 | 9570000 / 444 / 31100000 / 31100000 | 0 / 57.5 / 89 | 5.02e-16 | none | 488 | 0 |
| nx8_nu3_N20 | nominal | `qpoases` | 1/1 | 141 | 94.7 - 178 | 242 / 0.0700 / 141 | 1.42 | 2340000 / 484 / 1590000 / 1590000 | 0 / 14 / 23 | 2.39e-13 | none | 204 | 0 |
| nx8_nu3_N20 | nominal | `daqp` | 1/1 | 16.9 | 12.0 - 19.8 | 38.4 / 1.76 / 15.1 | 1.26 | 386000 / 18200 / 162000 / 180000 | 8 / 15.5 / 24 | 3.45e-5 | none | 0 | 0 |
| nx8_nu3_N20 | nominal | `proxqp_dense` | 4/5 | 156 | 135 - 182 | 257 / 3.74 / 152 | 1.67 | 2220000 / 35800 / 1240000 / 1280000 | 0 / 12 / 19 | 8.70e-3 | inaccurate 60 | 7 | 0 |
| nx8_nu3_N20 | nominal | `osqp` | 1/1 | 91.4 | 91.2 - 92.9 | 175 / 0.380 / 91.1 | 1.35 | 1390000 / 2990 / 840000 / 843000 | 25 / 25 / 25 | 1.99e-4 | none | 0 | 0 |
| nx8_nu3_N20 | nominal | `argmin_dense_admm_polished` | 1/1 | 157 | 157 - 158 | 350 / 0.0700 / 157 | 1.70 | 2760000 / 445 / 1540000 / 1540000 | 25 / 25 / 25 | 2.40e-5 | none | 0 | 0 |
| nx8_nu3_N20 | tight | `argmin_dense_admm` | 1/1 | 159 | 158 - 159 | 296 / 0.0800 / 159 | 1.68 | 2800000 / 595 / 1030000 / 1030000 | 25 / 50 / 75 | 4.92e-4 | none | 0 | 0 |
| nx8_nu3_N20 | tight | `argmin_dense_active_set` | 1/1 | 5830 | 5420 - 6090 | 6160 / 0.130 / 5830 | 2.11 | 59400000 / 594 / 56400000 / 56400000 | 0 / 60 / 60 | 3.91e-16 | none | 3808 | 265 |
| nx8_nu3_N20 | tight | `qpoases` | 1/1 | 181 | 148 - 213 | 1670 / 0.100 / 180 | 1.70 | 20800000 / 665 / 2250000 / 2260000 | 0 / 8 / 17 | 2.51e-13 | none | 257 | 0 |
| nx8_nu3_N20 | tight | `daqp` | 1/1 | 74.1 | 61.9 - 92.7 | 86.2 / 3.91 / 70.2 | 1.52 | 748000 / 32100 / 842000 / 874000 | 9 / 17 / 75 | 2.95e-5 | none | 0 | 0 |
| nx8_nu3_N20 | tight | `proxqp_dense` | 7/20 | 1110 | 696 - 1270 | 723 / 5.64 / 1100 | 2.20 | 6220000 / 50000 / 7970000 / 8020000 | 0 / 29.5 / 104 | 1.00e+0 | inaccurate 60, infeasible_claim 135 | 7 | 0 |
| nx8_nu3_N20 | tight | `osqp` | 1/1 | 274 | 269 - 282 | 316 / 0.570 / 274 | 1.65 | 2570000 / 4910 / 2730000 / 2740000 | 25 / 50 / 75 | 4.92e-4 | none | 0 | 0 |
| nx8_nu3_N20 | tight | `argmin_dense_admm_polished` | 14/15 | 403 | 399 - 478 | 560 / 0.0800 / 403 | 1.91 | 5500000 / 595 / 3740000 / 3740000 | 25 / 50 / 225 | 4.38e-1 | inaccurate 20 | 0 | 0 |
| nx8_nu3_N30 | nominal | `argmin_dense_admm` | 1/1 | 118 | 118 - 119 | 360 / 0.0900 / 118 | 3.40 | 3670000 / 598 / 773000 / 774000 | 25 / 25 / 25 | 7.33e-4 | none | 0 | 0 |
| nx8_nu3_N30 | nominal | `argmin_dense_active_set` | 1/1 | 8360 | 6500 - 10800 | 2710 / 0.130 / 8360 | 3.72 | 24200000 / 597 / 79800000 / 79800000 | 0 / 57.5 / 89 | 6.14e-16 | none | 488 | 0 |
| nx8_nu3_N30 | nominal | `qpoases` | 1/1 | 322 | 217 - 408 | 510 / 0.110 / 322 | 3.38 | 5140000 / 637 / 3660000 / 3660000 | 0 / 14 / 23 | 2.34e-13 | none | 204 | 0 |
| nx8_nu3_N30 | nominal | `daqp` | 1/1 | 30.6 | 20.8 - 36.9 | 88.4 / 3.83 / 26.7 | 2.99 | 1010000 / 33900 / 257000 / 291000 | 8 / 15.5 / 24 | 3.45e-5 | none | 0 | 0 |
| nx8_nu3_N30 | nominal | `proxqp_dense` | 4/5 | 264 | 232 - 310 | 553 / 8.27 / 254 | 3.68 | 4390000 / 78300 / 2020000 / 2100000 | 0 / 12 / 19 | 8.70e-3 | inaccurate 60 | 7 | 0 |
| nx8_nu3_N30 | nominal | `osqp` | 1/1 | 195 | 194 - 195 | 396 / 0.580 / 194 | 3.32 | 3230000 / 4150 / 1760000 / 1770000 | 25 / 25 / 25 | 7.33e-4 | none | 0 | 0 |
| nx8_nu3_N30 | nominal | `argmin_dense_admm_polished` | 1/1 | 360 | 359 - 361 | 831 / 0.100 / 359 | 3.54 | 6370000 / 598 / 3470000 / 3480000 | 25 / 25 / 25 | 1.33e-4 | none | 0 | 0 |
| nx8_nu3_N30 | tight | `argmin_dense_admm` | 1/1 | 305 | 304 - 306 | 640 / 0.120 / 305 | 3.73 | 6060000 / 796 / 1850000 / 1850000 | 25 / 50 / 75 | 3.98e-4 | none | 0 | 0 |
| nx8_nu3_N30 | tight | `argmin_dense_active_set` | 1/1 | 23500 | 22900 - 24100 | 24300 / 0.170 / 23500 | 4.05 | 225000000 / 795 / 218000000 / 218000000 | 0 / 90 / 90 | 3.70e-16 | none | 8416 | 295 |
| nx8_nu3_N30 | tight | `qpoases` | 1/1 | 431 | 341 - 492 | 5000 / 0.150 / 431 | 3.78 | 62400000 / 920 / 5450000 / 5450000 | 0 / 8 / 17 | 2.11e-13 | none | 257 | 0 |
| nx8_nu3_N30 | tight | `daqp` | 1/1 | 158 | 133 - 199 | 249 / 8.44 / 149 | 3.53 | 2140000 / 63800 / 1670000 / 1730000 | 9 / 17 / 105 | 2.95e-5 | none | 0 | 0 |
| nx8_nu3_N30 | tight | `proxqp_dense` | 7/20 | 2810 | 1930 - 3020 | 1500 / 12.4 / 2800 | 4.10 | 13500000 / 112000 / 20200000 / 20300000 | 0 / 34.5 / 94 | 1.00e+0 | inaccurate 55, infeasible_claim 140 | 8 | 0 |
| nx8_nu3_N30 | tight | `osqp` | 1/1 | 593 | 586 - 602 | 720 / 0.830 / 592 | 3.71 | 6000000 / 7000 / 5870000 / 5870000 | 25 / 50 / 75 | 4.91e-4 | none | 0 | 0 |
| nx8_nu3_N30 | tight | `argmin_dense_admm_polished` | 19/20 | 980 | 958 - 1110 | 1360 / 0.130 / 980 | 3.84 | 13100000 / 796 / 8880000 / 8880000 | 25 / 50 / 225 | 4.01e-1 | inaccurate 15 | 0 | 0 |
| nx12_nu4_N30 | nominal | `argmin_dense_admm` | 1/1 | 175 | 172 - 176 | 586 / 0.110 / 175 | 5.88 | 6100000 / 670 / 1060000 / 1060000 | 25 / 25 / 50 | 7.75e-4 | none | 0 | 0 |
| nx12_nu4_N30 | nominal | `argmin_dense_active_set` | 1/1 | 20300 | 18900 - 22200 | 8970 / 0.150 / 20300 | 6.09 | 84100000 / 669 / 195000000 / 195000000 | 0 / 74 / 93 | 5.69e-16 | none | 532 | 0 |
| nx12_nu4_N30 | nominal | `qpoases` | 1/1 | 735 | 654 - 803 | 1450 / 0.140 / 734 | 5.96 | 15200000 / 709 / 8130000 / 8130000 | 0 / 18.5 / 24 | 1.55e-13 | none | 217 | 0 |
| nx12_nu4_N30 | nominal | `daqp` | 1/1 | 57.4 | 52.3 - 65.6 | 172 / 6.66 / 50.3 | 5.68 | 2060000 / 53900 / 457000 / 511000 | 13 / 20 / 26 | 4.81e-4 | none | 0 | 0 |
| nx12_nu4_N30 | nominal | `proxqp_dense` | 4/5 | 423 | 388 - 481 | 765 / 13.3 / 410 | 6.18 | 7650000 / 121000 / 3170000 / 3290000 | 0 / 14 / 20 | 5.94e-3 | inaccurate 60 | 7 | 0 |
| nx12_nu4_N30 | nominal | `osqp` | 1/1 | 334 | 334 - 335 | 726 / 0.780 / 334 | 5.93 | 5970000 / 5180 / 2990000 / 3000000 | 25 / 25 / 50 | 7.75e-4 | none | 0 | 0 |
| nx12_nu4_N30 | nominal | `argmin_dense_admm_polished` | 1/1 | 663 | 661 - 667 | 1230 / 0.120 / 663 | 5.90 | 11400000 / 670 / 6400000 / 6400000 | 25 / 25 / 50 | 3.52e-4 | none | 0 | 0 |
| nx12_nu4_N30 | tight | `argmin_dense_admm` | 1/1 | 512 | 507 - 516 | 1740 / 0.150 / 512 | 6.27 | 10400000 / 970 / 2740000 / 2740000 | 25 / 50 / 75 | 4.92e-4 | none | 0 | 0 |
| nx12_nu4_N30 | tight | `argmin_dense_active_set` | 1/1 | 62000 | 59900 - 62900 | 64500 / 0.190 / 62000 | 6.62 | 562000000 / 969 / 552000000 / 552000000 | 0 / 120 / 120 | 4.01e-16 | none | 14851 | 295 |
| nx12_nu4_N30 | tight | `qpoases` | 1/1 | 920 | 826 - 1060 | 11900 / 0.200 / 920 | 6.30 | 145000000 / 1040 / 11200000 / 11200000 | 0 / 10 / 19 | 1.86e-13 | none | 294 | 0 |
| nx12_nu4_N30 | tight | `daqp` | 1/1 | 345 | 296 - 417 | 540 / 14.8 / 330 | 6.20 | 4550000 / 103000 / 3580000 / 3680000 | 13 / 21 / 137 | 5.43e-5 | none | 0 | 0 |
| nx12_nu4_N30 | tight | `proxqp_dense` | 1/4 | 5640 | 4190 - 6190 | 2940 / 19.5 / 5620 | 6.69 | 24700000 / 173000 / 40400000 / 40500000 | 0 / 41.5 / 130 | 1.00e+0 | inaccurate 57, infeasible_claim 168 | 8 | 0 |
| nx12_nu4_N30 | tight | `osqp` | 1/1 | 999 | 987 - 1010 | 1460 / 1.03 / 998 | 6.35 | 10200000 / 9020 / 9670000 / 9680000 | 25 / 50 / 75 | 4.93e-4 | none | 0 | 0 |
| nx12_nu4_N30 | tight | `argmin_dense_admm_polished` | 11/12 | 2310 | 1840 - 2330 | 3980 / 0.160 / 2310 | 6.29 | 24500000 / 970 / 19600000 / 19600000 | 25 / 100 / 250 | 3.80e-1 | inaccurate 25 | 4 | 0 |
| nx4_nu2_N50 | nominal | `argmin_dense_admm` | 1/1 | 132 | 132 - 134 | 418 / 0.0900 / 132 | 2.77 | 4250000 / 595 / 823000 / 824000 | 25 / 25 / 25 | 1.33e-4 | none | 0 | 0 |
| nx4_nu2_N50 | nominal | `argmin_dense_active_set` | 1/1 | 7220 | 4110 - 10300 | 2960 / 0.120 / 7220 | 2.98 | 28600000 / 594 / 70100000 / 70100000 | 0 / 36.5 / 74 | 3.48e-5 | none | 422 | 0 |
| nx4_nu2_N50 | nominal | `qpoases` | 1/1 | 267 | 209 - 440 | 596 / 0.100 / 267 | 2.82 | 5970000 / 634 / 2860000 / 2860000 | 0 / 8 / 19 | 3.83e-13 | none | 166 | 0 |
| nx4_nu2_N50 | nominal | `daqp` | 1/1 | 24.6 | 12.7 - 40.3 | 114 / 4.66 / 20.0 | 2.46 | 1310000 / 40000 / 179000 / 219000 | 6 / 9 / 20 | 3.60e-15 | none | 0 | 0 |
| nx4_nu2_N50 | nominal | `proxqp_dense` | 3/4 | 223 | 195 - 268 | 471 / 9.44 / 213 | 2.96 | 5020000 / 87100 / 1700000 / 1790000 | 0 / 9.5 / 16 | 1.74e-2 | inaccurate 75 | 8 | 0 |
| nx4_nu2_N50 | nominal | `osqp` | 1/1 | 245 | 244 - 245 | 505 / 0.590 / 244 | 2.70 | 4140000 / 4430 / 2190000 / 2190000 | 25 / 25 / 25 | 1.33e-4 | none | 0 | 0 |
| nx4_nu2_N50 | nominal | `argmin_dense_admm_polished` | 1/1 | 444 | 441 - 447 | 744 / 0.100 / 444 | 2.85 | 7680000 / 595 / 4250000 / 4250000 | 25 / 25 / 25 | 2.03e-5 | none | 0 | 0 |
| nx4_nu2_N50 | tight | `argmin_dense_admm` | 1/1 | 364 | 355 - 701 | 592 / 0.130 / 364 | 3.00 | 6200000 / 845 / 2110000 / 2110000 | 25 / 50 / 125 | 4.96e-4 | none | 1 | 0 |
| nx4_nu2_N50 | tight | `argmin_dense_active_set` | 1/1 | 33000 | 31600 - 33600 | 33700 / 0.170 / 33000 | 3.29 | 306000000 / 844 / 298000000 / 298000000 | 0 / 100 / 100 | 2.89e-16 | none | 10376 | 225 |
| nx4_nu2_N50 | tight | `qpoases` | 1/1 | 417 | 280 - 523 | 6060 / 0.160 / 416 | 3.02 | 74700000 / 915 / 5120000 / 5120000 | 0 / 6 / 15 | 2.49e-13 | none | 229 | 0 |
| nx4_nu2_N50 | tight | `daqp` | 1/1 | 141 | 107 - 178 | 326 / 10.3 / 131 | 2.90 | 2790000 / 75200 / 1430000 / 1510000 | 5 / 13 / 101 | 2.49e-15 | none | 0 | 0 |
| nx4_nu2_N50 | tight | `proxqp_dense` | 32/75 | 2560 | 1580 - 3470 | 1260 / 13.9 / 2540 | 3.30 | 12500000 / 124000 / 19900000 / 20000000 | 0 / 30 / 101 | 1.00e+0 | inaccurate 34, infeasible_claim 138 | 8 | 0 |
| nx4_nu2_N50 | tight | `osqp` | 1/1 | 789 | 771 - 1540 | 912 / 0.920 / 788 | 3.05 | 7880000 / 7670 / 7720000 / 7720000 | 25 / 50 / 125 | 4.90e-4 | none | 0 | 0 |
| nx4_nu2_N50 | tight | `argmin_dense_admm_polished` | 13/15 | 1360 | 1190 - 1530 | 1460 / 0.140 / 1360 | 3.04 | 15100000 / 845 / 12100000 / 12100000 | 25 / 62.5 / 175 | 5.39e-1 | inaccurate 40 | 2 | 0 |
| nx4_nu2_N100 | nominal | `argmin_dense_admm` | 1/1 | 431 | 429 - 439 | 2510 / 0.170 / 431 | 9.70 | 17800000 / 1000 / 2400000 / 2400000 | 25 / 25 / 25 | 1.32e-4 | none | 0 | 0 |
| nx4_nu2_N100 | nominal | `argmin_dense_active_set` | 1/1 | 41300 | 23200 - 59300 | 18700 / 0.215 / 41300 | 9.93 | 160000000 / 978 / 410000000 / 410000000 | 0 / 36.5 / 74 | 3.48e-5 | none | 758 | 0 |
| nx4_nu2_N100 | nominal | `qpoases` | 1/1 | 1230 | 1010 - 1930 | 2890 / 0.190 / 1230 | 9.64 | 25200000 / 1040 / 11300000 / 11300000 | 0 / 8 / 19 | 3.72e-13 | none | 166 | 0 |
| nx4_nu2_N100 | nominal | `daqp` | 1/1 | 100 | 47.2 - 156 | 653 / 19.0 / 78.8 | 9.61 | 7800000 / 130000 / 569000 / 699000 | 6 / 9 / 20 | 4.66e-15 | none | 0 | 0 |
| nx4_nu2_N100 | nominal | `proxqp_dense` | 3/4 | 640 | 556 - 762 | 2660 / 35.0 / 605 | 9.90 | 17400000 / 319000 / 4550000 / 4870000 | 0 / 9.5 / 16 | 1.74e-2 | inaccurate 75 | 8 | 0 |
| nx4_nu2_N100 | nominal | `osqp` | 1/1 | 928 | 925 - 937 | 2760 / 1.02 / 927 | 9.79 | 20300000 / 8090 / 8130000 / 8140000 | 25 / 25 / 25 | 1.32e-4 | none | 0 | 0 |
| nx4_nu2_N100 | nominal | `argmin_dense_admm_polished` | 1/1 | 2130 | 2120 - 2130 | 5620 / 0.180 / 2120 | 9.86 | 36000000 / 998 / 20600000 / 20600000 | 25 / 25 / 25 | 2.01e-5 | none | 2 | 0 |
| nx4_nu2_N100 | tight | `argmin_dense_admm` | 1/1 | 1380 | 1350 - 2690 | 2840 / 0.230 / 1380 | 10.2 | 27400000 / 1470 / 6680000 / 6680000 | 25 / 50 / 150 | 4.73e-4 | none | 2 | 0 |
| nx4_nu2_N100 | tight | `argmin_dense_active_set` | 1/1 | 372000 | 364000 - 377000 | 378000 / 0.280 / 372000 | 10.8 | 3300000000 / 1500 / 3250000000 / 3250000000 | 0 / 200 / 200 | 5.43e-5 | none | 41473 | 235 |
| nx4_nu2_N100 | tight | `qpoases` | 1/1 | 1910 | 1310 - 2390 | 49100 / 0.270 / 1910 | 10.0 | 564000000 / 1540 / 21500000 / 21500000 | 0 / 6 / 15 | 2.40e-13 | none | 229 | 0 |
| nx4_nu2_N100 | tight | `daqp` | 1/1 | 596 | 448 - 754 | 2320 / 46.6 / 550 | 10.0 | 18600000 / 260000 / 5140000 / 5400000 | 5 / 13 / 201 | 4.53e-15 | none | 0 | 0 |
| nx4_nu2_N100 | tight | `proxqp_dense` | 13/30 | 12100 | 7120 - 14900 | 6350 / 50.4 / 12000 | 10.5 | 51300000 / 451000 / 91900000 / 92400000 | 0 / 34 / 201 | 1.00e+0 | inaccurate 35, infeasible_claim 135 | 38 | 0 |
| nx4_nu2_N100 | tight | `osqp` | 1/1 | 2960 | 2880 - 5820 | 5250 / 1.52 / 2960 | 10.3 | 34100000 / 14500 / 28000000 / 28000000 | 25 / 50 / 150 | 4.81e-4 | none | 0 | 0 |
| nx4_nu2_N100 | tight | `argmin_dense_admm_polished` | 19/20 | 6820 | 6150 - 7500 | 10900 / 0.260 / 6820 | 10.3 | 79800000 / 1480 / 62300000 / 62300000 | 25 / 75 / 200 | 4.40e-1 | inaccurate 15 | 13 | 0 |
| nx8_nu3_N50 | nominal | `argmin_dense_admm` | 1/1 | 264 | 261 - 271 | 987 / 0.150 / 264 | 8.10 | 9980000 / 852 / 1670000 / 1670000 | 25 / 25 / 25 | 7.46e-4 | none | 0 | 0 |
| nx8_nu3_N50 | nominal | `argmin_dense_active_set` | 1/1 | 29000 | 22500 - 37600 | 8600 / 0.190 / 29000 | 8.41 | 82700000 / 822 / 281000000 / 281000000 | 0 / 57.5 / 89 | 2.13e-16 | none | 726 | 0 |
| nx8_nu3_N50 | nominal | `qpoases` | 1/1 | 973 | 672 - 1210 | 1510 / 0.160 / 973 | 8.13 | 14600000 / 862 / 10600000 / 10600000 | 0 / 14 / 23 | 2.39e-13 | none | 204 | 0 |
| nx8_nu3_N50 | nominal | `daqp` | 1/1 | 77.4 | 51.4 - 95.6 | 304 / 10.3 / 66.9 | 8.06 | 3660000 / 78600 / 539000 / 618000 | 8 / 15.5 / 24 | 3.45e-5 | none | 0 | 0 |
| nx8_nu3_N50 | nominal | `proxqp_dense` | 4/5 | 543 | 472 - 632 | 1030 / 21.6 / 520 | 8.38 | 10600000 / 196000 / 3950000 / 4150000 | 0 / 12 / 19 | 8.70e-3 | inaccurate 60 | 7 | 0 |
| nx8_nu3_N50 | nominal | `osqp` | 1/1 | 529 | 529 - 530 | 1260 / 0.880 / 528 | 8.26 | 10300000 / 6340 / 4690000 / 4700000 | 25 / 25 / 25 | 7.46e-4 | none | 0 | 0 |
| nx8_nu3_N50 | nominal | `argmin_dense_admm_polished` | 1/1 | 1090 | 1090 - 1100 | 1830 / 0.150 / 1090 | 8.25 | 18900000 / 823 / 10600000 / 10600000 | 25 / 25 / 25 | 1.65e-4 | none | 0 | 0 |
| nx8_nu3_N50 | tight | `argmin_dense_admm` | 1/1 | 785 | 779 - 789 | 1860 / 0.180 / 784 | 8.61 | 16900000 / 1200 / 4260000 / 4260000 | 25 / 50 / 50 | 3.24e-4 | none | 0 | 0 |
| nx8_nu3_N50 | tight | `argmin_dense_active_set` | 1/1 | 136000 | 134000 - 138000 | 139000 / 0.250 / 136000 | 9.23 | 1230000000 / 1220 / 1210000000 / 1210000000 | 0 / 150 / 150 | 3.70e-16 | none | 23335 | 295 |
| nx8_nu3_N50 | tight | `qpoases` | 1/1 | 1380 | 1110 - 1550 | 22000 / 0.230 / 1380 | 8.62 | 262000000 / 1300 / 16000000 / 16000000 | 0 / 8 / 17 | 2.07e-13 | none | 257 | 0 |
| nx8_nu3_N50 | tight | `daqp` | 1/1 | 428 | 369 - 547 | 1020 / 24.3 / 403 | 8.55 | 8420000 / 155000 / 4120000 / 4280000 | 9 / 17 / 165 | 2.95e-5 | none | 0 | 0 |
| nx8_nu3_N50 | tight | `proxqp_dense` | 47/150 | 8850 | 5440 - 10800 | 3900 / 31.1 / 8820 | 9.14 | 33900000 / 277000 / 61000000 / 61300000 | 0 / 29 / 164 | 1.00e+0 | inaccurate 48, infeasible_claim 158 | 16 | 0 |
| nx8_nu3_N50 | tight | `osqp` | 1/1 | 1630 | 1620 - 1660 | 2060 / 1.23 / 1630 | 8.72 | 17500000 / 11100 / 15700000 / 15700000 | 25 / 50 / 50 | 4.93e-4 | none | 0 | 0 |
| nx8_nu3_N50 | tight | `argmin_dense_admm_polished` | 29/30 | 3900 | 3150 - 3940 | 4270 / 0.200 / 3900 | 8.86 | 42000000 / 1180 / 33500000 / 33600000 | 25 / 100 / 175 | 3.97e-1 | inaccurate 10 | 8 | 0 |
| nx8_nu3_N100 | nominal | `argmin_dense_admm` | 1/1 | 965 | 962 - 970 | 4730 / 0.260 / 965 | 29.2 | 44400000 / 1380 / 4710000 / 4710000 | 25 / 25 / 25 | 8.29e-4 | none | 0 | 0 |
| nx8_nu3_N100 | nominal | `argmin_dense_active_set` | 1/1 | 185000 | 141000 - 238000 | 53500 / 0.270 / 185000 | 29.7 | 515000000 / 1350 / 1860000000 / 1860000000 | 0 / 57.5 / 89 | 2.87e-16 | none | 992 | 0 |
| nx8_nu3_N100 | nominal | `qpoases` | 1/1 | 4550 | 3190 - 5570 | 8090 / 0.270 / 4550 | 29.0 | 64300000 / 1410 / 43800000 / 43800000 | 0 / 14 / 23 | 2.46e-13 | none | 204 | 0 |
| nx8_nu3_N100 | nominal | `daqp` | 1/1 | 350 | 228 - 428 | 2060 / 47.1 / 303 | 29.1 | 23500000 / 269000 / 1780000 / 2050000 | 8 / 15.5 / 24 | 3.45e-5 | none | 0 | 0 |
| nx8_nu3_N100 | nominal | `proxqp_dense` | 49/60 | 1580 | 1390 - 1840 | 6200 / 79.8 / 1500 | 29.4 | 37900000 / 694000 / 10800000 / 11500000 | 0 / 12 / 19 | 8.70e-3 | inaccurate 55 | 25 | 0 |
| nx8_nu3_N100 | nominal | `osqp` | 1/1 | 2070 | 2060 - 2080 | 6640 / 1.34 / 2070 | 29.3 | 53800000 / 11700 / 17900000 / 17900000 | 25 / 25 / 25 | 8.29e-4 | none | 0 | 0 |
| nx8_nu3_N100 | nominal | `argmin_dense_admm_polished` | 1/1 | 5760 | 5750 - 5780 | 10600 / 0.280 / 5760 | 29.5 | 96400000 / 1370 / 56700000 / 56700000 | 25 / 25 / 25 | 1.83e-4 | none | 9 | 0 |
| nx8_nu3_N100 | tight | `argmin_dense_admm` | 1/1 | 2960 | 2950 - 2970 | 8830 / 0.330 / 2960 | 29.9 | 76700000 / 2160 / 13700000 / 13700000 | 25 / 50 / 50 | 4.05e-4 | none | 0 | 0 |
| nx8_nu3_N100 | tight | `argmin_dense_active_set` | 1/1 | 1700000 | 1680000 - 1710000 | 1720000 / 0.410 / 1700000 | 44.2 | 14400000000 / 2140 / 14300000000 / 14300000000 | 0 / 300 / 300 | 4.09e-16 | none | 92813 | 295 |
| nx8_nu3_N100 | tight | `qpoases` | 1/1 | 6020 | 4790 - 6940 | 172000 / 0.370 / 6020 | 30.0 | 1940000000 / 2190 / 66300000 / 66300000 | 0 / 8 / 17 | 2.06e-13 | none | 257 | 0 |
| nx8_nu3_N100 | tight | `daqp` | 1/1 | 1870 | 1620 - 2400 | 8130 / 115 / 1750 | 29.9 | 58700000 / 555000 / 14800000 / 15400000 | 9 / 17 / 315 | 2.95e-5 | none | 0 | 0 |
| nx8_nu3_N100 | tight | `proxqp_dense` | 41/150 | 33200 | 26100 - 46200 | 18000 / 113 / 33100 | 30.6 | 142000000 / 981000 / 262000000 / 263000000 | 0 / 39 / 1358310 | 1.00e+0 | inaccurate 35, max_iter 3, infeasible_claim 180 | 6714 | 0 |
| nx8_nu3_N100 | tight | `osqp` | 1/1 | 6290 | 6230 - 6410 | 12600 / 2.09 / 6280 | 30.0 | 84800000 / 21300 / 58200000 / 58200000 | 25 / 50 / 50 | 4.02e-4 | none | 0 | 0 |
| nx8_nu3_N100 | tight | `argmin_dense_admm_polished` | 29/30 | 19900 | 17100 - 20000 | 26700 / 0.340 / 19900 | 30.2 | 234000000 / 2120 / 184000000 / 184000000 | 25 / 100 / 175 | 3.97e-1 | inaccurate 10 | 21 | 0 |

### dense cohort, closed_loop regime, high tier

The dense cohort in the closed_loop regime, judged at the high tier's threshold of 1e-6.

| Cell | Variant | Arm | Pass rate | Median tick (us) | Tick p25 - p75 (us) | Median setup / update / solve (us) | Median condense (us) | Median instructions setup / update / solve / tick | Iterations min / median / max | Worst KKT residual | Failures | Allocations max | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nx2_nu1_N10 | nominal | `argmin_dense_admm` | 1/1 | 33.6 | 23.4 - 51.9 | 48.6 / 0.0300 / 33.6 | 0.160 | 291000 / 372 / 300000 / 301000 | 25 / 75 / 175 | 2.15e-16 | none | 0 | 0 |
| nx2_nu1_N10 | nominal | `argmin_dense_active_set` | 1/1 | 34.3 | 19.7 - 64.7 | 49.5 / 0.0400 / 34.2 | 0.170 | 308000 / 371 / 317000 / 318000 | 0 / 10 / 34 | 2.36e-16 | none | 178 | 0 |
| nx2_nu1_N10 | nominal | `qpoases` | 1/1 | 14.2 | 7.25 - 19.6 | 15.4 / 0.0400 / 14.2 | 0.160 | 44500 / 408 / 126000 / 126000 | 0 / 5 / 9 | 4.38e-16 | none | 154 | 0 |
| nx2_nu1_N10 | nominal | `daqp` | 1/1 | 1.17 | 0.990 - 1.80 | 3.98 / 0.220 / 0.930 | 0.150 | 13000 / 2320 / 7040 / 9350 | 1 / 6 / 10 | 2.89e-15 | none | 0 | 0 |
| nx2_nu1_N10 | nominal | `proxqp_dense` | 4/5 | 19.3 | 11.0 - 23.6 | 63.0 / 0.610 / 18.6 | 0.170 | 387000 / 5240 / 153000 / 159000 | 0 / 7 / 13 | 6.56e-2 | inaccurate 60 | 7 | 0 |
| nx2_nu1_N10 | nominal | `osqp` | 1/1 | 27.4 | 21.4 - 37.8 | 19.0 / 0.190 / 27.2 | 0.160 | 99000 / 1340 / 245000 / 246000 | 50 / 75 / 175 | 6.90e-8 | none | 60 | 0 |
| nx2_nu1_N10 | tight | `argmin_dense_admm` | 1/1 | 45.7 | 44.9 - 63.3 | 71.5 / 0.0300 / 45.6 | 0.200 | 408000 / 387 / 404000 / 404000 | 25 / 100 / 375 | 1.50e-16 | none | 0 | 0 |
| nx2_nu1_N10 | tight | `argmin_dense_active_set` | 1/1 | 39.4 | 15.3 - 45.6 | 68.4 / 0.0400 / 39.4 | 0.210 | 455000 / 386 / 377000 / 377000 | 0 / 10 / 10 | 4.05e-16 | none | 132 | 120 |
| nx2_nu1_N10 | tight | `qpoases` | 1/1 | 10.1 | 2.36 - 16.4 | 106 / 0.0400 / 10.0 | 0.200 | 795000 / 494 / 95100 / 95600 | 0 / 2 / 7 | 1.97e-16 | none | 159 | 0 |
| nx2_nu1_N10 | tight | `daqp` | 1/1 | 2.28 | 0.945 - 3.42 | 4.76 / 0.290 / 2.00 | 0.190 | 18900 / 3200 / 20200 / 23500 | 1 / 5 / 15 | 3.19e-15 | none | 0 | 0 |
| nx2_nu1_N10 | tight | `proxqp_dense` | 13/20 | 27.6 | 4.46 - 40.1 | 57.9 / 0.730 / 26.8 | 0.220 | 324000 / 6250 / 202000 / 208000 | 0 / 6.5 / 28 | 1.00e+0 | inaccurate 15, infeasible_claim 90 | 7 | 0 |
| nx2_nu1_N10 | tight | `osqp` | 1/1 | 48.9 | 42.3 - 58.4 | 26.1 / 0.230 / 48.6 | 0.200 | 140000 / 1630 / 493000 / 495000 | 50 / 100 / 350 | 3.50e-16 | none | 60 | 0 |
| nx4_nu2_N10 | nominal | `argmin_dense_admm` | 1/1 | 63.7 | 60.2 - 76.0 | 109 / 0.0300 / 63.7 | 0.240 | 809000 / 295 / 515000 / 516000 | 25 / 75 / 125 | 2.29e-15 | none | 0 | 0 |
| nx4_nu2_N10 | nominal | `argmin_dense_active_set` | 1/1 | 211 | 105 - 298 | 127 / 0.0400 / 211 | 0.280 | 976000 / 294 / 2040000 / 2040000 | 0 / 29.5 / 50 | 2.03e-16 | none | 294 | 0 |
| nx4_nu2_N10 | nominal | `qpoases` | 1/1 | 31.7 | 24.0 - 60.5 | 49.8 / 0.0400 / 31.6 | 0.250 | 351000 / 334 / 330000 / 331000 | 0 / 8 / 17 | 2.57e-16 | none | 303 | 0 |
| nx4_nu2_N10 | nominal | `daqp` | 1/1 | 4.18 | 3.02 - 5.04 | 7.86 / 0.400 / 3.78 | 0.220 | 40300 / 4380 / 34800 / 39200 | 5 / 9 / 18 | 5.25e-15 | none | 0 | 0 |
| nx4_nu2_N10 | nominal | `proxqp_dense` | 13/15 | 47.0 | 41.6 - 53.7 | 89.8 / 0.830 / 46.2 | 0.260 | 614000 / 7680 / 361000 / 368000 | 0 / 10 / 16 | 1.40e-2 | inaccurate 40 | 7 | 0 |
| nx4_nu2_N10 | nominal | `osqp` | 1/1 | 74.0 | 71.7 - 85.9 | 34.9 / 0.230 / 73.8 | 0.240 | 221000 / 1550 / 719000 / 721000 | 75 / 100 / 200 | 2.79e-16 | none | 60 | 0 |
| nx4_nu2_N10 | tight | `argmin_dense_admm` | 1/1 | 128 | 110 - 146 | 113 / 0.0400 / 128 | 0.330 | 914000 / 345 / 1110000 / 1110000 | 25 / 150 / 350 | 1.97e-16 | none | 0 | 0 |
| nx4_nu2_N10 | tight | `argmin_dense_active_set` | 1/1 | 202 | 112 - 233 | 269 / 0.0400 / 202 | 0.370 | 2380000 / 344 / 1990000 / 1990000 | 0 / 20 / 20 | 5.54e-16 | none | 462 | 165 |
| nx4_nu2_N10 | tight | `qpoases` | 1/1 | 34.8 | 22.2 - 49.0 | 374 / 0.0400 / 34.8 | 0.310 | 4030000 / 420 / 399000 / 400000 | 0 / 5 / 11 | 2.84e-16 | none | 239 | 0 |
| nx4_nu2_N10 | tight | `daqp` | 1/1 | 8.18 | 5.43 - 10.5 | 10.6 / 0.655 / 7.54 | 0.280 | 63300 / 6660 / 97100 / 104000 | 1 / 11 / 23 | 4.76e-15 | none | 0 | 0 |
| nx4_nu2_N10 | tight | `proxqp_dense` | 31/60 | 104 | 30.1 - 173 | 134 / 1.10 / 102 | 0.390 | 918000 / 10100 / 758000 / 768000 | 0 / 15 / 52 | 1.00e+0 | inaccurate 17, infeasible_claim 128 | 8 | 0 |
| nx4_nu2_N10 | tight | `osqp` | 1/1 | 167 | 136 - 195 | 55.0 / 0.300 / 167 | 0.320 | 352000 / 2200 / 1620000 / 1620000 | 75 / 150 / 275 | 4.69e-16 | none | 60 | 0 |
| nx4_nu2_N20 | nominal | `argmin_dense_admm` | 1/1 | 141 | 141 - 141 | 255 / 0.0500 / 141 | 0.600 | 1780000 / 370 / 1210000 / 1210000 | 25 / 75 / 100 | 1.92e-15 | none | 0 | 0 |
| nx4_nu2_N20 | nominal | `argmin_dense_active_set` | 1/1 | 894 | 527 - 1280 | 444 / 0.0500 / 893 | 0.650 | 3470000 / 369 / 8660000 / 8660000 | 0 / 36.5 / 74 | 4.44e-16 | none | 380 | 0 |
| nx4_nu2_N20 | nominal | `qpoases` | 1/1 | 125 | 97.7 - 241 | 325 / 0.0500 / 125 | 0.540 | 3530000 / 409 / 1430000 / 1430000 | 0 / 8 / 19 | 5.57e-16 | none | 316 | 0 |
| nx4_nu2_N20 | nominal | `daqp` | 1/1 | 6.82 | 4.21 - 9.61 | 19.3 / 0.920 / 5.83 | 0.500 | 157000 / 10300 / 57100 / 67400 | 6 / 9 / 20 | 8.36e-15 | none | 0 | 0 |
| nx4_nu2_N20 | nominal | `proxqp_dense` | 53/60 | 83.6 | 74.1 - 94.6 | 182 / 1.94 / 81.7 | 0.600 | 1330000 / 18900 / 664000 / 683000 | 0 / 11 / 16 | 2.34e-2 | inaccurate 35 | 7 | 0 |
| nx4_nu2_N20 | nominal | `osqp` | 1/1 | 176 | 174 - 179 | 88.2 / 0.310 / 175 | 0.560 | 670000 / 2270 / 1650000 / 1650000 | 50 / 75 / 100 | 3.45e-16 | none | 60 | 0 |
| nx4_nu2_N20 | tight | `argmin_dense_admm` | 1/1 | 363 | 258 - 416 | 299 / 0.0600 / 363 | 0.740 | 2800000 / 470 / 3080000 / 3080000 | 25 / 162.5 / 375 | 4.20e-16 | none | 0 | 0 |
| nx4_nu2_N20 | tight | `argmin_dense_active_set` | 1/1 | 1550 | 1400 - 1640 | 1700 / 0.0900 / 1550 | 0.800 | 16700000 / 469 / 15300000 / 15300000 | 0 / 40 / 40 | 2.97e-16 | none | 1723 | 210 |
| nx4_nu2_N20 | tight | `qpoases` | 1/1 | 147 | 93.8 - 175 | 2310 / 0.0600 / 147 | 0.650 | 30700000 / 540 / 1930000 / 1930000 | 0 / 6 / 15 | 4.58e-16 | none | 319 | 0 |
| nx4_nu2_N20 | tight | `daqp` | 1/1 | 25.3 | 19.5 - 31.7 | 34.3 / 1.72 / 23.6 | 0.580 | 281000 / 17200 / 308000 / 325000 | 5 / 13 / 41 | 3.51e-15 | none | 0 | 0 |
| nx4_nu2_N20 | tight | `proxqp_dense` | 7/15 | 449 | 341 - 530 | 283 / 2.90 / 446 | 0.810 | 2310000 / 25900 / 3350000 / 3370000 | 0 / 29.5 / 65 | 1.00e+0 | inaccurate 15, infeasible_claim 145 | 8 | 0 |
| nx4_nu2_N20 | tight | `osqp` | 1/1 | 546 | 430 - 690 | 150 / 0.440 / 545 | 0.690 | 1190000 / 3550 / 5590000 / 5600000 | 75 / 175 / 425 | 1.22e-15 | none | 60 | 0 |
| nx8_nu3_N20 | nominal | `argmin_dense_admm` | 1/1 | 275 | 275 - 276 | 429 / 0.0700 / 275 | 1.73 | 3590000 / 445 / 2370000 / 2370000 | 25 / 75 / 75 | 1.63e-15 | none | 0 | 0 |
| nx8_nu3_N20 | nominal | `argmin_dense_active_set` | 1/1 | 3240 | 2550 - 4190 | 1060 / 0.0800 / 3240 | 1.78 | 9570000 / 444 / 31100000 / 31100000 | 0 / 57.5 / 89 | 5.02e-16 | none | 488 | 0 |
| nx8_nu3_N20 | nominal | `qpoases` | 1/1 | 476 | 311 - 612 | 1200 / 0.0800 / 476 | 1.59 | 13900000 / 484 / 5300000 / 5300000 | 0 / 14 / 23 | 4.94e-16 | none | 395 | 0 |
| nx8_nu3_N20 | nominal | `daqp` | 1/1 | 16.9 | 11.9 - 19.8 | 38.5 / 1.77 / 15.1 | 1.28 | 386000 / 18200 / 163000 / 181000 | 8 / 15.5 / 24 | 5.43e-15 | none | 0 | 0 |
| nx8_nu3_N20 | nominal | `proxqp_dense` | 59/60 | 172 | 149 - 195 | 259 / 3.75 / 168 | 1.68 | 2390000 / 35800 / 1370000 / 1410000 | 0 / 14 / 22 | 4.79e-3 | inaccurate 5 | 7 | 0 |
| nx8_nu3_N20 | nominal | `osqp` | 1/1 | 369 | 366 - 374 | 174 / 0.410 / 369 | 1.56 | 1390000 / 2990 / 3340000 / 3350000 | 50 / 75 / 75 | 3.83e-16 | none | 60 | 0 |
| nx8_nu3_N20 | tight | `argmin_dense_admm` | 1/1 | 714 | 638 - 795 | 876 / 0.0800 / 714 | 1.93 | 7540000 / 595 / 5770000 / 5770000 | 25 / 150 / 575 | 2.21e-16 | none | 0 | 0 |
| nx8_nu3_N20 | tight | `argmin_dense_active_set` | 1/1 | 5820 | 5400 - 6070 | 6140 / 0.130 / 5820 | 2.10 | 59200000 / 594 / 56400000 / 56400000 | 0 / 60 / 60 | 3.91e-16 | none | 3808 | 265 |
| nx8_nu3_N20 | tight | `qpoases` | 1/1 | 390 | 319 - 464 | 8620 / 0.110 / 390 | 1.73 | 117000000 / 665 / 5460000 / 5460000 | 0 / 8 / 17 | 4.17e-16 | none | 359 | 0 |
| nx8_nu3_N20 | tight | `daqp` | 1/1 | 73.1 | 61.6 - 92.8 | 86.3 / 3.86 / 69.2 | 1.52 | 748000 / 32100 / 845000 / 877000 | 9 / 17 / 75 | 4.87e-15 | none | 0 | 0 |
| nx8_nu3_N20 | tight | `proxqp_dense` | 11/30 | 1350 | 1040 - 1700 | 765 / 5.67 / 1350 | 2.22 | 6490000 / 50000 / 9880000 / 9930000 | 0 / 39.5 / 99 | 1.00e+0 | inaccurate 30, infeasible_claim 160 | 8 | 0 |
| nx8_nu3_N20 | tight | `osqp` | 1/1 | 983 | 866 - 1050 | 321 / 0.640 / 982 | 1.91 | 2570000 / 4920 / 9730000 / 9730000 | 100 / 150 / 450 | 2.24e-15 | none | 60 | 0 |
| nx8_nu3_N30 | nominal | `argmin_dense_admm` | 1/1 | 700 | 590 - 705 | 999 / 0.0900 / 700 | 3.56 | 8620000 / 598 / 5720000 / 5720000 | 25 / 100 / 100 | 3.55e-7 | none | 0 | 0 |
| nx8_nu3_N30 | nominal | `argmin_dense_active_set` | 1/1 | 8360 | 6490 - 10800 | 2570 / 0.120 / 8360 | 3.72 | 24200000 / 597 / 79800000 / 79800000 | 0 / 57.5 / 89 | 4.77e-16 | none | 488 | 0 |
| nx8_nu3_N30 | nominal | `qpoases` | 1/1 | 1440 | 956 - 1850 | 5190 / 0.120 / 1440 | 3.52 | 58500000 / 637 / 14100000 / 14100000 | 0 / 14 / 23 | 7.18e-16 | none | 395 | 0 |
| nx8_nu3_N30 | nominal | `daqp` | 1/1 | 30.6 | 20.8 - 36.9 | 88.4 / 3.83 / 26.7 | 2.98 | 1010000 / 33900 / 258000 / 292000 | 8 / 15.5 / 24 | 7.62e-15 | none | 0 | 0 |
| nx8_nu3_N30 | nominal | `proxqp_dense` | 59/60 | 292 | 256 - 331 | 468 / 8.28 / 283 | 3.69 | 4690000 / 78300 / 2260000 / 2340000 | 0 / 14 / 22 | 4.79e-3 | inaccurate 5 | 7 | 0 |
| nx8_nu3_N30 | nominal | `osqp` | 1/1 | 1000 | 821 - 1010 | 397 / 0.610 / 1000 | 3.57 | 3230000 / 4180 / 9060000 / 9070000 | 75 / 100 / 100 | 3.55e-7 | none | 60 | 0 |
| nx8_nu3_N30 | tight | `argmin_dense_admm` | 1/1 | 1700 | 1420 - 1740 | 1910 / 0.130 / 1700 | 3.82 | 16700000 / 796 / 13400000 / 13400000 | 25 / 175 / 425 | 2.75e-16 | none | 0 | 0 |
| nx8_nu3_N30 | tight | `argmin_dense_active_set` | 1/1 | 23600 | 22900 - 24200 | 24400 / 0.160 / 23600 | 4.08 | 225000000 / 795 / 218000000 / 218000000 | 0 / 90 / 90 | 3.70e-16 | none | 8416 | 295 |
| nx8_nu3_N30 | tight | `qpoases` | 1/1 | 991 | 766 - 1140 | 35900 / 0.160 / 990 | 3.79 | 475000000 / 920 / 13500000 / 13600000 | 0 / 8 / 17 | 6.12e-16 | none | 359 | 0 |
| nx8_nu3_N30 | tight | `daqp` | 1/1 | 157 | 132 - 197 | 247 / 8.43 / 149 | 3.52 | 2140000 / 63800 / 1670000 / 1740000 | 9 / 17 / 105 | 3.51e-15 | none | 0 | 0 |
| nx8_nu3_N30 | tight | `proxqp_dense` | 9/25 | 3000 | 2410 - 3830 | 1540 / 12.4 / 2990 | 4.09 | 13700000 / 112000 / 22600000 / 22700000 | 0 / 43 / 118 | 1.00e+0 | inaccurate 20, infeasible_claim 172 | 8 | 0 |
| nx8_nu3_N30 | tight | `osqp` | 1/1 | 2130 | 1790 - 2200 | 712 / 0.890 / 2130 | 3.87 | 6000000 / 7000 / 21100000 / 21100000 | 100 / 150 / 400 | 3.07e-15 | none | 60 | 0 |
| nx12_nu4_N30 | nominal | `argmin_dense_admm` | 1/1 | 1010 | 995 - 1320 | 1770 / 0.130 / 1010 | 5.91 | 15600000 / 670 / 8460000 / 8460000 | 25 / 75 / 200 | 7.18e-16 | none | 0 | 0 |
| nx12_nu4_N30 | nominal | `argmin_dense_active_set` | 1/1 | 20200 | 18900 - 22200 | 8720 / 0.150 / 20200 | 6.10 | 84100000 / 669 / 195000000 / 195000000 | 0 / 74 / 94 | 6.37e-16 | none | 537 | 0 |
| nx12_nu4_N30 | nominal | `qpoases` | 1/1 | 3980 | 3590 - 4380 | 17100 / 0.150 / 3980 | 6.02 | 177000000 / 709 / 35900000 / 35900000 | 0 / 18.5 / 24 | 1.21e-15 | none | 426 | 0 |
| nx12_nu4_N30 | nominal | `daqp` | 1/1 | 57.4 | 52.6 - 65.7 | 172 / 6.64 / 50.6 | 5.68 | 2060000 / 53900 / 457000 / 511000 | 13 / 20 / 26 | 1.20e-14 | none | 0 | 0 |
| nx12_nu4_N30 | nominal | `proxqp_dense` | 1/1 | 487 | 445 - 536 | 792 / 13.3 / 473 | 6.22 | 7870000 / 121000 / 3610000 / 3730000 | 0 / 16 / 23 | 8.56e-7 | none | 7 | 0 |
| nx12_nu4_N30 | nominal | `osqp` | 1/1 | 1450 | 1440 - 1470 | 727 / 0.780 / 1450 | 6.12 | 5970000 / 5180 / 13100000 / 13100000 | 50 / 75 / 175 | 9.44e-16 | none | 60 | 0 |
| nx12_nu4_N30 | tight | `argmin_dense_admm` | 1/1 | 2860 | 2570 - 4370 | 5090 / 0.160 / 2860 | 6.37 | 31200000 / 970 / 22300000 / 22300000 | 25 / 150 / 450 | 2.98e-16 | none | 4 | 0 |
| nx12_nu4_N30 | tight | `argmin_dense_active_set` | 1/1 | 62000 | 59800 - 62800 | 64300 / 0.190 / 62000 | 6.61 | 561000000 / 969 / 551000000 / 551000000 | 0 / 120 / 120 | 4.01e-16 | none | 14851 | 295 |
| nx12_nu4_N30 | tight | `qpoases` | 1/1 | 2140 | 1940 - 2600 | 116000 / 0.210 / 2140 | 6.31 | 1470000000 / 1040 / 28000000 / 28000000 | 0 / 10 / 19 | 6.31e-16 | none | 405 | 0 |
| nx12_nu4_N30 | tight | `daqp` | 1/1 | 342 | 294 - 412 | 539 / 14.8 / 328 | 6.27 | 4550000 / 103000 / 3580000 / 3680000 | 13 / 21 / 137 | 2.69e-15 | none | 0 | 0 |
| nx12_nu4_N30 | tight | `proxqp_dense` | 79/300 | 5980 | 4810 - 6650 | n/a / 19.5 / 5960 | 6.65 | n/a / 173000 / 40800000 / 41000000 | 0 / 37 / 164 | 1.00e+0 | inaccurate 41, infeasible_claim 180 | 8 | 0 |
| nx12_nu4_N30 | tight | `osqp` | 1/1 | 4750 | 3760 - 5870 | 1450 / 1.09 / 4740 | 6.51 | 10200000 / 9030 / 45600000 / 45600000 | 125 / 200 / 600 | 6.38e-15 | none | 60 | 0 |
| nx4_nu2_N50 | nominal | `argmin_dense_admm` | 1/1 | 698 | 693 - 704 | 1000 / 0.0900 / 698 | 2.85 | 9290000 / 595 / 5860000 / 5860000 | 25 / 75 / 75 | 5.03e-16 | none | 0 | 0 |
| nx4_nu2_N50 | nominal | `argmin_dense_active_set` | 1/1 | 7210 | 4110 - 10200 | 2970 / 0.120 / 7210 | 2.98 | 28600000 / 594 / 70100000 / 70100000 | 0 / 36.5 / 74 | 5.87e-16 | none | 422 | 0 |
| nx4_nu2_N50 | nominal | `qpoases` | 1/1 | 1270 | 996 - 2240 | 6970 / 0.110 / 1270 | 2.88 | 77000000 / 631 / 12200000 / 12200000 | 0 / 8 / 19 | 1.00e-15 | none | 316 | 0 |
| nx4_nu2_N50 | nominal | `daqp` | 1/1 | 24.8 | 12.8 - 38.4 | 114 / 4.64 / 20.1 | 2.45 | 1310000 / 40000 / 179000 / 219000 | 6 / 9 / 20 | 3.60e-15 | none | 0 | 0 |
| nx4_nu2_N50 | nominal | `proxqp_dense` | 13/15 | 234 | 216 - 273 | 514 / 9.43 / 225 | 2.97 | 5310000 / 87100 / 1790000 / 1880000 | 0 / 10.5 / 16 | 2.34e-2 | inaccurate 40 | 7 | 0 |
| nx4_nu2_N50 | nominal | `osqp` | 1/1 | 1030 | 1020 - 1040 | 505 / 0.630 / 1030 | 2.90 | 4150000 / 4440 / 9270000 / 9270000 | 50 / 75 / 75 | 6.08e-16 | none | 60 | 0 |
| nx4_nu2_N50 | tight | `argmin_dense_admm` | 1/1 | 1720 | 1540 - 2430 | 1810 / 0.130 / 1720 | 3.02 | 17200000 / 845 / 14200000 / 14200000 | 25 / 125 / 425 | 2.84e-16 | none | 3 | 0 |
| nx4_nu2_N50 | tight | `argmin_dense_active_set` | 1/1 | 33000 | 31700 - 33600 | 33700 / 0.160 / 33000 | 3.30 | 306000000 / 844 / 298000000 / 298000000 | 0 / 100 / 100 | 2.89e-16 | none | 10376 | 225 |
| nx4_nu2_N50 | tight | `qpoases` | 1/1 | 930 | 610 - 1200 | 48300 / 0.160 / 929 | 3.04 | 639000000 / 915 / 12400000 / 12400000 | 0 / 6 / 15 | 6.43e-16 | none | 319 | 0 |
| nx4_nu2_N50 | tight | `daqp` | 1/1 | 142 | 107 - 178 | 326 / 10.3 / 131 | 2.91 | 2790000 / 75200 / 1430000 / 1510000 | 5 / 13 / 101 | 2.49e-15 | none | 0 | 0 |
| nx4_nu2_N50 | tight | `proxqp_dense` | 113/300 | 3390 | 1680 - 3660 | 1390 / 14.0 / 3380 | 3.31 | 13500000 / 124000 / 24400000 / 24500000 | 0 / 38.5 / 103 | 1.00e+0 | inaccurate 29, infeasible_claim 158 | 8 | 0 |
| nx4_nu2_N50 | tight | `osqp` | 1/1 | 2140 | 2120 - 3650 | 911 / 0.940 / 2140 | 3.11 | 7880000 / 7670 / 20800000 / 20800000 | 75 / 100 / 375 | 1.71e-15 | none | 60 | 0 |
| nx4_nu2_N100 | nominal | `argmin_dense_admm` | 1/1 | 2960 | 2950 - 2970 | 6380 / 0.180 / 2960 | 9.84 | 40600000 / 998 / 25200000 / 25200000 | 25 / 75 / 75 | 2.82e-16 | none | 2 | 0 |
| nx4_nu2_N100 | nominal | `argmin_dense_active_set` | 1/1 | 41400 | 23300 - 59400 | 18700 / 0.210 / 41400 | 10.1 | 160000000 / 986 / 410000000 / 410000000 | 0 / 36.5 / 74 | 2.13e-16 | none | 758 | 0 |
| nx4_nu2_N100 | nominal | `qpoases` | 1/1 | 9770 | 7610 - 16300 | 91200 / 0.210 / 9770 | 9.72 | 812000000 / 1040 / 68700000 / 68700000 | 0 / 8 / 19 | 1.26e-15 | none | 316 | 0 |
| nx4_nu2_N100 | nominal | `daqp` | 1/1 | 101 | 47.2 - 158 | 649 / 18.9 / 81.7 | 9.56 | 7800000 / 130000 / 569000 / 699000 | 6 / 9 / 20 | 4.66e-15 | none | 0 | 0 |
| nx4_nu2_N100 | nominal | `proxqp_dense` | 13/15 | 673 | 625 - 791 | 2290 / 35.0 / 638 | 9.88 | 18300000 / 319000 / 4780000 / 5100000 | 0 / 10.5 / 16 | 2.34e-2 | inaccurate 40 | 7 | 0 |
| nx4_nu2_N100 | nominal | `osqp` | 1/1 | 4330 | 4260 - 4340 | 2780 / 1.03 / 4330 | 9.87 | 20300000 / 8080 / 39300000 / 39300000 | 50 / 75 / 75 | 4.26e-16 | none | 60 | 0 |
| nx4_nu2_N100 | tight | `argmin_dense_admm` | 1/1 | 8200 | 7530 - 9510 | 11900 / 0.260 / 8200 | 10.4 | 86300000 / 1500 / 68900000 / 68900000 | 25 / 125 / 300 | 2.78e-16 | none | 13 | 0 |
| nx4_nu2_N100 | tight | `argmin_dense_active_set` | 1/1 | 376000 | 368000 - 380000 | 382000 / 0.280 / 376000 | 10.7 | 3300000000 / 1500 / 3260000000 / 3260000000 | 0 / 200 / 200 | 3.89e-16 | none | 41473 | 235 |
| nx4_nu2_N100 | tight | `qpoases` | 1/1 | 4380 | 2940 - 5540 | 874000 / 0.280 / 4380 | 10.1 | 8510000000 / 1560 / 53000000 / 53000000 | 0 / 6 / 15 | 1.06e-15 | none | 319 | 0 |
| nx4_nu2_N100 | tight | `daqp` | 1/1 | 595 | 448 - 749 | 2320 / 46.7 / 549 | 10.1 | 18600000 / 260000 / 5140000 / 5400000 | 5 / 13 / 201 | 4.53e-15 | none | 0 | 0 |
| nx4_nu2_N100 | tight | `proxqp_dense` | 103/300 | 15600 | 10000 - 21500 | 8070 / 50.5 / 15500 | 10.5 | 54000000 / 451000 / 121000000 / 121000000 | 0 / 34.5 / 10079 | 1.00e+0 | inaccurate 12, infeasible_claim 185 | 58 | 0 |
| nx4_nu2_N100 | tight | `osqp` | 1/1 | 11300 | 8540 - 15000 | 5240 / 1.56 / 11300 | 10.4 | 34100000 / 14500 / 109000000 / 109000000 | 75 / 150 / 325 | 3.10e-15 | none | 60 | 0 |
| nx8_nu3_N50 | nominal | `argmin_dense_admm` | 1/1 | 1870 | 1850 - 1900 | 2590 / 0.150 / 1870 | 8.20 | 23700000 / 823 / 15400000 / 15400000 | 25 / 100 / 125 | 3.33e-16 | none | 0 | 0 |
| nx8_nu3_N50 | nominal | `argmin_dense_active_set` | 1/1 | 29000 | 22500 - 37600 | 8580 / 0.190 / 29000 | 8.42 | 82700000 / 822 / 281000000 / 281000000 | 0 / 57.5 / 89 | 2.13e-16 | none | 726 | 0 |
| nx8_nu3_N50 | nominal | `qpoases` | 1/1 | 6690 | 4300 - 8420 | 32100 / 0.180 / 6690 | 8.18 | 318000000 / 862 / 54200000 / 54200000 | 0 / 14 / 23 | 7.61e-16 | none | 395 | 0 |
| nx8_nu3_N50 | nominal | `daqp` | 1/1 | 77.3 | 51.7 - 92.3 | 303 / 10.3 / 66.9 | 8.07 | 3660000 / 78600 / 540000 / 618000 | 8 / 15.5 / 24 | 5.40e-15 | none | 0 | 0 |
| nx8_nu3_N50 | nominal | `proxqp_dense` | 59/60 | 603 | 536 - 692 | 1120 / 21.6 / 582 | 8.39 | 11200000 / 196000 / 4380000 / 4570000 | 0 / 13.5 / 22 | 4.79e-3 | inaccurate 5 | 7 | 0 |
| nx8_nu3_N50 | nominal | `osqp` | 1/1 | 2870 | 2860 - 2880 | 1260 / 0.880 / 2870 | 8.28 | 10300000 / 6340 / 25900000 / 25900000 | 75 / 100 / 125 | 4.80e-16 | none | 60 | 0 |
| nx8_nu3_N50 | tight | `argmin_dense_admm` | 1/1 | 5020 | 4180 - 5090 | 5830 / 0.200 / 5020 | 8.84 | 50400000 / 1190 / 39800000 / 39800000 | 25 / 175 / 675 | 2.30e-16 | none | 10 | 0 |
| nx8_nu3_N50 | tight | `argmin_dense_active_set` | 1/1 | 136000 | 134000 - 138000 | 138000 / 0.250 / 136000 | 9.23 | 1230000000 / 1220 / 1210000000 / 1210000000 | 0 / 150 / 150 | 3.70e-16 | none | 23335 | 295 |
| nx8_nu3_N50 | tight | `qpoases` | 1/1 | 3170 | 2560 - 3810 | 258000 / 0.235 / 3170 | 8.65 | 3090000000 / 1300 / 39200000 / 39200000 | 0 / 8 / 17 | 7.11e-16 | none | 359 | 0 |
| nx8_nu3_N50 | tight | `daqp` | 1/1 | 428 | 369 - 547 | 1030 / 24.3 / 404 | 8.57 | 8420000 / 155000 / 4120000 / 4280000 | 9 / 17 / 165 | 2.90e-15 | none | 0 | 0 |
| nx8_nu3_N50 | tight | `proxqp_dense` | 27/100 | 10400 | 8970 - 10900 | 3980 / 31.2 / 10400 | 9.19 | 34400000 / 276000 / 72700000 / 73000000 | 0 / 38 / 155 | 1.00e+0 | inaccurate 31, infeasible_claim 188 | 15 | 0 |
| nx8_nu3_N50 | tight | `osqp` | 1/1 | 6260 | 4620 - 8650 | 2070 / 1.28 / 6260 | 8.82 | 17500000 / 11100 / 59800000 / 59800000 | 75 / 150 / 750 | 4.61e-15 | none | 60 | 0 |
| nx8_nu3_N100 | nominal | `argmin_dense_admm` | 1/1 | 8630 | 8560 - 9500 | 13600 / 0.280 / 8630 | 29.5 | 115000000 / 1370 / 70400000 / 70400000 | 25 / 100 / 125 | 3.72e-16 | none | 9 | 0 |
| nx8_nu3_N100 | nominal | `argmin_dense_active_set` | 1/1 | 185000 | 142000 - 238000 | 51800 / 0.280 / 185000 | 29.8 | 515000000 / 1350 / 1860000000 / 1860000000 | 0 / 57.5 / 89 | 2.96e-16 | none | 992 | 0 |
| nx8_nu3_N100 | nominal | `qpoases` | 1/1 | 55400 | 37100 - 69600 | 416000 / 0.280 / 55400 | 29.5 | 3200000000 / 1410 / 337000000 / 337000000 | 0 / 14 / 23 | 1.29e-15 | none | 395 | 0 |
| nx8_nu3_N100 | nominal | `daqp` | 1/1 | 350 | 224 - 419 | 2060 / 47.0 / 303 | 29.2 | 23500000 / 269000 / 1780000 / 2050000 | 8 / 15.5 / 24 | 4.97e-15 | none | 0 | 0 |
| nx8_nu3_N100 | nominal | `proxqp_dense` | 29/30 | 1740 | 1620 - 2050 | 4220 / 79.8 / 1660 | 29.5 | 39700000 / 694000 / 11600000 / 12300000 | 0 / 13 / 22 | 4.79e-3 | inaccurate 10 | 26 | 0 |
| nx8_nu3_N100 | nominal | `osqp` | 1/1 | 10700 | 8740 - 10900 | 6660 / 1.37 / 10700 | 29.4 | 53800000 / 11700 / 97500000 / 97500000 | 50 / 75 / 150 | 5.29e-16 | none | 60 | 0 |
| nx8_nu3_N100 | tight | `argmin_dense_admm` | 1/1 | 22900 | 21500 - 22900 | 30400 / 0.340 / 22900 | 30.2 | 268000000 / 2120 / 198000000 / 198000000 | 25 / 150 / 500 | 3.33e-16 | none | 21 | 0 |
| nx8_nu3_N100 | tight | `argmin_dense_active_set` | 1/1 | 1690000 | 1680000 - 1710000 | 1720000 / 0.420 / 1690000 | 45.7 | 14400000000 / 2140 / 14300000000 / 14300000000 | 0 / 300 / 300 | 4.09e-16 | none | 92813 | 295 |
| nx8_nu3_N100 | tight | `qpoases` | 1/1 | 13800 | 11300 - 17300 | 4680000 / 0.370 / 13800 | 30.1 | 38600000000 / 2190 / 162000000 / 162000000 | 0 / 8 / 17 | 1.32e-15 | none | 359 | 0 |
| nx8_nu3_N100 | tight | `daqp` | 1/1 | 1880 | 1620 - 2430 | 8160 / 115 / 1760 | 30.0 | 58700000 / 555000 / 14800000 / 15400000 | 9 / 17 / 315 | 3.65e-15 | none | 0 | 0 |
| nx8_nu3_N100 | tight | `proxqp_dense` | 22/75 | 43700 | 34300 - 65200 | 18200 / 115 / 43600 | 30.6 | 143000000 / 981000 / 320000000 / 321000000 | 0 / 45.5 / 304 | 1.00e+0 | inaccurate 17, infeasible_claim 195 | 309 | 0 |
| nx8_nu3_N100 | tight | `osqp` | 1/1 | 26400 | 23400 - 29800 | 12500 / 2.10 / 26400 | 30.2 | 84800000 / 21300 / 246000000 / 246000000 | 100 / 150 / 525 | 5.34e-15 | none | 60 | 0 |

### dense cohort, churning regime, mpc tier

The dense cohort following the hashed initial-state sequence seeded by the repetition index,
judged at the mpc tier's threshold of 1e-3.

| Cell | Variant | Arm | Pass rate | Median tick (us) | Tick p25 - p75 (us) | Median setup / update / solve (us) | Median condense (us) | Median instructions setup / update / solve / tick | Iterations min / median / max | Worst KKT residual | Failures | Allocations max | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nx2_nu1_N10 | nominal | `argmin_dense_admm` | 1/1 | 9.83 | 9.77 - 10.7 | 19.2 / 0.0300 / 9.79 | 0.150 | 161000 / 389 / 84600 / 85000 | 25 / 25 / 125 | 9.33e-4 | none | 0 | 0 |
| nx2_nu1_N10 | nominal | `argmin_dense_active_set` | 1/1 | 33.8 | 17.8 - 70.5 | 25.3 / 0.0300 / 33.8 | 0.170 | 120000 / 371 / 300000 / 301000 | 0 / 10 / 36 | 1.44e-4 | none | 195 | 0 |
| nx2_nu1_N10 | nominal | `qpoases` | 1/1 | 7.30 | 4.32 - 11.9 | 14.7 / 0.0300 / 7.26 | 0.150 | 59000 / 411 / 62500 / 62900 | 0 / 4 / 17 | 5.37e-13 | none | 164 | 0 |
| nx2_nu1_N10 | nominal | `daqp` | 1/1 | 1.04 | 0.818 - 1.50 | 3.96 / 0.220 / 0.820 | 0.150 | 13000 / 2320 / 7160 / 9570 | 1 / 5 / 18 | 2.18e-4 | none | 0 | 0 |
| nx2_nu1_N10 | nominal | `proxqp_dense` | 58/75 | 15.1 | 10.6 - 20.5 | 33.4 / 0.600 / 14.4 | 0.170 | 157000 / 5240 / 117000 / 122000 | 0 / 5 / 14 | 4.09e-1 | inaccurate 68 | 7 | 0 |
| nx2_nu1_N10 | nominal | `osqp` | 1/1 | 8.27 | 8.22 - 8.36 | 18.9 / 0.190 / 8.08 | 0.160 | 101000 / 1350 / 75300 / 76700 | 25 / 25 / 75 | 7.19e-4 | none | 0 | 0 |
| nx2_nu1_N10 | nominal | `argmin_dense_admm_polished` | 1/1 | 13.9 | 13.7 - 14.1 | 34.3 / 0.0300 / 13.8 | 0.160 | 207000 / 392 / 130000 / 130000 | 25 / 25 / 125 | 4.44e-16 | none | 0 | 0 |
| nx2_nu1_N10 | tight | `argmin_dense_admm` | 1/1 | 19.5 | 19.2 - 28.6 | 29.1 / 0.0300 / 19.5 | 0.190 | 243000 / 370 / 160000 / 160000 | 25 / 50 / 275 | 5.68e-4 | none | 0 | 0 |
| nx2_nu1_N10 | tight | `argmin_dense_active_set` | 1/1 | 39.8 | 30.8 - 44.1 | 66.2 / 0.0400 / 39.7 | 0.220 | 419000 / 369 / 380000 / 381000 | 0 / 10 / 10 | 8.19e-5 | none | 132 | 158 |
| nx2_nu1_N10 | tight | `qpoases` | 1/1 | 8.73 | 5.37 - 13.6 | 50.8 / 0.0400 / 8.70 | 0.200 | 315000 / 494 / 77400 / 77800 | 0 / 3 / 10 | 3.33e-13 | none | 186 | 0 |
| nx2_nu1_N10 | tight | `daqp` | 1/1 | 2.70 | 1.91 - 3.96 | 4.84 / 0.290 / 2.40 | 0.190 | 18900 / 3220 / 24400 / 27600 | 1 / 7 / 21 | 1.73e-4 | none | 0 | 0 |
| nx2_nu1_N10 | tight | `proxqp_dense` | 169/300 | 36.7 | 19.7 - 53.7 | 115 / 0.730 / 35.9 | 0.220 | 619000 / 6250 / 266000 / 272000 | 0 / 8.5 / 32 | 1.00e+0 | inaccurate 34, infeasible_claim 97 | 7 | 0 |
| nx2_nu1_N10 | tight | `osqp` | 1/1 | 21.8 | 21.3 - 31.2 | 26.7 / 0.210 / 21.6 | 0.190 | 142000 / 1630 / 215000 / 217000 | 25 / 50 / 175 | 5.55e-4 | none | 0 | 0 |
| nx2_nu1_N10 | tight | `argmin_dense_admm_polished` | 29/30 | 27.0 | 26.7 - 36.2 | 52.6 / 0.0300 / 27.0 | 0.200 | 331000 / 387 / 247000 / 248000 | 25 / 50 / 275 | 7.54e-1 | inaccurate 10 | 0 | 0 |
| nx4_nu2_N10 | nominal | `argmin_dense_admm` | 1/1 | 16.8 | 16.8 - 17.2 | 35.7 / 0.0300 / 16.8 | 0.240 | 300000 / 298 / 130000 / 130000 | 25 / 25 / 75 | 8.98e-4 | none | 0 | 0 |
| nx4_nu2_N10 | nominal | `argmin_dense_active_set` | 1/1 | 203 | 104 - 305 | 102 / 0.0300 / 203 | 0.280 | 787000 / 296 / 1960000 / 1960000 | 0 / 27 / 67 | 1.55e-4 | none | 435 | 0 |
| nx4_nu2_N10 | nominal | `qpoases` | 1/1 | 25.5 | 16.5 - 35.3 | 39.6 / 0.0400 / 25.5 | 0.250 | 300000 / 338 / 234000 / 235000 | 0 / 9 / 26 | 7.30e-13 | none | 249 | 0 |
| nx4_nu2_N10 | nominal | `daqp` | 1/1 | 3.06 | 2.41 - 4.28 | 7.90 / 0.400 / 2.66 | 0.220 | 40200 / 4390 / 22900 / 27300 | 1 / 10 / 27 | 4.03e-4 | none | 0 | 0 |
| nx4_nu2_N10 | nominal | `proxqp_dense` | 233/300 | 32.0 | 25.8 - 41.4 | 71.5 / 0.840 / 31.1 | 0.250 | 456000 / 7690 / 239000 / 247000 | 0 / 7 / 15 | 2.62e-1 | inaccurate 67 | 8 | 0 |
| nx4_nu2_N10 | nominal | `osqp` | 1/1 | 16.2 | 16.1 - 16.3 | 35.5 / 0.210 / 16.0 | 0.240 | 225000 / 1550 / 152000 / 154000 | 25 / 25 / 75 | 9.32e-4 | none | 0 | 0 |
| nx4_nu2_N10 | nominal | `argmin_dense_admm_polished` | 1/1 | 27.5 | 27.2 - 28.5 | 59.1 / 0.0300 / 27.5 | 0.250 | 430000 / 295 / 257000 / 257000 | 25 / 25 / 75 | 2.05e-4 | none | 0 | 0 |
| nx4_nu2_N10 | tight | `argmin_dense_admm` | 1/1 | 36.0 | 35.8 - 53.4 | 55.2 / 0.0400 / 35.9 | 0.290 | 504000 / 373 / 282000 / 283000 | 25 / 50 / 250 | 6.23e-4 | none | 0 | 0 |
| nx4_nu2_N10 | tight | `argmin_dense_active_set` | 1/1 | 207 | 175 - 223 | 265 / 0.0400 / 207 | 0.360 | 2330000 / 344 / 2050000 / 2050000 | 0 / 20 / 20 | 1.93e-4 | none | 462 | 204 |
| nx4_nu2_N10 | tight | `qpoases` | 1/1 | 25.6 | 16.9 - 37.1 | 137 / 0.0400 / 25.6 | 0.310 | 1190000 / 415 / 244000 / 244000 | 0 / 6 / 16 | 4.13e-13 | none | 274 | 0 |
| nx4_nu2_N10 | tight | `daqp` | 1/1 | 8.76 | 5.99 - 12.3 | 10.6 / 0.660 / 8.10 | 0.280 | 63300 / 6630 / 94300 / 101000 | 1 / 13 / 33 | 2.09e-4 | none | 0 | 0 |
| nx4_nu2_N10 | tight | `proxqp_dense` | 49/100 | 116 | 76.3 - 165 | 159 / 1.11 / 115 | 0.380 | 1060000 / 10100 / 837000 / 847000 | 0 / 17 / 70 | 1.00e+0 | inaccurate 36, infeasible_claim 117 | 8 | 0 |
| nx4_nu2_N10 | tight | `osqp` | 1/1 | 54.6 | 44.7 - 65.9 | 55.2 / 0.280 / 54.3 | 0.290 | 351000 / 2200 / 476000 / 479000 | 25 / 50 / 200 | 5.54e-4 | none | 0 | 0 |
| nx4_nu2_N10 | tight | `argmin_dense_admm_polished` | 19/20 | 75.2 | 58.0 - 92.3 | 95.5 / 0.0400 / 75.1 | 0.330 | 775000 / 347 / 690000 / 690000 | 25 / 75 / 275 | 6.09e-1 | inaccurate 15 | 0 | 0 |
| nx4_nu2_N20 | nominal | `argmin_dense_admm` | 1/1 | 34.4 | 34.3 - 34.5 | 91.7 / 0.0500 / 34.3 | 0.500 | 820000 / 370 / 252000 / 252000 | 25 / 25 / 50 | 8.77e-4 | none | 0 | 0 |
| nx4_nu2_N20 | nominal | `argmin_dense_active_set` | 1/1 | 732 | 362 - 1170 | 361 / 0.0500 / 732 | 0.640 | 2770000 / 369 / 7120000 / 7120000 | 0 / 29 / 126 | 1.55e-4 | none | 642 | 0 |
| nx4_nu2_N20 | nominal | `qpoases` | 1/1 | 61.2 | 39.8 - 87.9 | 100 / 0.0500 / 61.1 | 0.520 | 856000 / 409 / 653000 / 654000 | 0 / 9 / 26 | 6.04e-13 | none | 249 | 0 |
| nx4_nu2_N20 | nominal | `daqp` | 1/1 | 5.44 | 4.25 - 7.22 | 19.2 / 0.910 / 4.50 | 0.500 | 157000 / 10300 / 44300 / 54600 | 1 / 10 / 27 | 4.03e-4 | none | 0 | 0 |
| nx4_nu2_N20 | nominal | `proxqp_dense` | 58/75 | 53.0 | 43.4 - 67.1 | 150 / 1.92 / 51.1 | 0.590 | 1080000 / 18900 / 422000 / 441000 | 0 / 7 / 16 | 2.62e-1 | inaccurate 68 | 8 | 0 |
| nx4_nu2_N20 | nominal | `osqp` | 1/1 | 45.7 | 45.5 - 45.8 | 88.1 / 0.290 / 45.4 | 0.510 | 669000 / 2270 / 430000 / 432000 | 25 / 25 / 50 | 8.16e-4 | none | 0 | 0 |
| nx4_nu2_N20 | nominal | `argmin_dense_admm_polished` | 1/1 | 73.7 | 73.3 - 74.3 | 186 / 0.0500 / 73.6 | 0.590 | 1290000 / 370 / 715000 / 716000 | 25 / 25 / 50 | 1.62e-4 | none | 0 | 0 |
| nx4_nu2_N20 | tight | `argmin_dense_admm` | 1/1 | 80.5 | 79.7 - 80.9 | 143 / 0.0600 / 80.4 | 0.610 | 1390000 / 470 / 587000 / 587000 | 25 / 50 / 175 | 6.06e-4 | none | 0 | 0 |
| nx4_nu2_N20 | tight | `argmin_dense_active_set` | 1/1 | 1560 | 1490 - 1600 | 1680 / 0.0900 / 1560 | 0.790 | 16500000 / 469 / 15600000 / 15600000 | 0 / 40 / 40 | 1.81e-4 | none | 1723 | 226 |
| nx4_nu2_N20 | tight | `qpoases` | 1/1 | 73.0 | 47.8 - 109 | 533 / 0.0600 / 72.9 | 0.630 | 6210000 / 540 / 861000 / 862000 | 0 / 6 / 18 | 3.71e-13 | none | 325 | 0 |
| nx4_nu2_N20 | tight | `daqp` | 1/1 | 24.2 | 17.0 - 35.5 | 34.4 / 1.72 / 22.4 | 0.570 | 281000 / 17200 / 285000 / 302000 | 1 / 13 / 47 | 2.08e-4 | none | 0 | 0 |
| nx4_nu2_N20 | tight | `proxqp_dense` | 43/100 | 439 | 327 - 516 | 362 / 2.87 / 433 | 0.800 | 2980000 / 25900 / 3300000 / 3320000 | 0 / 28 / 91 | 1.00e+0 | inaccurate 46, infeasible_claim 125 | 9 | 0 |
| nx4_nu2_N20 | tight | `osqp` | 1/1 | 137 | 136 - 143 | 149 / 0.400 / 137 | 0.600 | 1190000 / 3550 / 1430000 / 1430000 | 25 / 50 / 125 | 6.06e-4 | none | 0 | 0 |
| nx4_nu2_N20 | tight | `argmin_dense_admm_polished` | 93/100 | 216 | 176 - 256 | 259 / 0.0700 / 216 | 0.740 | 2510000 / 470 / 1990000 / 2000000 | 25 / 75 / 225 | 6.09e-1 | inaccurate 21 | 0 | 0 |
| nx8_nu3_N20 | nominal | `argmin_dense_admm` | 1/1 | 60.8 | 60.7 - 61.0 | 160 / 0.0600 / 60.7 | 1.43 | 1650000 / 445 / 423000 / 424000 | 25 / 25 / 50 | 9.88e-4 | none | 0 | 0 |
| nx8_nu3_N20 | nominal | `argmin_dense_active_set` | 1/1 | 2570 | 1530 - 4000 | 951 / 0.0800 / 2570 | 1.78 | 7290000 / 444 / 24800000 / 24800000 | 0 / 42 / 180 | 3.49e-4 | none | 970 | 0 |
| nx8_nu3_N20 | nominal | `qpoases` | 1/1 | 164 | 112 - 211 | 187 / 0.0700 / 164 | 1.48 | 1780000 / 484 / 1800000 / 1800000 | 0 / 13 / 33 | 6.34e-13 | none | 313 | 0 |
| nx8_nu3_N20 | nominal | `daqp` | 1/1 | 13.1 | 10.0 - 16.8 | 38.8 / 1.76 / 11.2 | 1.27 | 386000 / 18200 / 106000 / 124000 | 2 / 14 / 34 | 4.34e-4 | none | 0 | 0 |
| nx8_nu3_N20 | nominal | `proxqp_dense` | 239/300 | 103 | 83.6 - 130 | 205 / 3.76 / 99.5 | 1.67 | 1930000 / 35800 / 805000 / 841000 | 0 / 8 / 17 | 1.51e-1 | inaccurate 61 | 8 | 0 |
| nx8_nu3_N20 | nominal | `osqp` | 1/1 | 91.3 | 91.0 - 95.2 | 175 / 0.380 / 90.9 | 1.36 | 1390000 / 2990 / 840000 / 843000 | 25 / 25 / 75 | 5.27e-4 | none | 0 | 0 |
| nx8_nu3_N20 | nominal | `argmin_dense_admm_polished` | 1/1 | 157 | 156 - 157 | 354 / 0.0700 / 157 | 1.69 | 2760000 / 445 / 1530000 / 1530000 | 25 / 25 / 50 | 4.89e-4 | none | 0 | 0 |
| nx8_nu3_N20 | tight | `argmin_dense_admm` | 1/1 | 159 | 158 - 164 | 297 / 0.0800 / 159 | 1.69 | 2800000 / 595 / 1030000 / 1030000 | 25 / 50 / 125 | 6.36e-4 | none | 0 | 0 |
| nx8_nu3_N20 | tight | `argmin_dense_active_set` | 1/1 | 5780 | 5530 - 5950 | 6120 / 0.130 / 5780 | 2.11 | 58200000 / 594 / 55800000 / 55800000 | 0 / 60 / 60 | 4.16e-5 | none | 3808 | 253 |
| nx8_nu3_N20 | tight | `qpoases` | 1/1 | 195 | 140 - 263 | 1580 / 0.100 / 195 | 1.68 | 19500000 / 665 / 2430000 / 2430000 | 0 / 9 / 22 | 3.45e-13 | none | 393 | 0 |
| nx8_nu3_N20 | tight | `daqp` | 1/1 | 71.6 | 52.5 - 98.5 | 86.5 / 3.87 / 67.7 | 1.53 | 748000 / 32100 / 799000 / 831000 | 3 / 19 / 73 | 2.12e-4 | none | 0 | 0 |
| nx8_nu3_N20 | tight | `proxqp_dense` | 9/25 | 1070 | 800 - 1290 | 711 / 5.63 / 1060 | 2.16 | 6210000 / 50000 / 7700000 / 7750000 | 0 / 31 / 121 | 1.00e+0 | inaccurate 49, infeasible_claim 143 | 8 | 0 |
| nx8_nu3_N20 | tight | `osqp` | 1/1 | 276 | 270 - 285 | 315 / 0.555 / 275 | 1.64 | 2570000 / 4910 / 2730000 / 2740000 | 25 / 50 / 125 | 6.40e-4 | none | 0 | 0 |
| nx8_nu3_N20 | tight | `argmin_dense_admm_polished` | 137/150 | 476 | 401 - 484 | 566 / 0.0800 / 476 | 1.90 | 5510000 / 595 / 4250000 / 4250000 | 25 / 75 / 250 | 5.10e-1 | inaccurate 26 | 0 | 0 |
| nx8_nu3_N30 | nominal | `argmin_dense_admm` | 1/1 | 119 | 118 - 119 | 362 / 0.0900 / 119 | 3.41 | 3670000 / 598 / 773000 / 774000 | 25 / 25 / 50 | 9.13e-4 | none | 0 | 0 |
| nx8_nu3_N30 | nominal | `argmin_dense_active_set` | 1/1 | 6660 | 3950 - 10400 | 2240 / 0.120 / 6660 | 3.71 | 18200000 / 597 / 63700000 / 63700000 | 0 / 42 / 179 | 3.49e-4 | none | 965 | 0 |
| nx8_nu3_N30 | nominal | `qpoases` | 1/1 | 351 | 240 - 455 | 401 / 0.120 / 351 | 3.40 | 3910000 / 634 / 3920000 / 3930000 | 0 / 13 / 33 | 6.37e-13 | none | 313 | 0 |
| nx8_nu3_N30 | nominal | `daqp` | 1/1 | 26.1 | 19.6 - 32.8 | 88.5 / 3.87 / 22.2 | 3.00 | 1010000 / 33900 / 190000 / 224000 | 2 / 14 / 34 | 4.34e-4 | none | 0 | 0 |
| nx8_nu3_N30 | nominal | `proxqp_dense` | 83/100 | 179 | 145 - 215 | 468 / 8.23 / 171 | 3.64 | 3910000 / 78300 / 1370000 / 1450000 | 0 / 8 / 17 | 1.64e-1 | inaccurate 51 | 8 | 0 |
| nx8_nu3_N30 | nominal | `osqp` | 1/1 | 196 | 195 - 197 | 400 / 0.570 / 196 | 3.30 | 3230000 / 4160 / 1760000 / 1770000 | 25 / 25 / 50 | 9.13e-4 | none | 0 | 0 |
| nx8_nu3_N30 | nominal | `argmin_dense_admm_polished` | 1/1 | 361 | 359 - 363 | 833 / 0.100 / 360 | 3.55 | 6360000 / 598 / 3460000 / 3460000 | 25 / 25 / 50 | 4.63e-4 | none | 0 | 0 |
| nx8_nu3_N30 | tight | `argmin_dense_admm` | 1/1 | 308 | 304 - 310 | 640 / 0.120 / 308 | 3.74 | 6060000 / 796 / 1850000 / 1850000 | 25 / 50 / 150 | 5.70e-4 | none | 0 | 0 |
| nx8_nu3_N30 | tight | `argmin_dense_active_set` | 1/1 | 23200 | 22500 - 23700 | 24000 / 0.160 / 23200 | 4.08 | 221000000 / 795 / 215000000 / 215000000 | 0 / 90 / 90 | 4.16e-5 | none | 8416 | 257 |
| nx8_nu3_N30 | tight | `qpoases` | 1/1 | 449 | 312 - 607 | 4830 / 0.160 / 449 | 3.74 | 59400000 / 920 / 5590000 / 5590000 | 0 / 9 / 22 | 3.28e-13 | none | 393 | 0 |
| nx8_nu3_N30 | tight | `daqp` | 1/1 | 156 | 113 - 214 | 247 / 8.42 / 148 | 3.49 | 2140000 / 63800 / 1660000 / 1720000 | 3 / 19 / 103 | 2.12e-4 | none | 0 | 0 |
| nx8_nu3_N30 | tight | `proxqp_dense` | 19/50 | 2640 | 1360 - 3020 | 1350 / 12.4 / 2630 | 4.09 | 11400000 / 112000 / 19400000 / 19500000 | 0 / 32.5 / 139 | 1.00e+0 | inaccurate 38, infeasible_claim 148 | 8 | 0 |
| nx8_nu3_N30 | tight | `osqp` | 1/1 | 579 | 578 - 592 | 710 / 0.830 / 579 | 3.72 | 5990000 / 7000 / 5810000 / 5820000 | 25 / 50 / 150 | 5.70e-4 | none | 0 | 0 |
| nx8_nu3_N30 | tight | `argmin_dense_admm_polished` | 139/150 | 1110 | 964 - 1250 | 1330 / 0.130 / 1110 | 3.83 | 13100000 / 796 / 9790000 / 9790000 | 25 / 75 / 200 | 5.78e-1 | inaccurate 22 | 0 | 0 |
| nx12_nu4_N30 | nominal | `argmin_dense_admm` | 1/1 | 175 | 174 - 176 | 591 / 0.110 / 175 | 5.90 | 6100000 / 674 / 1060000 / 1060000 | 25 / 25 / 50 | 8.99e-4 | none | 0 | 0 |
| nx12_nu4_N30 | nominal | `argmin_dense_active_set` | 1/1 | 16800 | 11600 - 24500 | 4270 / 0.150 / 16800 | 6.12 | 35400000 / 669 / 163000000 / 163000000 | 0 / 57 / 205 | 3.47e-4 | none | 1152 | 0 |
| nx12_nu4_N30 | nominal | `qpoases` | 1/1 | 798 | 607 - 1020 | 738 / 0.140 / 798 | 5.95 | 7000000 / 709 / 8760000 / 8760000 | 0 / 17 / 41 | 4.72e-13 | none | 384 | 0 |
| nx12_nu4_N30 | nominal | `daqp` | 1/1 | 54.4 | 42.0 - 70.2 | 172 / 6.66 / 47.7 | 5.65 | 2060000 / 53900 / 380000 / 434000 | 3 / 18 / 42 | 4.47e-4 | none | 0 | 0 |
| nx12_nu4_N30 | nominal | `proxqp_dense` | 239/300 | 305 | 255 - 356 | 584 / 13.3 / 292 | 6.19 | 6250000 / 121000 / 2270000 / 2390000 | 0 / 10 / 19 | 9.51e-2 | inaccurate 61 | 8 | 0 |
| nx12_nu4_N30 | nominal | `osqp` | 1/1 | 337 | 335 - 338 | 729 / 0.760 / 336 | 5.97 | 5970000 / 5180 / 2990000 / 3000000 | 25 / 25 / 50 | 8.99e-4 | none | 0 | 0 |
| nx12_nu4_N30 | nominal | `argmin_dense_admm_polished` | 1/1 | 665 | 662 - 669 | 1230 / 0.130 / 665 | 5.92 | 11400000 / 674 / 6380000 / 6380000 | 25 / 25 / 50 | 8.21e-4 | none | 0 | 0 |
| nx12_nu4_N30 | tight | `argmin_dense_admm` | 1/1 | 506 | 501 - 509 | 1580 / 0.150 / 506 | 6.22 | 10400000 / 970 / 2740000 / 2740000 | 25 / 50 / 100 | 4.85e-4 | none | 0 | 0 |
| nx12_nu4_N30 | tight | `argmin_dense_active_set` | 1/1 | 60800 | 59300 - 61800 | 62700 / 0.190 / 60800 | 6.64 | 552000000 / 969 / 541000000 / 541000000 | 0 / 120 / 120 | 3.21e-6 | none | 14851 | 272 |
| nx12_nu4_N30 | tight | `qpoases` | 1/1 | 998 | 743 - 1330 | 11300 / 0.200 / 998 | 6.33 | 138000000 / 1040 / 12200000 / 12200000 | 0 / 11 / 32 | 2.85e-13 | none | 497 | 0 |
| nx12_nu4_N30 | tight | `daqp` | 1/1 | 334 | 254 - 446 | 539 / 14.8 / 319 | 6.19 | 4550000 / 103000 / 3410000 / 3510000 | 5 / 23 / 135 | 2.12e-4 | none | 0 | 0 |
| nx12_nu4_N30 | tight | `proxqp_dense` | 1/3 | 4720 | 3080 - 5860 | 2300 / 19.5 / 4710 | 6.61 | 20000000 / 173000 / 33100000 / 33200000 | 0 / 41 / 153 | 1.00e+0 | inaccurate 53, infeasible_claim 147 | 8 | 0 |
| nx12_nu4_N30 | tight | `osqp` | 1/1 | 993 | 981 - 1000 | 1220 / 1.04 / 992 | 6.39 | 10200000 / 9020 / 9670000 / 9680000 | 25 / 50 / 100 | 4.94e-4 | none | 0 | 0 |
| nx12_nu4_N30 | tight | `argmin_dense_admm_polished` | 68/75 | 2060 | 1820 - 2080 | 3990 / 0.160 / 2060 | 6.28 | 24500000 / 970 / 18200000 / 18200000 | 25 / 75 / 225 | 5.40e-1 | inaccurate 28 | 4 | 0 |
| nx4_nu2_N50 | nominal | `argmin_dense_admm` | 1/1 | 132 | 132 - 137 | 410 / 0.0900 / 132 | 2.77 | 4250000 / 595 / 823000 / 824000 | 25 / 25 / 50 | 9.84e-4 | none | 0 | 0 |
| nx4_nu2_N50 | nominal | `argmin_dense_active_set` | 1/1 | 5930 | 2970 - 9330 | 2380 / 0.110 / 5930 | 2.98 | 23000000 / 594 / 58100000 / 58100000 | 0 / 29 / 129 | 1.55e-4 | none | 727 | 0 |
| nx4_nu2_N50 | nominal | `qpoases` | 1/1 | 316 | 198 - 454 | 502 / 0.110 / 316 | 2.81 | 4830000 / 634 / 3530000 / 3530000 | 0 / 9 / 26 | 6.12e-13 | none | 249 | 0 |
| nx4_nu2_N50 | nominal | `daqp` | 1/1 | 25.3 | 18.2 - 31.8 | 114 / 4.68 / 20.5 | 2.49 | 1310000 / 40000 / 166000 / 206000 | 1 / 10 / 27 | 4.03e-4 | none | 0 | 0 |
| nx4_nu2_N50 | nominal | `proxqp_dense` | 233/300 | 160 | 132 - 200 | 424 / 9.39 / 150 | 2.95 | 4630000 / 87100 / 1190000 / 1280000 | 0 / 7 / 16 | 2.62e-1 | inaccurate 67 | 8 | 0 |
| nx4_nu2_N50 | nominal | `osqp` | 1/1 | 245 | 244 - 248 | 504 / 0.590 / 244 | 2.69 | 4150000 / 4450 / 2190000 / 2190000 | 25 / 25 / 50 | 8.66e-4 | none | 0 | 0 |
| nx4_nu2_N50 | nominal | `argmin_dense_admm_polished` | 1/1 | 444 | 441 - 447 | 743 / 0.100 / 444 | 2.86 | 7680000 / 595 / 4250000 / 4250000 | 25 / 25 / 50 | 1.45e-4 | none | 0 | 0 |
| nx4_nu2_N50 | tight | `argmin_dense_admm` | 1/1 | 361 | 354 - 405 | 625 / 0.130 / 360 | 3.03 | 6200000 / 845 / 2110000 / 2110000 | 25 / 50 / 175 | 5.95e-4 | none | 1 | 0 |
| nx4_nu2_N50 | tight | `argmin_dense_active_set` | 1/1 | 32700 | 32100 - 33200 | 33300 / 0.160 / 32700 | 3.29 | 304000000 / 844 / 297000000 / 297000000 | 0 / 100 / 100 | 1.75e-4 | none | 10376 | 237 |
| nx4_nu2_N50 | tight | `qpoases` | 1/1 | 431 | 275 - 639 | 6180 / 0.160 / 431 | 3.02 | 76600000 / 924 / 5330000 / 5330000 | 0 / 6 / 18 | 3.50e-13 | none | 325 | 0 |
| nx4_nu2_N50 | tight | `daqp` | 1/1 | 140 | 100 - 207 | 326 / 10.2 / 129 | 2.89 | 2790000 / 75200 / 1380000 / 1460000 | 1 / 13 / 107 | 2.00e-4 | none | 0 | 0 |
| nx4_nu2_N50 | tight | `proxqp_dense` | 13/30 | 2590 | 1470 - 3390 | 1780 / 14.0 / 2580 | 3.31 | 16300000 / 124000 / 19500000 / 19600000 | 0 / 30.5 / 128 | 1.00e+0 | inaccurate 35, infeasible_claim 135 | 8 | 0 |
| nx4_nu2_N50 | tight | `osqp` | 1/1 | 782 | 770 - 795 | 915 / 0.890 / 781 | 3.04 | 7880000 / 7660 / 7650000 / 7660000 | 25 / 50 / 150 | 5.87e-4 | none | 0 | 0 |
| nx4_nu2_N50 | tight | `argmin_dense_admm_polished` | 23/25 | 1280 | 1190 - 1370 | 1450 / 0.150 / 1280 | 3.04 | 15100000 / 845 / 11600000 / 11600000 | 25 / 50 / 225 | 5.83e-1 | inaccurate 24 | 3 | 0 |
| nx4_nu2_N100 | nominal | `argmin_dense_admm` | 1/1 | 435 | 433 - 438 | 2510 / 0.180 / 435 | 9.69 | 17800000 / 1000 / 2400000 / 2400000 | 25 / 25 / 50 | 8.82e-4 | none | 0 | 0 |
| nx4_nu2_N100 | nominal | `argmin_dense_active_set` | 1/1 | 33800 | 16600 - 53600 | 15400 / 0.210 / 33800 | 10.1 | 128000000 / 978 / 335000000 / 335000000 | 0 / 29 / 129 | 1.55e-4 | none | 1307 | 0 |
| nx4_nu2_N100 | nominal | `qpoases` | 1/1 | 1380 | 826 - 2000 | 2430 / 0.200 / 1380 | 9.65 | 20600000 / 1040 / 13600000 / 13600000 | 0 / 9 / 26 | 6.13e-13 | none | 249 | 0 |
| nx4_nu2_N100 | nominal | `daqp` | 1/1 | 102 | 71.6 - 130 | 648 / 18.9 / 81.2 | 9.56 | 7800000 / 130000 / 531000 / 660000 | 1 / 10 / 27 | 4.03e-4 | none | 0 | 0 |
| nx4_nu2_N100 | nominal | `proxqp_dense` | 233/300 | 471 | 395 - 572 | 2540 / 34.9 / 436 | 9.80 | 16400000 / 319000 / 3280000 / 3600000 | 0 / 7 / 16 | 2.62e-1 | inaccurate 67 | 8 | 0 |
| nx4_nu2_N100 | nominal | `osqp` | 1/1 | 929 | 927 - 937 | 2790 / 1.02 / 928 | 9.84 | 20300000 / 8090 / 8130000 / 8140000 | 25 / 25 / 50 | 8.83e-4 | none | 0 | 0 |
| nx4_nu2_N100 | nominal | `argmin_dense_admm_polished` | 1/1 | 2130 | 2120 - 2140 | 5640 / 0.190 / 2130 | 9.84 | 36000000 / 998 / 20600000 / 20600000 | 25 / 25 / 50 | 1.43e-4 | none | 2 | 0 |
| nx4_nu2_N100 | tight | `argmin_dense_admm` | 1/1 | 1370 | 1340 - 1380 | 2850 / 0.240 / 1370 | 10.1 | 27400000 / 1490 / 6680000 / 6680000 | 25 / 50 / 175 | 5.79e-4 | none | 2 | 0 |
| nx4_nu2_N100 | tight | `argmin_dense_active_set` | 1/1 | 372000 | 368000 - 375000 | 376000 / 0.290 / 372000 | 10.7 | 3290000000 / 1490 / 3250000000 / 3250000000 | 0 / 200 / 200 | 1.74e-4 | none | 41473 | 242 |
| nx4_nu2_N100 | tight | `qpoases` | 1/1 | 1940 | 1220 - 2740 | 49200 / 0.270 / 1940 | 10.0 | 569000000 / 1550 / 21800000 / 21800000 | 0 / 6 / 18 | 3.49e-13 | none | 325 | 0 |
| nx4_nu2_N100 | tight | `daqp` | 1/1 | 594 | 428 - 861 | 2330 / 46.7 / 547 | 10.0 | 18600000 / 260000 / 5060000 / 5320000 | 1 / 13 / 207 | 1.98e-4 | none | 0 | 0 |
| nx4_nu2_N100 | tight | `proxqp_dense` | 23/60 | 10600 | 6010 - 15700 | 8210 / 50.2 / 10500 | 10.4 | 57700000 / 451000 / 80000000 / 80500000 | 0 / 27 / 204 | 1.00e+0 | inaccurate 30, infeasible_claim 155 | 66 | 0 |
| nx4_nu2_N100 | tight | `osqp` | 1/1 | 2890 | 2870 - 2910 | 5270 / 1.52 / 2890 | 10.2 | 34100000 / 14500 / 27700000 / 27800000 | 25 / 50 / 150 | 5.79e-4 | none | 0 | 0 |
| nx4_nu2_N100 | tight | `argmin_dense_admm_polished` | 93/100 | 6210 | 6170 - 6870 | 10900 / 0.270 / 6210 | 10.3 | 79800000 / 1490 / 59000000 / 59000000 | 25 / 50 / 225 | 5.69e-1 | inaccurate 21 | 15 | 0 |
| nx8_nu3_N50 | nominal | `argmin_dense_admm` | 1/1 | 266 | 262 - 272 | 990 / 0.150 / 266 | 8.12 | 9980000 / 851 / 1670000 / 1670000 | 25 / 25 / 50 | 6.50e-4 | none | 0 | 0 |
| nx8_nu3_N50 | nominal | `argmin_dense_active_set` | 1/1 | 22900 | 13600 - 36000 | 6410 / 0.190 / 22900 | 8.40 | 61700000 / 822 / 223000000 / 223000000 | 0 / 42 / 179 | 3.49e-4 | none | 1317 | 0 |
| nx8_nu3_N50 | nominal | `qpoases` | 1/1 | 985 | 669 - 1280 | 1200 / 0.160 / 985 | 8.14 | 11200000 / 862 / 10800000 / 10800000 | 0 / 13 / 33 | 6.32e-13 | none | 313 | 0 |
| nx8_nu3_N50 | nominal | `daqp` | 1/1 | 69.9 | 51.8 - 84.8 | 304 / 10.3 / 59.5 | 8.07 | 3660000 / 78600 / 444000 / 523000 | 2 / 14 / 34 | 4.34e-4 | none | 0 | 0 |
| nx8_nu3_N50 | nominal | `proxqp_dense` | 83/100 | 376 | 307 - 465 | 895 / 21.5 / 355 | 8.35 | 9670000 / 196000 / 2710000 / 2910000 | 0 / 8 / 16 | 1.64e-1 | inaccurate 51 | 8 | 0 |
| nx8_nu3_N50 | nominal | `osqp` | 1/1 | 530 | 528 - 532 | 1260 / 0.890 / 529 | 8.28 | 10300000 / 6350 / 4690000 / 4700000 | 25 / 25 / 50 | 6.50e-4 | none | 0 | 0 |
| nx8_nu3_N50 | nominal | `argmin_dense_admm_polished` | 1/1 | 1090 | 1080 - 1100 | 1830 / 0.150 / 1090 | 8.27 | 18900000 / 845 / 10600000 / 10600000 | 25 / 25 / 50 | 4.58e-4 | none | 0 | 0 |
| nx8_nu3_N50 | tight | `argmin_dense_admm` | 1/1 | 782 | 776 - 787 | 1850 / 0.180 / 782 | 8.59 | 16900000 / 1200 / 4260000 / 4260000 | 25 / 50 / 75 | 5.97e-4 | none | 0 | 0 |
| nx8_nu3_N50 | tight | `argmin_dense_active_set` | 1/1 | 135000 | 133000 - 137000 | 137000 / 0.240 / 135000 | 9.22 | 1220000000 / 1220 / 1200000000 / 1200000000 | 0 / 150 / 150 | 1.60e-5 | none | 23335 | 266 |
| nx8_nu3_N50 | tight | `qpoases` | 1/1 | 1390 | 959 - 1880 | 21400 / 0.230 / 1390 | 8.62 | 254000000 / 1300 / 16200000 / 16200000 | 0 / 9 / 22 | 3.14e-13 | none | 393 | 0 |
| nx8_nu3_N50 | tight | `daqp` | 1/1 | 443 | 319 - 608 | 1020 / 24.3 / 419 | 8.56 | 8420000 / 155000 / 4240000 / 4390000 | 3 / 19 / 163 | 2.12e-4 | none | 0 | 0 |
| nx8_nu3_N50 | tight | `proxqp_dense` | 49/150 | 7470 | 3720 - 10000 | 3130 / 31.3 / 7440 | 9.18 | 28100000 / 276000 / 54000000 / 54300000 | 0 / 40 / 160 | 1.00e+0 | inaccurate 50, infeasible_claim 152 | 17 | 0 |
| nx8_nu3_N50 | tight | `osqp` | 1/1 | 1630 | 1620 - 1650 | 2060 / 1.23 / 1630 | 8.70 | 17500000 / 11100 / 15700000 / 15700000 | 25 / 50 / 75 | 5.97e-4 | none | 0 | 0 |
| nx8_nu3_N50 | tight | `argmin_dense_admm_polished` | 14/15 | 3490 | 3130 - 3530 | 4250 / 0.200 / 3490 | 8.82 | 42000000 / 1180 / 31500000 / 31500000 | 25 / 50 / 175 | 5.78e-1 | inaccurate 20 | 8 | 0 |
| nx8_nu3_N100 | nominal | `argmin_dense_admm` | 1/1 | 965 | 961 - 971 | 4730 / 0.260 / 965 | 29.2 | 44400000 / 1380 / 4710000 / 4710000 | 25 / 25 / 50 | 5.09e-4 | none | 0 | 0 |
| nx8_nu3_N100 | nominal | `argmin_dense_active_set` | 1/1 | 144000 | 84100 - 226000 | 40900 / 0.270 / 144000 | 29.8 | 381000000 / 1360 / 1450000000 / 1450000000 | 0 / 42 / 179 | 3.49e-4 | none | 1937 | 0 |
| nx8_nu3_N100 | nominal | `qpoases` | 1/1 | 4410 | 2990 - 5780 | 6800 / 0.270 / 4410 | 29.1 | 50700000 / 1420 / 42400000 / 42400000 | 0 / 13 / 33 | 6.34e-13 | none | 313 | 0 |
| nx8_nu3_N100 | nominal | `daqp` | 1/1 | 321 | 235 - 392 | 2070 / 47.0 / 274 | 29.1 | 23500000 / 269000 / 1570000 / 1840000 | 2 / 14 / 34 | 4.34e-4 | none | 0 | 0 |
| nx8_nu3_N100 | nominal | `proxqp_dense` | 229/300 | 1140 | 961 - 1380 | 5140 / 79.7 / 1060 | 29.4 | 35400000 / 694000 / 7560000 / 8250000 | 0 / 8 / 17 | 1.72e-1 | inaccurate 71 | 23 | 0 |
| nx8_nu3_N100 | nominal | `osqp` | 1/1 | 2070 | 2060 - 2080 | 6680 / 1.33 / 2070 | 29.3 | 53800000 / 11700 / 17900000 / 17900000 | 25 / 25 / 50 | 5.09e-4 | none | 0 | 0 |
| nx8_nu3_N100 | nominal | `argmin_dense_admm_polished` | 1/1 | 5760 | 5740 - 5780 | 10600 / 0.270 / 5760 | 29.5 | 96400000 / 1370 / 56700000 / 56700000 | 25 / 25 / 50 | 5.91e-4 | none | 9 | 0 |
| nx8_nu3_N100 | tight | `argmin_dense_admm` | 1/1 | 2970 | 2940 - 2990 | 8420 / 0.330 / 2970 | 30.0 | 70000000 / 2140 / 13700000 / 13700000 | 25 / 50 / 50 | 5.72e-4 | none | 0 | 0 |
| nx8_nu3_N100 | tight | `argmin_dense_active_set` | 1/1 | 1680000 | 1660000 - 1690000 | 1680000 / 0.400 / 1680000 | 44.1 | 14400000000 / 2140 / 14200000000 / 14200000000 | 0 / 300 / 300 | 1.20e-5 | none | 92813 | 274 |
| nx8_nu3_N100 | tight | `qpoases` | 1/1 | 5960 | 4090 - 8020 | 168000 / 0.370 / 5960 | 30.0 | 1910000000 / 2190 / 66700000 / 66700000 | 0 / 9 / 22 | 2.97e-13 | none | 393 | 0 |
| nx8_nu3_N100 | tight | `daqp` | 1/1 | 1990 | 1440 - 2730 | 8130 / 115 / 1870 | 30.0 | 58700000 / 555000 / 15800000 / 16400000 | 3 / 19 / 313 | 2.12e-4 | none | 0 | 0 |
| nx8_nu3_N100 | tight | `proxqp_dense` | 97/300 | 29700 | 20700 - 51600 | 18400 / 112 / 29600 | 30.6 | 136000000 / 981000 / 212000000 / 213000000 | 0 / 41.5 / 301 | 1.00e+0 | inaccurate 43, infeasible_claim 160 | 307 | 0 |
| nx8_nu3_N100 | tight | `osqp` | 1/1 | 6270 | 6230 - 6390 | 12600 / 2.11 / 6270 | 30.1 | 84800000 / 21300 / 58200000 / 58200000 | 25 / 50 / 50 | 4.94e-4 | none | 0 | 0 |
| nx8_nu3_N100 | tight | `argmin_dense_admm_polished` | 23/25 | 17200 | 17100 - 18700 | 24200 / 0.340 / 17200 | 30.3 | 227000000 / 2120 / 171000000 / 171000000 | 25 / 50 / 250 | 5.78e-1 | inaccurate 24 | 21 | 0 |

### dense cohort, churning regime, high tier

The dense cohort in the churning regime, judged at the high tier's threshold of 1e-6.

| Cell | Variant | Arm | Pass rate | Median tick (us) | Tick p25 - p75 (us) | Median setup / update / solve (us) | Median condense (us) | Median instructions setup / update / solve / tick | Iterations min / median / max | Worst KKT residual | Failures | Allocations max | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nx2_nu1_N10 | nominal | `argmin_dense_admm` | 1/1 | 23.3 | 23.1 - 24.0 | 44.0 / 0.0300 / 23.2 | 0.160 | 288000 / 389 / 212000 / 212000 | 25 / 50 / 225 | 4.44e-16 | none | 0 | 0 |
| nx2_nu1_N10 | nominal | `argmin_dense_active_set` | 1/1 | 33.4 | 17.6 - 70.7 | 24.3 / 0.0400 / 33.4 | 0.170 | 118000 / 371 / 298000 / 299000 | 0 / 10 / 36 | 8.97e-16 | none | 195 | 0 |
| nx2_nu1_N10 | nominal | `qpoases` | 1/1 | 12.3 | 6.72 - 21.3 | 46.8 / 0.0400 / 12.2 | 0.160 | 272000 / 428 / 110000 / 110000 | 0 / 4 / 17 | 5.46e-16 | none | 326 | 0 |
| nx2_nu1_N10 | nominal | `daqp` | 1/1 | 1.04 | 0.818 - 1.51 | 3.99 / 0.220 / 0.820 | 0.150 | 13000 / 2320 / 7160 / 9580 | 1 / 5 / 18 | 2.52e-15 | none | 0 | 0 |
| nx2_nu1_N10 | nominal | `proxqp_dense` | 269/300 | 16.4 | 11.9 - 22.3 | 35.8 / 0.610 / 15.8 | 0.170 | 196000 / 5240 / 130000 / 135000 | 0 / 6 / 17 | 1.86e-1 | inaccurate 31 | 7 | 0 |
| nx2_nu1_N10 | nominal | `osqp` | 1/1 | 20.2 | 19.8 - 21.1 | 18.7 / 0.190 / 20.0 | 0.160 | 98600 / 1360 / 189000 / 191000 | 25 / 50 / 175 | 5.07e-7 | none | 60 | 0 |
| nx2_nu1_N10 | tight | `argmin_dense_admm` | 1/1 | 54.7 | 45.4 - 67.1 | 61.3 / 0.0300 / 54.6 | 0.200 | 409000 / 385 / 482000 / 482000 | 25 / 125 / 450 | 1.92e-16 | none | 0 | 0 |
| nx2_nu1_N10 | tight | `argmin_dense_active_set` | 1/1 | 39.3 | 30.5 - 43.3 | 66.5 / 0.0300 / 39.2 | 0.210 | 419000 / 386 / 380000 / 380000 | 0 / 10 / 10 | 3.85e-16 | none | 132 | 158 |
| nx2_nu1_N10 | tight | `qpoases` | 1/1 | 12.8 | 7.24 - 20.2 | 117 / 0.0400 / 12.7 | 0.200 | 851000 / 494 / 120000 / 120000 | 0 / 3 / 10 | 2.20e-16 | none | 236 | 0 |
| nx2_nu1_N10 | tight | `daqp` | 1/1 | 2.65 | 1.92 - 4.02 | 4.78 / 0.290 / 2.36 | 0.190 | 18900 / 3220 / 24400 / 27600 | 1 / 7 / 21 | 2.55e-15 | none | 0 | 0 |
| nx2_nu1_N10 | tight | `proxqp_dense` | 13/25 | 37.8 | 21.3 - 57.8 | 103 / 0.740 / 37.0 | 0.230 | 616000 / 6230 / 289000 / 295000 | 0 / 10 / 38 | 1.00e+0 | inaccurate 26, infeasible_claim 118 | 7 | 0 |
| nx2_nu1_N10 | tight | `osqp` | 1/1 | 59.8 | 49.7 - 77.8 | 26.2 / 0.230 / 59.5 | 0.200 | 140000 / 1610 / 600000 / 601000 | 50 / 125 / 325 | 5.70e-16 | none | 60 | 0 |
| nx4_nu2_N10 | nominal | `argmin_dense_admm` | 1/1 | 44.6 | 43.6 - 61.7 | 78.9 / 0.0300 / 44.6 | 0.240 | 679000 / 300 / 386000 / 386000 | 25 / 50 / 200 | 1.04e-15 | none | 0 | 0 |
| nx4_nu2_N10 | nominal | `argmin_dense_active_set` | 1/1 | 204 | 104 - 304 | 99.6 / 0.0300 / 204 | 0.270 | 788000 / 296 / 1960000 / 1960000 | 0 / 27.5 / 67 | 3.43e-16 | none | 435 | 0 |
| nx4_nu2_N10 | nominal | `qpoases` | 1/1 | 52.0 | 32.1 - 72.7 | 77.6 / 0.0400 / 52.0 | 0.240 | 598000 / 338 / 537000 / 538000 | 0 / 9 / 26 | 3.90e-16 | none | 501 | 0 |
| nx4_nu2_N10 | nominal | `daqp` | 1/1 | 3.16 | 2.41 - 4.39 | 7.88 / 0.400 / 2.76 | 0.220 | 40100 / 4420 / 22900 / 27300 | 1 / 10 / 27 | 4.22e-15 | none | 0 | 0 |
| nx4_nu2_N10 | nominal | `proxqp_dense` | 127/150 | 33.9 | 27.1 - 42.7 | 80.7 / 0.850 / 32.9 | 0.250 | 553000 / 7690 / 258000 / 266000 | 0 / 8 / 18 | 3.59e-1 | inaccurate 46 | 8 | 0 |
| nx4_nu2_N10 | nominal | `osqp` | 1/1 | 56.3 | 43.6 - 71.7 | 35.1 / 0.220 / 56.0 | 0.240 | 222000 / 1580 / 546000 / 548000 | 50 / 75 / 175 | 1.12e-8 | none | 60 | 0 |
| nx4_nu2_N10 | tight | `argmin_dense_admm` | 1/1 | 147 | 111 - 181 | 114 / 0.0400 / 146 | 0.330 | 915000 / 350 / 1240000 / 1250000 | 25 / 175 / 450 | 2.45e-16 | none | 0 | 0 |
| nx4_nu2_N10 | tight | `argmin_dense_active_set` | 1/1 | 205 | 174 - 224 | 265 / 0.0400 / 205 | 0.350 | 2330000 / 344 / 2050000 / 2050000 | 0 / 20 / 20 | 4.47e-16 | none | 462 | 204 |
| nx4_nu2_N10 | tight | `qpoases` | 1/1 | 42.7 | 27.7 - 62.5 | 401 / 0.0400 / 42.6 | 0.310 | 4170000 / 415 / 467000 / 467000 | 0 / 6 / 16 | 2.52e-16 | none | 349 | 0 |
| nx4_nu2_N10 | tight | `daqp` | 1/1 | 8.79 | 6.03 - 12.2 | 10.6 / 0.660 / 8.12 | 0.290 | 63300 / 6660 / 94300 / 101000 | 1 / 13 / 33 | 3.37e-15 | none | 0 | 0 |
| nx4_nu2_N10 | tight | `proxqp_dense` | 2/5 | 119 | 66.0 - 171 | 145 / 1.11 / 118 | 0.380 | 951000 / 10100 / 850000 / 860000 | 0 / 18.5 / 58 | 1.00e+0 | inaccurate 44, infeasible_claim 136 | 8 | 0 |
| nx4_nu2_N10 | tight | `osqp` | 1/1 | 152 | 130 - 179 | 55.1 / 0.300 / 152 | 0.310 | 352000 / 2220 / 1570000 / 1580000 | 75 / 150 / 400 | 9.24e-16 | none | 60 | 0 |
| nx4_nu2_N20 | nominal | `argmin_dense_admm` | 1/1 | 107 | 107 - 110 | 258 / 0.0500 / 107 | 0.590 | 1780000 / 370 / 961000 / 961000 | 25 / 50 / 175 | 2.07e-15 | none | 0 | 0 |
| nx4_nu2_N20 | nominal | `argmin_dense_active_set` | 1/1 | 734 | 359 - 1180 | 358 / 0.0500 / 734 | 0.640 | 2770000 / 369 / 7120000 / 7120000 | 0 / 29 / 126 | 5.32e-16 | none | 642 | 0 |
| nx4_nu2_N20 | nominal | `qpoases` | 1/1 | 177 | 110 - 258 | 407 / 0.0500 / 177 | 0.540 | 4420000 / 409 / 1920000 / 1920000 | 0 / 9 / 26 | 6.51e-16 | none | 501 | 0 |
| nx4_nu2_N20 | nominal | `daqp` | 1/1 | 5.42 | 4.20 - 7.28 | 19.3 / 0.920 / 4.46 | 0.500 | 157000 / 10300 / 44300 / 54600 | 1 / 10 / 27 | 4.37e-15 | none | 0 | 0 |
| nx4_nu2_N20 | nominal | `proxqp_dense` | 253/300 | 56.8 | 45.8 - 73.6 | 167 / 1.92 / 54.8 | 0.580 | 1250000 / 18900 / 462000 / 481000 | 0 / 8 / 19 | 3.59e-1 | inaccurate 47 | 8 | 0 |
| nx4_nu2_N20 | nominal | `osqp` | 1/1 | 132 | 131 - 175 | 87.6 / 0.310 / 132 | 0.550 | 669000 / 2270 / 1220000 / 1220000 | 50 / 50 / 175 | 3.32e-7 | none | 60 | 0 |
| nx4_nu2_N20 | tight | `argmin_dense_admm` | 1/1 | 332 | 292 - 380 | 306 / 0.0700 / 332 | 0.740 | 2800000 / 479 / 2860000 / 2860000 | 25 / 150 / 675 | 4.27e-16 | none | 0 | 0 |
| nx4_nu2_N20 | tight | `argmin_dense_active_set` | 1/1 | 1570 | 1500 - 1610 | 1670 / 0.0900 / 1570 | 0.790 | 16500000 / 469 / 15500000 / 15500000 | 0 / 40 / 40 | 4.55e-16 | none | 1723 | 226 |
| nx4_nu2_N20 | tight | `qpoases` | 1/1 | 153 | 97.1 - 230 | 2380 / 0.0700 / 153 | 0.650 | 30600000 / 540 / 2030000 / 2030000 | 0 / 6 / 18 | 4.78e-16 | none | 415 | 0 |
| nx4_nu2_N20 | tight | `daqp` | 1/1 | 24.2 | 17.2 - 35.5 | 34.4 / 1.71 / 22.3 | 0.570 | 281000 / 17200 / 285000 / 302000 | 1 / 13 / 47 | 2.95e-15 | none | 0 | 0 |
| nx4_nu2_N20 | tight | `proxqp_dense` | 127/300 | 448 | 350 - 567 | 384 / 2.86 / 445 | 0.800 | 3150000 / 25900 / 3350000 / 3380000 | 0 / 29 / 129 | 1.00e+0 | inaccurate 35, infeasible_claim 138 | 9 | 0 |
| nx4_nu2_N20 | tight | `osqp` | 1/1 | 445 | 414 - 577 | 149 / 0.450 / 444 | 0.670 | 1190000 / 3550 / 4490000 / 4490000 | 75 / 125 / 400 | 1.52e-15 | none | 60 | 0 |
| nx8_nu3_N20 | nominal | `argmin_dense_admm` | 1/1 | 216 | 215 - 225 | 425 / 0.0700 / 216 | 1.70 | 3580000 / 445 / 1950000 / 1950000 | 25 / 50 / 225 | 2.00e-15 | none | 0 | 0 |
| nx8_nu3_N20 | nominal | `argmin_dense_active_set` | 1/1 | 2570 | 1530 - 4010 | 778 / 0.0800 / 2570 | 1.79 | 7290000 / 444 / 24800000 / 24800000 | 0 / 42 / 180 | 6.56e-16 | none | 970 | 0 |
| nx8_nu3_N20 | nominal | `qpoases` | 1/1 | 579 | 400 - 790 | 1390 / 0.0800 / 579 | 1.57 | 16000000 / 481 / 6140000 / 6140000 | 0 / 13 / 33 | 1.03e-15 | none | 637 | 0 |
| nx8_nu3_N20 | nominal | `daqp` | 1/1 | 13.1 | 10.1 - 16.7 | 38.5 / 1.77 / 11.2 | 1.30 | 386000 / 18200 / 106000 / 124000 | 2 / 14 / 34 | 5.40e-15 | none | 0 | 0 |
| nx8_nu3_N20 | nominal | `proxqp_dense` | 257/300 | 116 | 97.3 - 149 | 228 / 3.74 / 112 | 1.65 | 2180000 / 35800 / 923000 / 959000 | 0 / 10 / 20 | 1.56e-1 | inaccurate 43 | 8 | 0 |
| nx8_nu3_N20 | nominal | `osqp` | 1/1 | 291 | 277 - 384 | 175 / 0.430 / 291 | 1.57 | 1390000 / 2990 / 2500000 / 2500000 | 50 / 50 / 225 | 5.99e-16 | none | 60 | 0 |
| nx8_nu3_N20 | tight | `argmin_dense_admm` | 1/1 | 714 | 560 - 863 | 781 / 0.0800 / 714 | 1.92 | 7030000 / 595 / 5770000 / 5770000 | 25 / 150 / 525 | 4.49e-7 | none | 0 | 0 |
| nx8_nu3_N20 | tight | `argmin_dense_active_set` | 1/1 | 5750 | 5510 - 5930 | 6070 / 0.140 / 5750 | 2.12 | 58200000 / 594 / 55700000 / 55700000 | 0 / 60 / 60 | 3.29e-16 | none | 3808 | 253 |
| nx8_nu3_N20 | tight | `qpoases` | 1/1 | 434 | 300 - 602 | 8240 / 0.100 / 434 | 1.72 | 112000000 / 665 / 6060000 / 6060000 | 0 / 9 / 22 | 5.08e-16 | none | 503 | 0 |
| nx8_nu3_N20 | tight | `daqp` | 1/1 | 71.4 | 51.9 - 97.4 | 86.6 / 3.90 / 67.4 | 1.54 | 748000 / 32100 / 801000 / 833000 | 3 / 19 / 73 | 3.22e-15 | none | 0 | 0 |
| nx8_nu3_N20 | tight | `proxqp_dense` | 109/300 | 1120 | 917 - 1390 | 757 / 5.62 / 1110 | 2.15 | 6490000 / 50000 / 8050000 / 8100000 | 0 / 36 / 115 | 1.00e+0 | inaccurate 33, infeasible_claim 158 | 8 | 0 |
| nx8_nu3_N20 | tight | `osqp` | 1/1 | 1010 | 868 - 1270 | 320 / 0.630 / 1010 | 1.92 | 2570000 / 4920 / 9900000 / 9900000 | 75 / 150 / 475 | 4.76e-7 | none | 60 | 0 |
| nx8_nu3_N30 | nominal | `argmin_dense_admm` | 1/1 | 475 | 474 - 482 | 843 / 0.100 / 475 | 3.54 | 7870000 / 598 / 4220000 / 4220000 | 25 / 50 / 200 | 2.81e-15 | none | 0 | 0 |
| nx8_nu3_N30 | nominal | `argmin_dense_active_set` | 1/1 | 6650 | 3950 - 10400 | 1940 / 0.120 / 6650 | 3.70 | 18200000 / 597 / 63700000 / 63700000 | 0 / 42 / 179 | 8.15e-16 | none | 965 | 0 |
| nx8_nu3_N30 | nominal | `qpoases` | 1/1 | 1680 | 1160 - 2260 | 5490 / 0.120 / 1680 | 3.53 | 60600000 / 637 / 16100000 / 16100000 | 0 / 13 / 33 | 8.70e-16 | none | 637 | 0 |
| nx8_nu3_N30 | nominal | `daqp` | 1/1 | 26.2 | 19.6 - 33.0 | 88.6 / 3.82 / 22.2 | 2.98 | 1010000 / 33900 / 190000 / 224000 | 2 / 14 / 34 | 4.83e-15 | none | 0 | 0 |
| nx8_nu3_N30 | nominal | `proxqp_dense` | 87/100 | 202 | 167 - 260 | 422 / 8.23 / 194 | 3.63 | 4360000 / 78300 / 1550000 / 1630000 | 0 / 10 / 20 | 1.56e-1 | inaccurate 39 | 8 | 0 |
| nx8_nu3_N30 | nominal | `osqp` | 1/1 | 633 | 625 - 830 | 395 / 0.600 / 632 | 3.54 | 3230000 / 4150 / 5600000 / 5600000 | 50 / 50 / 200 | 7.05e-16 | none | 60 | 0 |
| nx8_nu3_N30 | tight | `argmin_dense_admm` | 299/300 | 1570 | 1280 - 2160 | 1760 / 0.130 / 1570 | 3.82 | 15800000 / 796 / 12500000 / 12500000 | 25 / 150 / 525 | 8.53e-2 | inaccurate 1 | 0 | 0 |
| nx8_nu3_N30 | tight | `argmin_dense_active_set` | 1/1 | 23200 | 22500 - 23700 | 24000 / 0.170 / 23200 | 4.07 | 221000000 / 795 / 215000000 / 215000000 | 0 / 90 / 90 | 4.76e-16 | none | 8416 | 257 |
| nx8_nu3_N30 | tight | `qpoases` | 1/1 | 1050 | 730 - 1440 | 34600 / 0.160 / 1050 | 3.78 | 452000000 / 920 / 14300000 / 14300000 | 0 / 9 / 22 | 5.82e-16 | none | 503 | 0 |
| nx8_nu3_N30 | tight | `daqp` | 1/1 | 158 | 114 - 217 | 248 / 8.42 / 150 | 3.52 | 2140000 / 63800 / 1660000 / 1720000 | 3 / 19 / 103 | 2.81e-15 | none | 0 | 0 |
| nx8_nu3_N30 | tight | `proxqp_dense` | 47/150 | 2710 | 1820 - 3100 | 1500 / 12.3 / 2700 | 4.08 | 13000000 / 112000 / 20100000 / 20200000 | 0 / 35.5 / 114 | 1.00e+0 | inaccurate 50, infeasible_claim 156 | 8 | 0 |
| nx8_nu3_N30 | tight | `osqp` | 1/1 | 2440 | 1910 - 3080 | 712 / 0.880 / 2440 | 3.87 | 6000000 / 7000 / 24100000 / 24100000 | 75 / 175 / 500 | 4.68e-7 | none | 60 | 0 |
| nx12_nu4_N30 | nominal | `argmin_dense_admm` | 1/1 | 991 | 837 - 1000 | 1530 / 0.130 / 991 | 5.95 | 13500000 / 670 / 8430000 / 8430000 | 25 / 75 / 200 | 1.26e-15 | none | 0 | 0 |
| nx12_nu4_N30 | nominal | `argmin_dense_active_set` | 1/1 | 16800 | 11700 - 24500 | 3680 / 0.150 / 16800 | 6.11 | 35400000 / 669 / 163000000 / 163000000 | 0 / 57 / 205 | 9.27e-16 | none | 1152 | 0 |
| nx12_nu4_N30 | nominal | `qpoases` | 1/1 | 4900 | 3580 - 6290 | 15300 / 0.150 / 4900 | 6.06 | 158000000 / 709 / 41900000 / 41900000 | 0 / 17 / 41 | 1.03e-15 | none | 786 | 0 |
| nx12_nu4_N30 | nominal | `daqp` | 1/1 | 54.8 | 42.5 - 70.7 | 172 / 6.65 / 48.1 | 5.72 | 2060000 / 53900 / 380000 / 434000 | 3 / 18 / 42 | 6.16e-15 | none | 0 | 0 |
| nx12_nu4_N30 | nominal | `proxqp_dense` | 281/300 | 355 | 299 - 436 | 675 / 13.3 / 340 | 6.24 | 6860000 / 121000 / 2630000 / 2750000 | 0 / 12 / 22 | 1.32e-1 | inaccurate 19 | 8 | 0 |
| nx12_nu4_N30 | nominal | `osqp` | 1/1 | 1420 | 1110 - 1440 | 726 / 0.760 / 1420 | 6.11 | 5970000 / 5180 / 12800000 / 12800000 | 50 / 75 / 175 | 8.34e-16 | none | 60 | 0 |
| nx12_nu4_N30 | tight | `argmin_dense_admm` | 299/300 | 3020 | 2770 - 3730 | 4290 / 0.160 / 3020 | 6.32 | 29900000 / 970 / 23600000 / 23600000 | 25 / 175 / 525 | 2.47e-1 | inaccurate 1 | 8 | 0 |
| nx12_nu4_N30 | tight | `argmin_dense_active_set` | 1/1 | 60700 | 59200 - 61800 | 63300 / 0.190 / 60700 | 6.61 | 552000000 / 969 / 541000000 / 541000000 | 0 / 120 / 120 | 4.42e-16 | none | 14851 | 272 |
| nx12_nu4_N30 | tight | `qpoases` | 1/1 | 2390 | 1780 - 3150 | 107000 / 0.200 / 2390 | 6.31 | 1350000000 / 1040 / 31200000 / 31200000 | 0 / 11 / 32 | 7.82e-16 | none | 679 | 0 |
| nx12_nu4_N30 | tight | `daqp` | 1/1 | 334 | 255 - 459 | 540 / 14.8 / 319 | 6.21 | 4550000 / 103000 / 3430000 / 3530000 | 5 / 23 / 135 | 3.17e-15 | none | 0 | 0 |
| nx12_nu4_N30 | tight | `proxqp_dense` | 7/25 | 5590 | 4730 - 6310 | 2420 / 19.4 / 5570 | 6.68 | 22600000 / 173000 / 40300000 / 40500000 | 0 / 45 / 171 | 1.00e+0 | inaccurate 39, infeasible_claim 177 | 8 | 0 |
| nx12_nu4_N30 | tight | `osqp` | 1/1 | 4180 | 3240 - 5630 | 1460 / 1.09 / 4180 | 6.51 | 10200000 / 9020 / 40600000 / 40600000 | 75 / 175 / 475 | 4.60e-7 | none | 60 | 0 |
| nx4_nu2_N50 | nominal | `argmin_dense_admm` | 1/1 | 574 | 571 - 585 | 878 / 0.100 / 574 | 2.86 | 8480000 / 595 / 5050000 / 5050000 | 25 / 50 / 175 | 9.62e-16 | none | 0 | 0 |
| nx4_nu2_N50 | nominal | `argmin_dense_active_set` | 1/1 | 5940 | 2970 - 9320 | 2380 / 0.110 / 5940 | 2.98 | 23000000 / 594 / 58100000 / 58100000 | 0 / 29 / 129 | 8.76e-16 | none | 727 | 0 |
| nx4_nu2_N50 | nominal | `qpoases` | 1/1 | 1560 | 974 - 2380 | 7490 / 0.120 / 1560 | 2.88 | 81900000 / 631 / 14200000 / 14200000 | 0 / 9 / 26 | 1.21e-15 | none | 501 | 0 |
| nx4_nu2_N50 | nominal | `daqp` | 1/1 | 25.1 | 18.1 - 31.5 | 114 / 4.66 / 20.1 | 2.47 | 1310000 / 40000 / 166000 / 206000 | 1 / 10 / 27 | 4.08e-15 | none | 0 | 0 |
| nx4_nu2_N50 | nominal | `proxqp_dense` | 127/150 | 178 | 143 - 223 | 497 / 9.41 / 169 | 2.96 | 5100000 / 87100 / 1330000 / 1410000 | 0 / 8 / 19 | 3.59e-1 | inaccurate 46 | 8 | 0 |
| nx4_nu2_N50 | nominal | `osqp` | 1/1 | 797 | 791 - 1030 | 502 / 0.630 / 797 | 2.89 | 4150000 / 4460 / 7090000 / 7100000 | 50 / 50 / 175 | 1.87e-7 | none | 60 | 0 |
| nx4_nu2_N50 | tight | `argmin_dense_admm` | 1/1 | 1900 | 1550 - 2440 | 1810 / 0.130 / 1900 | 3.02 | 17200000 / 845 / 15200000 / 15200000 | 25 / 150 / 475 | 3.73e-16 | none | 3 | 0 |
| nx4_nu2_N50 | tight | `argmin_dense_active_set` | 1/1 | 32700 | 32100 - 33200 | 33400 / 0.170 / 32700 | 3.29 | 304000000 / 844 / 297000000 / 297000000 | 0 / 100 / 100 | 4.25e-16 | none | 10376 | 237 |
| nx4_nu2_N50 | tight | `qpoases` | 1/1 | 997 | 640 - 1480 | 49000 / 0.160 / 997 | 3.02 | 630000000 / 915 / 13300000 / 13300000 | 0 / 6 / 18 | 8.44e-16 | none | 415 | 0 |
| nx4_nu2_N50 | tight | `daqp` | 1/1 | 138 | 99.6 - 201 | 326 / 10.3 / 127 | 2.90 | 2790000 / 75200 / 1380000 / 1460000 | 1 / 13 / 107 | 2.97e-15 | none | 0 | 0 |
| nx4_nu2_N50 | tight | `proxqp_dense` | 28/75 | 3060 | 1950 - 3530 | 1080 / 14.0 / 3050 | 3.30 | 10200000 / 124000 / 22000000 / 22100000 | 0 / 32.5 / 118 | 1.00e+0 | inaccurate 41, infeasible_claim 147 | 8 | 0 |
| nx4_nu2_N50 | tight | `osqp` | 1/1 | 2890 | 2500 - 3340 | 912 / 0.950 / 2890 | 3.10 | 7880000 / 7660 / 28300000 / 28300000 | 75 / 150 / 450 | 2.27e-15 | none | 60 | 0 |
| nx4_nu2_N100 | nominal | `argmin_dense_admm` | 1/1 | 2550 | 2540 - 2570 | 6080 / 0.190 / 2550 | 9.76 | 38300000 / 995 / 22900000 / 22900000 | 25 / 50 / 175 | 3.38e-16 | none | 4 | 0 |
| nx4_nu2_N100 | nominal | `argmin_dense_active_set` | 1/1 | 33900 | 16600 - 53700 | 15400 / 0.210 / 33900 | 9.96 | 128000000 / 992 / 335000000 / 335000000 | 0 / 29 / 129 | 5.29e-16 | none | 1307 | 0 |
| nx4_nu2_N100 | nominal | `qpoases` | 1/1 | 11200 | 6760 - 16800 | 95700 / 0.210 / 11200 | 9.69 | 837000000 / 1040 / 76400000 / 76400000 | 0 / 9 / 26 | 1.29e-15 | none | 501 | 0 |
| nx4_nu2_N100 | nominal | `daqp` | 1/1 | 102 | 71.3 - 130 | 648 / 18.9 / 79.9 | 9.58 | 7800000 / 130000 / 531000 / 660000 | 1 / 10 / 27 | 4.44e-15 | none | 0 | 0 |
| nx4_nu2_N100 | nominal | `proxqp_dense` | 127/150 | 526 | 423 - 661 | 2170 / 35.0 / 490 | 9.88 | 17700000 / 319000 / 3650000 / 3970000 | 0 / 8 / 19 | 3.59e-1 | inaccurate 46 | 8 | 0 |
| nx4_nu2_N100 | nominal | `osqp` | 1/1 | 3440 | 3410 - 4350 | 2780 / 1.03 / 3440 | 9.90 | 20300000 / 8080 / 31200000 / 31200000 | 50 / 50 / 175 | 1.82e-7 | none | 60 | 0 |
| nx4_nu2_N100 | tight | `argmin_dense_admm` | 1/1 | 8900 | 8160 - 10200 | 12100 / 0.260 / 8900 | 10.4 | 86300000 / 1500 / 72100000 / 72100000 | 25 / 150 / 475 | 2.78e-16 | none | 15 | 0 |
| nx4_nu2_N100 | tight | `argmin_dense_active_set` | 1/1 | 372000 | 368000 - 375000 | 376000 / 0.280 / 372000 | 10.7 | 3290000000 / 1500 / 3250000000 / 3250000000 | 0 / 200 / 200 | 4.65e-16 | none | 41473 | 242 |
| nx4_nu2_N100 | tight | `qpoases` | 1/1 | 4480 | 2950 - 6580 | 802000 / 0.280 / 4480 | 10.1 | 8440000000 / 1540 / 54300000 / 54300000 | 0 / 6 / 18 | 1.11e-15 | none | 415 | 0 |
| nx4_nu2_N100 | tight | `daqp` | 1/1 | 595 | 429 - 848 | 2320 / 46.7 / 547 | 10.1 | 18600000 / 260000 / 5060000 / 5320000 | 1 / 13 / 207 | 3.47e-15 | none | 0 | 0 |
| nx4_nu2_N100 | tight | `proxqp_dense` | 1/3 | 13800 | 7930 - 18300 | 5980 / 50.6 / 13700 | 10.6 | 37000000 / 451000 / 96100000 / 96600000 | 0 / 28 / 10011 | 1.00e+0 | inaccurate 33, infeasible_claim 167 | 9574 | 0 |
| nx4_nu2_N100 | tight | `osqp` | 1/1 | 11400 | 9780 - 14700 | 5250 / 1.54 / 11400 | 10.4 | 34100000 / 14500 / 109000000 / 109000000 | 75 / 150 / 400 | 2.72e-15 | none | 60 | 0 |
| nx8_nu3_N50 | nominal | `argmin_dense_admm` | 1/1 | 1360 | 1340 - 1600 | 2380 / 0.150 / 1360 | 8.21 | 22100000 / 848 / 12200000 / 12200000 | 25 / 50 / 225 | 3.78e-16 | none | 1 | 0 |
| nx8_nu3_N50 | nominal | `argmin_dense_active_set` | 1/1 | 23000 | 13600 - 36000 | 6420 / 0.190 / 23000 | 8.40 | 61700000 / 822 / 223000000 / 223000000 | 0 / 42 / 179 | 2.77e-16 | none | 1317 | 0 |
| nx8_nu3_N50 | nominal | `qpoases` | 1/1 | 7110 | 4880 - 9500 | 33000 / 0.180 / 7110 | 8.18 | 321000000 / 862 / 56700000 / 56700000 | 0 / 13 / 33 | 8.51e-16 | none | 637 | 0 |
| nx8_nu3_N50 | nominal | `daqp` | 1/1 | 69.8 | 51.7 - 85.8 | 305 / 10.4 / 59.5 | 8.08 | 3660000 / 78600 / 444000 / 523000 | 2 / 14 / 34 | 5.77e-15 | none | 0 | 0 |
| nx8_nu3_N50 | nominal | `proxqp_dense` | 259/300 | 431 | 362 - 541 | 1040 / 21.6 / 410 | 8.36 | 10600000 / 196000 / 3070000 / 3270000 | 0 / 10 / 20 | 1.56e-1 | inaccurate 41 | 8 | 0 |
| nx8_nu3_N50 | nominal | `osqp` | 1/1 | 1830 | 1810 - 2340 | 1260 / 0.890 / 1820 | 8.29 | 10300000 / 6340 / 16500000 / 16500000 | 50 / 50 / 200 | 7.42e-16 | none | 60 | 0 |
| nx8_nu3_N50 | tight | `argmin_dense_admm` | 1/1 | 5260 | 4300 - 6350 | 5490 / 0.200 / 5260 | 8.80 | 48300000 / 1170 / 40900000 / 40900000 | 25 / 175 / 550 | 4.52e-7 | none | 10 | 0 |
| nx8_nu3_N50 | tight | `argmin_dense_active_set` | 1/1 | 135000 | 133000 - 137000 | 137000 / 0.260 / 135000 | 9.23 | 1220000000 / 1220 / 1200000000 / 1200000000 | 0 / 150 / 150 | 3.63e-16 | none | 23335 | 266 |
| nx8_nu3_N50 | tight | `qpoases` | 1/1 | 3350 | 2330 - 4550 | 239000 / 0.230 / 3350 | 8.64 | 2850000000 / 1300 / 41500000 / 41500000 | 0 / 9 / 22 | 7.84e-16 | none | 503 | 0 |
| nx8_nu3_N50 | tight | `daqp` | 1/1 | 448 | 321 - 613 | 1020 / 24.3 / 424 | 8.56 | 8420000 / 155000 / 4240000 / 4390000 | 3 / 19 / 163 | 3.36e-15 | none | 0 | 0 |
| nx8_nu3_N50 | tight | `proxqp_dense` | 23/75 | 7760 | 5640 - 10400 | 3510 / 31.2 / 7730 | 9.16 | 31500000 / 276000 / 55300000 / 55600000 | 0 / 41.5 / 169 | 1.00e+0 | inaccurate 43, infeasible_claim 165 | 14 | 0 |
| nx8_nu3_N50 | tight | `osqp` | 1/1 | 6950 | 5420 - 9170 | 2060 / 1.28 / 6950 | 8.82 | 17500000 / 11100 / 67400000 / 67400000 | 75 / 175 / 425 | 4.48e-7 | none | 60 | 0 |
| nx8_nu3_N100 | nominal | `argmin_dense_admm` | 1/1 | 6710 | 6670 - 7630 | 11500 / 0.270 / 6710 | 29.5 | 105000000 / 1370 / 61300000 / 61300000 | 25 / 50 / 175 | 1.55e-15 | none | 11 | 0 |
| nx8_nu3_N100 | nominal | `argmin_dense_active_set` | 1/1 | 144000 | 84100 - 226000 | 38200 / 0.270 / 144000 | 29.8 | 381000000 / 1370 / 1450000000 / 1450000000 | 0 / 42 / 179 | 4.73e-16 | none | 1937 | 0 |
| nx8_nu3_N100 | nominal | `qpoases` | 1/1 | 55000 | 37300 - 72900 | 431000 / 0.280 / 55000 | 29.4 | 3270000000 / 1410 / 328000000 / 328000000 | 0 / 13 / 33 | 1.41e-15 | none | 637 | 0 |
| nx8_nu3_N100 | nominal | `daqp` | 1/1 | 321 | 235 - 392 | 2070 / 47.1 / 274 | 29.1 | 23500000 / 269000 / 1570000 / 1840000 | 2 / 14 / 34 | 5.70e-15 | none | 0 | 0 |
| nx8_nu3_N100 | nominal | `proxqp_dense` | 43/50 | 1300 | 1090 - 1640 | 3990 / 79.8 / 1220 | 29.5 | 38000000 / 694000 / 8580000 / 9280000 | 0 / 10 / 20 | 1.56e-1 | inaccurate 42 | 25 | 0 |
| nx8_nu3_N100 | nominal | `osqp` | 1/1 | 10600 | 8760 - 10800 | 6660 / 1.36 / 10600 | 29.4 | 53800000 / 11700 / 96500000 / 96600000 | 50 / 75 / 200 | 8.05e-16 | none | 60 | 0 |
| nx8_nu3_N100 | tight | `argmin_dense_admm` | 299/300 | 24300 | 21400 - 28700 | 27400 / 0.340 / 24300 | 30.2 | 254000000 / 2120 / 205000000 / 205000000 | 25 / 175 / 550 | 8.51e-2 | inaccurate 1 | 23 | 0 |
| nx8_nu3_N100 | tight | `argmin_dense_active_set` | 1/1 | 1690000 | 1670000 - 1710000 | 1680000 / 0.400 / 1690000 | 46.2 | 14400000000 / 2140 / 14200000000 / 14200000000 | 0 / 300 / 300 | 4.82e-16 | none | 92813 | 274 |
| nx8_nu3_N100 | tight | `qpoases` | 1/1 | 14600 | 10000 - 19500 | 4460000 / 0.360 / 14600 | 30.1 | 38200000000 / 2190 / 174000000 / 174000000 | 0 / 9 / 22 | 1.48e-15 | none | 503 | 0 |
| nx8_nu3_N100 | tight | `daqp` | 1/1 | 1990 | 1440 - 2720 | 8130 / 115 / 1880 | 30.0 | 58700000 / 555000 / 15900000 / 16400000 | 3 / 19 / 313 | 4.31e-15 | none | 0 | 0 |
| nx8_nu3_N100 | tight | `proxqp_dense` | 89/300 | 37000 | 21400 - 56700 | 19300 / 115 / 36900 | 30.6 | 153000000 / 981000 / 272000000 / 273000000 | 0 / 44 / 10028 | 1.00e+0 | inaccurate 37, infeasible_claim 174 | 13047 | 0 |
| nx8_nu3_N100 | tight | `osqp` | 1/1 | 29500 | 23400 - 35000 | 12500 / 2.13 / 29500 | 30.3 | 84800000 / 21300 / 275000000 / 275000000 | 75 / 175 / 550 | 4.34e-7 | none | 60 | 0 |

### sparse cohort, fixed regime, mpc tier

The sparse cohort on the non-condensed problem, each arm re-solving identical data 60 times after
20 untimed warm resolves, judged at the mpc tier's threshold of 1e-3.

| Cell | Variant | Arm | Pass rate | Median tick (us) | Tick p25 - p75 (us) | Median setup / update / solve (us) | Median instructions setup / update / solve / tick | Iterations min / median / max | Worst KKT residual | Failures | Allocations max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nx2_nu1_N10 | nominal | `argmin_sparse_admm` | 1/1 | 17.8 | 17.7 - 17.9 | 91.9 / 0.0400 / 17.8 | 916000 / 389 / 232000 / 232000 | 25 / 25 / 25 | 4.69e-13 | none | 0 |
| nx2_nu1_N10 | nominal | `osqp` | 1/1 | 19.8 | 19.7 - 19.8 | 28.7 / 0.250 / 19.5 | 218000 / 2030 / 153000 / 155000 | 25 / 25 / 25 | 1.39e-13 | none | 0 |
| nx2_nu1_N10 | nominal | `proxqp_sparse` | 1/1 | 12.3 | 12.2 - 12.4 | 127 / 5.67 / 6.60 | 981000 / 56000 / 60800 / 117000 | 0 / 0 / 0 | 1.13e-4 | none | 23 |
| nx2_nu1_N10 | nominal | `hpipm` | 1/1 | 32.3 | 32.2 - 32.4 | 31.7 / 0.0500 / 32.2 | 64000 / 335 / 368000 / 368000 | 5 / 5 / 5 | 1.02e-8 | none | 0 |
| nx2_nu1_N10 | nominal | `argmin_sparse_admm_polished` | 1/1 | 22.0 | 21.9 - 22.3 | 130 / 0.0400 / 22.0 | 1150000 / 391 / 283000 / 283000 | 25 / 25 / 25 | 8.33e-17 | none | 0 |
| nx2_nu1_N10 | nominal | `osqp_polished` | 1/1 | 34.3 | 34.1 - 34.5 | 33.0 / 0.260 / 34.0 | 222000 / 2020 / 285000 / 287000 | 25 / 25 / 25 | 1.33e-16 | none | 60 |
| nx2_nu1_N10 | tight | `argmin_sparse_admm` | 1/1 | 18.0 | 18.0 - 18.1 | 53.4 / 0.0400 / 18.0 | 500000 / 392 / 269000 / 269000 | 25 / 25 / 25 | 1.56e-14 | none | 0 |
| nx2_nu1_N10 | tight | `osqp` | 1/1 | 20.9 | 20.8 - 21.1 | 30.7 / 0.280 / 20.6 | 232000 / 2360 / 173000 / 175000 | 25 / 25 / 25 | 1.03e-14 | none | 0 |
| nx2_nu1_N10 | tight | `proxqp_sparse` | 1/1 | 13.7 | 13.5 - 13.8 | 80.2 / 6.35 / 7.31 | 540000 / 62200 / 67700 / 130000 | 0 / 0 / 0 | 2.55e-4 | none | 23 |
| nx2_nu1_N10 | tight | `hpipm` | 1/1 | 27.5 | 27.4 - 27.7 | 15.0 / 0.0500 / 27.5 | 68400 / 335 / 310000 / 310000 | 4 / 4 / 4 | 1.26e-7 | none | 0 |
| nx2_nu1_N10 | tight | `argmin_sparse_admm_polished` | 1/1 | 22.2 | 22.0 - 22.4 | 92.3 / 0.0400 / 22.1 | 723000 / 395 / 320000 / 321000 | 25 / 25 / 25 | 4.44e-17 | none | 0 |
| nx2_nu1_N10 | tight | `osqp_polished` | 1/1 | 35.2 | 35.0 - 35.5 | 34.5 / 0.300 / 34.9 | 238000 / 2370 / 305000 / 308000 | 25 / 25 / 25 | 1.33e-16 | none | 60 |
| nx4_nu2_N10 | nominal | `argmin_sparse_admm` | 1/1 | 42.8 | 42.5 - 44.2 | 210 / 0.0500 / 42.8 | 2190000 / 459 / 545000 / 545000 | 25 / 25 / 25 | 1.67e-12 | none | 0 |
| nx4_nu2_N10 | nominal | `osqp` | 1/1 | 40.4 | 40.2 - 40.5 | 63.6 / 0.380 / 40.0 | 504000 / 3140 / 368000 / 372000 | 25 / 25 / 25 | 2.31e-13 | none | 0 |
| nx4_nu2_N10 | nominal | `proxqp_sparse` | 1/1 | 24.9 | 24.7 - 25.1 | 250 / 10.6 / 14.2 | 2300000 / 109000 / 139000 / 248000 | 0 / 0 / 0 | 8.87e-5 | none | 23 |
| nx4_nu2_N10 | nominal | `hpipm` | 1/1 | 34.2 | 34.0 - 34.5 | 26.5 / 0.0500 / 34.2 | 74400 / 352 / 364000 / 364000 | 4 / 4 / 4 | 2.16e-5 | none | 0 |
| nx4_nu2_N10 | nominal | `argmin_sparse_admm_polished` | 1/1 | 52.6 | 52.4 - 52.8 | 305 / 0.0500 / 52.5 | 2740000 / 459 / 666000 / 666000 | 25 / 25 / 25 | 1.39e-16 | none | 0 |
| nx4_nu2_N10 | nominal | `osqp_polished` | 1/1 | 75.2 | 74.6 - 75.9 | 63.7 / 0.390 / 74.8 | 510000 / 3140 / 697000 / 700000 | 25 / 25 / 25 | 2.02e-16 | none | 60 |
| nx4_nu2_N10 | tight | `argmin_sparse_admm` | 1/1 | 45.1 | 44.9 - 45.3 | 122 / 0.0500 / 45.0 | 1170000 / 509 / 605000 / 605000 | 25 / 25 / 25 | 8.94e-15 | none | 0 |
| nx4_nu2_N10 | tight | `osqp` | 1/1 | 43.5 | 43.4 - 43.8 | 64.0 / 0.420 / 43.1 | 535000 / 3800 / 403000 / 407000 | 25 / 25 / 25 | 9.58e-15 | none | 0 |
| nx4_nu2_N10 | tight | `proxqp_sparse` | 1/1 | 29.1 | 28.9 - 29.8 | 199 / 11.8 / 17.3 | 1600000 / 121000 / 171000 / 292000 | 0 / 0 / 0 | 6.90e-5 | none | 23 |
| nx4_nu2_N10 | tight | `hpipm` | 1/1 | 52.1 | 51.9 - 52.5 | 21.7 / 0.0500 / 52.1 | 80000 / 352 / 558000 / 558000 | 6 / 6 / 6 | 2.11e-8 | none | 0 |
| nx4_nu2_N10 | tight | `argmin_sparse_admm_polished` | 1/1 | 53.8 | 53.7 - 54.2 | 202 / 0.0500 / 53.7 | 1740000 / 509 / 713000 / 714000 | 25 / 25 / 25 | 4.44e-17 | none | 0 |
| nx4_nu2_N10 | tight | `osqp_polished` | 1/1 | 78.1 | 77.5 - 79.0 | 67.4 / 0.440 / 77.7 | 538000 / 3780 / 733000 / 737000 | 25 / 25 / 25 | 1.78e-16 | none | 60 |
| nx4_nu2_N20 | nominal | `argmin_sparse_admm` | 1/1 | 82.6 | 82.4 - 82.8 | 386 / 0.0800 / 82.5 | 4210000 / 684 / 1050000 / 1050000 | 25 / 25 / 25 | 2.12e-12 | none | 0 |
| nx4_nu2_N20 | nominal | `osqp` | 1/1 | 79.4 | 79.2 - 79.5 | 102 / 0.590 / 78.8 | 956000 / 5320 / 699000 / 704000 | 25 / 25 / 25 | 1.72e-13 | none | 0 |
| nx4_nu2_N20 | nominal | `proxqp_sparse` | 1/1 | 46.8 | 46.6 - 47.1 | 537 / 19.1 / 27.7 | 5160000 / 205000 / 276000 / 480000 | 0 / 0 / 0 | 8.87e-5 | none | 23 |
| nx4_nu2_N20 | nominal | `hpipm` | 1/1 | 66.0 | 65.8 - 66.1 | 56.7 / 0.0600 / 65.9 | 135000 / 352 / 706000 / 707000 | 4 / 4 / 4 | 4.30e-5 | none | 0 |
| nx4_nu2_N20 | nominal | `argmin_sparse_admm_polished` | 1/1 | 102 | 102 - 104 | 548 / 0.0900 / 102 | 5240000 / 684 / 1280000 / 1280000 | 25 / 25 / 25 | 2.30e-16 | none | 0 |
| nx4_nu2_N20 | nominal | `osqp_polished` | 1/1 | 143 | 143 - 144 | 104 / 0.620 / 143 | 955000 / 5320 / 1320000 / 1320000 | 25 / 25 / 25 | 3.07e-16 | none | 60 |
| nx4_nu2_N20 | tight | `argmin_sparse_admm` | 1/1 | 88.4 | 88.2 - 88.8 | 219 / 0.100 / 88.3 | 2260000 / 784 / 1170000 / 1170000 | 25 / 25 / 25 | 1.02e-14 | none | 0 |
| nx4_nu2_N20 | tight | `osqp` | 1/1 | 85.4 | 84.9 - 85.6 | 109 / 0.730 / 84.7 | 1020000 / 6600 / 768000 / 775000 | 25 / 25 / 25 | 1.10e-14 | none | 0 |
| nx4_nu2_N20 | tight | `proxqp_sparse` | 1/1 | 77.2 | 77.0 - 78.2 | 565 / 21.9 / 55.3 | 5100000 / 230000 / 561000 / 791000 | 0 / 0 / 0 | 2.84e-4 | none | 23 |
| nx4_nu2_N20 | tight | `hpipm` | 1/1 | 101 | 101 - 102 | 18.8 / 0.0600 / 101 | 147000 / 352 / 1080000 / 1080000 | 6 / 6 / 6 | 3.59e-8 | none | 0 |
| nx4_nu2_N20 | tight | `argmin_sparse_admm_polished` | 1/1 | 104 | 104 - 106 | 351 / 0.110 / 104 | 3350000 / 812 / 1380000 / 1380000 | 25 / 25 / 25 | 4.44e-17 | none | 0 |
| nx4_nu2_N20 | tight | `osqp_polished` | 1/1 | 153 | 152 - 153 | 112 / 0.760 / 152 | 1020000 / 6600 / 1410000 / 1420000 | 25 / 25 / 25 | 1.78e-16 | none | 60 |
| nx8_nu3_N20 | nominal | `argmin_sparse_admm` | 1/1 | 251 | 247 - 252 | 1090 / 0.160 / 250 | 11000000 / 1070 / 2680000 / 2680000 | 25 / 25 / 25 | 1.37e-12 | none | 0 |
| nx8_nu3_N20 | nominal | `osqp` | 1/1 | 209 | 207 - 214 | 248 / 1.06 / 208 | 2480000 / 9080 / 1960000 / 1970000 | 25 / 25 / 25 | 1.18e-13 | none | 0 |
| nx8_nu3_N20 | nominal | `proxqp_sparse` | 1/1 | 102 | 101 - 103 | 1540 / 35.6 / 66.1 | 15100000 / 381000 / 644000 / 1030000 | 0 / 0 / 0 | 8.76e-5 | none | 23 |
| nx8_nu3_N20 | nominal | `hpipm` | 1/1 | 99.8 | 99.6 - 99.9 | 22.1 / 0.0900 / 99.7 | 170000 / 365 / 991000 / 992000 | 4 / 4 / 4 | 1.08e-5 | none | 0 |
| nx8_nu3_N20 | nominal | `argmin_sparse_admm_polished` | 1/1 | 303 | 300 - 305 | 1470 / 0.170 / 303 | 13900000 / 1070 / 3290000 / 3290000 | 25 / 25 / 25 | 1.75e-16 | none | 0 |
| nx8_nu3_N20 | nominal | `osqp_polished` | 1/1 | 392 | 389 - 396 | 252 / 1.06 / 391 | 2480000 / 9080 / 3780000 / 3790000 | 25 / 25 / 25 | 1.75e-16 | none | 60 |
| nx8_nu3_N20 | tight | `argmin_sparse_admm` | 1/1 | 254 | 253 - 255 | 2370 / 0.180 / 253 | 25700000 / 1220 / 2860000 / 2860000 | 25 / 25 / 25 | 9.28e-14 | none | 0 |
| nx8_nu3_N20 | tight | `osqp` | 1/1 | 217 | 216 - 221 | 261 / 1.24 / 216 | 2580000 / 11000 / 2060000 / 2070000 | 25 / 25 / 25 | 1.02e-13 | none | 0 |
| nx8_nu3_N20 | tight | `proxqp_sparse` | 1/1 | 341 | 338 - 346 | 5210 / 39.4 / 300 | 54200000 / 418000 / 3110000 / 3530000 | 0 / 0 / 0 | 3.35e-4 | none | 23 |
| nx8_nu3_N20 | tight | `hpipm` | 1/1 | 153 | 152 - 154 | 23.1 / 0.0900 / 153 | 183000 / 365 / 1510000 / 1510000 | 6 / 6 / 6 | 5.29e-6 | none | 0 |
| nx8_nu3_N20 | tight | `argmin_sparse_admm_polished` | 1/1 | 300 | 299 - 301 | 2770 / 0.180 / 299 | 28700000 / 1220 / 3410000 / 3410000 | 25 / 25 / 25 | 1.11e-16 | none | 0 |
| nx8_nu3_N20 | tight | `osqp_polished` | 1/1 | 400 | 397 - 402 | 263 / 1.24 / 398 | 2580000 / 11000 / 3870000 / 3880000 | 25 / 25 / 25 | 1.78e-16 | none | 60 |
| nx8_nu3_N30 | nominal | `argmin_sparse_admm` | 1/1 | 371 | 368 - 374 | 1700 / 0.210 / 370 | 16300000 / 1530 / 4000000 / 4000000 | 25 / 25 / 25 | 1.71e-12 | none | 0 |
| nx8_nu3_N30 | nominal | `osqp` | 1/1 | 317 | 309 - 319 | 387 / 1.41 / 315 | 3670000 / 13100 / 2910000 / 2930000 | 25 / 25 / 25 | 1.34e-13 | none | 0 |
| nx8_nu3_N30 | nominal | `proxqp_sparse` | 1/1 | 151 | 150 - 155 | 2950 / 51.4 / 99.9 | 29000000 / 559000 / 973000 / 1530000 | 0 / 0 / 0 | 8.76e-5 | none | 23 |
| nx8_nu3_N30 | nominal | `hpipm` | 1/1 | 148 | 147 - 148 | 114 / 0.0900 / 148 | 249000 / 365 / 1470000 / 1470000 | 4 / 4 / 4 | 1.13e-5 | none | 0 |
| nx8_nu3_N30 | nominal | `argmin_sparse_admm_polished` | 1/1 | 452 | 448 - 455 | 2240 / 0.210 / 452 | 20700000 / 1530 / 4900000 / 4900000 | 25 / 25 / 25 | 3.15e-16 | none | 0 |
| nx8_nu3_N30 | nominal | `osqp_polished` | 1/1 | 580 | 575 - 587 | 389 / 1.41 / 579 | 3670000 / 13100 / 5610000 / 5620000 | 25 / 25 / 25 | 3.35e-16 | none | 60 |
| nx8_nu3_N30 | tight | `argmin_sparse_admm` | 1/1 | 381 | 380 - 382 | 3680 / 0.230 / 381 | 38300000 / 1720 / 4270000 / 4270000 | 25 / 25 / 25 | 1.09e-13 | none | 0 |
| nx8_nu3_N30 | tight | `osqp` | 1/1 | 335 | 326 - 340 | 479 / 1.62 / 333 | 3810000 / 16000 / 3070000 / 3080000 | 25 / 25 / 25 | 1.01e-13 | none | 0 |
| nx8_nu3_N30 | tight | `proxqp_sparse` | 1/1 | 882 | 877 - 888 | 12800 / 57.3 / 824 | 135000000 / 619000 / 8640000 / 9260000 | 0 / 0 / 0 | 3.35e-4 | none | 23 |
| nx8_nu3_N30 | tight | `hpipm` | 1/1 | 230 | 226 - 231 | 102 / 0.0900 / 230 | 267000 / 365 / 2240000 / 2240000 | 6 / 6 / 6 | 2.65e-6 | none | 0 |
| nx8_nu3_N30 | tight | `argmin_sparse_admm_polished` | 1/1 | 448 | 446 - 449 | 4360 / 0.240 / 448 | 42800000 / 1720 / 5060000 / 5060000 | 25 / 25 / 25 | 2.59e-16 | none | 0 |
| nx8_nu3_N30 | tight | `osqp_polished` | 1/1 | 596 | 590 - 600 | 455 / 1.62 / 594 | 3810000 / 16000 / 5730000 / 5740000 | 25 / 25 / 25 | 1.78e-16 | none | 60 |
| nx12_nu4_N30 | nominal | `argmin_sparse_admm` | 1/1 | 708 | 706 - 710 | 3110 / 0.270 / 708 | 31700000 / 2070 / 7620000 / 7620000 | 25 / 25 / 25 | 1.21e-12 | none | 0 |
| nx12_nu4_N30 | nominal | `osqp` | 1/1 | 602 | 597 - 604 | 675 / 1.89 / 600 | 7070000 / 18600 / 5750000 / 5770000 | 25 / 25 / 25 | 1.27e-13 | none | 0 |
| nx12_nu4_N30 | nominal | `proxqp_sparse` | 1/1 | 334 | 331 - 336 | 8440 / 75.5 / 258 | 86900000 / 819000 / 2570000 / 3390000 | 0 / 0 / 0 | 6.71e-5 | none | 23 |
| nx12_nu4_N30 | nominal | `hpipm` | 1/1 | 234 | 233 - 236 | 36.7 / 0.0900 / 234 | 291000 / 386 / 2380000 / 2380000 | 5 / 5 / 5 | 1.05e-5 | none | 0 |
| nx12_nu4_N30 | nominal | `argmin_sparse_admm_polished` | 1/1 | 836 | 834 - 839 | 4200 / 0.270 / 836 | 40500000 / 2060 / 9080000 / 9080000 | 25 / 25 / 25 | 1.30e-12 | none | 0 |
| nx12_nu4_N30 | nominal | `osqp_polished` | 1/1 | 1150 | 1150 - 1150 | 680 / 1.90 / 1150 | 7070000 / 18600 / 11400000 / 11400000 | 25 / 25 / 25 | 7.70e-16 | none | 60 |
| nx12_nu4_N30 | tight | `argmin_sparse_admm` | 1/1 | 733 | 731 - 737 | 4070 / 0.310 / 733 | 40600000 / 2400 / 7880000 / 7880000 | 25 / 25 / 25 | 1.51e-13 | none | 0 |
| nx12_nu4_N30 | tight | `osqp` | 1/1 | 617 | 615 - 619 | 727 / 2.17 / 615 | 7280000 / 22400 / 5960000 / 5980000 | 25 / 25 / 25 | 1.73e-13 | none | 0 |
| nx12_nu4_N30 | tight | `proxqp_sparse` | 1/1 | 1810 | 1800 - 1830 | 35500 / 83.6 / 1720 | 370000000 / 896000 / 17400000 / 18300000 | 0 / 0 / 0 | 3.38e-4 | none | 23 |
| nx12_nu4_N30 | tight | `hpipm` | 1/1 | 248 | 247 - 250 | 39.3 / 0.0900 / 248 | 311000 / 386 / 2500000 / 2500000 | 5 / 5 / 5 | 8.74e-6 | none | 0 |
| nx12_nu4_N30 | tight | `argmin_sparse_admm_polished` | 1/1 | 860 | 857 - 864 | 5230 / 0.320 / 860 | 49600000 / 2410 / 9380000 / 9380000 | 25 / 25 / 25 | 1.02e-13 | none | 0 |
| nx12_nu4_N30 | tight | `osqp_polished` | 1/1 | 1150 | 1150 - 1160 | 719 / 2.17 / 1150 | 7280000 / 22400 / 11400000 / 11400000 | 25 / 25 / 25 | 1.78e-16 | none | 60 |
| nx4_nu2_N50 | nominal | `argmin_sparse_admm` | 1/1 | 206 | 205 - 209 | 916 / 0.200 / 206 | 10300000 / 1360 / 2580000 / 2580000 | 25 / 25 / 25 | 1.76e-12 | none | 0 |
| nx4_nu2_N50 | nominal | `osqp` | 1/1 | 193 | 192 - 194 | 224 / 1.30 / 192 | 2300000 / 11800 / 1690000 / 1700000 | 25 / 25 / 25 | 3.69e-13 | none | 0 |
| nx4_nu2_N50 | nominal | `proxqp_sparse` | 1/1 | 117 | 117 - 118 | 1770 / 45.6 / 71.6 | 19100000 / 494000 / 686000 / 1180000 | 0 / 0 / 0 | 8.87e-5 | none | 23 |
| nx4_nu2_N50 | nominal | `hpipm` | 1/1 | 200 | 200 - 201 | 33.1 / 0.0900 / 200 | 316000 / 349 / 2140000 / 2140000 | 5 / 5 / 5 | 1.63e-10 | none | 0 |
| nx4_nu2_N50 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 1 / 1 / 2 | 5.35e-3 | inaccurate 300 | 203 |
| nx4_nu2_N50 | nominal | `argmin_sparse_admm_polished` | 1/1 | 252 | 251 - 255 | 1190 / 0.200 / 251 | 12700000 / 1360 / 3120000 / 3120000 | 25 / 25 / 25 | 2.22e-16 | none | 0 |
| nx4_nu2_N50 | nominal | `osqp_polished` | 1/1 | 346 | 344 - 347 | 225 / 1.32 / 344 | 2310000 / 11800 / 3180000 / 3190000 | 25 / 25 / 25 | 2.66e-16 | none | 60 |
| nx4_nu2_N50 | tight | `argmin_sparse_admm` | 1/1 | 220 | 220 - 224 | 511 / 0.230 / 220 | 5500000 / 1610 / 2880000 / 2880000 | 25 / 25 / 25 | 2.41e-14 | none | 0 |
| nx4_nu2_N50 | tight | `osqp` | 1/1 | 208 | 206 - 208 | 243 / 1.58 / 206 | 2470000 / 15000 / 1860000 / 1880000 | 25 / 25 / 25 | 1.52e-14 | none | 0 |
| nx4_nu2_N50 | tight | `proxqp_sparse` | 1/1 | 1090 | 1070 - 1110 | 3530 / 51.7 / 1030 | 36000000 / 556000 / 10900000 / 11500000 | 0 / 0 / 0 | 1.06e-4 | none | 23 |
| nx4_nu2_N50 | tight | `hpipm` | 1/1 | 246 | 245 - 247 | 35.7 / 0.0900 / 246 | 343000 / 352 / 2650000 / 2650000 | 6 / 6 / 6 | 1.81e-5 | none | 0 |
| nx4_nu2_N50 | tight | `tinympc` | 1/1 | 6.70 | 6.65 - 6.76 | 22.0 / 0.100 / 6.60 | 115000 / 397 / 76000 / 76400 | 1 / 1 / 2 | 1.19e-5 | none | 203 |
| nx4_nu2_N50 | tight | `argmin_sparse_admm_polished` | 1/1 | 258 | 258 - 262 | 797 / 0.230 / 258 | 8170000 / 1610 / 3380000 / 3380000 | 25 / 25 / 25 | 4.60e-17 | none | 0 |
| nx4_nu2_N50 | tight | `osqp_polished` | 1/1 | 367 | 366 - 368 | 242 / 1.58 / 365 | 2470000 / 15000 / 3420000 / 3430000 | 25 / 25 / 25 | 1.23e-16 | none | 60 |
| nx4_nu2_N100 | nominal | `argmin_sparse_admm` | 1/1 | 415 | 413 - 416 | 1820 / 0.330 / 414 | 20400000 / 2530 / 5120000 / 5120000 | 25 / 25 / 25 | 1.95e-12 | none | 0 |
| nx4_nu2_N100 | nominal | `osqp` | 1/1 | 388 | 387 - 391 | 428 / 2.26 / 386 | 4540000 / 22600 / 3340000 / 3370000 | 25 / 25 / 25 | 3.36e-13 | none | 0 |
| nx4_nu2_N100 | nominal | `proxqp_sparse` | 1/1 | 231 | 230 - 233 | 6230 / 87.5 / 143 | 60200000 / 973000 / 1370000 / 2340000 | 0 / 0 / 0 | 8.87e-5 | none | 23 |
| nx4_nu2_N100 | nominal | `hpipm` | 1/1 | 398 | 397 - 401 | 60.0 / 0.0900 / 398 | 618000 / 352 / 4260000 / 4260000 | 5 / 5 / 5 | 3.29e-11 | none | 0 |
| nx4_nu2_N100 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 1 / 1 / 2 | 5.35e-3 | inaccurate 300 | 403 |
| nx4_nu2_N100 | nominal | `argmin_sparse_admm_polished` | 1/1 | 503 | 501 - 505 | 2350 / 0.320 / 503 | 25200000 / 2530 / 6190000 / 6200000 | 25 / 25 / 25 | 2.22e-16 | none | 0 |
| nx4_nu2_N100 | nominal | `osqp_polished` | 1/1 | 691 | 689 - 694 | 426 / 2.28 / 689 | 4540000 / 22600 / 6260000 / 6290000 | 25 / 25 / 25 | 3.55e-16 | none | 60 |
| nx4_nu2_N100 | tight | `argmin_sparse_admm` | 1/1 | 446 | 445 - 447 | 1010 / 0.420 / 445 | 10900000 / 3030 / 5720000 / 5720000 | 25 / 25 / 25 | 2.21e-14 | none | 0 |
| nx4_nu2_N100 | tight | `osqp` | 1/1 | 417 | 413 - 419 | 459 / 2.66 / 414 | 4870000 / 29000 / 3690000 / 3720000 | 25 / 25 / 25 | 1.16e-14 | none | 0 |
| nx4_nu2_N100 | tight | `proxqp_sparse` | 1/1 | 6650 | 6600 - 6700 | 16200 / 100 / 6550 | 154000000 / 1100000 / 64000000 / 65100000 | 0 / 0 / 0 | 8.02e-5 | none | 23 |
| nx4_nu2_N100 | tight | `hpipm` | 1/1 | 494 | 493 - 500 | 196 / 0.0900 / 494 | 675000 / 352 / 5250000 / 5250000 | 6 / 6 / 6 | 2.35e-5 | none | 0 |
| nx4_nu2_N100 | tight | `tinympc` | 1/1 | 12.3 | 12.2 - 12.4 | 24.2 / 0.140 / 12.2 | 130000 / 397 / 150000 / 150000 | 1 / 1 / 1 | 1.11e-5 | none | 203 |
| nx4_nu2_N100 | tight | `argmin_sparse_admm_polished` | 1/1 | 523 | 522 - 531 | 1830 / 0.420 / 523 | 16200000 / 3030 / 6690000 / 6700000 | 25 / 25 / 25 | 4.09e-17 | none | 0 |
| nx4_nu2_N100 | tight | `osqp_polished` | 1/1 | 731 | 727 - 734 | 552 / 2.69 / 728 | 4870000 / 29000 / 6750000 / 6780000 | 25 / 25 / 25 | 1.64e-16 | none | 60 |
| nx8_nu3_N50 | nominal | `argmin_sparse_admm` | 1/1 | 619 | 616 - 621 | 2620 / 0.310 / 619 | 27000000 / 2380 / 6620000 / 6620000 | 25 / 25 / 25 | 3.63e-12 | none | 0 |
| nx8_nu3_N50 | nominal | `osqp` | 1/1 | 518 | 515 - 521 | 568 / 2.09 / 516 | 6040000 / 21000 / 4820000 / 4840000 | 25 / 25 / 25 | 3.35e-13 | none | 0 |
| nx8_nu3_N50 | nominal | `proxqp_sparse` | 1/1 | 255 | 253 - 259 | 6690 / 83.3 / 172 | 68200000 / 916000 / 1630000 / 2550000 | 0 / 0 / 0 | 8.76e-5 | none | 23 |
| nx8_nu3_N50 | nominal | `hpipm` | 1/1 | 245 | 243 - 246 | 44.8 / 0.100 / 245 | 403000 / 365 / 2440000 / 2440000 | 4 / 4 / 4 | 2.11e-5 | none | 0 |
| nx8_nu3_N50 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 1 / 1 / 2 | 3.85e-3 | inaccurate 300 | 203 |
| nx8_nu3_N50 | nominal | `argmin_sparse_admm_polished` | 1/1 | 736 | 734 - 740 | 3370 / 0.310 / 736 | 34100000 / 2380 / 7960000 / 7960000 | 25 / 25 / 25 | 3.47e-12 | none | 0 |
| nx8_nu3_N50 | nominal | `osqp_polished` | 1/1 | 957 | 952 - 963 | 576 / 2.09 / 955 | 6040000 / 21000 / 9270000 / 9290000 | 25 / 25 / 25 | 1.69e-15 | none | 60 |
| nx8_nu3_N50 | tight | `argmin_sparse_admm` | 1/1 | 642 | 639 - 644 | 3380 / 0.360 / 642 | 35700000 / 2750 / 7070000 / 7070000 | 25 / 25 / 25 | 6.79e-13 | none | 0 |
| nx8_nu3_N50 | tight | `osqp` | 1/1 | 539 | 537 - 543 | 605 / 2.45 / 536 | 6290000 / 25800 / 5080000 / 5100000 | 25 / 25 / 25 | 1.00e-12 | none | 0 |
| nx8_nu3_N50 | tight | `proxqp_sparse` | 1/1 | 2020 | 2010 - 2040 | 46600 / 92.8 / 1920 | 479000000 / 1010000 / 18900000 / 20000000 | 0 / 0 / 0 | 3.35e-4 | none | 23 |
| nx8_nu3_N50 | tight | `hpipm` | 1/1 | 380 | 375 - 386 | 47.8 / 0.0900 / 380 | 433000 / 365 / 3710000 / 3710000 | 6 / 6 / 6 | 1.23e-6 | none | 0 |
| nx8_nu3_N50 | tight | `tinympc` | 1/1 | 10.4 | 10.3 - 10.5 | 33.5 / 0.130 / 10.2 | 220000 / 412 / 127000 / 127000 | 1 / 1 / 2 | 2.25e-4 | none | 203 |
| nx8_nu3_N50 | tight | `argmin_sparse_admm_polished` | 1/1 | 737 | 733 - 739 | 4430 / 0.360 / 737 | 43200000 / 2750 / 8220000 / 8220000 | 25 / 25 / 25 | 6.89e-13 | none | 0 |
| nx8_nu3_N50 | tight | `osqp_polished` | 1/1 | 974 | 971 - 978 | 611 / 2.46 / 971 | 6290000 / 25800 / 9460000 / 9490000 | 25 / 25 / 25 | 2.39e-16 | none | 60 |
| nx8_nu3_N100 | nominal | `argmin_sparse_admm` | 1/1 | 1250 | 1250 - 1260 | 5410 / 0.640 / 1250 | 53700000 / 4420 / 13200000 / 13200000 | 25 / 25 / 25 | 3.60e-12 | none | 0 |
| nx8_nu3_N100 | nominal | `osqp` | 1/1 | 1040 | 1030 - 1050 | 1130 / 3.77 / 1040 | 12000000 / 40800 / 9580000 / 9620000 | 25 / 25 / 25 | 5.00e-13 | none | 0 |
| nx8_nu3_N100 | nominal | `proxqp_sparse` | 1/1 | 543 | 539 - 547 | 24200 / 162 / 380 | 238000000 / 1810000 / 3270000 / 5080000 | 0 / 0 / 0 | 8.76e-5 | none | 23 |
| nx8_nu3_N100 | nominal | `hpipm` | 1/1 | 492 | 490 - 493 | 84.0 / 0.110 / 491 | 790000 / 365 / 4850000 / 4850000 | 4 / 4 / 4 | 2.77e-5 | none | 0 |
| nx8_nu3_N100 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 1 / 1 / 2 | 5.07e-3 | inaccurate 300 | 403 |
| nx8_nu3_N100 | nominal | `argmin_sparse_admm_polished` | 1/1 | 1490 | 1480 - 1490 | 7220 / 0.630 / 1490 | 67800000 / 4420 / 15800000 / 15800000 | 25 / 25 / 25 | 3.24e-12 | none | 0 |
| nx8_nu3_N100 | nominal | `osqp_polished` | 1/1 | 1910 | 1910 - 1920 | 1130 / 3.78 / 1910 | 12000000 / 40800 / 18400000 / 18400000 | 25 / 25 / 25 | 1.64e-15 | none | 60 |
| nx8_nu3_N100 | tight | `argmin_sparse_admm` | 1/1 | 1300 | 1290 - 1310 | 6780 / 0.740 / 1300 | 71000000 / 5170 / 14100000 / 14100000 | 25 / 25 / 25 | 9.71e-13 | none | 0 |
| nx8_nu3_N100 | tight | `osqp` | 1/1 | 1100 | 1090 - 1100 | 1180 / 4.66 / 1090 | 12400000 / 50400 / 10100000 / 10100000 | 25 / 25 / 25 | 1.55e-12 | none | 0 |
| nx8_nu3_N100 | tight | `proxqp_sparse` | 1/1 | 23200 | 23200 - 23300 | 296000 / 182 / 23100 | 3020000000 / 1990000 / 227000000 / 229000000 | 0 / 0 / 0 | 3.35e-4 | none | 23 |
| nx8_nu3_N100 | tight | `hpipm` | 1/1 | 770 | 767 - 774 | 112 / 0.120 / 770 | 851000 / 365 / 7360000 / 7360000 | 6 / 6 / 6 | 1.28e-6 | none | 0 |
| nx8_nu3_N100 | tight | `tinympc` | 1/1 | 19.4 | 19.3 - 19.6 | 45.2 / 0.170 / 19.2 | 235000 / 412 / 250000 / 250000 | 1 / 1 / 2 | 2.25e-4 | none | 403 |
| nx8_nu3_N100 | tight | `argmin_sparse_admm_polished` | 1/1 | 1500 | 1480 - 1520 | 8730 / 0.740 / 1500 | 85800000 / 5170 / 16300000 / 16300000 | 25 / 25 / 25 | 1.07e-12 | none | 0 |
| nx8_nu3_N100 | tight | `osqp_polished` | 1/1 | 1960 | 1960 - 1970 | 1180 / 4.60 / 1960 | 12400000 / 50400 / 18700000 / 18800000 | 25 / 25 / 25 | 3.58e-16 | none | 60 |

### sparse cohort, fixed regime, high tier

The sparse cohort in the fixed regime, judged at the high tier's threshold of 1e-6.

| Cell | Variant | Arm | Pass rate | Median tick (us) | Tick p25 - p75 (us) | Median setup / update / solve (us) | Median instructions setup / update / solve / tick | Iterations min / median / max | Worst KKT residual | Failures | Allocations max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nx2_nu1_N10 | nominal | `argmin_sparse_admm` | 1/1 | 22.3 | 21.9 - 22.7 | 138 / 0.0400 / 22.2 | 1360000 / 391 / 283000 / 283000 | 25 / 25 / 25 | 8.33e-17 | none | 0 |
| nx2_nu1_N10 | nominal | `osqp` | 1/1 | 34.5 | 34.3 - 34.9 | 30.2 / 0.260 / 34.3 | 217000 / 2010 / 286000 / 288000 | 25 / 25 / 25 | 1.33e-16 | none | 60 |
| nx2_nu1_N10 | nominal | `proxqp_sparse` | 1/1 | 12.2 | 12.1 - 12.4 | 132 / 5.66 / 6.58 | 1110000 / 56000 / 60400 / 116000 | 0 / 0 / 0 | 4.10e-8 | none | 23 |
| nx2_nu1_N10 | nominal | `hpipm` | 1/1 | 52.8 | 52.6 - 52.9 | 29.5 / 0.0600 / 52.7 | 66700 / 335 / 611000 / 611000 | 6 / 6 / 6 | 5.38e-10 | none | 0 |
| nx2_nu1_N10 | tight | `argmin_sparse_admm` | 1/1 | 22.1 | 22.0 - 22.3 | 105 / 0.0400 / 22.1 | 979000 / 392 / 320000 / 321000 | 25 / 25 / 25 | 4.44e-17 | none | 0 |
| nx2_nu1_N10 | tight | `osqp` | 1/1 | 35.2 | 35.1 - 35.4 | 31.8 / 0.300 / 34.9 | 230000 / 2370 / 305000 / 308000 | 25 / 25 / 25 | 1.33e-16 | none | 60 |
| nx2_nu1_N10 | tight | `proxqp_sparse` | 1/1 | 13.6 | 13.5 - 13.7 | 98.6 / 6.36 / 7.28 | 777000 / 62200 / 67700 / 130000 | 0 / 0 / 0 | 4.53e-7 | none | 23 |
| nx2_nu1_N10 | tight | `hpipm` | 1/1 | 47.0 | 46.2 - 47.7 | 23.6 / 0.0500 / 47.0 | 71400 / 335 / 531000 / 532000 | 5 / 5 / 5 | 1.89e-11 | none | 0 |
| nx4_nu2_N10 | nominal | `argmin_sparse_admm` | 1/1 | 52.6 | 52.3 - 52.9 | 318 / 0.0500 / 52.5 | 3270000 / 459 / 666000 / 666000 | 25 / 25 / 25 | 1.39e-16 | none | 0 |
| nx4_nu2_N10 | nominal | `osqp` | 1/1 | 75.1 | 74.7 - 76.5 | 59.7 / 0.380 / 74.7 | 502000 / 3140 / 698000 / 701000 | 25 / 25 / 25 | 2.02e-16 | none | 60 |
| nx4_nu2_N10 | nominal | `proxqp_sparse` | 1/1 | 24.7 | 24.6 - 25.0 | 268 / 10.6 / 14.1 | 2570000 / 108000 / 139000 / 248000 | 0 / 0 / 0 | 2.21e-8 | none | 23 |
| nx4_nu2_N10 | nominal | `hpipm` | 1/1 | 55.2 | 54.9 - 56.1 | 28.2 / 0.0600 / 55.2 | 78000 / 352 / 613000 / 613000 | 5 / 5 / 5 | 1.22e-10 | none | 0 |
| nx4_nu2_N10 | tight | `argmin_sparse_admm` | 1/1 | 53.9 | 53.7 - 54.8 | 244 / 0.0500 / 53.9 | 2330000 / 509 / 713000 / 714000 | 25 / 25 / 25 | 4.44e-17 | none | 0 |
| nx4_nu2_N10 | tight | `osqp` | 1/1 | 78.3 | 77.9 - 78.6 | 65.4 / 0.440 / 77.8 | 536000 / 3780 / 732000 / 736000 | 25 / 25 / 25 | 1.78e-16 | none | 60 |
| nx4_nu2_N10 | tight | `proxqp_sparse` | 1/1 | 29.2 | 29.1 - 29.6 | 236 / 11.9 / 17.4 | 2060000 / 121000 / 171000 / 292000 | 0 / 0 / 0 | 7.52e-8 | none | 23 |
| nx4_nu2_N10 | tight | `hpipm` | 1/1 | 68.8 | 68.6 - 69.1 | 25.3 / 0.0600 / 68.7 | 83700 / 352 / 765000 / 766000 | 6 / 6 / 6 | 2.59e-8 | none | 0 |
| nx4_nu2_N20 | nominal | `argmin_sparse_admm` | 1/1 | 102 | 101 - 104 | 582 / 0.0900 / 102 | 6260000 / 684 / 1280000 / 1280000 | 25 / 25 / 25 | 2.30e-16 | none | 0 |
| nx4_nu2_N20 | nominal | `osqp` | 1/1 | 144 | 143 - 146 | 102 / 0.610 / 143 | 955000 / 5320 / 1320000 / 1330000 | 25 / 25 / 25 | 3.07e-16 | none | 60 |
| nx4_nu2_N20 | nominal | `proxqp_sparse` | 1/1 | 46.8 | 46.5 - 47.1 | 554 / 19.2 / 27.5 | 5620000 / 205000 / 275000 / 480000 | 0 / 0 / 0 | 2.21e-8 | none | 23 |
| nx4_nu2_N20 | nominal | `hpipm` | 1/1 | 106 | 106 - 108 | 37.1 / 0.0600 / 106 | 140000 / 352 / 1190000 / 1190000 | 5 / 5 / 5 | 1.02e-11 | none | 0 |
| nx4_nu2_N20 | tight | `argmin_sparse_admm` | 1/1 | 105 | 105 - 108 | 443 / 0.110 / 105 | 4500000 / 784 / 1380000 / 1380000 | 25 / 25 / 25 | 4.44e-17 | none | 0 |
| nx4_nu2_N20 | tight | `osqp` | 1/1 | 153 | 152 - 157 | 110 / 0.760 / 152 | 1030000 / 6600 / 1420000 / 1420000 | 25 / 25 / 25 | 1.78e-16 | none | 60 |
| nx4_nu2_N20 | tight | `proxqp_sparse` | 1/1 | 77.6 | 77.0 - 78.9 | 631 / 22.0 / 55.4 | 6070000 / 230000 / 560000 / 790000 | 0 / 0 / 0 | 4.34e-7 | none | 23 |
| nx4_nu2_N20 | tight | `hpipm` | 1/1 | 133 | 133 - 134 | 22.6 / 0.0700 / 133 | 151000 / 352 / 1480000 / 1480000 | 6 / 6 / 6 | 2.91e-8 | none | 0 |
| nx8_nu3_N20 | nominal | `argmin_sparse_admm` | 1/1 | 303 | 299 - 304 | 1980 / 0.170 / 303 | 19200000 / 1070 / 3290000 / 3290000 | 25 / 25 / 25 | 1.75e-16 | none | 0 |
| nx8_nu3_N20 | nominal | `osqp` | 1/1 | 393 | 390 - 396 | 250 / 1.06 / 391 | 2480000 / 9080 / 3780000 / 3790000 | 25 / 25 / 25 | 1.75e-16 | none | 60 |
| nx8_nu3_N20 | nominal | `proxqp_sparse` | 1/1 | 102 | 101 - 104 | 1640 / 35.6 / 66.5 | 16100000 / 381000 / 644000 / 1030000 | 0 / 0 / 0 | 2.14e-8 | none | 23 |
| nx8_nu3_N20 | nominal | `hpipm` | 1/1 | 158 | 158 - 161 | 23.0 / 0.0900 / 158 | 178000 / 365 / 1670000 / 1670000 | 5 / 5 / 5 | 2.22e-12 | none | 0 |
| nx8_nu3_N20 | tight | `argmin_sparse_admm` | 1/1 | 300 | 299 - 301 | 3520 / 0.180 / 300 | 37100000 / 1220 / 3410000 / 3410000 | 25 / 25 / 25 | 1.11e-16 | none | 0 |
| nx8_nu3_N20 | tight | `osqp` | 1/1 | 400 | 397 - 402 | 262 / 1.24 / 399 | 2580000 / 11000 / 3870000 / 3890000 | 25 / 25 / 25 | 1.78e-16 | none | 60 |
| nx8_nu3_N20 | tight | `proxqp_sparse` | 1/1 | 343 | 338 - 348 | 6680 / 39.4 / 304 | 69900000 / 417000 / 3110000 / 3530000 | 0 / 0 / 0 | 6.32e-8 | none | 23 |
| nx8_nu3_N20 | tight | `hpipm` | 1/1 | 262 | 262 - 263 | 24.4 / 0.0900 / 262 | 190000 / 365 / 2720000 / 2720000 | 8 / 8 / 8 | 1.23e-8 | none | 0 |
| nx8_nu3_N30 | nominal | `argmin_sparse_admm` | 1/1 | 452 | 450 - 453 | 3010 / 0.210 / 452 | 28500000 / 1530 / 4900000 / 4900000 | 25 / 25 / 25 | 3.15e-16 | none | 0 |
| nx8_nu3_N30 | nominal | `osqp` | 1/1 | 585 | 578 - 587 | 393 / 1.41 / 583 | 3670000 / 13100 / 5610000 / 5620000 | 25 / 25 / 25 | 3.35e-16 | none | 60 |
| nx8_nu3_N30 | nominal | `proxqp_sparse` | 1/1 | 151 | 150 - 153 | 3050 / 51.4 / 99.8 | 30400000 / 559000 / 973000 / 1530000 | 0 / 0 / 0 | 2.14e-8 | none | 23 |
| nx8_nu3_N30 | nominal | `hpipm` | 1/1 | 238 | 237 - 239 | 109 / 0.0900 / 238 | 259000 / 365 / 2490000 / 2490000 | 5 / 5 / 5 | 2.25e-12 | none | 0 |
| nx8_nu3_N30 | tight | `argmin_sparse_admm` | 1/1 | 447 | 445 - 453 | 5420 / 0.240 / 447 | 55300000 / 1720 / 5060000 / 5060000 | 25 / 25 / 25 | 2.59e-16 | none | 0 |
| nx8_nu3_N30 | tight | `osqp` | 1/1 | 593 | 588 - 604 | 477 / 1.63 / 592 | 3810000 / 16000 / 5730000 / 5740000 | 25 / 25 / 25 | 1.78e-16 | none | 60 |
| nx8_nu3_N30 | tight | `proxqp_sparse` | 1/1 | 891 | 882 - 898 | 16800 / 57.3 / 833 | 178000000 / 619000 / 8640000 / 9260000 | 0 / 0 / 0 | 6.32e-8 | none | 23 |
| nx8_nu3_N30 | tight | `hpipm` | 1/1 | 390 | 387 - 392 | 124 / 0.0900 / 390 | 278000 / 365 / 4030000 / 4030000 | 8 / 8 / 8 | 2.77e-9 | none | 0 |
| nx12_nu4_N30 | nominal | `argmin_sparse_admm` | 1/1 | 837 | 834 - 839 | 5560 / 0.270 / 837 | 54900000 / 2070 / 9080000 / 9080000 | 25 / 25 / 25 | 1.30e-12 | none | 0 |
| nx12_nu4_N30 | nominal | `osqp` | 1/1 | 1150 | 1150 - 1160 | 677 / 1.89 / 1150 | 7070000 / 18600 / 11400000 / 11400000 | 25 / 25 / 25 | 7.70e-16 | none | 60 |
| nx12_nu4_N30 | nominal | `proxqp_sparse` | 1/1 | 334 | 331 - 337 | 8700 / 75.4 / 259 | 90200000 / 819000 / 2570000 / 3390000 | 0 / 0 / 0 | 4.80e-8 | none | 23 |
| nx12_nu4_N30 | nominal | `hpipm` | 1/1 | 342 | 341 - 347 | 40.2 / 0.0900 / 342 | 307000 / 386 / 3660000 / 3660000 | 6 / 6 / 6 | 5.70e-12 | none | 0 |
| nx12_nu4_N30 | tight | `argmin_sparse_admm` | 1/1 | 859 | 856 - 864 | 7420 / 0.320 / 858 | 72700000 / 2410 / 9380000 / 9380000 | 25 / 25 / 25 | 1.02e-13 | none | 0 |
| nx12_nu4_N30 | tight | `osqp` | 1/1 | 1150 | 1150 - 1160 | 739 / 2.17 / 1150 | 7280000 / 22400 / 11400000 / 11400000 | 25 / 25 / 25 | 1.78e-16 | none | 60 |
| nx12_nu4_N30 | tight | `proxqp_sparse` | 1/1 | 1820 | 1800 - 1840 | 46600 / 83.8 / 1740 | 484000000 / 896000 / 17400000 / 18300000 | 0 / 0 / 0 | 8.73e-8 | none | 23 |
| nx12_nu4_N30 | tight | `hpipm` | 1/1 | 483 | 478 - 488 | 42.2 / 0.100 / 482 | 327000 / 386 / 5110000 / 5110000 | 8 / 8 / 8 | 2.79e-9 | none | 0 |
| nx4_nu2_N50 | nominal | `argmin_sparse_admm` | 1/1 | 252 | 251 - 255 | 1590 / 0.210 / 251 | 17800000 / 1360 / 3120000 / 3120000 | 25 / 25 / 25 | 2.22e-16 | none | 0 |
| nx4_nu2_N50 | nominal | `osqp` | 1/1 | 345 | 344 - 346 | 224 / 1.32 / 344 | 2300000 / 11800 / 3180000 / 3190000 | 25 / 25 / 25 | 2.66e-16 | none | 60 |
| nx4_nu2_N50 | nominal | `proxqp_sparse` | 1/1 | 117 | 116 - 118 | 1860 / 45.4 / 71.2 | 20300000 / 494000 / 686000 / 1180000 | 0 / 0 / 0 | 2.21e-8 | none | 23 |
| nx4_nu2_N50 | nominal | `hpipm` | 1/1 | 261 | 260 - 262 | 34.5 / 0.0900 / 261 | 328000 / 352 / 2920000 / 2920000 | 5 / 5 / 5 | 1.32e-11 | none | 0 |
| nx4_nu2_N50 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 1 / 1 / 2 | 5.35e-3 | inaccurate 300 | 203 |
| nx4_nu2_N50 | tight | `argmin_sparse_admm` | 1/1 | 259 | 258 - 263 | 999 / 0.230 / 259 | 10800000 / 1610 / 3380000 / 3380000 | 25 / 25 / 25 | 4.60e-17 | none | 0 |
| nx4_nu2_N50 | tight | `osqp` | 1/1 | 367 | 366 - 368 | 239 / 1.57 / 366 | 2470000 / 15000 / 3420000 / 3430000 | 25 / 25 / 25 | 1.23e-16 | none | 60 |
| nx4_nu2_N50 | tight | `proxqp_sparse` | 1/1 | 1090 | 1070 - 1110 | 3850 / 51.7 / 1030 | 39300000 / 556000 / 10900000 / 11500000 | 0 / 0 / 0 | 1.46e-7 | none | 23 |
| nx4_nu2_N50 | tight | `hpipm` | 1/1 | 324 | 323 - 325 | 37.4 / 0.0900 / 324 | 355000 / 352 / 3610000 / 3610000 | 6 / 6 / 6 | 7.65e-12 | none | 0 |
| nx4_nu2_N50 | tight | `tinympc` | 1/1 | 6.68 | 6.65 - 6.73 | 21.4 / 0.100 / 6.59 | 113000 / 397 / 76000 / 76400 | 1 / 1 / 1 | 1.20e-8 | none | 103 |
| nx4_nu2_N100 | nominal | `argmin_sparse_admm` | 1/1 | 504 | 501 - 505 | 3160 / 0.320 / 503 | 35200000 / 2530 / 6190000 / 6200000 | 25 / 25 / 25 | 2.22e-16 | none | 0 |
| nx4_nu2_N100 | nominal | `osqp` | 1/1 | 692 | 690 - 695 | 426 / 2.28 / 689 | 4540000 / 22600 / 6260000 / 6290000 | 25 / 25 / 25 | 3.55e-16 | none | 60 |
| nx4_nu2_N100 | nominal | `proxqp_sparse` | 1/1 | 231 | 230 - 233 | 6450 / 87.5 / 144 | 62700000 / 973000 / 1370000 / 2340000 | 0 / 0 / 0 | 2.21e-8 | none | 23 |
| nx4_nu2_N100 | nominal | `hpipm` | 1/1 | 523 | 520 - 526 | 205 / 0.0900 / 523 | 640000 / 349 / 5790000 / 5790000 | 5 / 5 / 5 | 1.36e-11 | none | 0 |
| nx4_nu2_N100 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 1 / 1 / 2 | 5.35e-3 | inaccurate 300 | 403 |
| nx4_nu2_N100 | tight | `argmin_sparse_admm` | 1/1 | 526 | 523 - 527 | 2170 / 0.420 / 526 | 21100000 / 3030 / 6690000 / 6700000 | 25 / 25 / 25 | 4.09e-17 | none | 0 |
| nx4_nu2_N100 | tight | `osqp` | 1/1 | 732 | 730 - 735 | 464 / 2.69 / 730 | 4870000 / 29000 / 6750000 / 6780000 | 25 / 25 / 25 | 1.64e-16 | none | 60 |
| nx4_nu2_N100 | tight | `proxqp_sparse` | 1/1 | 6690 | 6670 - 6720 | 17100 / 99.9 / 6590 | 164000000 / 1100000 / 64000000 / 65100000 | 0 / 0 / 0 | 1.12e-7 | none | 23 |
| nx4_nu2_N100 | tight | `hpipm` | 1/1 | 656 | 653 - 658 | 232 / 0.0900 / 656 | 695000 / 349 / 7160000 / 7160000 | 6 / 6 / 6 | 5.70e-12 | none | 0 |
| nx4_nu2_N100 | tight | `tinympc` | 1/1 | 12.3 | 12.2 - 12.4 | 24.2 / 0.120 / 12.2 | 129000 / 397 / 150000 / 150000 | 1 / 1 / 2 | 1.34e-8 | none | 403 |
| nx8_nu3_N50 | nominal | `argmin_sparse_admm` | 1/1 | 736 | 733 - 738 | 3970 / 0.310 / 735 | 40600000 / 2380 / 7960000 / 7960000 | 25 / 25 / 25 | 3.47e-12 | none | 0 |
| nx8_nu3_N50 | nominal | `osqp` | 1/1 | 956 | 952 - 962 | 576 / 2.09 / 954 | 6040000 / 21000 / 9270000 / 9290000 | 25 / 25 / 25 | 1.69e-15 | none | 60 |
| nx8_nu3_N50 | nominal | `proxqp_sparse` | 1/1 | 255 | 252 - 259 | 6460 / 83.1 / 172 | 70400000 / 916000 / 1630000 / 2550000 | 0 / 0 / 0 | 2.14e-8 | none | 23 |
| nx8_nu3_N50 | nominal | `hpipm` | 1/1 | 392 | 390 - 393 | 47.1 / 0.0900 / 392 | 419000 / 365 / 4120000 / 4120000 | 5 / 5 / 5 | 2.65e-12 | none | 0 |
| nx8_nu3_N50 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 1 / 1 / 2 | 3.85e-3 | inaccurate 300 | 203 |
| nx8_nu3_N50 | tight | `argmin_sparse_admm` | 1/1 | 737 | 735 - 741 | 7550 / 0.360 / 736 | 77600000 / 2750 / 8220000 / 8220000 | 25 / 25 / 25 | 6.00e-13 | none | 0 |
| nx8_nu3_N50 | tight | `osqp` | 1/1 | 974 | 972 - 979 | 603 / 2.48 / 972 | 6290000 / 25800 / 9460000 / 9490000 | 25 / 25 / 25 | 2.39e-16 | none | 60 |
| nx8_nu3_N50 | tight | `proxqp_sparse` | 1/1 | 2020 | 2010 - 2040 | 62800 / 93.0 / 1930 | 648000000 / 1010000 / 18900000 / 20000000 | 0 / 0 / 0 | 6.32e-8 | none | 23 |
| nx8_nu3_N50 | tight | `hpipm` | 1/1 | 659 | 652 - 673 | 53.0 / 0.0900 / 659 | 450000 / 365 / 6660000 / 6660000 | 8 / 8 / 8 | 2.61e-10 | none | 0 |
| nx8_nu3_N50 | tight | `tinympc` | 1/1 | 10.4 | 10.3 - 10.5 | 34.7 / 0.140 / 10.2 | 218000 / 412 / 127000 / 127000 | 1 / 1 / 2 | 2.31e-7 | none | 203 |
| nx8_nu3_N100 | nominal | `argmin_sparse_admm` | 1/1 | 1490 | 1480 - 1490 | 9590 / 0.640 / 1490 | 93600000 / 4420 / 15800000 / 15800000 | 25 / 25 / 25 | 3.24e-12 | none | 0 |
| nx8_nu3_N100 | nominal | `osqp` | 1/1 | 1910 | 1900 - 1920 | 1110 / 3.79 / 1910 | 12000000 / 40800 / 18400000 / 18400000 | 25 / 25 / 25 | 1.64e-15 | none | 60 |
| nx8_nu3_N100 | nominal | `proxqp_sparse` | 1/1 | 543 | 539 - 548 | 24500 / 162 / 380 | 243000000 / 1810000 / 3270000 / 5080000 | 0 / 0 / 0 | 2.14e-8 | none | 23 |
| nx8_nu3_N100 | nominal | `hpipm` | 1/1 | 800 | 797 - 805 | 334 / 0.110 / 800 | 821000 / 365 / 8180000 / 8180000 | 5 / 5 / 5 | 3.36e-12 | none | 0 |
| nx8_nu3_N100 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 1 / 1 / 2 | 5.07e-3 | inaccurate 300 | 403 |
| nx8_nu3_N100 | tight | `argmin_sparse_admm` | 1/1 | 1490 | 1480 - 1500 | 14900 / 0.740 / 1490 | 153000000 / 5170 / 16300000 / 16300000 | 25 / 25 / 25 | 1.07e-12 | none | 0 |
| nx8_nu3_N100 | tight | `osqp` | 1/1 | 1960 | 1950 - 1970 | 1180 / 4.61 / 1960 | 12400000 / 50400 / 18700000 / 18800000 | 25 / 25 / 25 | 3.58e-16 | none | 60 |
| nx8_nu3_N100 | tight | `proxqp_sparse` | 1/1 | 23400 | 23300 - 23500 | 415000 / 182 / 23200 | 4210000000 / 1990000 / 227000000 / 229000000 | 0 / 0 / 0 | 6.32e-8 | none | 23 |
| nx8_nu3_N100 | tight | `hpipm` | 1/1 | 1340 | 1340 - 1350 | 117 / 0.120 / 1340 | 881000 / 365 / 13200000 / 13200000 | 8 / 8 / 8 | 2.54e-10 | none | 0 |
| nx8_nu3_N100 | tight | `tinympc` | 1/1 | 19.5 | 19.4 - 19.6 | 45.2 / 0.170 / 19.3 | 235000 / 412 / 250000 / 250000 | 1 / 1 / 2 | 2.31e-7 | none | 403 |

### sparse cohort, closed_loop regime, mpc tier

The sparse cohort in the closed_loop regime, judged at the mpc tier's threshold of 1e-3; rules A
and C read this table.

| Cell | Variant | Arm | Pass rate | Median tick (us) | Tick p25 - p75 (us) | Median setup / update / solve (us) | Median instructions setup / update / solve / tick | Iterations min / median / max | Worst KKT residual | Failures | Allocations max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nx2_nu1_N10 | nominal | `argmin_sparse_admm` | 1/1 | 17.8 | 17.8 - 17.9 | 92.6 / 0.0400 / 17.8 | 917000 / 391 / 232000 / 232000 | 25 / 25 / 25 | 3.16e-4 | none | 0 |
| nx2_nu1_N10 | nominal | `osqp` | 1/1 | 38.8 | 37.7 - 40.0 | 28.9 / 0.260 / 38.5 | 217000 / 2010 / 309000 / 311000 | 25 / 50 / 100 | 4.34e-4 | none | 0 |
| nx2_nu1_N10 | nominal | `proxqp_sparse` | 43/60 | 37.7 | 25.7 - 53.4 | 120 / 5.83 / 31.9 | 973000 / 56000 / 311000 / 368000 | 0 / 3 / 9 | 8.70e-2 | inaccurate 85 | 35 |
| nx2_nu1_N10 | nominal | `hpipm` | 1/1 | 38.3 | 32.5 - 39.0 | 10.4 / 0.0500 / 38.3 | 63600 / 335 / 438000 / 438000 | 5 / 6 / 9 | 1.40e-4 | none | 0 |
| nx2_nu1_N10 | nominal | `argmin_sparse_admm_polished` | 1/1 | 41.3 | 40.4 - 42.1 | 125 / 0.0400 / 41.3 | 1140000 / 387 / 440000 / 440000 | 25 / 25 / 25 | 1.88e-16 | none | 81 |
| nx2_nu1_N10 | nominal | `osqp_polished` | 1/1 | 53.8 | 52.8 - 58.8 | 31.7 / 0.260 / 53.6 | 218000 / 2030 / 436000 / 438000 | 25 / 50 / 75 | 2.60e-16 | none | 60 |
| nx2_nu1_N10 | tight | `argmin_sparse_admm` | 1/1 | 35.3 | 35.1 - 35.6 | 54.6 / 0.0400 / 35.3 | 500000 / 391 / 530000 / 531000 | 25 / 50 / 100 | 5.19e-4 | none | 5 |
| nx2_nu1_N10 | tight | `osqp` | 1/1 | 39.8 | 39.7 - 40.1 | 29.8 / 0.290 / 39.5 | 231000 / 2360 / 338000 / 340000 | 25 / 50 / 200 | 4.78e-4 | none | 0 |
| nx2_nu1_N10 | tight | `proxqp_sparse` | 13/20 | 90.4 | 31.6 - 148 | 78.5 / 6.66 / 83.2 | 539000 / 62200 / 790000 / 852000 | 0 / 5.5 / 28 | 1.00e+0 | inaccurate 20, infeasible_claim 85 | 116 |
| nx2_nu1_N10 | tight | `hpipm` | 1/1 | 34.2 | 33.6 - 40.4 | 10.9 / 0.0500 / 34.1 | 68400 / 335 / 382000 / 383000 | 4 / 5 / 8 | 2.55e-5 | none | 0 |
| nx2_nu1_N10 | tight | `argmin_sparse_admm_polished` | 59/60 | 58.5 | 41.6 - 60.5 | 90.3 / 0.0400 / 58.4 | 719000 / 391 / 746000 / 746000 | 25 / 50 / 175 | 1.32e-1 | inaccurate 5 | 85 |
| nx2_nu1_N10 | tight | `osqp_polished` | 1/1 | 55.4 | 54.7 - 56.5 | 34.0 / 0.300 / 55.1 | 232000 / 2360 / 472000 / 475000 | 25 / 50 / 150 | 2.28e-4 | none | 60 |
| nx4_nu2_N10 | nominal | `argmin_sparse_admm` | 1/1 | 42.6 | 42.3 - 83.0 | 206 / 0.0500 / 42.5 | 2190000 / 459 / 545000 / 545000 | 25 / 25 / 50 | 3.33e-4 | none | 0 |
| nx4_nu2_N10 | nominal | `osqp` | 1/1 | 82.9 | 79.0 - 120 | 57.9 / 0.370 / 82.5 | 503000 / 3140 / 760000 / 763000 | 25 / 50 / 75 | 3.35e-4 | none | 0 |
| nx4_nu2_N10 | nominal | `proxqp_sparse` | 13/15 | 127 | 114 - 140 | 244 / 10.9 / 116 | 2290000 / 108000 / 1190000 / 1300000 | 0 / 7 / 11 | 6.05e-2 | inaccurate 40 | 35 |
| nx4_nu2_N10 | nominal | `hpipm` | 1/1 | 50.6 | 50.0 - 58.3 | 11.8 / 0.0500 / 50.5 | 74400 / 352 / 534000 / 535000 | 5 / 6 / 9 | 5.77e-5 | none | 0 |
| nx4_nu2_N10 | nominal | `argmin_sparse_admm_polished` | 1/1 | 104 | 102 - 143 | 279 / 0.0500 / 104 | 2740000 / 459 / 1100000 / 1100000 | 25 / 25 / 50 | 2.92e-6 | none | 83 |
| nx4_nu2_N10 | nominal | `osqp_polished` | 1/1 | 119 | 114 - 156 | 63.2 / 0.390 / 119 | 504000 / 3140 / 1100000 / 1100000 | 25 / 50 / 75 | 9.73e-5 | none | 60 |
| nx4_nu2_N10 | tight | `argmin_sparse_admm` | 1/1 | 132 | 89.5 - 175 | 121 / 0.0500 / 132 | 1170000 / 509 / 1790000 / 1790000 | 25 / 75 / 150 | 5.24e-4 | none | 5 |
| nx4_nu2_N10 | tight | `osqp` | 1/1 | 84.8 | 84.5 - 85.2 | 63.4 / 0.420 / 84.4 | 535000 / 3780 / 793000 / 797000 | 25 / 50 / 150 | 5.22e-4 | none | 0 |
| nx4_nu2_N10 | tight | `proxqp_sparse` | 7/30 | 453 | 319 - 498 | 199 / 13.0 / 440 | 1600000 / 121000 / 4340000 / 4460000 | 0 / 7 / 54 | 1.00e+0 | inaccurate 5, infeasible_claim 225 | 122 |
| nx4_nu2_N10 | tight | `hpipm` | 1/1 | 55.7 | 52.1 - 61.7 | 12.4 / 0.0600 / 55.6 | 79900 / 352 / 558000 / 558000 | 5 / 6 / 8 | 2.80e-5 | none | 0 |
| nx4_nu2_N10 | tight | `argmin_sparse_admm_polished` | 59/60 | 199 | 156 - 241 | 202 / 0.0600 / 198 | 1740000 / 510 / 2350000 / 2350000 | 25 / 75 / 150 | 6.63e-1 | inaccurate 5 | 83 |
| nx4_nu2_N10 | tight | `osqp_polished` | 1/1 | 122 | 121 - 124 | 67.1 / 0.440 / 121 | 536000 / 3780 / 1130000 / 1130000 | 25 / 50 / 150 | 5.01e-4 | none | 60 |
| nx4_nu2_N20 | nominal | `argmin_sparse_admm` | 1/1 | 84.6 | 82.6 - 108 | 384 / 0.0900 / 84.5 | 4210000 / 684 / 1050000 / 1050000 | 25 / 25 / 50 | 3.69e-4 | none | 0 |
| nx4_nu2_N20 | nominal | `osqp` | 1/1 | 163 | 155 - 170 | 101 / 0.590 / 163 | 955000 / 5320 / 1450000 / 1450000 | 50 / 50 / 75 | 3.76e-4 | none | 0 |
| nx4_nu2_N20 | nominal | `proxqp_sparse` | 4/5 | 245 | 213 - 280 | 503 / 19.9 / 225 | 5160000 / 205000 / 2390000 / 2590000 | 0 / 7 / 12 | 1.25e-3 | inaccurate 60 | 35 |
| nx4_nu2_N20 | nominal | `hpipm` | 1/1 | 98.2 | 97.0 - 118 | 17.1 / 0.0600 / 98.2 | 135000 / 352 / 1040000 / 1040000 | 5 / 6 / 9 | 7.11e-5 | none | 0 |
| nx4_nu2_N20 | nominal | `argmin_sparse_admm_polished` | 1/1 | 194 | 191 - 252 | 504 / 0.100 / 193 | 5240000 / 688 / 2070000 / 2070000 | 25 / 25 / 50 | 3.69e-4 | none | 83 |
| nx4_nu2_N20 | nominal | `osqp_polished` | 1/1 | 228 | 222 - 231 | 105 / 0.650 / 227 | 956000 / 5320 / 2060000 / 2060000 | 50 / 50 / 75 | 3.29e-4 | none | 60 |
| nx4_nu2_N20 | tight | `argmin_sparse_admm` | 1/1 | 261 | 175 - 347 | 224 / 0.100 / 261 | 2260000 / 784 / 3470000 / 3470000 | 25 / 75 / 125 | 5.95e-4 | none | 5 |
| nx4_nu2_N20 | tight | `osqp` | 1/1 | 167 | 166 - 172 | 109 / 0.720 / 166 | 1030000 / 6600 / 1510000 / 1520000 | 25 / 50 / 75 | 4.48e-4 | none | 0 |
| nx4_nu2_N20 | tight | `proxqp_sparse` | 1/12 | 1170 | 867 - 1530 | 549 / 24.3 / 1140 | 5090000 / 230000 / 12300000 / 12600000 | 0 / 2 / 53 | 1.00e+0 | infeasible_claim 275 | 71 |
| nx4_nu2_N20 | tight | `hpipm` | 1/1 | 102 | 100 - 125 | 18.4 / 0.0700 / 102 | 147000 / 352 / 1080000 / 1080000 | 5 / 6 / 10 | 2.85e-5 | none | 0 |
| nx4_nu2_N20 | tight | `argmin_sparse_admm_polished` | 29/30 | 383 | 304 - 465 | 352 / 0.120 / 383 | 3350000 / 784 / 4540000 / 4540000 | 25 / 75 / 125 | 6.63e-1 | inaccurate 10 | 84 |
| nx4_nu2_N20 | tight | `osqp_polished` | 1/1 | 236 | 235 - 239 | 113 / 0.760 / 235 | 1020000 / 6600 / 2160000 / 2170000 | 25 / 50 / 75 | 2.91e-4 | none | 60 |
| nx8_nu3_N20 | nominal | `argmin_sparse_admm` | 1/1 | 250 | 247 - 250 | 1080 / 0.170 / 249 | 11000000 / 1070 / 2680000 / 2680000 | 25 / 25 / 25 | 1.06e-4 | none | 0 |
| nx8_nu3_N20 | nominal | `osqp` | 1/1 | 420 | 410 - 498 | 248 / 1.06 / 419 | 2480000 / 9080 / 3850000 / 3860000 | 50 / 50 / 100 | 3.06e-4 | none | 0 |
| nx8_nu3_N20 | nominal | `proxqp_sparse` | 1/1 | 590 | 541 - 695 | 1550 / 36.9 / 554 | 15100000 / 381000 / 5960000 / 6340000 | 0 / 7 / 13 | 4.75e-4 | none | 35 |
| nx8_nu3_N20 | nominal | `hpipm` | 1/1 | 170 | 168 - 174 | 22.8 / 0.0900 / 170 | 170000 / 365 / 1690000 / 1690000 | 5 / 7 / 10 | 2.12e-5 | none | 0 |
| nx8_nu3_N20 | nominal | `argmin_sparse_admm_polished` | 1/1 | 559 | 556 - 563 | 1480 / 0.180 / 559 | 13900000 / 1070 / 5580000 / 5580000 | 25 / 25 / 25 | 3.59e-6 | none | 83 |
| nx8_nu3_N20 | nominal | `osqp_polished` | 1/1 | 606 | 595 - 741 | 258 / 1.07 / 605 | 2480000 / 9080 / 5680000 / 5690000 | 50 / 50 / 100 | 3.59e-6 | none | 60 |
| nx8_nu3_N20 | tight | `argmin_sparse_admm` | 1/1 | 502 | 500 - 505 | 2370 / 0.180 / 502 | 25700000 / 1220 / 5660000 / 5670000 | 25 / 50 / 75 | 5.68e-4 | none | 0 |
| nx8_nu3_N20 | tight | `osqp` | 1/1 | 427 | 425 - 434 | 261 / 1.25 / 425 | 2580000 / 11000 / 4060000 / 4070000 | 50 / 50 / 125 | 6.40e-4 | none | 0 |
| nx8_nu3_N20 | tight | `proxqp_sparse` | 1/20 | 2690 | 359 - 3650 | 5220 / 43.2 / 2650 | 54200000 / 418000 / 28500000 / 28900000 | 0 / 2 / 226 | 1.00e+0 | infeasible_claim 285 | 44 |
| nx8_nu3_N20 | tight | `hpipm` | 1/1 | 176 | 152 - 201 | 22.9 / 0.0900 / 176 | 183000 / 365 / 1750000 / 1750000 | 5 / 7 / 11 | 1.83e-5 | none | 0 |
| nx8_nu3_N20 | tight | `argmin_sparse_admm_polished` | 11/12 | 1070 | 825 - 1560 | 2770 / 0.190 / 1070 | 28700000 / 1220 / 11400000 / 11400000 | 25 / 75 / 300 | 6.66e-1 | inaccurate 25 | 87 |
| nx8_nu3_N20 | tight | `osqp_polished` | 1/1 | 614 | 609 - 619 | 265 / 1.26 / 613 | 2580000 / 11000 / 5880000 / 5890000 | 50 / 50 / 125 | 6.40e-4 | none | 60 |
| nx8_nu3_N30 | nominal | `argmin_sparse_admm` | 1/1 | 371 | 368 - 374 | 1710 / 0.210 / 371 | 16300000 / 1530 / 4000000 / 4000000 | 25 / 25 / 25 | 1.30e-4 | none | 0 |
| nx8_nu3_N30 | nominal | `osqp` | 1/1 | 657 | 621 - 682 | 394 / 1.42 / 656 | 3670000 / 13100 / 6200000 / 6220000 | 50 / 50 / 75 | 3.63e-4 | none | 0 |
| nx8_nu3_N30 | nominal | `proxqp_sparse` | 1/1 | 887 | 823 - 1070 | 2950 / 53.0 / 834 | 29000000 / 560000 / 8990000 / 9550000 | 0 / 7 / 13 | 4.80e-4 | none | 35 |
| nx8_nu3_N30 | nominal | `hpipm` | 1/1 | 253 | 251 - 286 | 90.8 / 0.0900 / 253 | 249000 / 365 / 2510000 / 2520000 | 5 / 7 / 10 | 2.75e-5 | none | 0 |
| nx8_nu3_N30 | nominal | `argmin_sparse_admm_polished` | 1/1 | 822 | 818 - 828 | 2260 / 0.220 / 822 | 20700000 / 1530 / 8280000 / 8290000 | 25 / 25 / 25 | 3.59e-6 | none | 83 |
| nx8_nu3_N30 | nominal | `osqp_polished` | 1/1 | 923 | 885 - 954 | 394 / 1.43 / 922 | 3670000 / 13100 / 8890000 / 8900000 | 50 / 50 / 75 | 3.59e-6 | none | 60 |
| nx8_nu3_N30 | tight | `argmin_sparse_admm` | 1/1 | 756 | 751 - 767 | 3710 / 0.240 / 756 | 38300000 / 1720 / 8440000 / 8450000 | 25 / 50 / 75 | 5.77e-4 | none | 0 |
| nx8_nu3_N30 | tight | `osqp` | 1/1 | 645 | 634 - 654 | 461 / 1.60 / 644 | 3810000 / 16000 / 6040000 / 6060000 | 50 / 50 / 125 | 6.36e-4 | none | 0 |
| nx8_nu3_N30 | tight | `proxqp_sparse` | 1/20 | 5640 | 898 - 8620 | 12800 / 61.6 / 5580 | 135000000 / 619000 / 61300000 / 61900000 | 0 / 2 / 23 | 1.00e+0 | infeasible_claim 285 | 44 |
| nx8_nu3_N30 | tight | `hpipm` | 1/1 | 263 | 232 - 300 | 102 / 0.100 / 263 | 267000 / 365 / 2600000 / 2600000 | 5 / 7 / 10 | 2.39e-5 | none | 0 |
| nx8_nu3_N30 | tight | `argmin_sparse_admm_polished` | 29/30 | 1970 | 1250 - 2330 | 4290 / 0.250 / 1960 | 42800000 / 1720 / 21200000 / 21200000 | 25 / 100 / 200 | 6.66e-1 | inaccurate 10 | 87 |
| nx8_nu3_N30 | tight | `osqp_polished` | 1/1 | 913 | 905 - 920 | 465 / 1.64 / 911 | 3810000 / 16000 / 8710000 / 8730000 | 50 / 50 / 125 | 6.36e-4 | none | 60 |
| nx12_nu4_N30 | nominal | `argmin_sparse_admm` | 1/1 | 708 | 705 - 710 | 3110 / 0.270 / 708 | 31700000 / 2070 / 7620000 / 7620000 | 25 / 25 / 25 | 7.13e-5 | none | 0 |
| nx12_nu4_N30 | nominal | `osqp` | 1/1 | 1310 | 1190 - 1780 | 678 / 1.88 / 1310 | 7070000 / 18600 / 12600000 / 12600000 | 50 / 50 / 75 | 3.48e-4 | none | 0 |
| nx12_nu4_N30 | nominal | `proxqp_sparse` | 1/1 | 1810 | 1670 - 2060 | 8430 / 77.8 / 1730 | 86900000 / 819000 / 18700000 / 19500000 | 0 / 8 / 12 | 2.95e-4 | none | 35 |
| nx12_nu4_N30 | nominal | `hpipm` | 1/1 | 333 | 325 - 371 | 37.0 / 0.0900 / 333 | 290000 / 386 / 3290000 / 3300000 | 6 / 7 / 11 | 1.27e-5 | none | 0 |
| nx12_nu4_N30 | nominal | `argmin_sparse_admm_polished` | 1/1 | 1600 | 1600 - 1610 | 4210 / 0.270 / 1600 | 40500000 / 2070 / 16300000 / 16300000 | 25 / 25 / 25 | 5.81e-5 | none | 82 |
| nx12_nu4_N30 | nominal | `osqp_polished` | 1/1 | 1860 | 1740 - 2330 | 683 / 1.89 / 1860 | 7070000 / 18600 / 18200000 / 18200000 | 50 / 50 / 75 | 1.76e-4 | none | 60 |
| nx12_nu4_N30 | tight | `argmin_sparse_admm` | 1/1 | 1470 | 1450 - 2190 | 4050 / 0.320 / 1470 | 40600000 / 2400 / 15600000 / 15600000 | 25 / 50 / 225 | 5.04e-4 | none | 5 |
| nx12_nu4_N30 | tight | `osqp` | 1/1 | 1220 | 1220 - 1230 | 714 / 2.17 / 1220 | 7280000 / 22400 / 11800000 / 11800000 | 50 / 50 / 125 | 5.88e-4 | none | 0 |
| nx12_nu4_N30 | tight | `proxqp_sparse` | 1/20 | 22200 | 1850 - 24100 | 35400 / 88.5 / 22100 | 370000000 / 896000 / 238000000 / 239000000 | 0 / 2 / 293 | 1.00e+0 | inaccurate 5, infeasible_claim 280 | 44 |
| nx12_nu4_N30 | tight | `hpipm` | 1/1 | 346 | 298 - 394 | 39.1 / 0.0900 / 346 | 311000 / 386 / 3460000 / 3460000 | 6 / 7 / 11 | 2.44e-5 | none | 0 |
| nx12_nu4_N30 | tight | `argmin_sparse_admm_polished` | 14/15 | 3110 | 2400 - 3830 | 5200 / 0.320 / 3110 | 49600000 / 2410 / 32200000 / 32200000 | 25 / 75 / 200 | 6.65e-1 | inaccurate 20 | 89 |
| nx12_nu4_N30 | tight | `osqp_polished` | 1/1 | 1770 | 1760 - 1780 | 716 / 2.17 / 1760 | 7280000 / 22400 / 17200000 / 17200000 | 50 / 50 / 125 | 5.90e-4 | none | 60 |
| nx4_nu2_N50 | nominal | `argmin_sparse_admm` | 1/1 | 206 | 205 - 404 | 908 / 0.210 / 205 | 10300000 / 1360 / 2580000 / 2580000 | 25 / 25 / 50 | 5.36e-4 | none | 0 |
| nx4_nu2_N50 | nominal | `osqp` | 1/1 | 403 | 383 - 433 | 220 / 1.30 / 402 | 2300000 / 11800 / 3500000 / 3520000 | 50 / 50 / 75 | 1.11e-4 | none | 0 |
| nx4_nu2_N50 | nominal | `proxqp_sparse` | 4/5 | 596 | 514 - 704 | 1750 / 46.1 / 550 | 19100000 / 494000 / 6010000 / 6500000 | 0 / 7 / 12 | 1.25e-3 | inaccurate 60 | 35 |
| nx4_nu2_N50 | nominal | `hpipm` | 1/1 | 243 | 237 - 291 | 32.9 / 0.0900 / 243 | 316000 / 352 / 2550000 / 2550000 | 5 / 6 / 9 | 8.22e-5 | none | 0 |
| nx4_nu2_N50 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 96 / 195.5 / 328 | 1.01e-2 | inaccurate 300 | 32803 |
| nx4_nu2_N50 | nominal | `argmin_sparse_admm_polished` | 1/1 | 465 | 462 - 660 | 1180 / 0.220 / 464 | 12700000 / 1360 / 5010000 / 5010000 | 25 / 25 / 50 | 3.94e-4 | none | 83 |
| nx4_nu2_N50 | nominal | `osqp_polished` | 1/1 | 557 | 539 - 587 | 224 / 1.32 / 556 | 2300000 / 11800 / 4980000 / 4990000 | 50 / 50 / 75 | 1.88e-15 | none | 60 |
| nx4_nu2_N50 | tight | `argmin_sparse_admm` | 1/1 | 857 | 645 - 870 | 513 / 0.230 / 857 | 5500000 / 1610 / 11300000 / 11300000 | 25 / 100 / 175 | 5.60e-4 | none | 5 |
| nx4_nu2_N50 | tight | `osqp` | 1/1 | 406 | 405 - 421 | 241 / 1.58 / 405 | 2470000 / 15000 / 3660000 / 3680000 | 25 / 50 / 75 | 5.97e-4 | none | 0 |
| nx4_nu2_N50 | tight | `proxqp_sparse` | 7/30 | 7020 | 4830 - 9910 | 3530 / 54.0 / 6960 | 36000000 / 556000 / 77300000 / 77900000 | 0 / 8 / 142 | 1.00e+0 | inaccurate 5, infeasible_claim 225 | 80 |
| nx4_nu2_N50 | tight | `hpipm` | 1/1 | 248 | 245 - 288 | 35.7 / 0.0900 / 248 | 343000 / 352 / 2650000 / 2650000 | 5 / 6 / 11 | 7.80e-5 | none | 0 |
| nx4_nu2_N50 | tight | `tinympc` | 19/20 | 2710 | 1100 - 4440 | 21.4 / 0.100 / 2710 | 113000 / 397 / 37700000 / 37700000 | 50 / 503.5 / 2217 | 6.46e-1 | inaccurate 15 | 221703 |
| nx4_nu2_N50 | tight | `argmin_sparse_admm_polished` | 29/30 | 1130 | 921 - 1150 | 802 / 0.240 / 1130 | 8170000 / 1610 / 14000000 / 14000000 | 25 / 100 / 175 | 6.63e-1 | inaccurate 10 | 84 |
| nx4_nu2_N50 | tight | `osqp_polished` | 1/1 | 568 | 566 - 572 | 241 / 1.58 / 566 | 2460000 / 15000 / 5220000 / 5240000 | 25 / 50 / 75 | 5.97e-4 | none | 60 |
| nx4_nu2_N100 | nominal | `argmin_sparse_admm` | 1/1 | 416 | 413 - 813 | 1800 / 0.340 / 415 | 20400000 / 2520 / 5120000 / 5120000 | 25 / 25 / 50 | 5.35e-4 | none | 0 |
| nx4_nu2_N100 | nominal | `osqp` | 1/1 | 799 | 763 - 815 | 424 / 2.27 / 797 | 4540000 / 22600 / 6930000 / 6960000 | 50 / 50 / 75 | 1.11e-4 | none | 0 |
| nx4_nu2_N100 | nominal | `proxqp_sparse` | 4/5 | 1220 | 1040 - 1420 | 6240 / 88.7 / 1130 | 60200000 / 973000 / 12100000 / 13000000 | 0 / 7 / 12 | 1.25e-3 | inaccurate 60 | 35 |
| nx4_nu2_N100 | nominal | `hpipm` | 1/1 | 482 | 473 - 563 | 59.6 / 0.0900 / 482 | 618000 / 352 / 5070000 / 5070000 | 5 / 6 / 9 | 9.64e-5 | none | 0 |
| nx4_nu2_N100 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 96 / 195.5 / 328 | 1.01e-2 | inaccurate 300 | 65603 |
| nx4_nu2_N100 | nominal | `argmin_sparse_admm_polished` | 1/1 | 918 | 913 - 1310 | 2360 / 0.330 / 918 | 25200000 / 2530 / 9880000 / 9880000 | 25 / 25 / 50 | 3.93e-4 | none | 83 |
| nx4_nu2_N100 | nominal | `osqp_polished` | 1/1 | 1100 | 1070 - 1120 | 427 / 2.29 / 1100 | 4540000 / 22600 / 9840000 / 9870000 | 50 / 50 / 75 | 1.25e-15 | none | 60 |
| nx4_nu2_N100 | tight | `argmin_sparse_admm` | 1/1 | 1740 | 1310 - 1770 | 1020 / 0.420 / 1740 | 10900000 / 3030 / 22500000 / 22500000 | 25 / 100 / 175 | 5.60e-4 | none | 5 |
| nx4_nu2_N100 | tight | `osqp` | 1/1 | 823 | 816 - 829 | 461 / 2.70 / 821 | 4870000 / 29000 / 7250000 / 7280000 | 25 / 50 / 75 | 5.97e-4 | none | 0 |
| nx4_nu2_N100 | tight | `proxqp_sparse` | 7/30 | 31200 | 20200 - 50100 | 16200 / 103 / 31100 | 154000000 / 1100000 / 331000000 / 333000000 | 0 / 5 / 52 | 1.00e+0 | infeasible_claim 230 | 89 |
| nx4_nu2_N100 | tight | `hpipm` | 1/1 | 499 | 494 - 580 | 206 / 0.0900 / 499 | 674000 / 352 / 5250000 / 5250000 | 5 / 6 / 11 | 7.74e-5 | none | 0 |
| nx4_nu2_N100 | tight | `tinympc` | 19/20 | 5350 | 2160 - 8760 | 26.6 / 0.130 / 5350 | 129000 / 397 / 74700000 / 74700000 | 50 / 503.5 / 2217 | 6.46e-1 | inaccurate 15 | 443403 |
| nx4_nu2_N100 | tight | `argmin_sparse_admm_polished` | 29/30 | 2290 | 1850 - 2320 | 1840 / 0.430 / 2290 | 16200000 / 3030 / 27700000 / 27700000 | 25 / 100 / 175 | 6.63e-1 | inaccurate 10 | 84 |
| nx4_nu2_N100 | tight | `osqp_polished` | 1/1 | 1140 | 1130 - 1150 | 565 / 2.72 / 1140 | 4870000 / 29000 / 10300000 / 10400000 | 25 / 50 / 75 | 5.97e-4 | none | 60 |
| nx8_nu3_N50 | nominal | `argmin_sparse_admm` | 1/1 | 619 | 616 - 627 | 2620 / 0.320 / 619 | 27000000 / 2380 / 6620000 / 6620000 | 25 / 25 / 50 | 3.79e-4 | none | 0 |
| nx8_nu3_N50 | nominal | `osqp` | 1/1 | 1020 | 1020 - 1030 | 565 / 2.08 / 1020 | 6040000 / 21000 / 9550000 / 9570000 | 50 / 50 / 100 | 2.40e-4 | none | 0 |
| nx8_nu3_N50 | nominal | `proxqp_sparse` | 1/1 | 1530 | 1420 - 1850 | 6250 / 85.4 / 1440 | 68200000 / 916000 / 15300000 / 16200000 | 0 / 7 / 13 | 4.70e-4 | none | 35 |
| nx8_nu3_N50 | nominal | `hpipm` | 1/1 | 421 | 416 - 477 | 44.3 / 0.0900 / 420 | 403000 / 365 / 4160000 / 4160000 | 5 / 7 / 11 | 7.46e-5 | none | 0 |
| nx8_nu3_N50 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 142 / 216.5 / 326 | 1.03e-2 | inaccurate 300 | 32603 |
| nx8_nu3_N50 | nominal | `argmin_sparse_admm_polished` | 1/1 | 1350 | 1350 - 1370 | 3370 / 0.320 / 1350 | 34100000 / 2380 / 13700000 / 13700000 | 25 / 25 / 50 | 3.59e-6 | none | 83 |
| nx8_nu3_N50 | nominal | `osqp_polished` | 1/1 | 1460 | 1460 - 1480 | 572 / 2.09 / 1460 | 6040000 / 21000 / 14000000 / 14000000 | 50 / 50 / 100 | 3.59e-6 | none | 60 |
| nx8_nu3_N50 | tight | `argmin_sparse_admm` | 1/1 | 1890 | 1270 - 2500 | 3360 / 0.360 / 1890 | 35700000 / 2750 / 20900000 / 20900000 | 25 / 75 / 100 | 5.80e-4 | none | 0 |
| nx8_nu3_N50 | tight | `osqp` | 1/1 | 1070 | 1060 - 1670 | 601 / 2.48 / 1070 | 6280000 / 25800 / 10000000 / 10000000 | 50 / 50 / 225 | 4.98e-4 | none | 0 |
| nx8_nu3_N50 | tight | `proxqp_sparse` | 1/20 | 14100 | 2110 - 27100 | 46600 / 97.4 / 14000 | 479000000 / 1010000 / 149000000 / 150000000 | 0 / 2 / 67 | 1.00e+0 | infeasible_claim 285 | 44 |
| nx8_nu3_N50 | tight | `hpipm` | 1/1 | 435 | 380 - 501 | 47.8 / 0.0900 / 435 | 433000 / 365 / 4300000 / 4300000 | 5 / 7 / 10 | 7.04e-5 | none | 0 |
| nx8_nu3_N50 | tight | `tinympc` | 11/12 | 4640 | 3100 - 9630 | 32.8 / 0.130 / 4640 | 218000 / 412 / 67200000 / 67200000 | 71 / 546.5 / 2367 | 6.66e-1 | inaccurate 25 | 236703 |
| nx8_nu3_N50 | tight | `argmin_sparse_admm_polished` | 29/30 | 2660 | 2040 - 3270 | 4440 / 0.360 / 2660 | 43200000 / 2750 / 28300000 / 28300000 | 25 / 75 / 225 | 6.65e-1 | inaccurate 10 | 87 |
| nx8_nu3_N50 | tight | `osqp_polished` | 1/1 | 1510 | 1500 - 2250 | 606 / 2.48 / 1510 | 6280000 / 25800 / 14400000 / 14400000 | 50 / 50 / 225 | 5.03e-4 | none | 60 |
| nx8_nu3_N100 | nominal | `argmin_sparse_admm` | 1/1 | 1260 | 1250 - 2480 | 5400 / 0.630 / 1260 | 53700000 / 4420 / 13200000 / 13200000 | 25 / 25 / 50 | 5.23e-4 | none | 0 |
| nx8_nu3_N100 | nominal | `osqp` | 1/1 | 2060 | 2050 - 2080 | 1130 / 3.78 / 2060 | 12000000 / 40800 / 18800000 / 18900000 | 50 / 50 / 100 | 3.16e-4 | none | 0 |
| nx8_nu3_N100 | nominal | `proxqp_sparse` | 1/1 | 3500 | 3230 - 4220 | 24200 / 165 / 3330 | 238000000 / 1810000 / 31400000 / 33200000 | 0 / 7 / 13 | 4.70e-4 | none | 35 |
| nx8_nu3_N100 | nominal | `hpipm` | 1/1 | 850 | 840 - 959 | 88.9 / 0.110 / 850 | 790000 / 365 / 8280000 / 8280000 | 6 / 7 / 10 | 7.13e-5 | none | 0 |
| nx8_nu3_N100 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 142 / 216.5 / 326 | 1.15e-2 | inaccurate 300 | 65203 |
| nx8_nu3_N100 | nominal | `argmin_sparse_admm_polished` | 1/1 | 2700 | 2680 - 3900 | 7200 / 0.640 / 2700 | 67800000 / 4420 / 27100000 / 27100000 | 25 / 25 / 50 | 3.59e-6 | none | 83 |
| nx8_nu3_N100 | nominal | `osqp_polished` | 1/1 | 2940 | 2930 - 3080 | 1110 / 3.80 / 2940 | 12000000 / 40800 / 27800000 / 27800000 | 50 / 50 / 125 | 4.90e-5 | none | 60 |
| nx8_nu3_N100 | tight | `argmin_sparse_admm` | 1/1 | 3850 | 2940 - 5100 | 6790 / 0.740 / 3850 | 71000000 / 5170 / 41600000 / 41600000 | 25 / 75 / 100 | 5.80e-4 | none | 0 |
| nx8_nu3_N100 | tight | `osqp` | 1/1 | 2170 | 2150 - 3400 | 1180 / 4.60 / 2160 | 12400000 / 50400 / 19900000 / 20000000 | 50 / 50 / 225 | 4.88e-4 | none | 0 |
| nx8_nu3_N100 | tight | `proxqp_sparse` | 1/20 | 71900 | 23500 - 177000 | 295000 / 188 / 71800 | 3020000000 / 1990000 / 763000000 / 765000000 | 0 / 2 / 69 | 1.00e+0 | infeasible_claim 285 | 44 |
| nx8_nu3_N100 | tight | `hpipm` | 1/1 | 891 | 771 - 1010 | 109 / 0.130 / 891 | 851000 / 365 / 8540000 / 8540000 | 5 / 7 / 10 | 8.26e-5 | none | 0 |
| nx8_nu3_N100 | tight | `tinympc` | 11/12 | 9340 | 6210 - 19200 | 45.1 / 0.180 / 9340 | 235000 / 412 / 133000000 / 133000000 | 71 / 546.5 / 2367 | 6.66e-1 | inaccurate 25 | 473403 |
| nx8_nu3_N100 | tight | `argmin_sparse_admm_polished` | 29/30 | 5360 | 4090 - 6600 | 8760 / 0.760 / 5360 | 85800000 / 5170 / 56200000 / 56200000 | 25 / 75 / 225 | 6.65e-1 | inaccurate 10 | 88 |
| nx8_nu3_N100 | tight | `osqp_polished` | 1/1 | 3040 | 3020 - 4490 | 1190 / 4.61 / 3040 | 12400000 / 50400 / 28600000 / 28600000 | 50 / 50 / 225 | 5.03e-4 | none | 60 |

### sparse cohort, closed_loop regime, high tier

The sparse cohort in the closed_loop regime, judged at the high tier's threshold of 1e-6.

| Cell | Variant | Arm | Pass rate | Median tick (us) | Tick p25 - p75 (us) | Median setup / update / solve (us) | Median instructions setup / update / solve / tick | Iterations min / median / max | Worst KKT residual | Failures | Allocations max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nx2_nu1_N10 | nominal | `argmin_sparse_admm` | 1/1 | 78.2 | 60.9 - 94.5 | 140 / 0.0400 / 78.2 | 1360000 / 374 / 923000 / 923000 | 25 / 75 / 150 | 1.88e-16 | none | 91 |
| nx2_nu1_N10 | nominal | `osqp` | 1/1 | 74.4 | 55.3 - 92.9 | 31.2 / 0.260 / 74.1 | 216000 / 2010 / 603000 / 605000 | 50 / 75 / 225 | 2.60e-16 | none | 60 |
| nx2_nu1_N10 | nominal | `proxqp_sparse` | 23/30 | 47.6 | 35.0 - 61.8 | 131 / 5.86 / 41.8 | 1110000 / 56000 / 428000 / 484000 | 0 / 5 / 20 | 8.70e-2 | inaccurate 70 | 107 |
| nx2_nu1_N10 | nominal | `hpipm` | 1/1 | 69.3 | 60.7 - 77.4 | 27.6 / 0.0500 / 69.2 | 66600 / 335 / 709000 / 710000 | 5 / 7 / 11 | 1.10e-8 | none | 0 |
| nx2_nu1_N10 | tight | `argmin_sparse_admm` | 1/1 | 110 | 91.8 - 128 | 108 / 0.0400 / 110 | 979000 / 391 / 1530000 / 1530000 | 25 / 125 / 325 | 1.11e-16 | none | 74 |
| nx2_nu1_N10 | tight | `osqp` | 1/1 | 93.8 | 92.0 - 113 | 33.4 / 0.300 / 93.5 | 230000 / 2360 / 805000 / 807000 | 50 / 100 / 325 | 2.22e-16 | none | 60 |
| nx2_nu1_N10 | tight | `proxqp_sparse` | 17/60 | 165 | 107 - 231 | 101 / 6.78 / 159 | 775000 / 62200 / 1530000 / 1590000 | 0 / 2 / 170 | 1.00e+0 | infeasible_claim 215 | 116 |
| nx2_nu1_N10 | tight | `hpipm` | 1/1 | 56.3 | 47.6 - 65.7 | 17.0 / 0.0500 / 56.3 | 71400 / 335 / 633000 / 634000 | 4 / 6 / 10 | 2.88e-8 | none | 0 |
| nx4_nu2_N10 | nominal | `argmin_sparse_admm` | 1/1 | 225 | 184 - 254 | 318 / 0.0500 / 224 | 3270000 / 459 / 2670000 / 2670000 | 25 / 100 / 125 | 2.82e-16 | none | 89 |
| nx4_nu2_N10 | nominal | `osqp` | 1/1 | 198 | 157 - 247 | 62.5 / 0.400 / 198 | 502000 / 3140 / 1830000 / 1830000 | 50 / 100 / 250 | 3.31e-16 | none | 60 |
| nx4_nu2_N10 | nominal | `proxqp_sparse` | 37/60 | 137 | 122 - 174 | 264 / 11.0 / 126 | 2540000 / 109000 / 1250000 / 1360000 | 0 / 8 / 29 | 1.03e-3 | inaccurate 115 | 116 |
| nx4_nu2_N10 | nominal | `hpipm` | 1/1 | 91.4 | 86.7 - 102 | 25.9 / 0.0500 / 91.3 | 78000 / 352 / 967000 / 967000 | 6 / 8 / 12 | 3.46e-9 | none | 0 |
| nx4_nu2_N10 | tight | `argmin_sparse_admm` | 1/1 | 325 | 272 - 411 | 245 / 0.0600 / 325 | 2330000 / 509 / 4120000 / 4120000 | 25 / 150 / 350 | 1.16e-16 | none | 84 |
| nx4_nu2_N10 | tight | `osqp` | 1/1 | 319 | 241 - 349 | 66.1 / 0.450 / 319 | 537000 / 3780 / 2900000 / 2900000 | 50 / 162.5 / 350 | 2.22e-16 | none | 60 |
| nx4_nu2_N10 | tight | `proxqp_sparse` | 1/20 | 399 | 38.4 - 897 | 239 / 13.9 / 385 | 2070000 / 121000 / 4090000 / 4220000 | 0 / 3 / 719 | 1.00e+0 | infeasible_claim 285 | 98 |
| nx4_nu2_N10 | tight | `hpipm` | 1/1 | 91.2 | 80.2 - 102 | 12.9 / 0.0600 / 91.1 | 83500 / 352 / 1010000 / 1010000 | 6 / 8 / 10 | 1.77e-8 | none | 0 |
| nx4_nu2_N20 | nominal | `argmin_sparse_admm` | 1/1 | 353 | 274 - 431 | 582 / 0.100 / 353 | 6260000 / 684 / 4110000 / 4110000 | 25 / 75 / 100 | 3.21e-16 | none | 83 |
| nx4_nu2_N20 | nominal | `osqp` | 1/1 | 314 | 297 - 372 | 104 / 0.620 / 313 | 954000 / 5320 / 2830000 / 2830000 | 50 / 75 / 125 | 5.38e-16 | none | 60 |
| nx4_nu2_N20 | nominal | `proxqp_sparse` | 23/60 | 265 | 237 - 285 | 549 / 19.8 / 245 | 5620000 / 205000 / 2570000 / 2770000 | 0 / 7 / 20 | 1.25e-3 | inaccurate 185 | 44 |
| nx4_nu2_N20 | nominal | `hpipm` | 1/1 | 170 | 167 - 207 | 17.8 / 0.0600 / 169 | 140000 / 352 / 1880000 / 1880000 | 7 / 8 / 12 | 2.01e-9 | none | 0 |
| nx4_nu2_N20 | tight | `argmin_sparse_admm` | 1/1 | 592 | 467 - 1060 | 448 / 0.120 / 592 | 4500000 / 784 / 6840000 / 6840000 | 25 / 125 / 550 | 1.22e-16 | none | 84 |
| nx4_nu2_N20 | tight | `osqp` | 1/1 | 564 | 405 - 732 | 114 / 0.790 / 563 | 1030000 / 6600 / 5130000 / 5140000 | 50 / 150 / 300 | 2.22e-16 | none | 60 |
| nx4_nu2_N20 | tight | `proxqp_sparse` | 1/12 | 1400 | 1230 - 1500 | 659 / 24.6 / 1380 | 6070000 / 230000 / 14500000 / 14700000 | 0 / 2 / 523 | 1.00e+0 | infeasible_claim 275 | 89 |
| nx4_nu2_N20 | tight | `hpipm` | 1/1 | 177 | 154 - 199 | 23.4 / 0.0700 / 177 | 151000 / 352 / 1950000 / 1950000 | 6 / 8 / 11 | 2.68e-8 | none | 0 |
| nx8_nu3_N20 | nominal | `argmin_sparse_admm` | 1/1 | 1270 | 1080 - 1350 | 1930 / 0.180 / 1270 | 19200000 / 1070 / 13400000 / 13400000 | 25 / 100 / 200 | 1.24e-10 | none | 88 |
| nx8_nu3_N20 | nominal | `osqp` | 1/1 | 793 | 597 - 822 | 249 / 1.06 / 791 | 2480000 / 9080 / 7620000 / 7630000 | 50 / 75 / 300 | 4.24e-16 | none | 60 |
| nx8_nu3_N20 | nominal | `proxqp_sparse` | 3/20 | 588 | 548 - 1340 | 1620 / 37.1 / 551 | 16100000 / 381000 / 5900000 / 6280000 | 0 / 7 / 18 | 4.75e-4 | inaccurate 255 | 44 |
| nx8_nu3_N20 | nominal | `hpipm` | 1/1 | 310 | 280 - 340 | 22.9 / 0.0900 / 310 | 178000 / 365 / 3290000 / 3290000 | 7 / 10 / 13 | 1.25e-9 | none | 0 |
| nx8_nu3_N20 | tight | `argmin_sparse_admm` | 1/1 | 2060 | 1580 - 2070 | 3510 / 0.190 / 2060 | 37100000 / 1220 / 22600000 / 22600000 | 25 / 175 / 525 | 1.48e-16 | none | 86 |
| nx8_nu3_N20 | tight | `osqp` | 1/1 | 1360 | 1230 - 1860 | 264 / 1.26 / 1360 | 2580000 / 11000 / 13000000 / 13000000 | 100 / 137.5 / 450 | 2.22e-16 | none | 60 |
| nx8_nu3_N20 | tight | `proxqp_sparse` | 1/60 | 352 | 350 - 356 | 6640 / 45.6 / 307 | 69900000 / 418000 / 3110000 / 3530000 | 0 / 2 / 34 | 1.00e+0 | infeasible_claim 295 | 41 |
| nx8_nu3_N20 | tight | `hpipm` | 1/1 | 264 | 262 - 298 | 24.3 / 0.0900 / 264 | 190000 / 365 / 2720000 / 2720000 | 6 / 8 / 11 | 1.38e-8 | none | 0 |
| nx8_nu3_N30 | nominal | `argmin_sparse_admm` | 1/1 | 1200 | 1180 - 1570 | 2960 / 0.220 / 1200 | 28500000 / 1530 / 12200000 / 12200000 | 25 / 50 / 125 | 2.74e-11 | none | 88 |
| nx8_nu3_N30 | nominal | `osqp` | 1/1 | 1230 | 1200 - 1400 | 413 / 1.43 / 1220 | 3670000 / 13100 / 11300000 / 11400000 | 50 / 75 / 125 | 7.33e-16 | none | 60 |
| nx8_nu3_N30 | nominal | `proxqp_sparse` | 3/20 | 910 | 850 - 2290 | 3070 / 53.4 / 858 | 30400000 / 560000 / 9240000 / 9800000 | 0 / 7 / 18 | 4.80e-4 | inaccurate 255 | 44 |
| nx8_nu3_N30 | nominal | `hpipm` | 1/1 | 465 | 419 - 510 | 113 / 0.0900 / 465 | 259000 / 365 / 4890000 / 4890000 | 7 / 10 / 13 | 1.05e-9 | none | 0 |
| nx8_nu3_N30 | tight | `argmin_sparse_admm` | 1/1 | 3090 | 2380 - 3120 | 5420 / 0.250 / 3090 | 55300000 / 1720 / 33800000 / 33800000 | 25 / 175 / 525 | 4.44e-16 | none | 86 |
| nx8_nu3_N30 | tight | `osqp` | 1/1 | 2170 | 1850 - 2830 | 464 / 1.63 / 2170 | 3810000 / 16000 / 20600000 / 20600000 | 100 / 150 / 475 | 1.86e-16 | none | 60 |
| nx8_nu3_N30 | tight | `proxqp_sparse` | 1/60 | 898 | 892 - 898 | 16900 / 63.6 / 834 | 178000000 / 619000 / 8640000 / 9260000 | 0 / 2 / 27 | 1.00e+0 | infeasible_claim 295 | 32 |
| nx8_nu3_N30 | tight | `hpipm` | 1/1 | 397 | 380 - 443 | 124 / 0.0900 / 396 | 278000 / 365 / 4030000 / 4030000 | 7 / 8 / 11 | 1.34e-8 | none | 0 |
| nx12_nu4_N30 | nominal | `argmin_sparse_admm` | 1/1 | 3580 | 3120 - 3850 | 5560 / 0.270 / 3580 | 54900000 / 2090 / 35900000 / 36000000 | 25 / 87.5 / 150 | 1.19e-9 | none | 91 |
| nx12_nu4_N30 | nominal | `osqp` | 1/1 | 2510 | 2330 - 2590 | 678 / 1.89 / 2510 | 7070000 / 18600 / 24500000 / 24500000 | 50 / 75 / 225 | 7.81e-16 | none | 60 |
| nx12_nu4_N30 | nominal | `proxqp_sparse` | 1/15 | 5500 | 4100 - 5690 | 8720 / 79.4 / 5420 | 90200000 / 819000 / 59800000 / 60600000 | 0 / 8 / 17 | 2.74e-4 | inaccurate 280 | 53 |
| nx12_nu4_N30 | nominal | `hpipm` | 1/1 | 621 | 566 - 675 | 40.2 / 0.0900 / 621 | 306000 / 386 / 6640000 / 6640000 | 7 / 11 / 14 | 6.10e-10 | none | 0 |
| nx12_nu4_N30 | tight | `argmin_sparse_admm` | 1/1 | 8860 | 5120 - 9950 | 7380 / 0.320 / 8860 | 72700000 / 2400 / 93300000 / 93300000 | 25 / 275 / 600 | 1.06e-13 | none | 90 |
| nx12_nu4_N30 | tight | `osqp` | 1/1 | 4750 | 4160 - 5350 | 730 / 2.19 / 4750 | 7280000 / 22400 / 46200000 / 46200000 | 75 / 175 / 550 | 2.28e-16 | none | 60 |
| nx12_nu4_N30 | tight | `proxqp_sparse` | 1/12 | 28000 | 25000 - 59100 | 46600 / 88.4 / 27900 | 484000000 / 896000 / 299000000 / 300000000 | 0 / 2.5 / 88 | 1.00e+0 | infeasible_claim 275 | 116 |
| nx12_nu4_N30 | tight | `hpipm` | 1/1 | 538 | 483 - 551 | 42.6 / 0.0900 / 538 | 328000 / 386 / 5740000 / 5740000 | 7 / 9 / 11 | 1.49e-8 | none | 0 |
| nx4_nu2_N50 | nominal | `argmin_sparse_admm` | 1/1 | 862 | 665 - 1060 | 1580 / 0.220 / 862 | 17800000 / 1360 / 10000000 / 10000000 | 25 / 75 / 100 | 1.02e-15 | none | 83 |
| nx4_nu2_N50 | nominal | `osqp` | 1/1 | 763 | 722 - 908 | 225 / 1.32 / 762 | 2300000 / 11800 / 6840000 / 6850000 | 50 / 75 / 175 | 1.88e-15 | none | 60 |
| nx4_nu2_N50 | nominal | `proxqp_sparse` | 23/60 | 642 | 594 - 685 | 1870 / 46.7 / 595 | 20300000 / 494000 / 6450000 / 6950000 | 0 / 7 / 20 | 1.25e-3 | inaccurate 185 | 44 |
| nx4_nu2_N50 | nominal | `hpipm` | 1/1 | 414 | 400 - 490 | 34.5 / 0.0900 / 414 | 328000 / 352 / 4600000 / 4600000 | 6 / 8 / 12 | 2.56e-9 | none | 0 |
| nx4_nu2_N50 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 174 / 472 / 741 | 1.01e-2 | inaccurate 300 | 74103 |
| nx4_nu2_N50 | tight | `argmin_sparse_admm` | 1/1 | 1590 | 1340 - 2500 | 1010 / 0.240 / 1590 | 10800000 / 1610 / 19600000 / 19600000 | 25 / 150 / 400 | 3.68e-16 | none | 94 |
| nx4_nu2_N50 | tight | `osqp` | 1/1 | 1070 | 965 - 1560 | 242 / 1.57 / 1070 | 2460000 / 15000 / 9730000 / 9740000 | 50 / 112.5 / 275 | 6.38e-16 | none | 60 |
| nx4_nu2_N50 | tight | `proxqp_sparse` | 1/12 | 8800 | 8000 - 12900 | 3850 / 54.1 / 8750 | 39300000 / 556000 / 96300000 / 96800000 | 0 / 5 / 139 | 1.00e+0 | infeasible_claim 275 | 89 |
| nx4_nu2_N50 | tight | `hpipm` | 1/1 | 427 | 375 - 479 | 37.4 / 0.0900 / 427 | 355000 / 352 / 4770000 / 4770000 | 6 / 8 / 10 | 5.78e-8 | none | 0 |
| nx4_nu2_N50 | tight | `tinympc` | 19/20 | 4660 | 1520 - 7070 | 21.6 / 0.110 / 4660 | 113000 / 397 / 65700000 / 65700000 | 88 / 933 / 4000 | 1.31e-1 | max_iter 15 | 400003 |
| nx4_nu2_N100 | nominal | `argmin_sparse_admm` | 1/1 | 1720 | 1320 - 2110 | 3170 / 0.330 / 1720 | 35200000 / 2520 / 19900000 / 19900000 | 25 / 75 / 100 | 1.01e-15 | none | 83 |
| nx4_nu2_N100 | nominal | `osqp` | 1/1 | 1540 | 1450 - 1830 | 426 / 2.29 / 1530 | 4540000 / 22600 / 13500000 / 13600000 | 50 / 75 / 175 | 1.25e-15 | none | 60 |
| nx4_nu2_N100 | nominal | `proxqp_sparse` | 23/60 | 1300 | 1200 - 1400 | 6470 / 88.8 / 1210 | 62700000 / 973000 / 12700000 / 13700000 | 0 / 7 / 20 | 1.25e-3 | inaccurate 185 | 44 |
| nx4_nu2_N100 | nominal | `hpipm` | 1/1 | 831 | 804 - 970 | 205 / 0.0900 / 831 | 640000 / 352 / 9150000 / 9150000 | 6 / 8 / 12 | 3.18e-9 | none | 0 |
| nx4_nu2_N100 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 174 / 472 / 741 | 1.01e-2 | inaccurate 300 | 148203 |
| nx4_nu2_N100 | tight | `argmin_sparse_admm` | 1/1 | 3220 | 2690 - 5700 | 2190 / 0.430 / 3220 | 21100000 / 3030 / 38700000 / 38700000 | 25 / 150 / 400 | 4.05e-16 | none | 93 |
| nx4_nu2_N100 | tight | `osqp` | 1/1 | 2140 | 1940 - 3130 | 513 / 2.72 / 2140 | 4870000 / 29000 / 19200000 / 19300000 | 50 / 112.5 / 275 | 4.49e-16 | none | 60 |
| nx4_nu2_N100 | tight | `proxqp_sparse` | 1/15 | 59300 | 34200 - 80700 | 17100 / 104 / 59200 | 164000000 / 1100000 / 610000000 / 611000000 | 0 / 2 / 31 | 1.00e+0 | infeasible_claim 280 | 89 |
| nx4_nu2_N100 | tight | `hpipm` | 1/1 | 868 | 755 - 969 | 233 / 0.0900 / 868 | 695000 / 352 / 9480000 / 9480000 | 6 / 8 / 10 | 5.15e-8 | none | 0 |
| nx4_nu2_N100 | tight | `tinympc` | 19/20 | 9250 | 3000 - 13900 | 27.3 / 0.120 / 9250 | 129000 / 397 / 130000000 / 130000000 | 87 / 933 / 4000 | 1.31e-1 | max_iter 15 | 800003 |
| nx8_nu3_N50 | nominal | `argmin_sparse_admm` | 1/1 | 2560 | 1960 - 2570 | 3980 / 0.320 / 2560 | 40600000 / 2380 / 26600000 / 26600000 | 25 / 75 / 100 | 3.92e-11 | none | 83 |
| nx8_nu3_N50 | nominal | `osqp` | 1/1 | 2140 | 1970 - 2490 | 570 / 2.08 / 2140 | 6040000 / 21000 / 20200000 / 20300000 | 50 / 75 / 125 | 1.98e-15 | none | 60 |
| nx8_nu3_N50 | nominal | `proxqp_sparse` | 3/20 | 1520 | 1400 - 4680 | 6460 / 85.9 / 1440 | 70400000 / 916000 / 15200000 / 16100000 | 0 / 7 / 18 | 4.70e-4 | inaccurate 255 | 44 |
| nx8_nu3_N50 | nominal | `hpipm` | 1/1 | 771 | 694 - 847 | 47.0 / 0.0900 / 771 | 419000 / 365 / 8090000 / 8090000 | 7 / 10 / 13 | 9.65e-10 | none | 0 |
| nx8_nu3_N50 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 308 / 541.5 / 734 | 9.82e-3 | inaccurate 300 | 73403 |
| nx8_nu3_N50 | tight | `argmin_sparse_admm` | 1/1 | 4550 | 3920 - 5200 | 7580 / 0.370 / 4550 | 77600000 / 2750 / 49000000 / 49000000 | 25 / 150 / 525 | 3.74e-13 | none | 93 |
| nx8_nu3_N50 | tight | `osqp` | 1/1 | 3580 | 2560 - 4650 | 605 / 2.48 / 3580 | 6280000 / 25800 / 34100000 / 34100000 | 75 / 150 / 400 | 6.30e-16 | none | 60 |
| nx8_nu3_N50 | tight | `proxqp_sparse` | 1/60 | 2050 | 2040 - 2080 | 62700 / 99.3 / 1950 | 648000000 / 1010000 / 18900000 / 20000000 | 0 / 3 / 440 | 1.00e+0 | infeasible_claim 295 | 86 |
| nx8_nu3_N50 | tight | `hpipm` | 1/1 | 674 | 642 - 740 | 51.5 / 0.0900 / 674 | 450000 / 365 / 6660000 / 6660000 | 6 / 8 / 11 | 2.74e-8 | none | 0 |
| nx8_nu3_N50 | tight | `tinympc` | 19/20 | 7540 | 5310 - 13200 | 33.4 / 0.140 / 7540 | 219000 / 412 / 109000000 / 109000000 | 108 / 923.5 / 4000 | 6.65e-1 | max_iter 15 | 400003 |
| nx8_nu3_N100 | nominal | `argmin_sparse_admm` | 1/1 | 5160 | 5110 - 6380 | 9630 / 0.640 / 5160 | 93600000 / 4420 / 52900000 / 52900000 | 25 / 75 / 100 | 5.33e-11 | none | 83 |
| nx8_nu3_N100 | nominal | `osqp` | 1/1 | 4300 | 3960 - 5010 | 1120 / 3.79 / 4290 | 12000000 / 40800 / 40200000 / 40300000 | 50 / 75 / 125 | 3.14e-15 | none | 60 |
| nx8_nu3_N100 | nominal | `proxqp_sparse` | 3/20 | 3470 | 3240 - 14300 | 24500 / 165 / 3310 | 243000000 / 1810000 / 31000000 / 32900000 | 0 / 7 / 18 | 4.70e-4 | inaccurate 255 | 44 |
| nx8_nu3_N100 | nominal | `hpipm` | 1/1 | 1580 | 1420 - 1730 | 325 / 0.110 / 1580 | 821000 / 365 / 16100000 / 16100000 | 7 / 10 / 13 | 9.00e-10 | none | 0 |
| nx8_nu3_N100 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 308 / 541.5 / 734 | 1.10e-2 | inaccurate 300 | 146803 |
| nx8_nu3_N100 | tight | `argmin_sparse_admm` | 1/1 | 9120 | 7900 - 10400 | 14900 / 0.760 / 9110 | 153000000 / 5170 / 96400000 / 96400000 | 25 / 150 / 525 | 6.01e-13 | none | 93 |
| nx8_nu3_N100 | tight | `osqp` | 1/1 | 6380 | 5160 - 10700 | 1180 / 4.61 / 6380 | 12400000 / 50400 / 59600000 / 59600000 | 75 / 125 / 400 | 5.68e-16 | none | 60 |
| nx8_nu3_N100 | tight | `proxqp_sparse` | 1/60 | 23500 | 23400 - 23500 | 413000 / 189 / 23300 | 4210000000 / 1990000 / 227000000 / 229000000 | 0 / 2 / 172 | 1.00e+0 | infeasible_claim 295 | 50 |
| nx8_nu3_N100 | tight | `hpipm` | 1/1 | 1350 | 1320 - 1510 | 129 / 0.130 / 1350 | 881000 / 365 / 13200000 / 13200000 | 6 / 8 / 11 | 2.90e-8 | none | 0 |
| nx8_nu3_N100 | tight | `tinympc` | 19/20 | 15100 | 10600 - 26600 | 45.3 / 0.180 / 15100 | 235000 / 412 / 217000000 / 217000000 | 108 / 923.5 / 4000 | 6.65e-1 | max_iter 15 | 800003 |

### sparse cohort, churning regime, mpc tier

The sparse cohort in the churning regime, judged at the mpc tier's threshold of 1e-3.

| Cell | Variant | Arm | Pass rate | Median tick (us) | Tick p25 - p75 (us) | Median setup / update / solve (us) | Median instructions setup / update / solve / tick | Iterations min / median / max | Worst KKT residual | Failures | Allocations max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nx2_nu1_N10 | nominal | `argmin_sparse_admm` | 1/1 | 17.9 | 17.8 - 18.1 | 88.7 / 0.0400 / 17.9 | 916000 / 391 / 232000 / 232000 | 25 / 25 / 75 | 2.72e-4 | none | 5 |
| nx2_nu1_N10 | nominal | `osqp` | 1/1 | 37.6 | 37.3 - 38.0 | 27.5 / 0.250 / 37.4 | 217000 / 2010 / 298000 / 300000 | 25 / 50 / 100 | 5.55e-4 | none | 0 |
| nx2_nu1_N10 | nominal | `proxqp_sparse` | 69/100 | 38.7 | 27.6 - 48.8 | 67.0 / 5.87 / 32.9 | 484000 / 56100 / 329000 / 385000 | 0 / 4 / 11 | 1.09e-1 | inaccurate 93 | 44 |
| nx2_nu1_N10 | nominal | `hpipm` | 1/1 | 38.4 | 32.7 - 43.9 | 10.6 / 0.0500 / 38.4 | 63600 / 335 / 438000 / 438000 | 3 / 6 / 11 | 3.11e-4 | none | 0 |
| nx2_nu1_N10 | nominal | `argmin_sparse_admm_polished` | 1/1 | 40.1 | 39.4 - 41.2 | 121 / 0.0400 / 40.0 | 1110000 / 391 / 431000 / 431000 | 25 / 25 / 75 | 1.85e-5 | none | 86 |
| nx2_nu1_N10 | nominal | `osqp_polished` | 1/1 | 51.9 | 51.3 - 52.7 | 31.8 / 0.260 / 51.6 | 217000 / 2040 / 422000 / 424000 | 25 / 50 / 100 | 1.56e-5 | none | 60 |
| nx2_nu1_N10 | tight | `argmin_sparse_admm` | 1/1 | 36.2 | 35.4 - 52.7 | 77.2 / 0.0400 / 36.2 | 763000 / 395 / 530000 / 531000 | 25 / 50 / 200 | 6.45e-4 | none | 10 |
| nx2_nu1_N10 | tight | `osqp` | 1/1 | 40.0 | 39.7 - 58.3 | 29.4 / 0.280 / 39.7 | 231000 / 2360 / 339000 / 341000 | 25 / 50 / 250 | 6.36e-4 | none | 0 |
| nx2_nu1_N10 | tight | `proxqp_sparse` | 7/75 | 122 | 58.8 - 185 | 139 / 6.99 / 115 | 1060000 / 62100 / 1160000 / 1220000 | 0 / 2 / 84 | 1.00e+0 | inaccurate 11, infeasible_claim 261 | 86 |
| nx2_nu1_N10 | tight | `hpipm` | 1/1 | 40.5 | 34.3 - 46.8 | 11.0 / 0.0500 / 40.4 | 68400 / 335 / 455000 / 455000 | 4 / 6 / 10 | 2.92e-5 | none | 0 |
| nx2_nu1_N10 | tight | `argmin_sparse_admm_polished` | 289/300 | 60.0 | 58.7 - 76.5 | 111 / 0.0400 / 59.9 | 984000 / 395 / 749000 / 750000 | 25 / 50 / 200 | 6.65e-1 | inaccurate 11 | 90 |
| nx2_nu1_N10 | tight | `osqp_polished` | 1/1 | 55.4 | 54.7 - 73.0 | 34.0 / 0.300 / 55.1 | 233000 / 2360 / 473000 / 476000 | 25 / 50 / 225 | 6.34e-4 | none | 60 |
| nx4_nu2_N10 | nominal | `argmin_sparse_admm` | 1/1 | 42.4 | 42.2 - 42.9 | 205 / 0.0500 / 42.4 | 2190000 / 459 / 545000 / 545000 | 25 / 25 / 75 | 6.52e-4 | none | 5 |
| nx4_nu2_N10 | nominal | `osqp` | 1/1 | 78.5 | 78.0 - 82.9 | 57.2 / 0.370 / 78.2 | 503000 / 3160 / 727000 / 730000 | 25 / 50 / 100 | 5.28e-4 | none | 0 |
| nx4_nu2_N10 | nominal | `proxqp_sparse` | 61/100 | 99.9 | 87.2 - 124 | 176 / 10.8 / 89.0 | 1580000 / 109000 / 914000 / 1020000 | 0 / 5 / 15 | 8.15e-2 | inaccurate 117 | 53 |
| nx4_nu2_N10 | nominal | `hpipm` | 1/1 | 52.8 | 50.4 - 59.1 | 11.6 / 0.0500 / 52.7 | 74400 / 352 / 534000 / 535000 | 4 / 6 / 11 | 1.44e-4 | none | 0 |
| nx4_nu2_N10 | nominal | `argmin_sparse_admm_polished` | 1/1 | 102 | 100 - 106 | 279 / 0.0500 / 102 | 2730000 / 459 / 1070000 / 1070000 | 25 / 25 / 75 | 4.24e-4 | none | 88 |
| nx4_nu2_N10 | nominal | `osqp_polished` | 1/1 | 114 | 113 - 119 | 62.6 / 0.400 / 114 | 504000 / 3140 / 1050000 / 1050000 | 25 / 50 / 100 | 6.65e-7 | none | 60 |
| nx4_nu2_N10 | tight | `argmin_sparse_admm` | 1/1 | 132 | 88.8 - 175 | 214 / 0.0500 / 132 | 2350000 / 509 / 1790000 / 1790000 | 25 / 75 / 175 | 6.39e-4 | none | 5 |
| nx4_nu2_N10 | tight | `osqp` | 1/1 | 84.8 | 84.2 - 124 | 62.7 / 0.420 / 84.4 | 536000 / 3780 / 793000 / 797000 | 25 / 50 / 175 | 6.57e-4 | none | 0 |
| nx4_nu2_N10 | tight | `proxqp_sparse` | 13/150 | 306 | 231 - 622 | 307 / 13.3 / 293 | 2680000 / 121000 / 2950000 / 3070000 | 0 / 3 / 1505 | 1.00e+0 | inaccurate 5, infeasible_claim 269 | 104 |
| nx4_nu2_N10 | tight | `hpipm` | 1/1 | 60.2 | 52.3 - 68.4 | 12.3 / 0.0500 / 60.2 | 79900 / 352 / 647000 / 647000 | 5 / 7 / 11 | 2.92e-5 | none | 0 |
| nx4_nu2_N10 | tight | `argmin_sparse_admm_polished` | 24/25 | 195 | 154 - 237 | 296 / 0.0600 / 195 | 2920000 / 509 / 2340000 / 2340000 | 25 / 75 / 175 | 6.66e-1 | inaccurate 12 | 90 |
| nx4_nu2_N10 | tight | `osqp_polished` | 1/1 | 122 | 121 - 142 | 67.0 / 0.450 / 121 | 537000 / 3780 / 1130000 / 1130000 | 25 / 50 / 175 | 5.41e-4 | none | 60 |
| nx4_nu2_N20 | nominal | `argmin_sparse_admm` | 1/1 | 82.7 | 82.4 - 83.0 | 380 / 0.0900 / 82.6 | 4210000 / 684 / 1050000 / 1050000 | 25 / 25 / 50 | 6.10e-4 | none | 0 |
| nx4_nu2_N20 | nominal | `osqp` | 1/1 | 162 | 155 - 228 | 103 / 0.580 / 161 | 955000 / 5320 / 1380000 / 1390000 | 25 / 50 / 100 | 5.11e-4 | none | 0 |
| nx4_nu2_N20 | nominal | `proxqp_sparse` | 91/150 | 194 | 166 - 239 | 344 / 19.7 / 174 | 3480000 / 205000 / 1900000 / 2100000 | 0 / 5 / 13 | 1.39e-2 | inaccurate 118 | 53 |
| nx4_nu2_N20 | nominal | `hpipm` | 1/1 | 97.8 | 96.8 - 113 | 17.0 / 0.0700 / 97.7 | 135000 / 352 / 1040000 / 1040000 | 4 / 6 / 10 | 1.22e-4 | none | 0 |
| nx4_nu2_N20 | nominal | `argmin_sparse_admm_polished` | 1/1 | 191 | 190 - 196 | 500 / 0.110 / 191 | 5230000 / 684 / 2050000 / 2050000 | 25 / 25 / 50 | 1.99e-4 | none | 83 |
| nx4_nu2_N20 | nominal | `osqp_polished` | 1/1 | 226 | 219 - 294 | 103 / 0.610 / 225 | 954000 / 5300 / 2000000 / 2010000 | 25 / 50 / 100 | 2.75e-4 | none | 60 |
| nx4_nu2_N20 | tight | `argmin_sparse_admm` | 1/1 | 259 | 175 - 272 | 406 / 0.100 / 259 | 4550000 / 784 / 3470000 / 3470000 | 25 / 75 / 250 | 6.63e-4 | none | 5 |
| nx4_nu2_N20 | tight | `osqp` | 1/1 | 166 | 166 - 176 | 108 / 0.700 / 166 | 1020000 / 6600 / 1510000 / 1520000 | 25 / 50 / 300 | 5.93e-4 | none | 0 |
| nx4_nu2_N20 | tight | `proxqp_sparse` | 2/15 | 894 | 664 - 1460 | 858 / 23.9 / 870 | 8400000 / 230000 / 8960000 / 9190000 | 0 / 3 / 47736 | 1.00e+0 | inaccurate 2, max_iter 1, infeasible_claim 257 | 36020 |
| nx4_nu2_N20 | tight | `hpipm` | 1/1 | 116 | 101 - 132 | 18.7 / 0.0700 / 116 | 147000 / 352 / 1250000 / 1250000 | 5 / 7 / 12 | 2.65e-5 | none | 0 |
| nx4_nu2_N20 | tight | `argmin_sparse_admm_polished` | 283/300 | 374 | 291 - 459 | 527 / 0.120 / 374 | 5640000 / 784 / 4540000 / 4540000 | 25 / 75 / 250 | 6.66e-1 | inaccurate 17 | 89 |
| nx4_nu2_N20 | tight | `osqp_polished` | 1/1 | 236 | 234 - 241 | 112 / 0.760 / 235 | 1020000 / 6600 / 2160000 / 2170000 | 25 / 50 / 300 | 6.52e-4 | none | 60 |
| nx8_nu3_N20 | nominal | `argmin_sparse_admm` | 1/1 | 250 | 249 - 253 | 1090 / 0.170 / 249 | 11000000 / 1070 / 2680000 / 2680000 | 25 / 25 / 50 | 3.07e-4 | none | 0 |
| nx8_nu3_N20 | nominal | `osqp` | 1/1 | 419 | 413 - 461 | 249 / 1.05 / 418 | 2480000 / 9090 / 3880000 / 3890000 | 25 / 50 / 125 | 5.16e-4 | none | 0 |
| nx8_nu3_N20 | nominal | `proxqp_sparse` | 49/60 | 473 | 405 - 614 | 915 / 36.5 / 437 | 9560000 / 381000 / 4770000 / 5150000 | 0 / 6 / 14 | 2.67e-3 | inaccurate 55 | 53 |
| nx8_nu3_N20 | nominal | `hpipm` | 1/1 | 170 | 147 - 175 | 21.8 / 0.0900 / 170 | 170000 / 365 / 1690000 / 1690000 | 5 / 7 / 11 | 8.06e-5 | none | 0 |
| nx8_nu3_N20 | nominal | `argmin_sparse_admm_polished` | 1/1 | 556 | 552 - 561 | 1440 / 0.180 / 556 | 13900000 / 1070 / 5560000 / 5560000 | 25 / 25 / 50 | 2.34e-4 | none | 83 |
| nx8_nu3_N20 | nominal | `osqp_polished` | 1/1 | 602 | 596 - 674 | 250 / 1.06 / 601 | 2480000 / 9090 / 5700000 / 5710000 | 25 / 50 / 125 | 3.23e-4 | none | 60 |
| nx8_nu3_N20 | tight | `argmin_sparse_admm` | 1/1 | 503 | 500 - 507 | 1130 / 0.180 / 503 | 11700000 / 1220 / 5660000 / 5660000 | 25 / 50 / 150 | 6.49e-4 | none | 5 |
| nx8_nu3_N20 | tight | `osqp` | 1/1 | 428 | 425 - 434 | 265 / 1.24 / 426 | 2580000 / 11000 / 4060000 / 4070000 | 50 / 50 / 150 | 6.63e-4 | none | 0 |
| nx8_nu3_N20 | tight | `proxqp_sparse` | 1/15 | 4160 | 1500 - 6330 | 4710 / 43.1 / 4110 | 49500000 / 418000 / 45400000 / 45900000 | 0 / 2 / 153 | 1.00e+0 | inaccurate 1, infeasible_claim 279 | 53 |
| nx8_nu3_N20 | tight | `hpipm` | 1/1 | 178 | 153 - 202 | 23.0 / 0.0900 / 178 | 183000 / 365 / 1750000 / 1750000 | 5 / 7 / 12 | 2.93e-5 | none | 0 |
| nx8_nu3_N20 | tight | `argmin_sparse_admm_polished` | 283/300 | 1310 | 1070 - 1340 | 1500 / 0.190 / 1310 | 14700000 / 1220 / 14200000 / 14200000 | 25 / 100 / 200 | 6.66e-1 | inaccurate 17 | 88 |
| nx8_nu3_N20 | tight | `osqp_polished` | 1/1 | 613 | 609 - 619 | 265 / 1.25 / 612 | 2580000 / 11000 / 5870000 / 5880000 | 50 / 50 / 150 | 6.63e-4 | none | 60 |
| nx8_nu3_N30 | nominal | `argmin_sparse_admm` | 1/1 | 371 | 369 - 375 | 1720 / 0.210 / 371 | 16300000 / 1530 / 4000000 / 4000000 | 25 / 25 / 50 | 5.33e-4 | none | 0 |
| nx8_nu3_N30 | nominal | `osqp` | 1/1 | 633 | 615 - 914 | 388 / 1.41 / 632 | 3670000 / 13100 / 5770000 / 5780000 | 25 / 50 / 100 | 4.84e-4 | none | 0 |
| nx8_nu3_N30 | nominal | `proxqp_sparse` | 251/300 | 717 | 625 - 927 | 1880 / 52.6 / 664 | 17500000 / 560000 / 7320000 / 7880000 | 0 / 6 / 14 | 2.72e-3 | inaccurate 49 | 53 |
| nx8_nu3_N30 | nominal | `hpipm` | 1/1 | 252 | 218 - 254 | 90.4 / 0.0900 / 252 | 249000 / 365 / 2510000 / 2520000 | 5 / 7 / 10 | 1.38e-4 | none | 0 |
| nx8_nu3_N30 | nominal | `argmin_sparse_admm_polished` | 1/1 | 823 | 818 - 830 | 2260 / 0.220 / 823 | 20700000 / 1530 / 8270000 / 8270000 | 25 / 25 / 50 | 2.34e-4 | none | 83 |
| nx8_nu3_N30 | nominal | `osqp_polished` | 1/1 | 905 | 888 - 1180 | 393 / 1.43 / 904 | 3670000 / 13100 / 8470000 / 8480000 | 25 / 50 / 100 | 3.21e-4 | none | 60 |
| nx8_nu3_N30 | tight | `argmin_sparse_admm` | 1/1 | 760 | 753 - 767 | 1730 / 0.230 / 760 | 17400000 / 1720 / 8440000 / 8450000 | 25 / 50 / 150 | 6.49e-4 | none | 5 |
| nx8_nu3_N30 | tight | `osqp` | 1/1 | 651 | 644 - 654 | 458 / 1.62 / 649 | 3810000 / 16000 / 6040000 / 6060000 | 50 / 50 / 150 | 6.63e-4 | none | 0 |
| nx8_nu3_N30 | tight | `proxqp_sparse` | 1/15 | 9350 | 3270 - 13800 | 11600 / 61.5 / 9290 | 125000000 / 619000 / 103000000 / 103000000 | 0 / 2 / 61 | 1.00e+0 | inaccurate 2, infeasible_claim 278 | 53 |
| nx8_nu3_N30 | tight | `hpipm` | 1/1 | 266 | 235 - 305 | 102 / 0.100 / 266 | 267000 / 365 / 2600000 / 2600000 | 5 / 7 / 12 | 2.84e-5 | none | 0 |
| nx8_nu3_N30 | tight | `argmin_sparse_admm_polished` | 283/300 | 1970 | 1600 - 2000 | 2310 / 0.240 / 1970 | 21900000 / 1720 / 21200000 / 21200000 | 25 / 100 / 200 | 6.66e-1 | inaccurate 17 | 88 |
| nx8_nu3_N30 | tight | `osqp_polished` | 1/1 | 919 | 912 - 927 | 462 / 1.64 / 917 | 3810000 / 16000 / 8700000 / 8720000 | 50 / 50 / 150 | 6.63e-4 | none | 60 |
| nx12_nu4_N30 | nominal | `argmin_sparse_admm` | 1/1 | 709 | 705 - 711 | 3120 / 0.280 / 708 | 31700000 / 2060 / 7620000 / 7620000 | 25 / 25 / 50 | 4.00e-4 | none | 0 |
| nx12_nu4_N30 | nominal | `osqp` | 1/1 | 1310 | 1190 - 1780 | 685 / 1.90 / 1310 | 7070000 / 18600 / 12600000 / 12600000 | 25 / 50 / 100 | 5.19e-4 | none | 0 |
| nx12_nu4_N30 | nominal | `proxqp_sparse` | 19/20 | 1420 | 1240 - 1700 | 3900 / 77.8 / 1340 | 36300000 / 819000 / 14500000 / 15300000 | 0 / 6 / 14 | 2.17e-3 | inaccurate 15 | 44 |
| nx12_nu4_N30 | nominal | `hpipm` | 1/1 | 326 | 322 - 369 | 36.6 / 0.0900 / 326 | 290000 / 386 / 3290000 / 3300000 | 5 / 7 / 11 | 6.11e-5 | none | 0 |
| nx12_nu4_N30 | nominal | `argmin_sparse_admm_polished` | 1/1 | 1600 | 1590 - 1610 | 4180 / 0.270 / 1600 | 40500000 / 2060 / 16300000 / 16300000 | 25 / 25 / 50 | 3.23e-4 | none | 85 |
| nx12_nu4_N30 | nominal | `osqp_polished` | 1/1 | 1860 | 1740 - 2340 | 679 / 1.89 / 1850 | 7070000 / 18600 / 18100000 / 18200000 | 25 / 50 / 100 | 3.06e-4 | none | 60 |
| nx12_nu4_N30 | tight | `argmin_sparse_admm` | 1/1 | 1470 | 1450 - 2890 | 4060 / 0.310 / 1470 | 40600000 / 2400 / 15600000 / 15600000 | 25 / 50 / 200 | 5.97e-4 | none | 5 |
| nx12_nu4_N30 | tight | `osqp` | 1/1 | 1230 | 1220 - 1810 | 719 / 2.19 / 1220 | 7280000 / 22400 / 11800000 / 11800000 | 50 / 50 / 150 | 6.66e-4 | none | 0 |
| nx12_nu4_N30 | tight | `proxqp_sparse` | 7/150 | 22300 | 4330 - 26600 | 23700 / 89.1 / 22200 | 248000000 / 896000 / 239000000 / 240000000 | 0 / 2 / 1573 | 1.00e+0 | inaccurate 1, infeasible_claim 285 | 62 |
| nx12_nu4_N30 | tight | `hpipm` | 1/1 | 353 | 342 - 409 | 39.4 / 0.0900 / 353 | 312000 / 386 / 3460000 / 3460000 | 5 / 7 / 12 | 2.86e-5 | none | 0 |
| nx12_nu4_N30 | tight | `argmin_sparse_admm_polished` | 281/300 | 3820 | 3120 - 4540 | 5190 / 0.310 / 3820 | 49600000 / 2390 / 39900000 / 39900000 | 25 / 100 / 200 | 6.66e-1 | inaccurate 19 | 89 |
| nx12_nu4_N30 | tight | `osqp_polished` | 1/1 | 1770 | 1760 - 2370 | 738 / 2.20 / 1770 | 7280000 / 22400 / 17200000 / 17200000 | 50 / 50 / 150 | 5.91e-4 | none | 60 |
| nx4_nu2_N50 | nominal | `argmin_sparse_admm` | 1/1 | 206 | 206 - 210 | 914 / 0.200 / 206 | 10300000 / 1360 / 2580000 / 2580000 | 25 / 25 / 75 | 5.49e-4 | none | 0 |
| nx4_nu2_N50 | nominal | `osqp` | 1/1 | 385 | 378 - 565 | 225 / 1.29 / 383 | 2300000 / 11800 / 3340000 / 3350000 | 25 / 50 / 100 | 4.45e-4 | none | 0 |
| nx4_nu2_N50 | nominal | `proxqp_sparse` | 89/150 | 464 | 398 - 595 | 1120 / 46.0 / 418 | 11900000 / 494000 / 4680000 / 5170000 | 0 / 5 / 13 | 1.42e-2 | inaccurate 122 | 53 |
| nx4_nu2_N50 | nominal | `hpipm` | 1/1 | 243 | 237 - 276 | 33.0 / 0.0900 / 242 | 316000 / 352 / 2550000 / 2550000 | 4 / 6 / 11 | 1.77e-4 | none | 0 |
| nx4_nu2_N50 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 8 / 102 / 378 | 2.42e-2 | inaccurate 300 | 37803 |
| nx4_nu2_N50 | nominal | `argmin_sparse_admm_polished` | 1/1 | 464 | 461 - 467 | 1180 / 0.210 / 464 | 12700000 / 1360 / 4990000 / 4990000 | 25 / 25 / 75 | 1.04e-15 | none | 83 |
| nx4_nu2_N50 | nominal | `osqp_polished` | 1/1 | 541 | 532 - 717 | 226 / 1.32 / 540 | 2300000 / 11800 / 4830000 / 4850000 | 25 / 50 / 100 | 2.75e-4 | none | 60 |
| nx4_nu2_N50 | tight | `argmin_sparse_admm` | 1/1 | 645 | 443 - 680 | 1240 / 0.220 / 644 | 14300000 / 1610 / 8510000 / 8510000 | 25 / 75 / 175 | 6.54e-4 | none | 5 |
| nx4_nu2_N50 | tight | `osqp` | 1/1 | 407 | 405 - 411 | 237 / 1.59 / 405 | 2470000 / 15000 / 3660000 / 3680000 | 50 / 50 / 175 | 6.53e-4 | none | 0 |
| nx4_nu2_N50 | tight | `proxqp_sparse` | 17/150 | 4220 | 3140 - 8460 | 5450 / 55.1 / 4160 | 58700000 / 556000 / 47000000 / 47500000 | 0 / 2 / 179 | 1.00e+0 | inaccurate 4, infeasible_claim 262 | 89 |
| nx4_nu2_N50 | tight | `hpipm` | 1/1 | 284 | 246 - 324 | 35.7 / 0.100 / 284 | 344000 / 352 / 3070000 / 3070000 | 5 / 7 / 12 | 9.14e-5 | none | 0 |
| nx4_nu2_N50 | tight | `tinympc` | 131/150 | 2510 | 1520 - 3850 | 21.6 / 0.100 / 2510 | 113000 / 397 / 35200000 / 35200000 | 39 / 466.5 / 2056 | 6.66e-1 | inaccurate 38 | 205603 |
| nx4_nu2_N50 | tight | `argmin_sparse_admm_polished` | 93/100 | 928 | 718 - 1150 | 1540 / 0.230 / 928 | 16900000 / 1610 / 11100000 / 11100000 | 25 / 75 / 250 | 6.66e-1 | inaccurate 21 | 93 |
| nx4_nu2_N50 | tight | `osqp_polished` | 1/1 | 570 | 567 - 580 | 243 / 1.58 / 569 | 2470000 / 15000 / 5220000 / 5240000 | 50 / 50 / 175 | 6.53e-4 | none | 60 |
| nx4_nu2_N100 | nominal | `argmin_sparse_admm` | 1/1 | 415 | 412 - 418 | 1810 / 0.345 / 415 | 20400000 / 2530 / 5120000 / 5120000 | 25 / 25 / 75 | 5.48e-4 | none | 0 |
| nx4_nu2_N100 | nominal | `osqp` | 1/1 | 779 | 765 - 1140 | 438 / 2.27 / 777 | 4540000 / 22600 / 6610000 / 6630000 | 25 / 50 / 100 | 4.45e-4 | none | 0 |
| nx4_nu2_N100 | nominal | `proxqp_sparse` | 89/150 | 914 | 774 - 1190 | 3970 / 88.6 / 825 | 35600000 / 973000 / 9080000 / 10100000 | 0 / 5 / 13 | 1.42e-2 | inaccurate 122 | 53 |
| nx4_nu2_N100 | nominal | `hpipm` | 1/1 | 481 | 474 - 551 | 62.0 / 0.0900 / 481 | 618000 / 352 / 5070000 / 5070000 | 4 / 6 / 10 | 1.45e-4 | none | 0 |
| nx4_nu2_N100 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 8 / 102 / 378 | 2.42e-2 | inaccurate 300 | 75603 |
| nx4_nu2_N100 | nominal | `argmin_sparse_admm_polished` | 1/1 | 915 | 909 - 922 | 2350 / 0.340 / 914 | 25200000 / 2530 / 9860000 / 9860000 | 25 / 25 / 75 | 1.86e-8 | none | 83 |
| nx4_nu2_N100 | nominal | `osqp_polished` | 1/1 | 1080 | 1070 - 1440 | 426 / 2.28 / 1080 | 4540000 / 22600 / 9540000 / 9560000 | 25 / 50 / 100 | 2.75e-4 | none | 60 |
| nx4_nu2_N100 | tight | `argmin_sparse_admm` | 1/1 | 1310 | 888 - 1330 | 2640 / 0.430 / 1310 | 28300000 / 3030 / 16900000 / 16900000 | 25 / 75 / 175 | 6.54e-4 | none | 5 |
| nx4_nu2_N100 | tight | `osqp` | 1/1 | 826 | 817 - 834 | 460 / 2.68 / 823 | 4870000 / 29000 / 7250000 / 7280000 | 50 / 50 / 175 | 6.53e-4 | none | 0 |
| nx4_nu2_N100 | tight | `proxqp_sparse` | 9/100 | 14600 | 11500 - 38900 | 33800 / 103 / 14500 | 343000000 / 1100000 / 156000000 / 157000000 | 0 / 3 / 191 | 1.00e+0 | inaccurate 2, infeasible_claim 271 | 752 |
| nx4_nu2_N100 | tight | `hpipm` | 1/1 | 571 | 494 - 651 | 200 / 0.0900 / 571 | 674000 / 349 / 6090000 / 6090000 | 5 / 7 / 12 | 9.02e-5 | none | 0 |
| nx4_nu2_N100 | tight | `tinympc` | 131/150 | 4970 | 3000 - 7620 | 25.0 / 0.130 / 4970 | 129000 / 397 / 69800000 / 69800000 | 39 / 466.5 / 2056 | 6.66e-1 | inaccurate 38 | 411203 |
| nx4_nu2_N100 | tight | `argmin_sparse_admm_polished` | 93/100 | 1870 | 1440 - 2300 | 3270 / 0.430 / 1870 | 33500000 / 3030 / 22100000 / 22100000 | 25 / 75 / 250 | 6.66e-1 | inaccurate 21 | 98 |
| nx4_nu2_N100 | tight | `osqp_polished` | 1/1 | 1140 | 1130 - 1160 | 560 / 2.70 / 1140 | 4870000 / 29000 / 10300000 / 10400000 | 50 / 50 / 175 | 6.53e-4 | none | 60 |
| nx8_nu3_N50 | nominal | `argmin_sparse_admm` | 1/1 | 619 | 616 - 620 | 2630 / 0.320 / 618 | 27000000 / 2380 / 6620000 / 6620000 | 25 / 25 / 50 | 4.56e-4 | none | 0 |
| nx8_nu3_N50 | nominal | `osqp` | 1/1 | 1040 | 1020 - 1510 | 572 / 2.08 / 1040 | 6040000 / 21000 / 9550000 / 9570000 | 25 / 50 / 100 | 5.22e-4 | none | 0 |
| nx8_nu3_N50 | nominal | `proxqp_sparse` | 253/300 | 1210 | 1040 - 1550 | 3510 / 85.2 / 1130 | 38700000 / 916000 / 12100000 / 13100000 | 0 / 6 / 14 | 2.75e-3 | inaccurate 47 | 53 |
| nx8_nu3_N50 | nominal | `hpipm` | 1/1 | 418 | 363 - 424 | 44.1 / 0.0900 / 418 | 403000 / 365 / 4160000 / 4160000 | 5 / 7 / 11 | 1.11e-4 | none | 0 |
| nx8_nu3_N50 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 24 / 116 / 338 | 1.58e-2 | inaccurate 300 | 33803 |
| nx8_nu3_N50 | nominal | `argmin_sparse_admm_polished` | 1/1 | 1350 | 1340 - 1350 | 3400 / 0.330 / 1350 | 34200000 / 2380 / 13700000 / 13700000 | 25 / 25 / 50 | 3.37e-4 | none | 83 |
| nx8_nu3_N50 | nominal | `osqp_polished` | 1/1 | 1480 | 1460 - 1950 | 569 / 2.09 / 1480 | 6040000 / 21000 / 14000000 / 14000000 | 25 / 50 / 100 | 3.93e-4 | none | 60 |
| nx8_nu3_N50 | tight | `argmin_sparse_admm` | 1/1 | 1270 | 1260 - 1900 | 4620 / 0.370 / 1270 | 49600000 / 2750 / 14000000 / 14000000 | 25 / 50 / 200 | 6.21e-4 | none | 10 |
| nx8_nu3_N50 | tight | `osqp` | 1/1 | 1070 | 1060 - 1260 | 609 / 2.46 / 1060 | 6280000 / 25800 / 10000000 / 10000000 | 50 / 50 / 225 | 5.85e-4 | none | 0 |
| nx8_nu3_N50 | tight | `proxqp_sparse` | 19/300 | 28500 | 8180 - 41700 | 42300 / 96.8 / 28400 | 444000000 / 1010000 / 301000000 / 302000000 | 0 / 2 / 322 | 1.00e+0 | inaccurate 3, infeasible_claim 278 | 53 |
| nx8_nu3_N50 | tight | `hpipm` | 1/1 | 442 | 382 - 504 | 49.8 / 0.0900 / 442 | 433000 / 365 / 4300000 / 4300000 | 5 / 7 / 12 | 8.69e-5 | none | 0 |
| nx8_nu3_N50 | tight | `tinympc` | 89/100 | 4440 | 3040 - 6540 | 38.4 / 0.140 / 4440 | 219000 / 412 / 63800000 / 63800000 | 111 / 504.5 / 2962 | 6.67e-1 | inaccurate 33 | 296203 |
| nx8_nu3_N50 | tight | `argmin_sparse_admm_polished` | 14/15 | 3270 | 2660 - 3310 | 5750 / 0.360 / 3270 | 57000000 / 2750 / 35200000 / 35200000 | 25 / 100 / 225 | 6.66e-1 | inaccurate 20 | 88 |
| nx8_nu3_N50 | tight | `osqp_polished` | 1/1 | 1510 | 1500 - 1620 | 605 / 2.48 / 1500 | 6280000 / 25800 / 14400000 / 14400000 | 50 / 50 / 225 | 5.84e-4 | none | 60 |
| nx8_nu3_N100 | nominal | `argmin_sparse_admm` | 1/1 | 1250 | 1250 - 1260 | 5540 / 0.630 / 1250 | 53700000 / 4420 / 13200000 / 13200000 | 25 / 25 / 75 | 5.19e-4 | none | 0 |
| nx8_nu3_N100 | nominal | `osqp` | 1/1 | 2190 | 2060 - 3060 | 1110 / 3.78 / 2180 | 12000000 / 40800 / 20400000 / 20500000 | 50 / 50 / 100 | 3.99e-4 | none | 0 |
| nx8_nu3_N100 | nominal | `proxqp_sparse` | 253/300 | 2620 | 2260 - 3470 | 13300 / 164 / 2450 | 128000000 / 1810000 / 24200000 / 26100000 | 0 / 6 / 14 | 2.75e-3 | inaccurate 47 | 53 |
| nx8_nu3_N100 | nominal | `hpipm` | 1/1 | 843 | 734 - 856 | 90.4 / 0.110 / 843 | 790000 / 365 / 8280000 / 8280000 | 5 / 7 / 11 | 1.17e-4 | none | 0 |
| nx8_nu3_N100 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 24 / 116 / 338 | 1.63e-2 | inaccurate 300 | 67603 |
| nx8_nu3_N100 | nominal | `argmin_sparse_admm_polished` | 1/1 | 2690 | 2680 - 2700 | 7330 / 0.650 / 2690 | 67800000 / 4420 / 27100000 / 27100000 | 25 / 25 / 75 | 3.35e-4 | none | 83 |
| nx8_nu3_N100 | nominal | `osqp_polished` | 1/1 | 3060 | 2930 - 3930 | 1110 / 3.80 / 3050 | 12000000 / 40800 / 29200000 / 29200000 | 50 / 50 / 100 | 3.93e-4 | none | 60 |
| nx8_nu3_N100 | tight | `argmin_sparse_admm` | 1/1 | 2600 | 2580 - 3880 | 9330 / 0.740 / 2600 | 98600000 / 5170 / 27900000 / 27900000 | 25 / 50 / 200 | 6.21e-4 | none | 10 |
| nx8_nu3_N100 | tight | `osqp` | 1/1 | 2160 | 2160 - 2560 | 1180 / 4.59 / 2160 | 12400000 / 50400 / 19900000 / 20000000 | 50 / 50 / 225 | 5.85e-4 | none | 0 |
| nx8_nu3_N100 | tight | `proxqp_sparse` | 3/50 | 167000 | 33000 - 206000 | 275000 / 187 / 167000 | 2860000000 / 1990000 / 1710000000 / 1720000000 | 0 / 2 / 423 | 1.00e+0 | inaccurate 2, infeasible_claim 280 | 86 |
| nx8_nu3_N100 | tight | `hpipm` | 1/1 | 900 | 779 - 1030 | 113 / 0.130 / 900 | 851000 / 365 / 8550000 / 8550000 | 5 / 7 / 14 | 8.48e-5 | none | 0 |
| nx8_nu3_N100 | tight | `tinympc` | 89/100 | 8890 | 6100 - 13100 | 45.3 / 0.180 / 8890 | 235000 / 412 / 127000000 / 127000000 | 111 / 504.5 / 2962 | 6.67e-1 | inaccurate 33 | 592403 |
| nx8_nu3_N100 | tight | `argmin_sparse_admm_polished` | 14/15 | 6610 | 5350 - 6690 | 11300 / 0.750 / 6610 | 113000000 / 5170 / 70000000 / 70000000 | 25 / 100 / 250 | 6.66e-1 | inaccurate 20 | 94 |
| nx8_nu3_N100 | tight | `osqp_polished` | 1/1 | 3030 | 3020 - 3220 | 1180 / 4.59 / 3030 | 12400000 / 50400 / 28600000 / 28600000 | 50 / 50 / 225 | 5.84e-4 | none | 60 |

### sparse cohort, churning regime, high tier

The sparse cohort in the churning regime, judged at the high tier's threshold of 1e-6.

| Cell | Variant | Arm | Pass rate | Median tick (us) | Tick p25 - p75 (us) | Median setup / update / solve (us) | Median instructions setup / update / solve / tick | Iterations min / median / max | Worst KKT residual | Failures | Allocations max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nx2_nu1_N10 | nominal | `argmin_sparse_admm` | 1/1 | 57.6 | 56.6 - 58.6 | 132 / 0.0400 / 57.6 | 1340000 / 391 / 656000 / 657000 | 25 / 50 / 100 | 2.25e-16 | none | 86 |
| nx2_nu1_N10 | nominal | `osqp` | 1/1 | 52.8 | 51.8 - 71.2 | 31.0 / 0.260 / 52.5 | 216000 / 2040 / 426000 / 428000 | 50 / 50 / 250 | 3.27e-16 | none | 60 |
| nx2_nu1_N10 | nominal | `proxqp_sparse` | 33/50 | 40.5 | 30.3 - 52.5 | 91.2 / 5.86 / 34.6 | 733000 / 56100 / 351000 / 407000 | 0 / 4 / 20 | 1.10e-1 | inaccurate 102 | 62 |
| nx2_nu1_N10 | nominal | `hpipm` | 1/1 | 61.7 | 54.0 - 70.1 | 17.0 / 0.0500 / 61.6 | 66600 / 335 / 709000 / 710000 | 5 / 7 / 12 | 1.99e-7 | none | 0 |
| nx2_nu1_N10 | tight | `argmin_sparse_admm` | 1/1 | 111 | 93.7 - 148 | 132 / 0.0400 / 110 | 1270000 / 421 / 1530000 / 1530000 | 25 / 125 / 425 | 9.99e-17 | none | 79 |
| nx2_nu1_N10 | tight | `osqp` | 1/1 | 101 | 92.3 - 130 | 33.7 / 0.300 / 101 | 230000 / 2390 / 808000 / 810000 | 50 / 100 / 400 | 2.66e-16 | none | 60 |
| nx2_nu1_N10 | tight | `proxqp_sparse` | 7/60 | 147 | 77.0 - 203 | 171 / 6.98 / 140 | 1410000 / 62100 / 1370000 / 1430000 | 0 / 3 / 55 | 1.00e+0 | inaccurate 4, infeasible_claim 261 | 104 |
| nx2_nu1_N10 | tight | `hpipm` | 1/1 | 65.6 | 56.6 - 73.4 | 11.6 / 0.0500 / 65.5 | 71400 / 335 / 736000 / 736000 | 5 / 7 / 13 | 3.02e-8 | none | 0 |
| nx4_nu2_N10 | nominal | `argmin_sparse_admm` | 1/1 | 143 | 142 - 184 | 365 / 0.0500 / 143 | 3850000 / 459 / 1610000 / 1610000 | 25 / 50 / 150 | 2.89e-16 | none | 93 |
| nx4_nu2_N10 | nominal | `osqp` | 1/1 | 154 | 118 - 194 | 62.3 / 0.390 / 153 | 501000 / 3140 / 1410000 / 1410000 | 50 / 75 / 225 | 3.52e-16 | none | 60 |
| nx4_nu2_N10 | nominal | `proxqp_sparse` | 23/50 | 103 | 88.9 - 134 | 248 / 10.9 / 92.3 | 2310000 / 109000 / 957000 / 1070000 | 0 / 5 / 24 | 1.35e-2 | inaccurate 162 | 116 |
| nx4_nu2_N10 | nominal | `hpipm` | 1/1 | 87.5 | 77.6 - 98.0 | 17.7 / 0.0600 / 87.5 | 78000 / 352 / 967000 / 967000 | 6 / 8 / 14 | 8.75e-9 | none | 0 |
| nx4_nu2_N10 | tight | `argmin_sparse_admm` | 1/1 | 367 | 283 - 459 | 302 / 0.0600 / 367 | 2920000 / 509 / 4710000 / 4710000 | 25 / 175 / 500 | 1.30e-16 | none | 89 |
| nx4_nu2_N10 | tight | `osqp` | 1/1 | 273 | 241 - 299 | 66.3 / 0.440 / 272 | 536000 / 3780 / 2510000 / 2510000 | 75 / 137.5 / 475 | 2.66e-16 | none | 60 |
| nx4_nu2_N10 | tight | `proxqp_sparse` | 7/100 | 541 | 450 - 836 | 553 / 13.5 / 527 | 5190000 / 121000 / 5110000 / 5230000 | 0 / 3 / 523 | 1.00e+0 | inaccurate 3, infeasible_claim 276 | 107 |
| nx4_nu2_N10 | tight | `hpipm` | 1/1 | 91.2 | 90.2 - 102 | 20.4 / 0.0500 / 91.1 | 83600 / 352 / 1010000 / 1010000 | 6 / 8 / 13 | 2.69e-8 | none | 0 |
| nx4_nu2_N20 | nominal | `argmin_sparse_admm` | 1/1 | 276 | 271 - 349 | 591 / 0.110 / 276 | 6250000 / 684 / 3080000 / 3080000 | 25 / 50 / 250 | 4.49e-16 | none | 84 |
| nx4_nu2_N20 | nominal | `osqp` | 1/1 | 306 | 228 - 387 | 104 / 0.620 / 305 | 952000 / 5300 / 2720000 / 2720000 | 50 / 75 / 225 | 5.39e-16 | none | 60 |
| nx4_nu2_N20 | nominal | `proxqp_sparse` | 11/25 | 191 | 163 - 257 | 504 / 19.9 / 171 | 5230000 / 205000 / 1850000 / 2050000 | 0 / 5 / 21 | 1.39e-2 | inaccurate 168 | 71 |
| nx4_nu2_N20 | nominal | `hpipm` | 1/1 | 168 | 148 - 188 | 17.8 / 0.0600 / 168 | 140000 / 352 / 1880000 / 1880000 | 6 / 8 / 13 | 9.09e-9 | none | 0 |
| nx4_nu2_N20 | tight | `argmin_sparse_admm` | 1/1 | 632 | 545 - 1020 | 555 / 0.120 / 632 | 5640000 / 784 / 7980000 / 7980000 | 25 / 150 / 525 | 9.99e-17 | none | 89 |
| nx4_nu2_N20 | tight | `osqp` | 1/1 | 484 | 420 - 635 | 112 / 0.780 / 483 | 1020000 / 6600 / 4390000 / 4400000 | 75 / 125 / 425 | 2.22e-16 | none | 60 |
| nx4_nu2_N20 | tight | `proxqp_sparse` | 1/20 | 1260 | 838 - 2010 | 1590 / 24.0 / 1240 | 16300000 / 230000 / 13600000 / 13800000 | 0 / 3 / 208 | 1.00e+0 | inaccurate 3, infeasible_claim 282 | 1175 |
| nx4_nu2_N20 | tight | `hpipm` | 1/1 | 176 | 174 - 198 | 19.4 / 0.0700 / 176 | 151000 / 352 / 1950000 / 1950000 | 6 / 8 / 12 | 2.80e-8 | none | 0 |
| nx8_nu3_N20 | nominal | `argmin_sparse_admm` | 1/1 | 804 | 798 - 810 | 1660 / 0.180 / 804 | 16500000 / 1070 / 8190000 / 8190000 | 25 / 50 / 100 | 5.69e-10 | none | 84 |
| nx8_nu3_N20 | nominal | `osqp` | 1/1 | 812 | 612 - 893 | 253 / 1.07 / 811 | 2480000 / 9080 / 7610000 / 7620000 | 50 / 75 / 275 | 5.40e-16 | none | 60 |
| nx8_nu3_N20 | nominal | `proxqp_sparse` | 97/300 | 505 | 396 - 643 | 1400 / 36.7 / 469 | 15100000 / 381000 / 5020000 / 5400000 | 0 / 6 / 21 | 2.67e-3 | inaccurate 203 | 80 |
| nx8_nu3_N20 | nominal | `hpipm` | 1/1 | 280 | 249 - 310 | 23.2 / 0.0900 / 280 | 178000 / 365 / 2970000 / 2970000 | 6 / 9 / 13 | 3.94e-9 | none | 0 |
| nx8_nu3_N20 | tight | `argmin_sparse_admm` | 299/300 | 1830 | 1570 - 2830 | 2240 / 0.190 / 1830 | 23100000 / 1220 / 19900000 / 19900000 | 25 / 150 / 550 | 7.56e-2 | inaccurate 1 | 91 |
| nx8_nu3_N20 | tight | `osqp` | 1/1 | 1440 | 1230 - 1680 | 264 / 1.26 / 1440 | 2580000 / 11000 / 13900000 / 13900000 | 75 / 150 / 500 | 5.79e-7 | none | 60 |
| nx8_nu3_N20 | tight | `proxqp_sparse` | 3/50 | 4380 | 2930 - 5820 | 4870 / 42.7 / 4340 | 51900000 / 418000 / 48100000 / 48500000 | 0 / 2 / 40 | 1.00e+0 | inaccurate 2, infeasible_claim 280 | 62 |
| nx8_nu3_N20 | tight | `hpipm` | 1/1 | 294 | 262 - 325 | 24.3 / 0.0900 / 294 | 190000 / 365 / 3050000 / 3050000 | 6 / 9 / 13 | 2.90e-8 | none | 0 |
| nx8_nu3_N30 | nominal | `argmin_sparse_admm` | 1/1 | 1180 | 1180 - 1190 | 2700 / 0.220 / 1180 | 24600000 / 1530 / 12200000 / 12200000 | 25 / 50 / 100 | 4.50e-11 | none | 83 |
| nx8_nu3_N30 | nominal | `osqp` | 1/1 | 1210 | 932 - 1510 | 412 / 1.43 / 1200 | 3670000 / 13100 / 11300000 / 11300000 | 50 / 75 / 300 | 9.02e-16 | none | 60 |
| nx8_nu3_N30 | nominal | `proxqp_sparse` | 97/300 | 751 | 644 - 954 | 2880 / 52.8 / 693 | 28600000 / 560000 / 7630000 / 8190000 | 0 / 6 / 21 | 2.72e-3 | inaccurate 203 | 62 |
| nx8_nu3_N30 | nominal | `hpipm` | 1/1 | 419 | 378 - 467 | 113 / 0.0900 / 419 | 259000 / 365 / 4410000 / 4410000 | 6 / 9 / 14 | 3.55e-9 | none | 0 |
| nx8_nu3_N30 | tight | `argmin_sparse_admm` | 299/300 | 2740 | 2340 - 4250 | 3510 / 0.240 / 2740 | 34400000 / 1720 / 29600000 / 29600000 | 25 / 150 / 625 | 7.11e-2 | inaccurate 1 | 93 |
| nx8_nu3_N30 | tight | `osqp` | 1/1 | 1910 | 1590 - 2520 | 453 / 1.65 / 1910 | 3810000 / 16000 / 17700000 / 17700000 | 75 / 125 / 450 | 5.61e-7 | none | 60 |
| nx8_nu3_N30 | tight | `proxqp_sparse` | 17/300 | 9540 | 5480 - 12400 | 12200 / 61.6 / 9480 | 128000000 / 619000 / 105000000 / 106000000 | 0 / 2 / 64 | 1.00e+0 | inaccurate 2, infeasible_claim 281 | 86 |
| nx8_nu3_N30 | tight | `hpipm` | 1/1 | 438 | 391 - 482 | 124 / 0.0900 / 438 | 278000 / 365 / 4530000 / 4530000 | 6 / 9 / 13 | 2.91e-8 | none | 0 |
| nx12_nu4_N30 | nominal | `argmin_sparse_admm` | 1/1 | 2250 | 2240 - 2270 | 5540 / 0.270 / 2250 | 54900000 / 2070 / 23200000 / 23200000 | 25 / 50 / 125 | 1.54e-7 | none | 89 |
| nx12_nu4_N30 | nominal | `osqp` | 1/1 | 2450 | 1870 - 3040 | 679 / 1.89 / 2450 | 7070000 / 18600 / 23900000 / 23900000 | 50 / 75 / 225 | 1.10e-15 | none | 60 |
| nx12_nu4_N30 | nominal | `proxqp_sparse` | 29/150 | 1480 | 1260 - 3430 | 6270 / 78.6 / 1400 | 61900000 / 819000 / 15200000 / 16000000 | 0 / 7 / 22 | 2.17e-3 | inaccurate 242 | 62 |
| nx12_nu4_N30 | nominal | `hpipm` | 1/1 | 562 | 508 - 617 | 40.3 / 0.0900 / 561 | 307000 / 386 / 6040000 / 6040000 | 6 / 10 / 14 | 1.70e-9 | none | 0 |
| nx12_nu4_N30 | tight | `argmin_sparse_admm` | 1/1 | 6690 | 5250 - 10300 | 6620 / 0.315 / 6690 | 65000000 / 2400 / 70800000 / 70800000 | 25 / 200 / 650 | 1.07e-15 | none | 95 |
| nx12_nu4_N30 | tight | `osqp` | 1/1 | 4720 | 3580 - 5970 | 739 / 2.19 / 4720 | 7280000 / 22400 / 46100000 / 46100000 | 75 / 175 / 525 | 2.22e-16 | none | 60 |
| nx12_nu4_N30 | tight | `proxqp_sparse` | 13/300 | 21700 | 2060 - 24800 | 30700 / 89.4 / 21600 | 326000000 / 896000 / 235000000 / 236000000 | 0 / 2 / 1573 | 1.00e+0 | inaccurate 3, infeasible_claim 284 | 62 |
| nx12_nu4_N30 | tight | `hpipm` | 1/1 | 542 | 488 - 599 | 42.3 / 0.0900 / 541 | 328000 / 386 / 5740000 / 5740000 | 7 / 9 / 14 | 2.87e-8 | none | 0 |
| nx4_nu2_N50 | nominal | `argmin_sparse_admm` | 1/1 | 668 | 662 - 862 | 1380 / 0.210 / 668 | 15200000 / 1360 / 7510000 / 7510000 | 25 / 50 / 325 | 1.04e-15 | none | 83 |
| nx4_nu2_N50 | nominal | `osqp` | 1/1 | 728 | 555 - 927 | 226 / 1.32 / 727 | 2300000 / 11800 / 6480000 / 6500000 | 50 / 75 / 225 | 1.03e-15 | none | 60 |
| nx4_nu2_N50 | nominal | `proxqp_sparse` | 131/300 | 454 | 388 - 651 | 1730 / 46.2 / 409 | 18900000 / 494000 / 4520000 / 5010000 | 0 / 5 / 21 | 1.42e-2 | inaccurate 169 | 71 |
| nx4_nu2_N50 | nominal | `hpipm` | 1/1 | 413 | 364 - 463 | 34.5 / 0.0900 / 413 | 328000 / 352 / 4600000 / 4600000 | 6 / 8 / 14 | 1.23e-8 | none | 0 |
| nx4_nu2_N50 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 16 / 224 / 856 | 2.42e-2 | inaccurate 300 | 85603 |
| nx4_nu2_N50 | tight | `argmin_sparse_admm` | 1/1 | 1560 | 1350 - 1830 | 2610 / 0.230 / 1560 | 31200000 / 1610 / 19600000 / 19600000 | 25 / 150 / 575 | 3.67e-16 | none | 93 |
| nx4_nu2_N50 | tight | `osqp` | 1/1 | 1170 | 973 - 1570 | 241 / 1.58 / 1170 | 2470000 / 15000 / 10600000 / 10600000 | 75 / 125 / 400 | 5.98e-16 | none | 60 |
| nx4_nu2_N50 | tight | `proxqp_sparse` | 17/300 | 7310 | 4850 - 13700 | 10400 / 55.3 / 7250 | 114000000 / 556000 / 79600000 / 80200000 | 0 / 3 / 1710 | 1.00e+0 | inaccurate 3, infeasible_claim 280 | 761 |
| nx4_nu2_N50 | tight | `hpipm` | 1/1 | 429 | 386 - 481 | 37.4 / 0.0900 / 429 | 355000 / 352 / 4770000 / 4770000 | 6 / 8 / 12 | 4.38e-8 | none | 0 |
| nx4_nu2_N50 | tight | `tinympc` | 19/20 | 3950 | 2490 - 6710 | 21.7 / 0.100 / 3950 | 113000 / 397 / 55500000 / 55500000 | 153 / 794.5 / 4000 | 1.33e-1 | max_iter 15 | 400003 |
| nx4_nu2_N100 | nominal | `argmin_sparse_admm` | 1/1 | 1330 | 1320 - 1720 | 2780 / 0.340 / 1330 | 30200000 / 2530 / 14900000 / 14900000 | 25 / 50 / 325 | 1.10e-15 | none | 83 |
| nx4_nu2_N100 | nominal | `osqp` | 1/1 | 1450 | 1110 - 1860 | 425 / 2.28 / 1450 | 4540000 / 22600 / 12800000 / 12800000 | 50 / 75 / 225 | 1.34e-15 | none | 60 |
| nx4_nu2_N100 | nominal | `proxqp_sparse` | 131/300 | 910 | 779 - 1350 | 6160 / 88.7 / 822 | 59200000 / 973000 / 8840000 / 9820000 | 0 / 5 / 21 | 1.42e-2 | inaccurate 169 | 71 |
| nx4_nu2_N100 | nominal | `hpipm` | 1/1 | 825 | 727 - 921 | 206 / 0.0900 / 824 | 640000 / 352 / 9150000 / 9150000 | 6 / 8 / 13 | 1.17e-8 | none | 0 |
| nx4_nu2_N100 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 16 / 224 / 856 | 2.42e-2 | inaccurate 300 | 171203 |
| nx4_nu2_N100 | tight | `argmin_sparse_admm` | 1/1 | 2720 | 2290 - 3610 | 5440 / 0.430 / 2720 | 61600000 / 3030 / 33000000 / 33000000 | 25 / 125 / 500 | 3.67e-16 | none | 93 |
| nx4_nu2_N100 | tight | `osqp` | 1/1 | 2400 | 1970 - 3220 | 514 / 2.70 / 2400 | 4870000 / 29000 / 21500000 / 21500000 | 50 / 125 / 400 | 4.08e-16 | none | 60 |
| nx4_nu2_N100 | tight | `proxqp_sparse` | 19/300 | 42900 | 25200 - 70500 | 62300 / 103 / 42800 | 646000000 / 1100000 / 442000000 / 443000000 | 0 / 3 / 191 | 1.00e+0 | inaccurate 3, infeasible_claim 278 | 752 |
| nx4_nu2_N100 | tight | `hpipm` | 1/1 | 874 | 781 - 979 | 235 / 0.0900 / 874 | 695000 / 352 / 9480000 / 9480000 | 6 / 8 / 12 | 6.27e-8 | none | 0 |
| nx4_nu2_N100 | tight | `tinympc` | 19/20 | 7840 | 4990 - 13400 | 28.3 / 0.120 / 7840 | 129000 / 397 / 110000000 / 110000000 | 153 / 794.5 / 4000 | 1.33e-1 | max_iter 15 | 800003 |
| nx8_nu3_N50 | nominal | `argmin_sparse_admm` | 1/1 | 1960 | 1950 - 1970 | 3990 / 0.320 / 1960 | 40600000 / 2380 / 20100000 / 20100000 | 25 / 50 / 125 | 9.01e-11 | none | 83 |
| nx8_nu3_N50 | nominal | `osqp` | 1/1 | 2040 | 1550 - 2490 | 567 / 2.09 / 2040 | 6040000 / 21000 / 19400000 / 19400000 | 50 / 75 / 300 | 2.15e-15 | none | 60 |
| nx8_nu3_N50 | nominal | `proxqp_sparse` | 8/25 | 1270 | 1070 - 1720 | 6050 / 85.1 / 1180 | 66100000 / 916000 / 12800000 / 13700000 | 0 / 6 / 21 | 2.75e-3 | inaccurate 204 | 62 |
| nx8_nu3_N50 | nominal | `hpipm` | 1/1 | 696 | 621 - 772 | 47.3 / 0.0900 / 696 | 419000 / 365 / 7300000 / 7300000 | 6 / 9 / 14 | 5.50e-9 | none | 0 |
| nx8_nu3_N50 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 34 / 257 / 781 | 1.58e-2 | inaccurate 300 | 78103 |
| nx8_nu3_N50 | tight | `argmin_sparse_admm` | 299/300 | 3940 | 3310 - 6400 | 7760 / 0.360 / 3940 | 78700000 / 2750 / 42100000 / 42100000 | 25 / 125 / 475 | 6.77e-2 | inaccurate 1 | 93 |
| nx8_nu3_N50 | tight | `osqp` | 1/1 | 3080 | 2560 - 4120 | 605 / 2.49 / 3070 | 6280000 / 25800 / 29200000 / 29200000 | 75 / 125 / 400 | 6.06e-7 | none | 60 |
| nx8_nu3_N50 | tight | `proxqp_sparse` | 19/300 | 31600 | 20700 - 39800 | 42900 / 97.4 / 31500 | 450000000 / 1010000 / 332000000 / 333000000 | 0 / 2 / 461 | 1.00e+0 | inaccurate 2, infeasible_claim 279 | 62 |
| nx8_nu3_N50 | tight | `hpipm` | 1/1 | 723 | 654 - 755 | 50.8 / 0.0900 / 723 | 450000 / 365 / 7470000 / 7470000 | 7 / 9 / 13 | 4.30e-8 | none | 0 |
| nx8_nu3_N50 | tight | `tinympc` | 19/20 | 6640 | 4770 - 9700 | 33.2 / 0.150 / 6640 | 219000 / 412 / 96600000 / 96600000 | 149 / 809 / 4000 | 6.67e-1 | max_iter 15 | 400003 |
| nx8_nu3_N100 | nominal | `argmin_sparse_admm` | 1/1 | 3920 | 3910 - 5130 | 8560 / 0.640 / 3920 | 80700000 / 4420 / 40000000 / 40000000 | 25 / 50 / 175 | 3.30e-10 | none | 83 |
| nx8_nu3_N100 | nominal | `osqp` | 1/1 | 4070 | 3110 - 5010 | 1110 / 3.78 / 4070 | 12000000 / 40800 / 38700000 / 38700000 | 50 / 75 / 300 | 3.68e-15 | none | 60 |
| nx8_nu3_N100 | nominal | `proxqp_sparse` | 19/60 | 2890 | 2260 - 3990 | 22900 / 165 / 2730 | 227000000 / 1810000 / 25800000 / 27600000 | 0 / 6 / 21 | 2.75e-3 | inaccurate 205 | 62 |
| nx8_nu3_N100 | nominal | `hpipm` | 1/1 | 1420 | 1270 - 1580 | 322 / 0.120 / 1420 | 821000 / 365 / 14500000 / 14500000 | 6 / 9 / 14 | 4.76e-9 | none | 0 |
| nx8_nu3_N100 | nominal | `tinympc` | 0/1 | no passing tick | no passing tick | no passing tick / no passing tick / no passing tick | no passing tick / no passing tick / no passing tick / no passing tick | 34 / 257 / 781 | 1.63e-2 | inaccurate 300 | 156203 |
| nx8_nu3_N100 | tight | `argmin_sparse_admm` | 299/300 | 7900 | 6660 - 12800 | 15300 / 0.760 / 7900 | 155000000 / 5170 / 83300000 / 83300000 | 25 / 125 / 475 | 6.82e-2 | inaccurate 1 | 93 |
| nx8_nu3_N100 | tight | `osqp` | 1/1 | 6220 | 5180 - 8340 | 1180 / 4.60 / 6220 | 12400000 / 50400 / 58000000 / 58100000 | 75 / 125 / 425 | 5.91e-7 | none | 60 |
| nx8_nu3_N100 | tight | `proxqp_sparse` | 3/50 | 186000 | 86600 - 213000 | 290000 / 186 / 186000 | 2950000000 / 1990000 / 1930000000 / 1940000000 | 0 / 3 / 313 | 1.00e+0 | inaccurate 2, infeasible_claim 280 | 95 |
| nx8_nu3_N100 | tight | `hpipm` | 1/1 | 1500 | 1340 - 1530 | 127 / 0.130 / 1500 | 881000 / 365 / 14800000 / 14800000 | 7 / 9 / 13 | 3.72e-8 | none | 0 |
| nx8_nu3_N100 | tight | `tinympc` | 19/20 | 13300 | 9540 - 19300 | 45.8 / 0.180 / 13300 | 235000 / 412 / 192000000 / 192000000 | 149 / 809 / 4000 | 6.67e-1 | max_iter 15 | 800003 |

## Verdicts

### Block-structured storage

Outcome: **NO-GO**.

| Cell | Variant | argmin median tick (ns) | argmin pass rate | Best reference | Reference median tick (ns) | Ratio | At least 3 |
|---|---|---|---|---|---|---|---|
| nx4_nu2_N50 | nominal | 206000 | 1/1 | hpipm | 243000 | 0.845 (41117/48677) | no |
| nx4_nu2_N50 | tight | 857000 | 1/1 | hpipm | 248000 | 3.45 (1714037/496259) | yes |
| nx4_nu2_N100 | nominal | 416000 | 1/1 | hpipm | 482000 | 0.863 (415585/481704) | no |
| nx4_nu2_N100 | tight | 1740000 | 1/1 | hpipm | 499000 | 3.49 (3480094/998559) | yes |
| nx8_nu3_N50 | nominal | 619000 | 1/1 | hpipm | 421000 | 1.47 (1238729/841100) | no |
| nx8_nu3_N50 | tight | 1890000 | 1/1 | hpipm | 435000 | 4.35 (3787407/870910) | yes |
| nx8_nu3_N100 | nominal | 1260000 | 1/1 | hpipm | 850000 | 1.48 (2512437/1699328) | no |
| nx8_nu3_N100 | tight | 3850000 | 1/1 | hpipm | 891000 | 4.32 (7708459/1782938) | yes |

- nx4_nu2_N50 nominal: ratio 41117/48677 (0.845) is below 3
- nx4_nu2_N100 nominal: ratio 415585/481704 (0.863) is below 3
- nx8_nu3_N50 nominal: ratio 1238729/841100 (1.47) is below 3
- nx8_nu3_N100 nominal: ratio 2512437/1699328 (1.48) is below 3
- nx8_nu3 tight: ratio at N = 100 does not exceed the ratio at N = 50

The rule computed NO-GO. HPIPM passed every tick on every in-scope cell-variant and so did argmin's
sparse ADMM solver, so all eight ratios are defined. The four nominal ratios (0.845 to 1.48) are
below 3 and the four tight ratios (3.45 to 4.35) meet it. Of the four families, three have a larger
ratio at N = 100 than at N = 50; the nx8_nu3 tight family does not (4.35 at N = 50, 4.32 at
N = 100). As the rule states, the outcome is recorded as a follow-up investigation of argmin's sparse
solver; this benchmark implements nothing.

### Dense default

#### fixed, mpc tier

Outcome: **portfolio** - portfolio: argmin_dense_admm wins 20, argmin_dense_active_set wins 1, 0 tied.

| Cell | Variant | argmin_dense_admm median tick (ns) | argmin_dense_admm pass rate | argmin_dense_active_set median tick (ns) | argmin_dense_active_set pass rate | Ratio | Winner |
|---|---|---|---|---|---|---|---|
| nx2_nu1_N10 | nominal | 9850 | 1/1 | 10100 | 1/1 | 0.978 (985/1007) | argmin_dense_admm |
| nx2_nu1_N10 | tight | 9850 | 1/1 | 16000 | 1/1 | 0.615 (985/1601) | argmin_dense_admm |
| nx4_nu2_N10 | nominal | 16800 | 1/1 | 25400 | 1/1 | 0.660 (1676/2539) | argmin_dense_admm |
| nx4_nu2_N10 | tight | 18400 | 1/1 | 70200 | 1/1 | 0.262 (108/413) | argmin_dense_admm |
| nx4_nu2_N20 | nominal | 34200 | 1/1 | 70900 | 1/1 | 0.483 (3425/7092) | argmin_dense_admm |
| nx4_nu2_N20 | tight | 41200 | 1/1 | 474000 | 1/1 | 0.0867 (41150/474489) | argmin_dense_admm |
| nx8_nu3_N20 | nominal | 60900 | 1/1 | 233000 | 1/1 | 0.262 (12185/46594) | argmin_dense_admm |
| nx8_nu3_N20 | tight | 81000 | 1/1 | 1710000 | 1/1 | 0.0474 (161969/3415600) | argmin_dense_admm |
| nx8_nu3_N30 | nominal | 118000 | 1/1 | 506000 | 1/1 | 0.234 (236359/1011810) | argmin_dense_admm |
| nx8_nu3_N30 | tight | 156000 | 1/1 | 7440000 | 1/1 | 0.0209 (155615/7444461) | argmin_dense_admm |
| nx12_nu4_N30 | nominal | 175000 | 1/1 | 1160000 | 1/1 | 0.151 (350659/2318310) | argmin_dense_admm |
| nx12_nu4_N30 | tight | 258000 | 1/1 | 19500000 | 1/1 | 0.0133 (172260/12975959) | argmin_dense_admm |
| nx4_nu2_N50 | nominal | 132000 | 1/1 | 608000 | 1/1 | 0.217 (264380/1216499) | argmin_dense_admm |
| nx4_nu2_N50 | tight | 184000 | 1/1 | 10800000 | 1/1 | 0.0170 (183605/10797266) | argmin_dense_admm |
| nx4_nu2_N100 | nominal | 435000 | 1/1 | 2910000 | 1/1 | 0.150 (870109/5816145) | argmin_dense_admm |
| nx4_nu2_N100 | tight | 694000 | 1/1 | 133000000 | 1/1 | 0.00520 (347197/66723116) | argmin_dense_admm |
| nx8_nu3_N50 | nominal | 265000 | 1/1 | 1550000 | 1/1 | 0.171 (529740/3099527) | argmin_dense_admm |
| nx8_nu3_N50 | tight | 399000 | 1/1 | 45300000 | 1/1 | 0.00882 (79841/9051998) | argmin_dense_admm |
| nx8_nu3_N100 | nominal | 974000 | 1/1 | 8180000 | 1/1 | 0.119 (1947329/16351478) | argmin_dense_admm |
| nx8_nu3_N100 | tight | 1510000 | 1/1 | 669000000 | 1/1 | 0.00226 (3024746/1338718197) | argmin_dense_admm |
| dense_mpc_shaped | nominal | 9330 | 1/1 | 6370 | 1/1 | 1.46 (933/637) | argmin_dense_active_set |

#### fixed, high tier

Outcome: **portfolio** - portfolio: argmin_dense_admm wins 16, argmin_dense_active_set wins 5, 0 tied.

| Cell | Variant | argmin_dense_admm median tick (ns) | argmin_dense_admm pass rate | argmin_dense_active_set median tick (ns) | argmin_dense_active_set pass rate | Ratio | Winner |
|---|---|---|---|---|---|---|---|
| nx2_nu1_N10 | nominal | 14100 | 1/1 | 10000 | 1/1 | 1.41 (2822/2005) | argmin_dense_active_set |
| nx2_nu1_N10 | tight | 17300 | 1/1 | 16000 | 1/1 | 1.09 (165/152) | argmin_dense_active_set |
| nx4_nu2_N10 | nominal | 27400 | 1/1 | 25300 | 1/1 | 1.09 (5484/5051) | argmin_dense_active_set |
| nx4_nu2_N10 | tight | 40200 | 1/1 | 70000 | 1/1 | 0.574 (2009/3499) | argmin_dense_admm |
| nx4_nu2_N20 | nominal | 73100 | 1/1 | 71200 | 1/1 | 1.03 (146160/142349) | argmin_dense_active_set |
| nx4_nu2_N20 | tight | 137000 | 1/1 | 469000 | 1/1 | 0.291 (273100/937779) | argmin_dense_admm |
| nx8_nu3_N20 | nominal | 157000 | 1/1 | 234000 | 1/1 | 0.670 (1957/2921) | argmin_dense_admm |
| nx8_nu3_N20 | tight | 322000 | 1/1 | 1700000 | 1/1 | 0.189 (644299/3400995) | argmin_dense_admm |
| nx8_nu3_N30 | nominal | 356000 | 1/1 | 506000 | 1/1 | 0.704 (177830/252777) | argmin_dense_admm |
| nx8_nu3_N30 | tight | 808000 | 1/1 | 7420000 | 1/1 | 0.109 (807599/7423977) | argmin_dense_admm |
| nx12_nu4_N30 | nominal | 664000 | 1/1 | 1160000 | 1/1 | 0.573 (1327080/2314867) | argmin_dense_admm |
| nx12_nu4_N30 | tight | 1570000 | 1/1 | 19600000 | 1/1 | 0.0803 (1572334/19582183) | argmin_dense_admm |
| nx4_nu2_N50 | nominal | 444000 | 1/1 | 607000 | 1/1 | 0.731 (887128/1214409) | argmin_dense_admm |
| nx4_nu2_N50 | tight | 1010000 | 1/1 | 10800000 | 1/1 | 0.0935 (1008049/10782986) | argmin_dense_admm |
| nx4_nu2_N100 | nominal | 2120000 | 1/1 | 2910000 | 1/1 | 0.729 (2122179/2909359) | argmin_dense_admm |
| nx4_nu2_N100 | tight | 5500000 | 1/1 | 132000000 | 1/1 | 0.0416 (10997747/264058090) | argmin_dense_admm |
| nx8_nu3_N50 | nominal | 1090000 | 1/1 | 1550000 | 1/1 | 0.705 (2186768/3101739) | argmin_dense_admm |
| nx8_nu3_N50 | tight | 2750000 | 1/1 | 45200000 | 1/1 | 0.0608 (5497385/90486952) | argmin_dense_admm |
| nx8_nu3_N100 | nominal | 5760000 | 1/1 | 8170000 | 1/1 | 0.705 (5762347/8174948) | argmin_dense_admm |
| nx8_nu3_N100 | tight | 15600000 | 1/1 | 675000000 | 1/1 | 0.0231 (31211579/1349577066) | argmin_dense_admm |
| dense_mpc_shaped | nominal | 12200 | 1/1 | 6330 | 1/1 | 1.94 (1225/633) | argmin_dense_active_set |

#### closed_loop, mpc tier

Outcome: **default** - argmin_dense_admm is the default: it wins 20 cell-variants, loses none, 0 tied.

| Cell | Variant | argmin_dense_admm median tick (ns) | argmin_dense_admm pass rate | argmin_dense_active_set median tick (ns) | argmin_dense_active_set pass rate | Ratio | Winner |
|---|---|---|---|---|---|---|---|
| nx2_nu1_N10 | nominal | 9840 | 1/1 | 37300 | 1/1 | 0.264 (6560/24863) | argmin_dense_admm |
| nx2_nu1_N10 | tight | 19700 | 1/1 | 40000 | 1/1 | 0.493 (1970/3999) | argmin_dense_admm |
| nx4_nu2_N10 | nominal | 16800 | 1/1 | 212000 | 1/1 | 0.0792 (33560/423959) | argmin_dense_admm |
| nx4_nu2_N10 | tight | 35900 | 1/1 | 202000 | 1/1 | 0.177 (71880/404999) | argmin_dense_admm |
| nx4_nu2_N20 | nominal | 34400 | 1/1 | 894000 | 1/1 | 0.0384 (68749/1788202) | argmin_dense_admm |
| nx4_nu2_N20 | tight | 120000 | 1/1 | 1560000 | 1/1 | 0.0773 (120455/1557473) | argmin_dense_admm |
| nx8_nu3_N20 | nominal | 61000 | 1/1 | 3240000 | 1/1 | 0.0188 (61040/3242101) | argmin_dense_admm |
| nx8_nu3_N20 | tight | 159000 | 1/1 | 5830000 | 1/1 | 0.0273 (317939/11666668) | argmin_dense_admm |
| nx8_nu3_N30 | nominal | 118000 | 1/1 | 8360000 | 1/1 | 0.0142 (236700/16715063) | argmin_dense_admm |
| nx8_nu3_N30 | tight | 305000 | 1/1 | 23500000 | 1/1 | 0.0130 (2585/199427) | argmin_dense_admm |
| nx12_nu4_N30 | nominal | 175000 | 1/1 | 20300000 | 1/1 | 0.00861 (9982/1159385) | argmin_dense_admm |
| nx12_nu4_N30 | tight | 512000 | 1/1 | 62000000 | 1/1 | 0.00827 (102443/12393923) | argmin_dense_admm |
| nx4_nu2_N50 | nominal | 132000 | 1/1 | 7220000 | 1/1 | 0.0183 (26493/1444379) | argmin_dense_admm |
| nx4_nu2_N50 | tight | 364000 | 1/1 | 33000000 | 1/1 | 0.0110 (90940/8239299) | argmin_dense_admm |
| nx4_nu2_N100 | nominal | 431000 | 1/1 | 41300000 | 1/1 | 0.0104 (86201/8253407) | argmin_dense_admm |
| nx4_nu2_N100 | tight | 1380000 | 1/1 | 372000000 | 1/1 | 0.00370 (2750617/744123512) | argmin_dense_admm |
| nx8_nu3_N50 | nominal | 264000 | 1/1 | 29000000 | 1/1 | 0.00911 (264295/29006128) | argmin_dense_admm |
| nx8_nu3_N50 | tight | 785000 | 1/1 | 136000000 | 1/1 | 0.00576 (1569238/272509569) | argmin_dense_admm |
| nx8_nu3_N100 | nominal | 965000 | 1/1 | 185000000 | 1/1 | 0.00522 (241356/46203625) | argmin_dense_admm |
| nx8_nu3_N100 | tight | 2960000 | 1/1 | 1700000000 | 1/1 | 0.00175 (5928036/3392277337) | argmin_dense_admm |

#### closed_loop, high tier

Outcome: **portfolio** - portfolio: argmin_dense_admm wins 19, argmin_dense_active_set wins 1, 0 tied.

| Cell | Variant | argmin_dense_admm median tick (ns) | argmin_dense_admm pass rate | argmin_dense_active_set median tick (ns) | argmin_dense_active_set pass rate | Ratio | Winner |
|---|---|---|---|---|---|---|---|
| nx2_nu1_N10 | nominal | 33600 | 1/1 | 34300 | 1/1 | 0.981 (6722/6855) | argmin_dense_admm |
| nx2_nu1_N10 | tight | 45700 | 1/1 | 39400 | 1/1 | 1.16 (4567/3945) | argmin_dense_active_set |
| nx4_nu2_N10 | nominal | 63700 | 1/1 | 211000 | 1/1 | 0.302 (12739/42157) | argmin_dense_admm |
| nx4_nu2_N10 | tight | 128000 | 1/1 | 202000 | 1/1 | 0.634 (256069/404120) | argmin_dense_admm |
| nx4_nu2_N20 | nominal | 141000 | 1/1 | 894000 | 1/1 | 0.158 (28181/178710) | argmin_dense_admm |
| nx4_nu2_N20 | tight | 363000 | 1/1 | 1550000 | 1/1 | 0.234 (362825/1547958) | argmin_dense_admm |
| nx8_nu3_N20 | nominal | 275000 | 1/1 | 3240000 | 1/1 | 0.0849 (275255/3243506) | argmin_dense_admm |
| nx8_nu3_N20 | tight | 714000 | 1/1 | 5820000 | 1/1 | 0.123 (713799/5817713) | argmin_dense_admm |
| nx8_nu3_N30 | nominal | 700000 | 1/1 | 8360000 | 1/1 | 0.0838 (1400039/16714972) | argmin_dense_admm |
| nx8_nu3_N30 | tight | 1700000 | 1/1 | 23600000 | 1/1 | 0.0721 (1703137/23609363) | argmin_dense_admm |
| nx12_nu4_N30 | nominal | 1010000 | 1/1 | 20200000 | 1/1 | 0.0498 (671956/13486985) | argmin_dense_admm |
| nx12_nu4_N30 | tight | 2860000 | 1/1 | 62000000 | 1/1 | 0.0461 (5713922/123985827) | argmin_dense_admm |
| nx4_nu2_N50 | nominal | 698000 | 1/1 | 7210000 | 1/1 | 0.0969 (73502/758859) | argmin_dense_admm |
| nx4_nu2_N50 | tight | 1720000 | 1/1 | 33000000 | 1/1 | 0.0520 (1715383/32972919) | argmin_dense_admm |
| nx4_nu2_N100 | nominal | 2960000 | 1/1 | 41400000 | 1/1 | 0.0716 (5919772/82724949) | argmin_dense_admm |
| nx4_nu2_N100 | tight | 8200000 | 1/1 | 376000000 | 1/1 | 0.0218 (16407472/751677765) | argmin_dense_admm |
| nx8_nu3_N50 | nominal | 1870000 | 1/1 | 29000000 | 1/1 | 0.0644 (1867989/29016817) | argmin_dense_admm |
| nx8_nu3_N50 | tight | 5020000 | 1/1 | 136000000 | 1/1 | 0.0368 (10038767/272606119) | argmin_dense_admm |
| nx8_nu3_N100 | nominal | 8630000 | 1/1 | 185000000 | 1/1 | 0.0466 (17257170/370058123) | argmin_dense_admm |
| nx8_nu3_N100 | tight | 22900000 | 1/1 | 1690000000 | 1/1 | 0.0135 (45727117/3384851418) | argmin_dense_admm |

#### churning, mpc tier

Outcome: **default** - argmin_dense_admm is the default: it wins 20 cell-variants, loses none, 0 tied.

| Cell | Variant | argmin_dense_admm median tick (ns) | argmin_dense_admm pass rate | argmin_dense_active_set median tick (ns) | argmin_dense_active_set pass rate | Ratio | Winner |
|---|---|---|---|---|---|---|---|
| nx2_nu1_N10 | nominal | 9830 | 1/1 | 33800 | 1/1 | 0.290 (19660/67679) | argmin_dense_admm |
| nx2_nu1_N10 | tight | 19500 | 1/1 | 39800 | 1/1 | 0.490 (1949/3975) | argmin_dense_admm |
| nx4_nu2_N10 | nominal | 16800 | 1/1 | 203000 | 1/1 | 0.0831 (99/1192) | argmin_dense_admm |
| nx4_nu2_N10 | tight | 36000 | 1/1 | 207000 | 1/1 | 0.174 (71960/413739) | argmin_dense_admm |
| nx4_nu2_N20 | nominal | 34400 | 1/1 | 732000 | 1/1 | 0.0470 (68760/1464179) | argmin_dense_admm |
| nx4_nu2_N20 | tight | 80500 | 1/1 | 1560000 | 1/1 | 0.0517 (8049/155725) | argmin_dense_admm |
| nx8_nu3_N20 | nominal | 60800 | 1/1 | 2570000 | 1/1 | 0.0236 (121580/5141067) | argmin_dense_admm |
| nx8_nu3_N20 | tight | 159000 | 1/1 | 5780000 | 1/1 | 0.0275 (318019/11554887) | argmin_dense_admm |
| nx8_nu3_N30 | nominal | 119000 | 1/1 | 6660000 | 1/1 | 0.0178 (237411/13327292) | argmin_dense_admm |
| nx8_nu3_N30 | tight | 308000 | 1/1 | 23200000 | 1/1 | 0.0133 (616019/46432671) | argmin_dense_admm |
| nx12_nu4_N30 | nominal | 175000 | 1/1 | 16800000 | 1/1 | 0.0104 (35030/3356541) | argmin_dense_admm |
| nx12_nu4_N30 | tight | 506000 | 1/1 | 60800000 | 1/1 | 0.00833 (1012249/121554208) | argmin_dense_admm |
| nx4_nu2_N50 | nominal | 132000 | 1/1 | 5930000 | 1/1 | 0.0223 (52952/2372883) | argmin_dense_admm |
| nx4_nu2_N50 | tight | 361000 | 1/1 | 32700000 | 1/1 | 0.0110 (72103/6541854) | argmin_dense_admm |
| nx4_nu2_N100 | nominal | 435000 | 1/1 | 33800000 | 1/1 | 0.0129 (869939/67663917) | argmin_dense_admm |
| nx4_nu2_N100 | tight | 1370000 | 1/1 | 372000000 | 1/1 | 0.00368 (2734978/743070241) | argmin_dense_admm |
| nx8_nu3_N50 | nominal | 266000 | 1/1 | 22900000 | 1/1 | 0.0116 (266195/22939748) | argmin_dense_admm |
| nx8_nu3_N50 | tight | 782000 | 1/1 | 135000000 | 1/1 | 0.00581 (521523/89782283) | argmin_dense_admm |
| nx8_nu3_N100 | nominal | 965000 | 1/1 | 144000000 | 1/1 | 0.00671 (1929778/287427663) | argmin_dense_admm |
| nx8_nu3_N100 | tight | 2970000 | 1/1 | 1680000000 | 1/1 | 0.00177 (5934683/3362021428) | argmin_dense_admm |

#### churning, high tier

Outcome: **default** - argmin_dense_active_set is the default: the only candidate passing every tick.

| Cell | Variant | argmin_dense_admm median tick (ns) | argmin_dense_admm pass rate | argmin_dense_active_set median tick (ns) | argmin_dense_active_set pass rate | Ratio | Winner |
|---|---|---|---|---|---|---|---|
| nx2_nu1_N10 | nominal | 23300 | 1/1 | 33400 | 1/1 | 0.697 (4655/6677) | - |
| nx2_nu1_N10 | tight | 54700 | 1/1 | 39300 | 1/1 | 1.39 (10935/7853) | - |
| nx4_nu2_N10 | nominal | 44600 | 1/1 | 204000 | 1/1 | 0.218 (89220/408659) | - |
| nx4_nu2_N10 | tight | 147000 | 1/1 | 205000 | 1/1 | 0.714 (7326/10267) | - |
| nx4_nu2_N20 | nominal | 107000 | 1/1 | 734000 | 1/1 | 0.146 (107140/734039) | - |
| nx4_nu2_N20 | tight | 332000 | 1/1 | 1570000 | 1/1 | 0.212 (663610/3136747) | - |
| nx8_nu3_N20 | nominal | 216000 | 1/1 | 2570000 | 1/1 | 0.0841 (432571/5144929) | - |
| nx8_nu3_N20 | tight | 714000 | 1/1 | 5750000 | 1/1 | 0.124 (1428579/11505487) | - |
| nx8_nu3_N30 | nominal | 475000 | 1/1 | 6650000 | 1/1 | 0.0714 (67872/949993) | - |
| nx8_nu3_N30 | tight | 1570000 | 299/300 | 23200000 | 1/1 | 0.0675 (3135298/46466751) | - |
| nx12_nu4_N30 | nominal | 991000 | 1/1 | 16800000 | 1/1 | 0.0589 (990913/16825145) | - |
| nx12_nu4_N30 | tight | 3020000 | 299/300 | 60700000 | 1/1 | 0.0498 (6043694/121469599) | - |
| nx4_nu2_N50 | nominal | 574000 | 1/1 | 5940000 | 1/1 | 0.0967 (1148459/11879377) | - |
| nx4_nu2_N50 | tight | 1900000 | 1/1 | 32700000 | 1/1 | 0.0580 (3794435/65473378) | - |
| nx4_nu2_N100 | nominal | 2550000 | 1/1 | 33900000 | 1/1 | 0.0751 (5093738/67814967) | - |
| nx4_nu2_N100 | tight | 8900000 | 1/1 | 372000000 | 1/1 | 0.0239 (4448281/185975755) | - |
| nx8_nu3_N50 | nominal | 1360000 | 1/1 | 23000000 | 1/1 | 0.0591 (903909/15301228) | - |
| nx8_nu3_N50 | tight | 5260000 | 1/1 | 135000000 | 1/1 | 0.0390 (10512459/269530327) | - |
| nx8_nu3_N100 | nominal | 6710000 | 1/1 | 144000000 | 1/1 | 0.0467 (13411318/287469973) | - |
| nx8_nu3_N100 | tight | 24300000 | 299/300 | 1690000000 | 1/1 | 0.0144 (6081539/421667261) | - |

The rule computed one outcome per regime and tier. In five combinations both candidates passed
every tick on every in-scope cell-variant, so the medians decided: `argmin_dense_admm` is the default
in closed_loop and churning at the mpc tier, winning all 20 cell-variants, and the other three are
portfolios. Their boundaries: in fixed/mpc `argmin_dense_active_set` wins only `dense_mpc_shaped`
(n = 10, m = 5), and `argmin_dense_admm` wins every MPC cell, the smallest (n = 10) included; in
fixed/high `argmin_dense_active_set` wins both variants of nx2_nu1_N10, the nominal variant of
nx4_nu2_N10 and of nx4_nu2_N20 (n = N nu of 10, 20 and 40) and `dense_mpc_shaped`, and
`argmin_dense_admm` wins the tight variant at n = 20 and n = 40 and every cell-variant from n = 60
up; in closed_loop/high `argmin_dense_active_set` wins only the tight variant of nx2_nu1_N10. In
churning/high `argmin_dense_admm` passed 299 of 300 ticks on the tight variant of nx8_nu3_N30,
nx12_nu4_N30 and nx8_nu3_N100, so `argmin_dense_active_set`, which passed every tick, is the default
there without a timing comparison; that table's ratio column is shown for reference and was not read
by the rule. The outcome is documentation only: argmin's library defaults are unchanged.

### OSQP replacement

Outcome: **keep**.

| Cell | Variant | argmin median tick (ns) | argmin pass rate | OSQP median tick (ns) | OSQP pass rate | Ratio | Satisfied |
|---|---|---|---|---|---|---|---|
| nx2_nu1_N10 | nominal | 17800 | 1/1 | 38800 | 1/1 | 0.459 (1783/3881) | yes |
| nx2_nu1_N10 | tight | 35300 | 1/1 | 39800 | 1/1 | 0.887 (3531/3983) | yes |
| nx4_nu2_N10 | nominal | 42600 | 1/1 | 82900 | 1/1 | 0.513 (4255/8294) | yes |
| nx4_nu2_N10 | tight | 132000 | 1/1 | 84800 | 1/1 | 1.56 (264169/169620) | no |
| nx4_nu2_N20 | nominal | 84600 | 1/1 | 163000 | 1/1 | 0.518 (169160/326849) | yes |
| nx4_nu2_N20 | tight | 261000 | 1/1 | 167000 | 1/1 | 1.56 (52193/33390) | no |
| nx8_nu3_N20 | nominal | 250000 | 1/1 | 420000 | 1/1 | 0.594 (166360/280093) | yes |
| nx8_nu3_N20 | tight | 502000 | 1/1 | 427000 | 1/1 | 1.18 (1003879/853548) | yes |
| nx8_nu3_N30 | nominal | 371000 | 1/1 | 657000 | 1/1 | 0.564 (371089/657444) | yes |
| nx8_nu3_N30 | tight | 756000 | 1/1 | 645000 | 1/1 | 1.17 (216027/184384) | yes |
| nx12_nu4_N30 | nominal | 708000 | 1/1 | 1310000 | 1/1 | 0.540 (1415909/2621477) | yes |
| nx12_nu4_N30 | tight | 1470000 | 1/1 | 1220000 | 1/1 | 1.20 (2932195/2448017) | yes |
| nx4_nu2_N50 | nominal | 206000 | 1/1 | 403000 | 1/1 | 0.510 (205585/403359) | yes |
| nx4_nu2_N50 | tight | 857000 | 1/1 | 406000 | 1/1 | 2.11 (1714037/812660) | no |
| nx4_nu2_N100 | nominal | 416000 | 1/1 | 799000 | 1/1 | 0.520 (415585/799459) | yes |
| nx4_nu2_N100 | tight | 1740000 | 1/1 | 823000 | 1/1 | 2.11 (1740047/823185) | no |
| nx8_nu3_N50 | nominal | 619000 | 1/1 | 1020000 | 1/1 | 0.606 (1238729/2044497) | yes |
| nx8_nu3_N50 | tight | 1890000 | 1/1 | 1070000 | 1/1 | 1.77 (3787407/2143798) | no |
| nx8_nu3_N100 | nominal | 1260000 | 1/1 | 2060000 | 1/1 | 0.608 (2512437/4128905) | yes |
| nx8_nu3_N100 | tight | 3850000 | 1/1 | 2170000 | 1/1 | 1.78 (7708459/4330758) | no |

- nx4_nu2_N10 tight: ratio 264169/169620 is not within 5/4
- nx4_nu2_N20 tight: ratio 52193/33390 is not within 5/4
- nx4_nu2_N50 tight: ratio 1714037/812660 is not within 5/4
- nx4_nu2_N100 tight: ratio 1740047/823185 is not within 5/4
- nx8_nu3_N50 tight: ratio 3787407/2143798 is not within 5/4
- nx8_nu3_N100 tight: ratio 7708459/4330758 is not within 5/4

The rule computed keep. argmin's sparse ADMM solver and OSQP both passed every tick on all 20
in-scope cell-variants, so every ratio is defined and the outcome rests on the ratio alone. All ten
nominal cell-variants are satisfied (0.459 to 0.608). On the tight variant four are satisfied (0.887
to 1.20) and six exceed 5/4: nx4_nu2_N10 and nx4_nu2_N20 (1.56), nx4_nu2_N50 and nx4_nu2_N100
(2.11), nx8_nu3_N50 (1.77) and nx8_nu3_N100 (1.78). The outcome is advice to ctrlpp: whether to
replace OSQP, and the regression work a replacement would need, are ctrlpp's decisions and ctrlpp's
work. The OSQP arm pinned `adaptive_rho_interval` to 25 iterations, whereas ctrlpp's deployed OSQP
wrapper keeps OSQP's default, which sets the interval from the measured setup time; the OSQP timings
here are of the pinned configuration.

## Run-specific limitations

- No tight family was refused. Every family passed the binding gate with a median binding fraction
  of 0.5 on every cell (the minimum is 0.1) and a state-row binding step fraction of 0.65 on
  nx2_nu1_N10 and 1.0 on every other cell (the minimum is 0.5).
- `argmin_dense_active_set` fell back to the feasible start only on the tight variant in the
  closed_loop and churning regimes: on all 40 of those rows, 9,578 of their 12,000 ticks, at most
  295 of 300 (98.3%) on one row. It never fell back in the fixed regime or on the nominal variant.
  Every fallback tick passed, and its time is inside the tick. The per-row counts are the last
  column of the dense tables.
- No tick of any row had a KKT residual within one ulp of its tier threshold.
- Several competitor arms fail ticks under the common judge at their pre-registered settings, which
  were not tuned per regime. ProxQP (dense and sparse, warm-started from its previous result)
  passes every tick on every fixed-regime row except the high-tier tight variant of nx12_nu4_N30
  (dense, `inaccurate` on all 300 ticks). In the closed_loop and churning regimes it fails 34% to
  42% of the ticks of each regime and tier on the dense problem and 47% to 80% on the sparse
  problem, as `inaccurate` and `infeasible_claim`, and passes every tick on at most 5 of the 20
  rows of any regime and tier. These failures were not investigated; another ProxQP configuration
  may behave differently.
- TinyMPC is judged on multipliers recovered by least squares and solves a QP whose stage weights
  include `rho I`; it decides no rule, and its pass rates should be read with both facts in mind.
- The polished sub-arms (`argmin_dense_admm_polished`, `argmin_sparse_admm_polished`,
  `osqp_polished`) are reported only and are excluded from every rule.
- One host at one fixed clock. These numbers are a fresh baseline and are not comparable with
  earlier argmin benchmark numbers measured with boost enabled.
- The closed_loop and churning regimes replay prescribed initial-state sequences; they are not
  closed-loop simulations, so no arm's solution feeds back into the next problem.
- The tight variant uses hard state boxes, where ctrlpp's deployed linear MPC softens its state
  bounds with slack variables.

## Provenance

- argmin library under test: `ca7ec854a02dc90f0b423cec52bda9a4f306c43a`, fetched by the harness.
  The argmin commits after it, which add the verdict command and this page, touch only
  `benchmarks/scripts/` and `docs/`: `git diff --stat ca7ec854a02dc90f0b423cec52bda9a4f306c43a --
  lib/` prints nothing.
- Pre-registration freeze commit: `b91820d929bbad8edb73c1e890ca75935523b25f`, committed
  2026-10-08T20:00:27Z, before the run started. The run's manifest cites it, and the verdict command
  checked it with `--expect-preregistration-commit`.
- ctrlpp harness commit: `706d5262e9a458b5683d80c1565257eba5a5b46f`, with a clean working tree
  under the paths the benchmark build reads.
- Competitor pins:

  | Component | Commit |
  |---|---|
  | osqp | `236713ce9a56c182ac3230d52108f952afce1523` |
  | qpoases | `680f18c8ef0018a120e1604b769f056e8368df97` |
  | daqp | `29a8987d8b9a596c1ab6acf84c17d38b068b73bc` |
  | hpipm | `16a8db59f595bda642f460aa0f62e308f4b76db3` |
  | blasfeo | `9628393214623b29e5dd2a8784f7423edbc93e04` |
  | tinympc | `023f36bd5b27267e1a96ed47a774d120b5707f13` |
  | proxsuite | `b93d7778ffc3299d84b5cb0851022a29bf24a596` |

  The harness fetches and verifies every pin above except proxsuite, which it takes from the host's
  system package (proxsuite 0.7.3, header-only) and records under the upstream commit of that
  release. Eigen also comes from the host's system package, Eigen 3.4.1. Neither package version is
  in the run manifest; both were read from the host's package database after the run.

- BLASFEO target `X64_INTEL_HASWELL` with `LA=HIGH_PERFORMANCE` and `MF=PANELMAJ`; HPIPM target
  `AVX`; DAQP with `FASTER_MATH=OFF` and `NATIVE_OPTIMIZATION=OFF`. The harness confirms each from
  the configure log before it builds.
- Compiler: gcc (GCC) 16.2.1 20260810, Release, C and C++ flags `-march=x86-64-v3 -O3 -DNDEBUG`.
  Effective flags per component, read from the compilation database of the measured build:

  | Component | Effective flags |
  |---|---|
  | osqp | `-march=x86-64-v3 -O3 -O3 -DNDEBUG -fPIC` |
  | qpoases | `-DNDEBUG -march=x86-64-v3 -D__NO_COPYRIGHT__ -DLINUX -g -O3 -finline-functions -std=c++23` |
  | daqp | `-DPROFILING -Ddaqp_EXPORTS -march=x86-64-v3 -O3 -O3 -DNDEBUG -fPIC` |
  | blasfeo | `-march=x86-64-v3 -DLA_HIGH_PERFORMANCE -DBLASFEO_REF_API -DMF_PANELMAJ -DDP_ROUTINES -DSP_ROUTINES -DK_MAX_STACK=300 -DBLAS_API -DUSE_C99_MATH -DEXT_DEP -DEXT_DEP_MALLOC -DOS_LINUX -DEXTERNAL_BLAS_NONE -O2 -fPIC -DTARGET_X64_INTEL_HASWELL -DTARGET_PHANTOM -m64 -mavx -mavx2 -mfma -O3 -DNDEBUG` |
  | hpipm | `-march=x86-64-v3 -O2 -fPIC -DTARGET_AVX -m64 -mavx -DUSE_C99_MATH -DRUNTIME_CHECKS -O3 -DNDEBUG` |
  | tinympc | `-march=x86-64-v3 -O3 -DNDEBUG -std=c++17 -fPIC` |
  | tinympc wrapper in the harness | `-Dctrlpp_qp_solvers_tinympc_EXPORTS -march=x86-64-v3 -O3 -DNDEBUG -std=c++23 -fPIC -fvisibility-inlines-hidden` |
  | argmin and proxsuite (header-only, in the harness) | `-march=x86-64-v3 -O3 -DNDEBUG -std=c++23` |

- Host: AMD Ryzen 7 5800X3D 8-Core Processor, Linux 6.18.53-1-lts; `amd-pstate` in passive mode,
  governor `performance`; global and per-policy boost 0; SMT control off; measurement pinned to
  CPU 2; `scaling_min_freq` = `scaling_max_freq` = `cpuinfo_max_freq` = 3,401,000 kHz; observed
  `scaling_cur_freq` over 205 samples taken during the run: median 3,375,195 kHz, minimum
  3,374,261 kHz, maximum 3,375,207 kHz; `perf_event_paranoid` 2.
- Run `20261009T045427Z-706d5262e9`, started 2026-10-09T04:54:27Z and finished
  2026-10-09T06:37:21Z.
- The run ledger holds two recorded runs. The earlier one was refused because its clock was not
  fixed: the frequency bounds were set, but the CPU frequency driver ran in its active mode, which
  ignores them, and the frequency observed during that run varied. The verdict command's clock
  check reads the clock state the harness derives from the configured bounds, so it accepted that
  run; the refusal rests on the observed frequency. The two runs gave the same rule outcomes. The
  published run is the first recorded run whose observed clock held.

## Reproduce

The raw per-tick records and the run manifest are reproducible from the harness commit; a new run
gets its own run id and its own timings.

1. Check out ctrlpp at `706d5262e9a458b5683d80c1565257eba5a5b46f`, and argmin at
   `b91820d929bbad8edb73c1e890ca75935523b25f` or later beside it (the harness reads the verdict
   command from `../argmin`, or from `ARGMIN_CHECKOUT`). Install Eigen 3.4.1 and proxsuite 0.7.3
   where CMake finds them; the harness does not fetch either. At that commit the harness is
   `benchmarks/comparison/qp_solvers/run_qp_solvers.sh`; the directory named in the introduction
   above, `benchmarks/comparison/qp_head_to_head/`, does not exist under that name.
2. Prepare: `benchmarks/comparison/qp_solvers/run_qp_solvers.sh prepare`. It configures, builds the
   two drivers, checks every fetched pin and the BLASFEO, HPIPM and DAQP targets, and runs a short
   smoke pass and the schema check. The configure command it runs is:

   ```
   cmake -S <ctrlpp>/benchmarks -B <ctrlpp>/build/qp-solvers -G "Unix Makefiles" \
     -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
     -DCMAKE_C_FLAGS=-march=x86-64-v3 -DCMAKE_CXX_FLAGS=-march=x86-64-v3 \
     -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_DISABLE_FIND_PACKAGE_ct_optcon=ON \
     -DCTRLPP_FETCH_BENCHMARK_DEPS=ON -DCTRLPP_ARGMIN_SOURCE_DIR= \
     -DCTRLPP_ARGMIN_GIT_TAG=ca7ec854a02dc90f0b423cec52bda9a4f306c43a -DCTRLPP_BENCH_QP_SOLVERS=ON
   ```

3. Fix the host state: boost 0 globally and on every policy; SMT control off; governor
   `performance`; `scaling_min_freq` = `scaling_max_freq` = `cpuinfo_max_freq` on every policy;
   under `amd-pstate`, passive mode, because the active mode ignores the bounds; confirm that
   `scaling_cur_freq` stays fixed before and during the run; the instruction counter must
   open (it did at `perf_event_paranoid` 2). The measurement is pinned to CPU 2 by default
   (`QP_SOLVERS_CPU`, never CPU 0).
4. Measure: `benchmarks/comparison/qp_solvers/run_qp_solvers.sh measure <destination>`. It refuses
   to start on an unfixed clock, a dirty harness tree or a mismatched pin, writes the run to
   `<destination>/<run id>` and appends it to the ledger `<destination>/runs.jsonl`.
5. Compute the verdicts in argmin:

   ```
   python3 benchmarks/scripts/qp_head_to_head_verdicts.py --run-dir <destination>/<run id> \
     --expect-preregistration-commit b91820d929bbad8edb73c1e890ca75935523b25f \
     --output-json verdicts.json --output-md verdicts.md
   ```
