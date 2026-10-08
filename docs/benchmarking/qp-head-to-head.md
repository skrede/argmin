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
| argmin | `ca7ec854a02dc90f0b423cec52bda9a4f306c43a` |
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
