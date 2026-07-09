# Publication benchmark methodology

This document defines what argmin benchmark claims mean and how publication
runs are produced. The benchmark harness is useful only when the run contract is
explicit: constrained optimization results must be feasible, timing must be
measured under a recorded environment, and cross-library comparisons must not
pretend that different solver stop rules are equivalent without proof.

## Headline result

The headline benchmark result is a feasibility-qualified trace-first
Dolan-More performance profile. For each solver/problem/seed row, the profile
uses the first trace point that satisfies both:

- objective accuracy: `|f_best - f*| <= tau * max(1, |f*|)`
- feasibility: `constraint_violation <= eps_feas`

For constrained problems, missing or non-finite feasibility is not a hit. A
final feasible result does not retroactively make an earlier trace point
feasible. Adapter traces must therefore carry true feasibility values at the
trace point, or the affected row is excluded from the constrained profile.

Common-stop summary tables are secondary. A solver row can enter a common-stop
table only after the native stop rule is mapped to the argmin-equivalent rule
and a small sanity test shows that the mapping fires at the intended scale.
Until that proof exists, breadth belongs in the trace-first profile.

## Success criteria

A publication row is eligible for profile and baseline use only when the harness
can record:

- the final objective, known optimum, objective accuracy, final feasibility,
  native status, solver iterations, and evaluation counts
- trace entries containing best objective, feasibility, counts, and elapsed
  solve time
- whether an evaluation cap, wall cap, native convergence, or native failure
  stopped the solve
- the run provenance sidecar for machine, compiler, build, seed, thread, and
  wall-gate policy

Rows that lack required problem semantics or required signals are excluded with
a concrete reason rather than kept as weak caveats.

## Budget semantics

Every benchmark row is governed by two independent caps:

- a uniform function-evaluation cap enforced through the harness counting
  wrapper
- a harness wall cap used to prevent runaway publication runs

The recorded status must preserve which cap fired. A native solver status is
not relabeled as an argmin status unless the adapter has a documented
equivalence proof.

Publication timing is solve-only timing. Problem construction, adapter setup,
trace allocation, and other setup work happen outside the primary timed region.
End-to-end timing may be recorded for diagnostics, but it is not part of the
main profile or publication summary claim.

## Environment controls

`benchmarks/scripts/run_publish_bench.sh` is the publication wrapper. It exports
single-thread settings for OpenMP and common BLAS implementations:

- `OMP_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `VECLIB_MAXIMUM_THREADS=1`

The wrapper records the current CPU model, host identifier, benchmark binary
path and hash, compiler path/version, CMake build type and flags, current CPU
affinity if `taskset` is available, frequency-governor information if
`cpupower` is available, turbo/boost policy where the kernel exposes it, and
the wall-gate policy. Missing optional tools are recorded as unavailable; they
do not abort advisory or smoke runs.

Full local publication runs enforce wall gates. CI or advisory runs may disable
or skip wall gates, but the sidecar must record that policy. Correctness and
iteration checks remain meaningful even when wall gates are advisory.

## Cohorts and exclusions

Comparators are capability-gated by cohort. A library appears only in cohorts
where its adapter supports the required problem semantics, cost signal,
feasibility signal, and cap behavior. Optional libraries can be absent without
aborting the entire benchmark run.

The headline constrained cohort is a pooled constrained Hock-Schittkowski
cohort. Smaller bound-only, inequality-only, equality-only, or mixed-class plots
are diagnostic or appendix material. New constrained fixtures require derivative
checks, known-optimum validation, adapter coverage, and profile generation
before inclusion.

NLopt COBYLA is included wherever it is methodology-eligible. If it is slow or
dominates a profile, that is a result, not a removal reason.

## Provenance and release eligibility

Every wrapper run writes `provenance.json` under the timestamped output
directory and updates `latest-run.txt` under the chosen output root. The sidecar
contains a `provenance_id` used by baseline rows and moved-number ledger rows.

Publication baseline generation requires Release build provenance. Debug or
advisory smoke runs can still produce summaries and profiles, but their
sidecars are stamped baseline-ineligible with a concrete reason. A
baseline-eligible local run must also use an enforced wall-gate policy.

Recommended local publication command:

```bash
cmake --preset bench
cmake --build build/bench --target publish_bench regression_check

SEED_START=42 \
SEED_COUNT=11 \
OUT_ROOT=bench-out/publish \
EXPECT_BASELINE_ELIGIBLE=1 \
REQUIRE_RELEASE_PUBLICATION=1 \
bash benchmarks/scripts/run_publish_bench.sh
```

For a smoke run, lower `SEED_COUNT` and disable expensive post-steps explicitly:

```bash
SEED_COUNT=1 \
OUT_ROOT=/tmp/argmin-publish-smoke \
RUN_REGRESSION_CHECK=0 \
RUN_DM_PROFILE=0 \
bash benchmarks/scripts/run_publish_bench.sh
```

Smoke-run numbers are not publication numbers.

## Common-stop side tables

Common-stop tables can be useful for readers who want a single scalar at a
matched stopping rule, but they require proof. For each admitted row, document:

- the native stop controls used by the solver
- the argmin-equivalent criterion being approximated
- why the native controls express that criterion for the selected problem
  class
- a sanity test showing the stop fires at the intended objective/feasibility
  scale

Rows without this proof stay in trace-first profiles only.

## Regression baselines and ledger policy

Regression baselines use explicit row dispositions:

- `pass`: status, accuracy, feasibility, iteration, and local wall bounds are
  enforced
- `expected_fail`: correctness-only fix detection is enforced, and wall time is
  ignored
- `excluded`: the cell is outside the active cohort and includes a reason

Re-baselining records results; it does not prove correctness. Every moved
published number needs an independent correctness witness and a ledger row that
records the old value, new value, methodology change, provenance ID, and notes.
If a corrected run reproduces the shipped value, record that fact instead of
changing the default or baseline.

## Outputs

A completed wrapper run produces:

- `publish_summary.csv`
- `traces/`
- `provenance.json`
- `run.log`
- optional `regression_check.log`
- optional `dm_profile.log` and `profiles/`

Keep the output directory together with the exact commit and sidecar whenever a
benchmark result is cited.
