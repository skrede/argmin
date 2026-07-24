# QP validation data — provenance and license

These generated C++ headers are the committed dense convex-QP subset the
operator-splitting solver is measured against. Each header is emitted by
`scripts/mm_to_header.py` from a QPS (fixed-position MPS + `QUADOBJ`) source and
carries plain aggregate `P`, `q`, `A`, `l`, `u` data plus the verified optimal
objective. The converter is offline: it is not wired into CMake or into any
test, and no QPS parser runs at test time.

## Selection rule (the commitment)

A problem is converted only if, measured against the actual QPS file:

- `n ≤ 128` — the fixed-N inline ceiling for `double` (an `n × n` condensed-KKT
  member must fit Eigen's 131072-byte stack-allocation limit), and
- `A` is dense-representable, `n · m ≤ 100000` stored entries.

The rule, not any fixed roster, is the commitment. `mm_to_header.py` verifies
`n` and `m` against each parsed file and drops whatever fails.

## Source of the committed subset

The field-standard reference set for dense convex QP validation is the
Maros–Mészáros repository (Maros & Mészáros, "A repository of convex quadratic
programming problems," *Optimization Methods and Software* 11 (1999), 671–681).
That set is distributed as an academic collection and is widely mirrored, but it
ships **without an explicit redistribution license** covering the commission of
converted data into a third-party source repository. Under this project's
rigor posture, redistributing converted third-party data whose redistribution
terms cannot be confirmed is not acceptable.

Therefore the committed subset here is **self-authored**: a set of small dense
convex QPs written for this repository, spanning the constraint structures the
solver must handle — diagonal and dense-SPD objectives; box-only, equality-only,
mixed equality-plus-slack-box, active general-inequality, and two-sided ranged
rows. These problems are original to this repository and carry no external
redistribution encumbrance. They exercise the same converter and the same
`l ≤ A x ≤ u` mapping the Maros–Mészáros files would.

Each problem's optimal objective was computed by an independent KKT oracle
(pure-python active-set solve of the equality-constrained KKT system, followed
by verification of stationarity, primal feasibility of every row, and
dual-sign feasibility) — not by the solver under measurement. The converter was
then run on the authored QPS files and its output round-trip-checked against the
intended `P`, `q`, `A`, `l`, `u` and the verified optima.

The authored QPS source files are **not committed** (only the converter, the
generated headers, and this note), matching the lean, committed-baseline
posture; no raw `.qps`/`.mps` files live under `tests/`.

## Regenerating from a license-clean mirror

`mm_to_header.py` reads any directory of QPS files:

```
scripts/mm_to_header.py <qps_dir> tests/unit/mm_data --optima <optima.json>
```

If and when a Maros–Mészáros mirror carrying a confirmed, compatible
redistribution license is available, point the converter at it: every file
surviving the selection rule is emitted as an additional header, and
`--optima` supplies the published optima. No test or build change is required —
the runner registers whatever headers the converter produced via
`mm_problems.inc`.

## Why the converter is unchanged now that a sparse solver exists

The selection rule above is deliberately derived from the dense fixed-size
ceiling — `n ≤ 128` exists because a condensed-KKT member must fit Eigen's stack
allocation limit, and `n · m ≤ 100000` exists because the emitted `A` is a dense
aggregate. Neither bound describes what a sparse solver could accept, and the
rule is not claimed to.

Extending the converter to emit sparse data would not unblock anything. The
blocker on the public reference collection is *redistribution licensing*, not
file format, and that blocker is identical regardless of the format converted
to. A sparse emitter would still have nothing license-clean to read.

The regeneration path is therefore unchanged: should a mirror carrying a
confirmed, compatible redistribution license become available, `mm_to_header.py`
and its selection rule are where that work starts.

The structured sparse validation data is not converted at all. It is generated
in-tree by `tests/unit/sparse_control_qp_family.h` — a spring-coupled
double-integrator family with banded equality dynamics over a runtime horizon —
so a reader looking for committed sparse fixture files under `tests/` will not
find any, and none are meant to exist.

## Recorded date

Committed subset authored and converted: 2026-07-22.
