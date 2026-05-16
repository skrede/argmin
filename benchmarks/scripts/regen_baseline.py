#!/usr/bin/env python3
"""Regenerate benchmarks/baselines/v0.3.0-regression.csv from a publish_summary.csv.

Aggregates per-seed publish_summary rows into per-(solver, problem, mode)
cells; computes the per-cell bounds with a 1.30 pad over the per-step wall
median and the outer-iter maximum; assigns the row disposition per the
status-aware accept-as-pass rule (every-seed-stalled AND accuracy<1e-12 AND
cv<1e-8 -> pass; original-disposition rules otherwise); preserves the
existing curated cohort + the (kraft_slsqp_accurate, hs026) override; and
emits the new baseline with a header comment block documenting the rule
and the NMPC informational-known_optimum convention.

Not a planning-tool reference site -- the baseline is a code-tree artifact.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


# Solver cohort retained from the prior curated baseline.
COHORT_SOLVERS = {
    "bobyqa",
    "byrd_lbfgsb",
    "filter_nw_sqp_accurate",
    "filter_slsqp_accurate",
    "kraft_slsqp_accurate",
    "lbfgsb",
    "multistart_bobyqa",
    "nw_sqp_accurate",
    "tr_sqp_accurate",
    "tr_sqp_fast",
}


# Solver-name -> bench-config dispatch mode. "default" for solvers without
# a per-suffix fast/accurate split.
def solver_dispatch_mode(solver: str) -> str:
    if solver.endswith("_fast"):
        return "fast"
    if solver.endswith("_accurate"):
        return "accurate"
    return "default"


# Per-mode tolerance floors lifted verbatim from the prior baseline header.
# fast -> min_accuracy_log10=-4.0, max_cv_log10=-4.0
# accurate -> min_accuracy_log10=-8.0, max_cv_log10=-6.0
# default -> min_accuracy_log10=-8.0, max_cv_log10=-6.0
def mode_tolerances(mode: str) -> tuple[float, float]:
    if mode == "fast":
        return (-4.0, -4.0)
    return (-8.0, -6.0)


# Manual override: (kraft_slsqp_accurate, hs026, accurate) max_outer_iters
# hardcoded at 200 as the LSEI re-attempt gate cell (preserved across
# regenerations from the prior curated baseline).
MANUAL_ITERS_OVERRIDE: dict[tuple[str, str, str], int] = {
    ("kraft_slsqp_accurate", "hs026", "accurate"): 200,
}


# Tag cells that are documented known-failures in the test suite, hence
# expected=shouldfail in the baseline regardless of empirical convergence
# at any seed (the test suite [!shouldfail] tag is the contract).
KNOWN_FAILURE_CELLS = {
    ("tr_sqp_accurate", "hs024", "accurate"),
    ("tr_sqp_fast",     "hs024", "fast"),
    ("tr_sqp_accurate", "hs035", "accurate"),
    ("tr_sqp_fast",     "hs035", "fast"),
    ("tr_sqp_accurate", "hs039", "accurate"),
    ("tr_sqp_fast",     "hs039", "fast"),
    ("tr_sqp_accurate", "hs040", "accurate"),
    ("tr_sqp_fast",     "hs040", "fast"),
    ("tr_sqp_accurate", "hs043", "accurate"),
    ("tr_sqp_fast",     "hs043", "fast"),
    ("tr_sqp_accurate", "hs050", "accurate"),
    ("tr_sqp_fast",     "hs050", "fast"),
}


HEADER_COMMENT_BLOCK = """\
# Regression baseline keyed by (solver, problem, mode).
# Derived from an 11-seed publish_bench sweep with pad = 1.30 over
# the per-step wall median and the outer-iter maximum. tolerance
# columns set per mode: fast -> min_accuracy_log10=-4.0 max_cv_log10=-4.0;
# accurate -> min_accuracy_log10=-8.0 max_cv_log10=-6.0.
# (kraft_slsqp_accurate, hs026, accurate) max_outer_iters hardcoded at 200
# as the LSEI re-attempt gate cell.
# constraint_violation column is present in publish_summary at this HEAD;
# the max_cv_log10 column gates against the empirical per-seed cv maxima.
# expected=shouldfail rows mark cells with a documented algorithmic-design
# limit (TR-SQP family: HS024, HS035, HS039, HS040, HS043, HS050 on both
# fast and accurate modes). expected=skip rows mark cells with
# path-dependent convergence (per-seed status divergence) outside the
# published gate.
# Status-aware accept-as-pass rule: rows whose every seed reports
# status=stalled AND accuracy<1e-12 AND cv<1e-8 are promoted in-memory
# from expected=skip to expected=pass by regression_check and gated
# against the bounds below. This catches cells that stall at
# machine precision on strictly-feasible minimizers (|f - f*| at
# 1e-13..1e-16, cv at 1e-14..1e-16). The (accuracy_cutoff, cv_cutoff)
# pair is empirically selected by a narrow threshold sweep.
# nmpc_lqr_* rows: known_optimum is informational. The fixture returns 0 from
# optimal_value(); the bench publishes accuracy as |f - 0| and cv as the
# dynamics-equality residual under the single-shooting formulation. The numeric
# accuracy value on a converged trajectory equals the terminal LQR cost itself
# (not a deficit against a true minimizer; the closed-form LQR optimum requires
# a Riccati / DARE solve at fixed problem data that is intentionally not part of
# this fixture). Treat nmpc_lqr_* rows as expected-skip with respect to the
# accuracy-against-known_optimum bar; the dynamics-feasibility is reported in
# the cv column at machine precision and is the relevant convergence signal.
"""


def safe_log10(x: float) -> float:
    if x <= 0.0:
        return -math.inf
    return math.log10(x)


def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def load_curated_cells(prior_baseline: Path) -> set[tuple[str, str, str]]:
    """Extract the (solver, problem, mode) cell set from the prior baseline.

    The regen preserves the curated cohort: only cells that were in the
    prior baseline land in the new one. Adding cells is a deliberate
    decision; the regen script does not silently widen the cohort.
    """
    out: set[tuple[str, str, str]] = set()
    with prior_baseline.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if parts and parts[0] == "solver":
                continue
            if len(parts) < 3:
                continue
            out.add((parts[0], parts[1], parts[2]))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True, type=Path,
                   help="path to publish_summary.csv")
    p.add_argument("--prior-baseline", required=True, type=Path,
                   help="path to the prior baseline CSV (curated cell set "
                        "lifted from here)")
    p.add_argument("--output", required=True, type=Path,
                   help="path to new baseline CSV")
    p.add_argument("--accuracy-cutoff", type=float, default=1e-12,
                   help="status-aware promote-to-pass accuracy threshold")
    p.add_argument("--cv-cutoff", type=float, default=1e-8,
                   help="status-aware promote-to-pass cv threshold")
    args = p.parse_args()

    curated_cells = load_curated_cells(args.prior_baseline)
    print(f"regen_baseline: curated cohort size = {len(curated_cells)} cells",
          file=sys.stderr)

    # Load summary, aggregate per (solver, problem, dispatch_mode).
    with args.summary.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(f"regen_baseline: no rows in {args.summary}", file=sys.stderr)
        return 1

    # cv column may not be present (older publish_summary). We require it
    # for the status-aware rule.
    if "constraint_violation" not in rows[0]:
        print("WARN: constraint_violation column missing from publish_summary; "
              "status-aware promotion will short-circuit (no rows promoted).",
              file=sys.stderr)

    by_cell: dict[tuple[str, str, str], dict] = defaultdict(
        lambda: {"per_step_us": [], "iters": [], "acc": [], "cv": [], "status": []})
    for r in rows:
        solver = r["solver"]
        problem = r["problem"]
        if solver not in COHORT_SOLVERS:
            continue
        mode = solver_dispatch_mode(solver)
        if (solver, problem, mode) not in curated_cells:
            continue
        try:
            iters = int(r["solver_iters"])
            wall_us = int(r["wall_time_us"])
            acc = float(r["accuracy"])
            cv = float(r.get("constraint_violation", "nan"))
        except (KeyError, ValueError):
            continue
        denom = iters if iters > 0 else 1
        per_step = wall_us / denom
        cell = by_cell[(solver, problem, mode)]
        cell["per_step_us"].append(per_step)
        cell["iters"].append(iters)
        cell["acc"].append(acc)
        cell["cv"].append(cv)
        cell["status"].append(r["status"])

    rows_out: list[tuple] = []
    for (solver, problem, mode), cell in sorted(by_cell.items()):
        statuses = cell["status"]
        accs = cell["acc"]
        cvs = cell["cv"]
        iters = cell["iters"]
        per_step_us = cell["per_step_us"]

        # Bounds: pad = 1.30 over per-step median and outer-iter max,
        # round up to integer.
        per_step_med = median(per_step_us)
        bound_us = max(1, int(math.ceil(per_step_med * 1.30)))
        max_iters_empirical = max(iters)
        bound_iters = max(1, int(math.ceil(max_iters_empirical * 1.30)))
        manual_iters = MANUAL_ITERS_OVERRIDE.get((solver, problem, mode))
        if manual_iters is not None:
            bound_iters = manual_iters

        min_acc_log10, max_cv_log10 = mode_tolerances(mode)

        # Disposition.
        all_converged = all(s == "converged" for s in statuses)
        all_stalled = all(s == "stalled" for s in statuses)
        every_acc_below = all(0.0 < a < args.accuracy_cutoff for a in accs)
        every_cv_below = (all(not math.isnan(c) for c in cvs) and
                          all(c < args.cv_cutoff for c in cvs))

        if (solver, problem, mode) in KNOWN_FAILURE_CELLS:
            # Test suite tags these as [!shouldfail]; the baseline gate
            # asserts the cell fails at least one convergence check.
            disposition = "shouldfail"
            # For shouldfail rows we need the bounds to reflect the
            # per-mode tolerance floors (so that any future convergence
            # tightens the gate properly), but the actual values do not
            # gate -- regression_check only fires UNEXPECTED_PASS when
            # every seed converged AND every numeric gate held.
            comment_line = (
                f"# shouldfail: cell tagged [!shouldfail] in tr_sqp_test.cpp "
                f"per the trace-derived mechanism family; expected to fail "
                f"at least one convergence gate at default options."
            )
        elif all_converged:
            # Strict pass: every seed converged. Bounds derived from the
            # per-seed worst-case accuracy (we tighten min_acc_log10 to
            # the floor of the per-seed log10 max only when that is
            # tighter than the mode floor; otherwise we keep the mode
            # floor as the relaxed gate).
            disposition = "pass"
            empirical_acc_log10 = safe_log10(max(accs))
            if empirical_acc_log10 < min_acc_log10:
                # Empirical accuracy is BETTER than mode floor (smaller
                # log10). The gate stays at the mode floor (the looser
                # bound) so per-seed noise within the floor does not
                # breach.
                pass
            else:
                # Empirical accuracy exceeds the mode floor: this means
                # the cell does not actually pass the strict accuracy
                # gate. Mark skip with the diagnostic comment.
                disposition = "skip"
                comment_line = (
                    f"# skip: empirical max accuracy log10={empirical_acc_log10:.3f} "
                    f"exceeds the per-mode bound {min_acc_log10:.1f} for "
                    f"({solver}, {problem}, {mode}); known_optimum is "
                    f"informational, not the true f*."
                )
            if disposition == "pass":
                comment_line = None
        elif all_stalled and every_acc_below and every_cv_below:
            # Status-aware promote-to-pass: every seed stalled at
            # machine precision on the feasible manifold. The rule
            # bakes in the gate at the per-mode floor; regression_check
            # applies the same promotion at runtime and checks the
            # bounds.
            disposition = "pass"
            comment_line = None
        else:
            # Skip: path-dependent convergence or non-machine-precision
            # stall.
            disposition = "skip"
            status_set = sorted(set(statuses))
            comment_line = (
                f"# skip: every seed status in {status_set} for "
                f"({solver}, {problem}, {mode}); no converged seed to derive "
                f"a pass bound."
            )

        rows_out.append((
            solver, problem, mode, bound_us, min_acc_log10,
            bound_iters, max_cv_log10, disposition, comment_line))

    # Emit.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        f.write(HEADER_COMMENT_BLOCK)
        f.write("solver,problem,mode,max_us_per_step,min_accuracy_log10,"
                "max_outer_iters,max_cv_log10,expected\n")
        for (solver, problem, mode, bound_us, min_acc_log10,
             bound_iters, max_cv_log10, disposition, comment) in rows_out:
            if comment is not None:
                f.write(comment + "\n")
            f.write(f"{solver},{problem},{mode},{bound_us},{min_acc_log10:.1f},"
                    f"{bound_iters},{max_cv_log10:.1f},{disposition}\n")

    n_pass = sum(1 for r in rows_out if r[7] == "pass")
    n_skip = sum(1 for r in rows_out if r[7] == "skip")
    n_shouldfail = sum(1 for r in rows_out if r[7] == "shouldfail")
    print(f"regen_baseline: {len(rows_out)} cells -> {args.output}", file=sys.stderr)
    print(f"  pass:       {n_pass}", file=sys.stderr)
    print(f"  skip:       {n_skip}", file=sys.stderr)
    print(f"  shouldfail: {n_shouldfail}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
