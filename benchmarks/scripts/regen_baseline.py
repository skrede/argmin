#!/usr/bin/env python3
"""Regenerate or migrate the benchmark regression baseline.

The active baseline schema uses explicit row dispositions:
``pass``, ``expected_fail``, and ``excluded``. New publication baselines
require a provenance sidecar whose build type is Release. Legacy baselines
can be migrated in place without moving measured numeric bounds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


COHORT_SOLVERS = {
    "bobyqa",
    "byrd_lbfgsb",
    "filter_nw_sqp_accurate",
    "filter_slsqp_accurate",
    "filter_trsqp_accurate",
    "filter_trsqp_fast",
    "kraft_slsqp_accurate",
    "lbfgsb",
    "multistart_bobyqa",
    "nw_sqp_accurate",
    "tr_sqp_accurate",
    "tr_sqp_fast",
}

DISPOSITION_PASS = "pass"
DISPOSITION_EXPECTED_FAIL = "expected_fail"
DISPOSITION_EXCLUDED = "excluded"
DISPOSITIONS = {
    DISPOSITION_PASS,
    DISPOSITION_EXPECTED_FAIL,
    DISPOSITION_EXCLUDED,
}

BASELINE_FIELDS = [
    "solver",
    "problem",
    "mode",
    "max_us_per_step",
    "min_accuracy_log10",
    "max_outer_iters",
    "max_cv_log10",
    "disposition",
    "provenance_id",
    "wall_gate_policy",
    "exclusion_reason",
    "baseline_instr_per_iter",
    "instr_ceiling",
]

NUMERIC_BASELINE_FIELDS = [
    "max_us_per_step",
    "min_accuracy_log10",
    "max_outer_iters",
    "max_cv_log10",
]

LEDGER_FIELDS = [
    "metric",
    "solver",
    "problem",
    "mode",
    "old_value",
    "new_value",
    "methodology_change",
    "correctness_witness",
    "provenance_id",
    "commit_kind",
    "notes",
    "disposition",
    "disposition_change",
]

# Independent oracle verdict schema. The verdict is produced by the analytic
# oracle validator and is the disposition source of truth: a cell's committed
# disposition is derived from these per-seed pass/fail verdicts, never from the
# same run the gate then checks.
VERDICT_FIELDS = [
    "solver",
    "problem",
    "mode",
    "seed",
    "oracle_pass",
    "objective_distance",
    "feasibility_residual",
    "kkt_residual",
    "witness",
]

ORACLE_PASS_TOKEN = "pass"
ORACLE_FAIL_TOKEN = "fail"
ORACLE_PASS_VALUES = {ORACLE_PASS_TOKEN, ORACLE_FAIL_TOKEN}

DISPOSITION_LEDGER_METRIC = "disposition"

# Real-time-critical solver families: the SQP families ctrlpp drives for NMPC.
# They alone carry an absolute instructions/iter ceiling in the baseline; the
# population-based and derivative-free solvers get the ratio gate only, because
# their per-iteration cost varies by design (evaluations/iter scale with the
# population or simplex). The regression gate holds the same set.
REALTIME_CRITICAL_FAMILIES = {
    "nw_sqp",
    "filter_nw_sqp",
    "kraft_slsqp",
    "filter_slsqp",
}

# Absolute instructions/iter ceiling for a real-time-critical cell, as a margin
# above the measured worst-seed instructions/iter on the current tree. The
# measured seed-to-seed spread on the gated cohort is ~2%, so a 1.25x margin
# clears same-host noise and modest cross-host instruction-count drift while
# staying a genuine upper bound (a per-iteration cost regression on the padded
# factorization class is multiples above the measured cost, not percent).
INSTR_CEILING_MARGIN = 1.25


def solver_family(solver: str) -> str:
    if solver.endswith("_fast"):
        return solver[:-len("_fast")]
    if solver.endswith("_accurate"):
        return solver[:-len("_accurate")]
    return solver


def is_realtime_critical_solver(solver: str) -> bool:
    return solver_family(solver) in REALTIME_CRITICAL_FAMILIES


def solver_dispatch_mode(solver: str) -> str:
    if solver.endswith("_fast"):
        return "fast"
    if solver.endswith("_accurate"):
        return "accurate"
    return "default"


def mode_tolerances(mode: str) -> tuple[float, float]:
    if mode == "fast":
        return (-4.0, -4.0)
    return (-8.0, -6.0)


MANUAL_ITERS_OVERRIDE: dict[tuple[str, str, str], int] = {
    ("kraft_slsqp_accurate", "hs026", "accurate"): 200,
}


# Curated Hock-Schittkowski fixtures that were measured but never carried a
# baseline row, so they tripped the "cell not in baseline (informational)"
# warning instead of being gated. They are folded into the cohort here so every
# measured cohort cell for these problems is gated with an oracle-sourced
# disposition. A problem is only added for a solver/mode that actually measured
# it (the summary aggregation drops any expanded cell with no measurement).
CURATED_FIXTURE_PROBLEMS = [
    "hs021", "hs023", "hs027", "hs029", "hs030", "hs031",
    "hs034", "hs036", "hs037", "hs038", "hs044", "hs052",
]


HEADER_COMMENT_BLOCK = """\
# Regression baseline keyed by (solver, problem, mode).
# Numeric bounds are historical gate values. The disposition column controls
# how each row participates in the gate:
# pass: status, accuracy, feasibility, iteration, and wall bounds are enforced.
# expected_fail: correctness-only fix detection is enforced; wall time is ignored.
# excluded: the cell is outside the active cohort and must include a reason.
# wall_gate_policy describes whether the wall gate is enforced locally or made
# advisory by the caller through REGRESSION_CHECK_DISABLE_WALL_GATE.
# baseline_instr_per_iter is the committed reference userspace instructions/iter
# for the cell; the gate breaches when the measured instructions/iter exceeds it
# times the ratio factor (an upper bound applied to every gated cell).
# instr_ceiling is an absolute instructions/iter upper bound, populated only for
# the real-time-critical SQP families (nw_sqp, filter_nw_sqp, kraft_slsqp,
# filter_slsqp); it is empty for the population-based and derivative-free rows.
"""


def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def require_fields(row: dict[str, str], fields: list[str], context: str) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"{context}: missing required column(s): {', '.join(missing)}")


def parse_float(value: str, context: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{context}: invalid numeric value {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{context}: non-finite numeric value {value!r}")
    return parsed


def parse_summary_metric(value: str, context: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{context}: invalid summary metric {value!r}") from exc
    if math.isnan(parsed):
        return math.inf
    if parsed < 0.0:
        raise ValueError(f"{context}: negative summary metric {value!r}")
    return parsed


def parse_int(value: str, context: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{context}: invalid integer value {value!r}") from exc


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        data_lines = [line for line in f if line.strip() and not line.startswith("#")]
    if not data_lines:
        raise ValueError(f"{path}: no CSV header")
    reader = csv.DictReader(data_lines)
    return [dict(row) for row in reader]


def legacy_disposition(value: str) -> str:
    if value in DISPOSITIONS:
        return value
    if value == "shouldfail" or value == "skip":
        return DISPOSITION_EXPECTED_FAIL
    raise ValueError(f"invalid legacy disposition {value!r}")


def load_baseline_rows(path: Path,
                       default_provenance_id: str = "historical-baseline",
                       default_wall_policy: str = "enforced") -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    out: list[dict[str, str]] = []
    for row in rows:
        require_fields(row, ["solver", "problem", "mode", *NUMERIC_BASELINE_FIELDS],
                       str(path))
        source_disp = row.get("disposition", row.get("expected", ""))
        disposition = legacy_disposition(source_disp)
        migrated = {
            "solver": row["solver"],
            "problem": row["problem"],
            "mode": row["mode"],
            "max_us_per_step": row["max_us_per_step"],
            "min_accuracy_log10": row["min_accuracy_log10"],
            "max_outer_iters": row["max_outer_iters"],
            "max_cv_log10": row["max_cv_log10"],
            "disposition": disposition,
            "provenance_id": row.get("provenance_id") or default_provenance_id,
            "wall_gate_policy": row.get("wall_gate_policy") or default_wall_policy,
            "exclusion_reason": row.get("exclusion_reason") or "",
            "baseline_instr_per_iter": row.get("baseline_instr_per_iter") or "",
            "instr_ceiling": row.get("instr_ceiling") or "",
        }
        for field in NUMERIC_BASELINE_FIELDS:
            parse_float(migrated[field], f"{path}:{row['solver']}:{row['problem']}")
        if disposition == DISPOSITION_EXCLUDED and not migrated["exclusion_reason"]:
            migrated["exclusion_reason"] = "historical non-cohort cell"
        out.append(migrated)
    return out


def load_curated_cells(prior_baseline: Path) -> set[tuple[str, str, str]]:
    rows = load_baseline_rows(prior_baseline)
    return {(row["solver"], row["problem"], row["mode"]) for row in rows}


def expand_cohort_for_new_solvers(
    curated: set[tuple[str, str, str]],
    ref_solver: str,
    new_solvers: list[str],
) -> set[tuple[str, str, str]]:
    ref_cells = [(p, m) for (s, p, m) in curated if s == ref_solver]
    if not ref_cells:
        print(f"regen_baseline: WARN: reference solver {ref_solver!r} not "
              "present in curated cohort; expansion is a no-op",
              file=sys.stderr)
        return set(curated)
    out = set(curated)
    added = 0
    for new_solver in new_solvers:
        dispatch_mode = solver_dispatch_mode(new_solver)
        for problem, _ref_mode in ref_cells:
            tup = (new_solver, problem, dispatch_mode)
            if tup not in out:
                out.add(tup)
                added += 1
    print(f"regen_baseline: cohort expansion added {added} new cells "
          f"(ref_solver={ref_solver}, new_solvers={new_solvers})",
          file=sys.stderr)
    return out


def expand_cohort_for_new_problems(
    curated: set[tuple[str, str, str]],
    new_problems: list[str],
) -> set[tuple[str, str, str]]:
    """Fold measured-but-ungated fixture problems into the curated cohort.

    Adds ``(solver, problem, dispatch_mode)`` for every cohort solver and each
    new problem. The summary aggregation only emits a cell that has an actual
    measurement, so an expanded combination a solver never ran produces no row.
    """
    out = set(curated)
    added = 0
    for solver in sorted(COHORT_SOLVERS):
        dispatch_mode = solver_dispatch_mode(solver)
        for problem in new_problems:
            tup = (solver, problem, dispatch_mode)
            if tup not in out:
                out.add(tup)
                added += 1
    print(f"regen_baseline: cohort expansion added {added} new fixture cells "
          f"(problems={new_problems})",
          file=sys.stderr)
    return out


def load_release_provenance(path: Path) -> dict[str, str] | None:
    try:
        with path.open("r") as f:
            data = json.load(f)
    except OSError as exc:
        print(f"regen_baseline: cannot read provenance sidecar {path}: {exc}",
              file=sys.stderr)
        return None
    except json.JSONDecodeError as exc:
        print(f"regen_baseline: invalid provenance JSON {path}: {exc}",
              file=sys.stderr)
        return None

    build_data = data.get("build", {})
    build_type = (
        data.get("build_type")
        or data.get("cmake_build_type")
        or (build_data.get("type") if isinstance(build_data, dict) else None)
    )
    if build_type != "Release":
        print("regen_baseline: publication baseline requires Release provenance "
              f"(got {build_type!r})",
              file=sys.stderr)
        return None

    provenance_id = str(data.get("provenance_id") or data.get("run_id") or path.stem)
    wall_gate_policy = str(data.get("wall_gate_policy") or "enforced")
    return {
        "provenance_id": provenance_id,
        "wall_gate_policy": wall_gate_policy,
    }


def load_oracle_verdict(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    """Load the independent oracle verdict, keyed by baseline cell.

    Returns a mapping ``(solver, problem, dispatch_mode) -> {seed: oracle_pass}``.
    The verdict's own ``mode`` column records the publication run mode and is not
    the cell-join axis; the cell mode is derived from the solver name so the
    per-seed verdict aligns with the ``(solver, problem, mode)`` baseline cell key
    that the gate later enforces.
    """
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"{path}: no oracle verdict rows")
    by_cell: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
    for row in rows:
        require_fields(row, VERDICT_FIELDS, str(path))
        solver = row["solver"]
        problem = row["problem"]
        seed = row["seed"]
        oracle_pass = row["oracle_pass"]
        if oracle_pass not in ORACLE_PASS_VALUES:
            raise ValueError(
                f"{path}: invalid oracle_pass {oracle_pass!r} for "
                f"solver={solver} problem={problem} seed={seed}")
        cell = (solver, problem, solver_dispatch_mode(solver))
        seeds = by_cell[cell]
        if seed in seeds and seeds[seed] != oracle_pass:
            raise ValueError(
                f"{path}: conflicting oracle_pass for solver={solver} "
                f"problem={problem} seed={seed}: "
                f"{seeds[seed]!r} vs {oracle_pass!r}")
        seeds[seed] = oracle_pass
    return dict(by_cell)


def oracle_disposition(verdict: dict[tuple[str, str, str], dict[str, str]],
                       key: tuple[str, str, str],
                       cell_seeds: list[str],
                       context: str) -> str:
    """Resolve a cell disposition from the independent oracle verdict.

    Applies the pinned seed-aggregation rule: the cell is ``pass`` iff every
    seed of that cell cleared the oracle, otherwise ``expected_fail``. A cell
    with no verdict, or a cell missing any of its measured seeds, fails closed
    (raises) rather than silently falling back to a self-derived disposition.
    """
    seeds = verdict.get(key)
    if seeds is None:
        raise ValueError(
            f"{context}: no oracle verdict for cell {key}; refusing to "
            "fall back to self-derived disposition")
    missing = sorted(s for s in set(cell_seeds) if s not in seeds)
    if missing:
        raise ValueError(
            f"{context}: oracle verdict for cell {key} is missing "
            f"seed(s) {', '.join(missing)}; failing closed")
    all_pass = all(seeds[s] == ORACLE_PASS_TOKEN for s in cell_seeds)
    return DISPOSITION_PASS if all_pass else DISPOSITION_EXPECTED_FAIL


def write_baseline(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        f.write(HEADER_COMMENT_BLOCK)
        writer = csv.DictWriter(f, fieldnames=BASELINE_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in BASELINE_FIELDS})


def migrate_existing_baseline(prior_baseline: Path, output: Path) -> int:
    try:
        rows = load_baseline_rows(prior_baseline)
    except ValueError as exc:
        print(f"regen_baseline: {exc}", file=sys.stderr)
        return 1
    write_baseline(output, rows)
    print(f"regen_baseline: migrated {len(rows)} rows -> {output}", file=sys.stderr)
    return 0


def aggregate_summary(args: argparse.Namespace,
                      curated_cells: set[tuple[str, str, str]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    with args.summary.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"no rows in {args.summary}")

    required = [
        "solver",
        "problem",
        "seed",
        "solver_iters",
        "wall_time_us",
        "accuracy",
        "constraint_violation",
        "status",
        "row_disposition",
        "exclusion_reason",
        "provenance_id",
        "instructions",
    ]
    for row in rows:
        require_fields(row, required, str(args.summary))

    by_cell: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "per_step_us": [],
            "iters": [],
            "acc": [],
            "cv": [],
            "status": [],
            "seeds": [],
            "instr_per_iter": [],
            "excluded": False,
            "exclusion_reason": "",
            "provenance_id": "",
        }
    )
    for r in rows:
        solver = r["solver"]
        problem = r["problem"]
        if solver not in COHORT_SOLVERS:
            continue
        mode = solver_dispatch_mode(solver)
        key = (solver, problem, mode)
        if key not in curated_cells:
            continue

        cell = by_cell[key]
        row_disposition = r["row_disposition"]
        if row_disposition == "excluded":
            cell["excluded"] = True
            cell["exclusion_reason"] = r["exclusion_reason"] or "adapter capability"
            cell["provenance_id"] = r["provenance_id"]
            continue
        if row_disposition != "included":
            raise ValueError(f"{args.summary}: invalid row_disposition {row_disposition!r}")

        context = f"{args.summary}:{solver}:{problem}"
        iters = parse_int(r["solver_iters"], context)
        wall_us = parse_int(r["wall_time_us"], context)
        acc = parse_summary_metric(r["accuracy"], context)
        cv = parse_summary_metric(r["constraint_violation"], context)
        instructions = parse_int(r["instructions"], context)
        denom = iters if iters > 0 else 1
        cell["per_step_us"].append(wall_us / denom)
        cell["iters"].append(iters)
        cell["acc"].append(acc)
        cell["cv"].append(cv)
        cell["status"].append(r["status"])
        cell["seeds"].append(r["seed"])
        # A non-positive instructions value is the counter-unavailable sentinel;
        # excluding it here keeps a run without perf counters from committing a
        # zero/negative reference. A cell left with no measured instructions/iter
        # simply carries an empty reference (the gate then does not arm on it).
        if instructions > 0:
            cell["instr_per_iter"].append(instructions / denom)
        cell["provenance_id"] = r["provenance_id"]
    return by_cell


def generated_rows(args: argparse.Namespace,
                   provenance: dict[str, str],
                   verdict: dict[tuple[str, str, str], dict[str, str]]) -> list[dict[str, str]]:
    curated_cells = load_curated_cells(args.prior_baseline)
    print(f"regen_baseline: curated cohort size = {len(curated_cells)} cells",
          file=sys.stderr)

    prior_solvers = {s for (s, _p, _m) in curated_cells}
    missing_new_solvers = [s for s in sorted(COHORT_SOLVERS) if s not in prior_solvers]
    if missing_new_solvers:
        curated_cells = expand_cohort_for_new_solvers(
            curated_cells,
            ref_solver="filter_nw_sqp_accurate",
            new_solvers=missing_new_solvers,
        )

    prior_problems = {p for (_s, p, _m) in curated_cells}
    missing_fixture_problems = [p for p in CURATED_FIXTURE_PROBLEMS
                                if p not in prior_problems]
    if missing_fixture_problems:
        curated_cells = expand_cohort_for_new_problems(
            curated_cells,
            new_problems=missing_fixture_problems,
        )

    by_cell = aggregate_summary(args, curated_cells)
    rows_out: list[dict[str, str]] = []
    for (solver, problem, mode), cell in sorted(by_cell.items()):
        min_acc_log10, max_cv_log10 = mode_tolerances(mode)
        if cell["excluded"]:
            rows_out.append({
                "solver": solver,
                "problem": problem,
                "mode": mode,
                "max_us_per_step": "0",
                "min_accuracy_log10": f"{min_acc_log10:.1f}",
                "max_outer_iters": "0",
                "max_cv_log10": f"{max_cv_log10:.1f}",
                "disposition": DISPOSITION_EXCLUDED,
                "provenance_id": cell["provenance_id"] or provenance["provenance_id"],
                "wall_gate_policy": provenance["wall_gate_policy"],
                "exclusion_reason": cell["exclusion_reason"] or "adapter capability",
                "baseline_instr_per_iter": "",
                "instr_ceiling": "",
            })
            continue

        if not cell["per_step_us"]:
            continue

        # Numeric bounds stay measured advisory envelope numbers, but they no
        # longer decide pass/expected_fail. The disposition comes from the
        # independent oracle verdict (per-seed pass/fail aggregated per cell),
        # not from this run's own converged-status/accuracy/cv.
        bound_us = max(1, int(math.ceil(median(cell["per_step_us"]) * 1.30)))
        max_iters_empirical = max(cell["iters"])
        bound_iters = max(1, int(math.ceil(max_iters_empirical * 1.30)))
        manual_iters = MANUAL_ITERS_OVERRIDE.get((solver, problem, mode))
        if manual_iters is not None:
            bound_iters = manual_iters

        disposition = oracle_disposition(
            verdict,
            (solver, problem, mode),
            cell["seeds"],
            f"{args.summary}:{solver}:{problem}:{mode}",
        )

        # Committed reference instructions/iter (median across seeds) plus, for
        # the real-time-critical families, an absolute ceiling a margin above
        # the measured worst-seed cost. A cell with no measured instructions/iter
        # (counter unavailable on the whole run) carries an empty reference so
        # the gate does not arm on a fabricated number.
        ipi_samples = cell["instr_per_iter"]
        if ipi_samples:
            baseline_ipi = str(int(round(median(ipi_samples))))
            if is_realtime_critical_solver(solver):
                ceiling = str(int(math.ceil(max(ipi_samples) * INSTR_CEILING_MARGIN)))
            else:
                ceiling = ""
        else:
            baseline_ipi = ""
            ceiling = ""

        rows_out.append({
            "solver": solver,
            "problem": problem,
            "mode": mode,
            "max_us_per_step": str(bound_us),
            "min_accuracy_log10": f"{min_acc_log10:.1f}",
            "max_outer_iters": str(bound_iters),
            "max_cv_log10": f"{max_cv_log10:.1f}",
            "disposition": disposition,
            "provenance_id": provenance["provenance_id"],
            "wall_gate_policy": provenance["wall_gate_policy"],
            "exclusion_reason": "",
            "baseline_instr_per_iter": baseline_ipi,
            "instr_ceiling": ceiling,
        })
    return rows_out


def generate_baseline(args: argparse.Namespace) -> int:
    provenance = load_release_provenance(args.provenance) if args.provenance else None
    if provenance is None:
        return 1
    if getattr(args, "oracle_verdict", None) is None:
        print("regen_baseline: --oracle-verdict is required to source "
              "dispositions from the independent oracle", file=sys.stderr)
        return 1
    try:
        verdict = load_oracle_verdict(args.oracle_verdict)
        rows = generated_rows(args, provenance, verdict)
    except (OSError, ValueError) as exc:
        print(f"regen_baseline: {exc}", file=sys.stderr)
        return 1
    write_baseline(args.output, rows)
    counts = {name: sum(1 for row in rows if row["disposition"] == name)
              for name in sorted(DISPOSITIONS)}
    print(f"regen_baseline: {len(rows)} cells -> {args.output}", file=sys.stderr)
    for name, count in counts.items():
        print(f"  {name}: {count}", file=sys.stderr)
    return 0


def ledger_rows(path: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    for row in rows:
        require_fields(row, LEDGER_FIELDS, str(path))
    return rows


def ledger_has_row(rows: list[dict[str, str]],
                   metric: str,
                   solver: str,
                   problem: str,
                   mode: str,
                   old_value: str,
                   new_value: str) -> bool:
    for row in rows:
        if (row["metric"] == metric
            and row["solver"] == solver
            and row["problem"] == problem
            and row["mode"] == mode
            and row["old_value"] == old_value
            and row["new_value"] == new_value
            and row["provenance_id"]
            and row["correctness_witness"]
            and row["commit_kind"]):
            return True
    return False


def baseline_moves(prior: Path, current: Path) -> list[tuple[str, str, str, str, str, str]]:
    before = {
        (row["solver"], row["problem"], row["mode"]): row
        for row in load_baseline_rows(prior)
    }
    after = {
        (row["solver"], row["problem"], row["mode"]): row
        for row in load_baseline_rows(current)
    }
    moves: list[tuple[str, str, str, str, str, str]] = []
    for key, old_row in before.items():
        new_row = after.get(key)
        if new_row is None:
            continue
        solver, problem, mode = key
        for metric in NUMERIC_BASELINE_FIELDS:
            old_value = old_row[metric]
            new_value = new_row[metric]
            if parse_float(old_value, f"{prior}:{metric}") != parse_float(new_value, f"{current}:{metric}"):
                moves.append((metric, solver, problem, mode, old_value, new_value))
        # Disposition flips (pass<->expected_fail) were previously invisible
        # because only numeric fields were diffed. Surface them as a move so the
        # ledger-completeness check requires an explicit reconciliation row.
        old_disp = old_row["disposition"]
        new_disp = new_row["disposition"]
        if old_disp != new_disp:
            moves.append((DISPOSITION_LEDGER_METRIC, solver, problem, mode,
                          old_disp, new_disp))
    return moves


def moved_rows_from_file(path: Path) -> list[tuple[str, str, str, str, str, str]]:
    rows = read_csv_rows(path)
    moves: list[tuple[str, str, str, str, str, str]] = []
    for row in rows:
        require_fields(row, ["metric", "solver", "problem", "mode", "old_value", "new_value"],
                       str(path))
        if row["old_value"] != row["new_value"]:
            moves.append((
                row["metric"],
                row["solver"],
                row["problem"],
                row["mode"],
                row["old_value"],
                row["new_value"],
            ))
    return moves


def disabled_register_moves(path: Path) -> list[tuple[str, str, str, str, str, str]]:
    rows = read_csv_rows(path)
    moves: list[tuple[str, str, str, str, str, str]] = []
    for row in rows:
        require_fields(row, [
            "item",
            "grid_id",
            "selected_value",
            "old_value",
            "correctness_witness",
            "provenance_id",
            "action",
        ], str(path))
        if row["selected_value"] != row["old_value"]:
            moves.append((
                row["item"],
                row["grid_id"],
                "disabled_register",
                row["action"],
                row["old_value"],
                row["selected_value"],
            ))
    return moves


def validate_ledger_completeness(args: argparse.Namespace) -> int:
    try:
        ledger = ledger_rows(args.ledger)
        moves = baseline_moves(args.prior_baseline, args.current_baseline)
        for profile in args.profile_output or []:
            moves.extend(moved_rows_from_file(profile))
        for register in args.disabled_register or []:
            moves.extend(disabled_register_moves(register))
    except (OSError, ValueError) as exc:
        print(f"regen_baseline: {exc}", file=sys.stderr)
        return 1

    missing = []
    for metric, solver, problem, mode, old_value, new_value in moves:
        if not ledger_has_row(ledger, metric, solver, problem, mode,
                              old_value, new_value):
            missing.append((metric, solver, problem, mode, old_value, new_value))

    if missing:
        for metric, solver, problem, mode, old_value, new_value in missing:
            print("MISSING_LEDGER: "
                  f"metric={metric} solver={solver} problem={problem} "
                  f"mode={mode} old_value={old_value} new_value={new_value}",
                  file=sys.stderr)
        return 1
    print(f"regen_baseline: ledger covers {len(moves)} moved value(s)",
          file=sys.stderr)
    return 0


def assert_dispositions_from_oracle(args: argparse.Namespace) -> int:
    """Assert every committed baseline disposition equals the oracle verdict.

    Re-derives each non-excluded cell's disposition from the independent oracle
    verdict under the pinned seed-aggregation rule and compares it to the
    committed disposition. A mismatch means a self-derived (or stale)
    disposition survived in the shipped baseline; the check fails closed. This
    is the whole-cohort invariant that no disposition remains a product of the
    self-referential 1.3x derivation.
    """
    try:
        committed = load_baseline_rows(args.current_baseline)
        verdict = load_oracle_verdict(args.oracle_verdict)
        curated_cells = {(r["solver"], r["problem"], r["mode"]) for r in committed}
        by_cell = aggregate_summary(args, curated_cells)
    except (OSError, ValueError) as exc:
        print(f"regen_baseline: {exc}", file=sys.stderr)
        return 1

    mismatches: list[str] = []
    checked = 0
    for row in committed:
        key = (row["solver"], row["problem"], row["mode"])
        committed_disp = row["disposition"]
        if committed_disp == DISPOSITION_EXCLUDED:
            continue
        cell = by_cell.get(key)
        if cell is None or not cell["seeds"]:
            mismatches.append(
                f"DISPOSITION_UNMEASURED: cell {key} is committed "
                f"{committed_disp!r} but has no measurement to witness it")
            continue
        try:
            expected = oracle_disposition(
                verdict, key, cell["seeds"], f"{args.current_baseline}:{key}")
        except ValueError as exc:
            mismatches.append(f"DISPOSITION_NO_VERDICT: {exc}")
            continue
        checked += 1
        if expected != committed_disp:
            mismatches.append(
                f"DISPOSITION_MISMATCH: cell {key} committed {committed_disp!r} "
                f"but the oracle verdict aggregates to {expected!r}")

    if mismatches:
        for line in mismatches:
            print(line, file=sys.stderr)
        return 1
    print(f"regen_baseline: all {checked} committed dispositions equal the "
          "independent oracle verdict (no self-derived disposition survives)",
          file=sys.stderr)
    return 0


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


SELF_TEST_SUMMARY = """\
solver,library,problem,class,dimension,seed,mode,solver_iters,f_evals,g_evals,c_evals,J_evals,wall_time_us,final_objective,known_optimum,accuracy,constraint_violation,status,row_disposition,cap_status,exclusion_reason,solve_wall_time_us,end_to_end_wall_time_us,provenance_id,instructions
bobyqa,argmin,problem_a,inequality,2,42,publication,2,2,0,0,0,20,0.0,0.0,1e-10,1e-10,converged,included,none,,20,20,selftest-run,200
bobyqa,argmin,problem_b,inequality,2,42,publication,0,0,0,0,0,8,inf,0.0,inf,5e-1,failed,included,none,,8,8,selftest-run,800
"""

SELF_TEST_PRIOR = """\
solver,problem,mode,max_us_per_step,min_accuracy_log10,max_outer_iters,max_cv_log10,expected
bobyqa,problem_a,default,10,-8.0,10,-8.0,pass
bobyqa,problem_b,default,10,-8.0,10,-8.0,pass
"""


SELF_TEST_VERDICT = """\
solver,problem,mode,seed,oracle_pass,objective_distance,feasibility_residual,kkt_residual,witness
bobyqa,problem_a,publication,42,pass,1e-12,0.0,0.0,feasibility-at-x*
bobyqa,problem_b,publication,42,fail,5e-1,5e-1,0.0,feasibility-at-x*
"""

SELF_TEST_VERDICT_MISSING_CELL = """\
solver,problem,mode,seed,oracle_pass,objective_distance,feasibility_residual,kkt_residual,witness
bobyqa,problem_a,publication,42,pass,1e-12,0.0,0.0,feasibility-at-x*
"""


def run_self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="argmin-regen-baseline-") as tmp_s:
        tmp = Path(tmp_s)
        summary = tmp / "summary.csv"
        prior = tmp / "prior.csv"
        provenance = tmp / "provenance.json"
        verdict = tmp / "verdict.csv"
        output = tmp / "baseline.csv"
        write_text(summary, SELF_TEST_SUMMARY)
        write_text(prior, SELF_TEST_PRIOR)
        write_text(verdict, SELF_TEST_VERDICT)
        write_text(provenance, json.dumps({
            "build_type": "Release",
            "provenance_id": "selftest-release",
            "wall_gate_policy": "enforced",
        }))
        args = argparse.Namespace(
            summary=summary,
            prior_baseline=prior,
            output=output,
            provenance=provenance,
            oracle_verdict=verdict,
        )
        if generate_baseline(args) != 0:
            return 1
        rows = load_baseline_rows(output)
        if len(rows) != 2:
            print("regen_baseline self-test: expected two output rows", file=sys.stderr)
            return 1
        by_key = {(r["solver"], r["problem"]): r for r in rows}
        # Disposition must track the oracle verdict, not the checked run.
        if by_key[("bobyqa", "problem_a")]["disposition"] != DISPOSITION_PASS:
            print("regen_baseline self-test: oracle-pass cell not marked pass",
                  file=sys.stderr)
            return 1
        if by_key[("bobyqa", "problem_b")]["disposition"] != DISPOSITION_EXPECTED_FAIL:
            print("regen_baseline self-test: oracle-fail cell not marked "
                  "expected_fail", file=sys.stderr)
            return 1
        for row in rows:
            if row["disposition"] not in DISPOSITIONS or not row["provenance_id"]:
                print("regen_baseline self-test: invalid disposition or provenance",
                      file=sys.stderr)
                return 1

        # The instruction reference is emitted (median instructions/iter), and a
        # non-real-time family carries no absolute ceiling.
        if by_key[("bobyqa", "problem_a")]["baseline_instr_per_iter"] != "100":
            print("regen_baseline self-test: baseline_instr_per_iter not the "
                  "median instructions/iter", file=sys.stderr)
            return 1
        if by_key[("bobyqa", "problem_a")]["instr_ceiling"] != "":
            print("regen_baseline self-test: non-real-time cell carries an "
                  "absolute instruction ceiling", file=sys.stderr)
            return 1

        # A verdict missing a measured cell must fail closed, never silently
        # fall back to the old self-derived disposition.
        write_text(verdict, SELF_TEST_VERDICT_MISSING_CELL)
        if generate_baseline(args) == 0:
            print("regen_baseline self-test: missing-cell verdict did not fail "
                  "closed", file=sys.stderr)
            return 1

        # An absent verdict file must also fail closed.
        verdict.unlink()
        if generate_baseline(args) == 0:
            print("regen_baseline self-test: absent verdict did not fail closed",
                  file=sys.stderr)
            return 1
    return 0


def run_release_guard_self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="argmin-regen-guard-") as tmp_s:
        tmp = Path(tmp_s)
        summary = tmp / "summary.csv"
        prior = tmp / "prior.csv"
        provenance = tmp / "provenance.json"
        output = tmp / "baseline.csv"
        write_text(summary, SELF_TEST_SUMMARY)
        write_text(prior, SELF_TEST_PRIOR)
        write_text(provenance, json.dumps({
            "build_type": "Debug",
            "provenance_id": "selftest-debug",
        }))
        args = argparse.Namespace(
            summary=summary,
            prior_baseline=prior,
            output=output,
            provenance=provenance,
        )
        if generate_baseline(args) == 0:
            print("regen_baseline release-guard self-test: Debug provenance passed",
                  file=sys.stderr)
            return 1
    return 0


def run_ledger_self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="argmin-ledger-") as tmp_s:
        tmp = Path(tmp_s)
        prior = tmp / "prior.csv"
        current = tmp / "current.csv"
        ledger = tmp / "ledger.csv"
        profile = tmp / "profile.csv"
        register = tmp / "register.csv"
        # The current baseline both moves a numeric bound (10 -> 11) and flips
        # the disposition (pass -> expected_fail). The disposition flip must be
        # a ledger-visible move, not only the numeric bound.
        write_text(prior, """\
solver,problem,mode,max_us_per_step,min_accuracy_log10,max_outer_iters,max_cv_log10,disposition,provenance_id,wall_gate_policy,exclusion_reason
solver_a,problem_a,default,10,-8.0,10,-8.0,pass,old,enforced,
""")
        write_text(current, """\
solver,problem,mode,max_us_per_step,min_accuracy_log10,max_outer_iters,max_cv_log10,disposition,provenance_id,wall_gate_policy,exclusion_reason
solver_a,problem_a,default,11,-8.0,10,-8.0,expected_fail,new,enforced,
""")
        write_text(ledger, ",".join(LEDGER_FIELDS) + "\n")
        write_text(profile, """\
metric,solver,problem,mode,old_value,new_value
t_tau,solver_a,problem_a,default,5,6
""")
        write_text(register, """\
item,grid_id,selected_value,old_value,correctness_witness,provenance_id,selected_numeric,action,notes
default_x,grid_a,2,1,witness,new,true,updated,
""")
        args = argparse.Namespace(
            prior_baseline=prior,
            current_baseline=current,
            ledger=ledger,
            profile_output=[profile],
            disabled_register=[register],
        )
        if validate_ledger_completeness(args) == 0:
            print("regen_baseline ledger self-test: missing row passed",
                  file=sys.stderr)
            return 1

        # A ledger that reconciles the numeric moves but omits the disposition
        # flip must still fail: the flip is a first-class move now.
        write_text(ledger, ",".join(LEDGER_FIELDS) + "\n"
                   "max_us_per_step,solver_a,problem_a,default,10,11,"
                   "methodology,correctness,new,methodology-repair,,,\n"
                   "t_tau,solver_a,problem_a,default,5,6,"
                   "methodology,correctness,new,methodology-repair,,,\n"
                   "default_x,grid_a,disabled_register,updated,1,2,"
                   "methodology,correctness,new,methodology-repair,,,\n")
        if validate_ledger_completeness(args) == 0:
            print("regen_baseline ledger self-test: disposition flip was not "
                  "required by the completeness check", file=sys.stderr)
            return 1

        # With the disposition flip reconciled, the ledger is complete.
        write_text(ledger, ",".join(LEDGER_FIELDS) + "\n"
                   "max_us_per_step,solver_a,problem_a,default,10,11,"
                   "methodology,correctness,new,methodology-repair,,,\n"
                   "t_tau,solver_a,problem_a,default,5,6,"
                   "methodology,correctness,new,methodology-repair,,,\n"
                   "default_x,grid_a,disabled_register,updated,1,2,"
                   "methodology,correctness,new,methodology-repair,,,\n"
                   "disposition,solver_a,problem_a,default,pass,expected_fail,"
                   "methodology,oracle-verdict,new,methodology-repair,"
                   "expected_fail,pass->expected_fail\n")
        if validate_ledger_completeness(args) != 0:
            print("regen_baseline ledger self-test: complete ledger failed",
                  file=sys.stderr)
            return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path)
    p.add_argument("--prior-baseline", type=Path)
    p.add_argument("--current-baseline", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--provenance", type=Path)
    p.add_argument("--oracle-verdict", type=Path)
    p.add_argument("--ledger", type=Path)
    p.add_argument("--profile-output", action="append", type=Path)
    p.add_argument("--disabled-register", action="append", type=Path)
    p.add_argument("--accuracy-cutoff", type=float, default=1e-12)
    p.add_argument("--cv-cutoff", type=float, default=1e-8)
    p.add_argument("--migrate-existing-baseline", action="store_true")
    p.add_argument("--validate-ledger-completeness", action="store_true")
    p.add_argument("--assert-dispositions-from-oracle", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--self-test-release-guard", action="store_true")
    p.add_argument("--self-test-ledger-completeness", action="store_true")
    args = p.parse_args()

    if args.self_test:
        return run_self_test()
    if args.self_test_release_guard:
        return run_release_guard_self_test()
    if args.self_test_ledger_completeness:
        return run_ledger_self_test()

    if args.migrate_existing_baseline:
        if args.prior_baseline is None or args.output is None:
            print("regen_baseline: --prior-baseline and --output are required for migration",
                  file=sys.stderr)
            return 1
        return migrate_existing_baseline(args.prior_baseline, args.output)

    if args.validate_ledger_completeness:
        if args.prior_baseline is None or args.current_baseline is None or args.ledger is None:
            print("regen_baseline: --prior-baseline, --current-baseline, and --ledger "
                  "are required for ledger validation",
                  file=sys.stderr)
            return 1
        return validate_ledger_completeness(args)

    if args.assert_dispositions_from_oracle:
        if (args.current_baseline is None or args.oracle_verdict is None
                or args.summary is None):
            print("regen_baseline: --current-baseline, --oracle-verdict, and "
                  "--summary are required to assert dispositions from the oracle",
                  file=sys.stderr)
            return 1
        return assert_dispositions_from_oracle(args)

    if (args.summary is None or args.prior_baseline is None
        or args.output is None or args.provenance is None
        or args.oracle_verdict is None):
        print("regen_baseline: --summary, --prior-baseline, --output, "
              "--provenance, and --oracle-verdict are required",
              file=sys.stderr)
        return 1
    return generate_baseline(args)


if __name__ == "__main__":
    sys.exit(main())
