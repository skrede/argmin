#!/usr/bin/env python3
"""Sweep benchmark publication thresholds.

Default mode preserves the regression gate's (accuracy_cutoff, cv_cutoff)
sweep:

Runs regression_check across the 3x3 grid
(accuracy_cutoff in {1e-10, 1e-12, 1e-14}) x (cv_cutoff in {1e-6, 1e-8, 1e-10}),
reads the PROMOTE: stderr lines emitted under
ARGMIN_REGRESSION_CHECK_PROMOTE_LOG=1, and aggregates the per-pair promoted
set + boundary diffs. Emits a JSON record + markdown writeup at the paths
specified by --output-json and --output-md.

Feasibility mode sweeps a single publication eps_feas grid against the same
trace-first first-hit logic used by dm_profile.py and emits a JSON provenance
record before final publication profiles are generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

import dm_profile


PROMOTE_LINE_RE = re.compile(
    r"^PROMOTE:\s+solver=(?P<solver>\S+)\s+problem=(?P<problem>\S+)\s+"
    r"mode=(?P<mode>\S+)\s+max_accuracy=(?P<max_acc>\S+)\s+"
    r"max_cv=(?P<max_cv>\S+)\s*$"
)

SELF_TEST_SUMMARY = (
    "solver,library,problem,class,dimension,seed,mode,solver_iters,"
    "f_evals,g_evals,c_evals,J_evals,wall_time_us,final_objective,"
    "known_optimum,accuracy,constraint_violation,status,row_disposition,"
    "cap_status,exclusion_reason,solve_wall_time_us,end_to_end_wall_time_us,"
    "provenance_id\n"
    "solver_a,argmin,constrained,inequality,1,42,publication,3,"
    "3,0,0,0,3,0.0,0.0,0.0,1e-9,converged,included,"
    "none,,3,3,selftest\n"
)

SELF_TEST_TRACE = (
    "iter,f_evals,g_evals,c_evals,J_evals,wall_us,"
    "f_current,f_best,accuracy,cv,step_norm,kkt_residual\n"
    "0,1,0,0,0,1,0.0,0.0,0.0,5e-8,0.0,0.0\n"
    "1,2,0,0,0,2,0.0,0.0,0.0,1e-9,0.0,0.0\n"
)


def run_sweep(binary: Path, summary: Path, baseline: Path,
              acc_cutoff: float, cv_cutoff: float) -> tuple[list[dict], int]:
    env = os.environ.copy()
    env["ARGMIN_REGRESSION_CHECK_PROMOTE_LOG"] = "1"
    cmd = [
        str(binary), str(summary), str(baseline),
        "--accuracy-cutoff", f"{acc_cutoff:.16e}",
        "--cv-cutoff", f"{cv_cutoff:.16e}",
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          check=False)
    promoted = []
    for line in proc.stderr.splitlines():
        m = PROMOTE_LINE_RE.match(line)
        if not m:
            continue
        promoted.append({
            "solver": m.group("solver"),
            "problem": m.group("problem"),
            "mode": m.group("mode"),
            "max_accuracy": float(m.group("max_acc")),
            "max_cv": float(m.group("max_cv")),
        })
    return promoted, proc.returncode


def _parse_grid(s: str) -> list[float]:
    values = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return sorted(set(values), reverse=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _write_self_test_run(run_dir: Path) -> None:
    traces = run_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    (run_dir / "publish_summary.csv").write_text(SELF_TEST_SUMMARY)
    (traces / "solver_a_constrained_seed42.csv").write_text(SELF_TEST_TRACE)


def _constrained_records(summary: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for record in summary.to_dict("records"):
        if str(record.get("row_disposition", "")) != "included":
            continue
        if not dm_profile._is_constrained_class(str(record["class"])):
            continue
        records.append(record)
    return records


def _trace_path(traces_dir: Path, record: dict[str, object]) -> Path:
    return traces_dir / f"{record['solver']}_{record['problem']}_seed{int(record['seed'])}.csv"


def _observed_feasibility_breakpoints(run_dir: Path,
                                      summary: pd.DataFrame,
                                      tau: float) -> list[float]:
    traces_dir = run_dir / "traces"
    breakpoints: set[float] = set()
    for record in _constrained_records(summary):
        trace_path = _trace_path(traces_dir, record)
        if not trace_path.is_file():
            continue
        trace = pd.read_csv(trace_path)
        if trace.empty or "f_best" not in trace.columns or "cv" not in trace.columns:
            continue
        f_star = float(record["known_optimum"])
        objective_mask = (
            (pd.to_numeric(trace["f_best"], errors="coerce") - f_star).abs()
            <= tau * max(1.0, abs(f_star)))
        cv = pd.to_numeric(trace["cv"], errors="coerce")
        for value in cv[objective_mask]:
            finite_value = float(value)
            if math.isfinite(finite_value) and finite_value > 0.0:
                breakpoints.add(finite_value)
    return sorted(breakpoints, reverse=True)


def _input_provenance(run_dir: Path, self_test: bool) -> dict[str, object]:
    if self_test:
        return {
            "kind": "self-test-feasibility",
            "summary_sha256": hashlib.sha256(SELF_TEST_SUMMARY.encode()).hexdigest(),
            "trace_sha256": hashlib.sha256(SELF_TEST_TRACE.encode()).hexdigest(),
        }
    summary = run_dir / "publish_summary.csv"
    traces = sorted((run_dir / "traces").glob("*.csv"))
    return {
        "kind": "publish_bench_run",
        "run_dir": str(run_dir),
        "summary_sha256": _sha256(summary),
        "trace_count": len(traces),
        "trace_sha256": [
            {"path": str(p.relative_to(run_dir)), "sha256": _sha256(p)}
            for p in traces
        ],
    }


def _collect_feasibility_stats(run_dir: Path,
                               eps_grid: list[float],
                               tau: float,
                               summary: pd.DataFrame) -> tuple[list[dict[str, object]], dict[str, object]]:
    traces_dir = run_dir / "traces"
    if not traces_dir.is_dir():
        raise FileNotFoundError(f"missing traces directory: {traces_dir}")

    rows_by_eps: list[dict[str, object]] = []
    for eps in eps_grid:
        rows = dm_profile._collect_t_tau_rows(run_dir, summary, (tau,), eps)
        if rows.empty:
            constrained_rows = pd.DataFrame({"t_tau": []})
        else:
            constrained_rows = rows[
                (rows["metric"] == "f_evals")
                & rows["problem"].map(lambda p: dm_profile._is_constrained_class(
                    str(summary[summary["problem"] == p]["class"].iloc[0])))]
        hit_count = int(constrained_rows["t_tau"].map(math.isfinite).sum())

        objective_hit_points = 0
        rejected_infeasible_points = 0
        for record in _constrained_records(summary):
            trace_path = _trace_path(traces_dir, record)
            if not trace_path.is_file():
                continue
            trace = pd.read_csv(trace_path)
            if trace.empty:
                continue
            f_star = float(record["known_optimum"])
            objective_mask = (
                (pd.to_numeric(trace["f_best"], errors="coerce") - f_star).abs()
                <= tau * max(1.0, abs(f_star)))
            if "cv" in trace.columns:
                cv = pd.to_numeric(trace["cv"], errors="coerce")
            else:
                cv = pd.Series([float("nan")] * len(trace))
            rejected_mask = objective_mask & (~cv.map(math.isfinite) | (cv > eps))
            objective_hit_points += int(objective_mask.sum())
            rejected_infeasible_points += int(rejected_mask.sum())

        rows_by_eps.append({
            "eps_feas": eps,
            "constrained_hit_count": hit_count,
            "objective_accurate_constrained_hits": objective_hit_points,
            "rejected_objective_accurate_infeasible_hits": rejected_infeasible_points,
        })

    selected = 1e-8 if any(abs(v - 1e-8) <= 0.0 for v in eps_grid) else eps_grid[len(eps_grid) // 2]
    selected_stats = next(row for row in rows_by_eps
                          if abs(float(row["eps_feas"]) - selected) <= 0.0)
    looser = min((row for row in rows_by_eps
                  if float(row["eps_feas"]) > selected),
                 key=lambda row: float(row["eps_feas"]),
                 default=None)
    stricter = max((row for row in rows_by_eps
                    if float(row["eps_feas"]) < selected),
                   key=lambda row: float(row["eps_feas"]),
                   default=None)
    neighbor_parts = []
    if looser is not None:
        neighbor_parts.append(
            f"nearest looser hit count {looser['constrained_hit_count']} "
            f"at eps={looser['eps_feas']:.3e}")
    if stricter is not None:
        neighbor_parts.append(
            f"nearest stricter hit count {stricter['constrained_hit_count']} "
            f"at eps={stricter['eps_feas']:.3e}")
    neighbor_evidence = "; ".join(neighbor_parts) if neighbor_parts else "no neighboring thresholds"
    rationale = (
        "Selected 1e-8 when present because it is the centered publication "
        "candidate and the sweep records adjacent stricter, looser, and "
        "observed-breakpoint classifications. Selected-threshold constrained "
        f"hit count is {selected_stats['constrained_hit_count']}; "
        f"{neighbor_evidence}. Objective-accurate trace points with feasibility "
        "above the selected threshold remain counted as rejected, so the "
        "record exposes any material neighboring-threshold movement instead "
        "of hiding it."
    )
    selection = {
        "selected_eps_feas": selected,
        "selection_rationale": rationale,
        "selected_threshold_stats": selected_stats,
    }
    return rows_by_eps, selection


def run_feasibility_sweep(args: argparse.Namespace) -> int:
    eps_grid = _parse_grid(args.eps_feas_grid)
    if not eps_grid:
        print("threshold_pair_sweep: empty --eps-feas-grid", file=sys.stderr)
        return 1
    if not any(abs(v - 1e-8) <= 0.0 for v in eps_grid):
        print("threshold_pair_sweep: grid must contain 1e-8", file=sys.stderr)
        return 1
    if not any(v > 1e-8 for v in eps_grid) or not any(v < 1e-8 for v in eps_grid):
        print("threshold_pair_sweep: grid must contain stricter and looser values around 1e-8",
              file=sys.stderr)
        return 1

    if args.self_test_feasibility:
        tmp = tempfile.TemporaryDirectory(prefix="argmin-eps-feas-")
        run_dir = Path(tmp.name)
        _write_self_test_run(run_dir)
        self_test = True
    else:
        tmp = None
        if args.run_dir is None:
            print("threshold_pair_sweep: --run-dir is required outside self-test mode",
                  file=sys.stderr)
            return 1
        run_dir = args.run_dir
        self_test = False

    try:
        summary = dm_profile._load_summary(run_dir)
        observed_breakpoints = [] if self_test else _observed_feasibility_breakpoints(
            run_dir, summary, args.tau)
        eps_grid = sorted(set(eps_grid + observed_breakpoints), reverse=True)
        by_threshold, selection = _collect_feasibility_stats(
            run_dir, eps_grid, args.tau, summary)
        output = args.output if args.output is not None else args.output_json
        if output is None:
            print("threshold_pair_sweep: --output is required in feasibility mode",
                  file=sys.stderr)
            return 1
        result = {
            "script_version": "eps-feas-sweep-v1",
            "harness_version": "publish-bench-contract-v1",
            "tau": args.tau,
            "eps_feas_grid": eps_grid,
            "observed_cv_breakpoints": observed_breakpoints,
            "per_threshold": by_threshold,
            "selected_eps_feas": selection["selected_eps_feas"],
            "selection_rationale": selection["selection_rationale"],
            "selected_threshold_stats": selection["selected_threshold_stats"],
            "input_provenance": _input_provenance(run_dir, self_test),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"threshold_pair_sweep: emitted {output}", file=sys.stderr)
        return 0
    finally:
        if tmp is not None:
            tmp.cleanup()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=Path)
    p.add_argument("--summary", type=Path)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--head-sha", default="HEAD", type=str)
    p.add_argument("--output-json", type=Path)
    p.add_argument("--output-md", type=Path)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--self-test-feasibility", action="store_true")
    p.add_argument("--eps-feas-grid", default="1e-6,1e-7,1e-8,1e-9,1e-10")
    p.add_argument("--tau", default=1e-8, type=float)
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    if args.self_test_feasibility or args.run_dir is not None:
        return run_feasibility_sweep(args)

    if args.binary is None or args.summary is None or args.baseline is None:
        print("threshold_pair_sweep: --binary, --summary, and --baseline are required",
              file=sys.stderr)
        return 1
    if args.output_json is None or args.output_md is None:
        print("threshold_pair_sweep: --output-json and --output-md are required",
              file=sys.stderr)
        return 1

    accuracy_axis = [1e-10, 1e-12, 1e-14]
    cv_axis = [1e-6, 1e-8, 1e-10]

    by_pair = []
    pair_matrix = {}
    for acc in accuracy_axis:
        for cv in cv_axis:
            promoted, rc = run_sweep(args.binary, args.summary, args.baseline,
                                     acc, cv)
            promoted_count = len(promoted)
            pair_matrix[(acc, cv)] = promoted_count
            by_pair.append({
                "accuracy_cutoff": acc,
                "cv_cutoff": cv,
                "promoted_count": promoted_count,
                "regression_check_rc": rc,
                "rows": sorted(promoted,
                               key=lambda r: (r["solver"], r["problem"], r["mode"])),
            })
            print(f"  (acc={acc:.0e}, cv={cv:.0e}): {promoted_count} promoted, "
                  f"rc={rc}", file=sys.stderr)

    # Selection: pick the (1e-12, 1e-8) pair as primary (per the
    # canonical-references analysis -- matches the per-iter machine-precision
    # stall trace evidence). Verify the boundary behavior is sane.
    chosen_acc = 1e-12
    chosen_cv = 1e-8
    chosen_entry = next(e for e in by_pair
                        if e["accuracy_cutoff"] == chosen_acc
                        and e["cv_cutoff"] == chosen_cv)

    # Find the cells that flip between adjacent (chosen vs neighbors).
    chosen_rows = {(r["solver"], r["problem"], r["mode"])
                   for r in chosen_entry["rows"]}
    boundary = {}
    for entry in by_pair:
        rows = {(r["solver"], r["problem"], r["mode"]) for r in entry["rows"]}
        added = sorted(rows - chosen_rows)
        dropped = sorted(chosen_rows - rows)
        key = f"acc={entry['accuracy_cutoff']:.0e}_cv={entry['cv_cutoff']:.0e}"
        boundary[key] = {
            "vs_chosen_added": [list(t) for t in added],
            "vs_chosen_dropped": [list(t) for t in dropped],
        }

    rationale = (
        "Selected (accuracy_cutoff=1e-12, cv_cutoff=1e-8) as the canonical "
        "pair. The accuracy axis values are the per-mode tolerance floor "
        "(accurate mode is 10^-8 on min_accuracy_log10; 1e-12 is four "
        "orders below) and the cv axis is the standard infeasibility-tolerance "
        "expected for machine-precision-stalled iterates on the HS-suite. "
        "Looser pairs (1e-10, 1e-6) admit cells with cv up to 1e-6, which "
        "overlaps with genuinely-stalled mid-precision iterates rather than "
        "machine-precision feasible ones. Tighter pairs (1e-14, 1e-10) drop "
        "cells whose worst-seed accuracy lands in [1e-14, 1e-12], i.e. just "
        "above the double-precision noise floor for low-dimensional Hessians."
    )

    result = {
        "head_sha": args.head_sha,
        "grid": {
            "accuracy": accuracy_axis,
            "cv": cv_axis,
        },
        "by_pair": by_pair,
        "selection": {
            "accuracy_cutoff": chosen_acc,
            "cv_cutoff": chosen_cv,
            "promoted_count": chosen_entry["promoted_count"],
            "promoted_rows": [[r["solver"], r["problem"], r["mode"]]
                              for r in chosen_entry["rows"]],
            "rationale": rationale,
        },
        "boundary_diffs": boundary,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(result, f, indent=2)

    # Markdown.
    lines = []
    lines.append(f"# Threshold-pair sweep for regression_check status-aware promote-to-pass\n")
    lines.append(f"**HEAD:** `{args.head_sha}`\n")
    lines.append(f"**Summary input:** `{args.summary}`\n")
    lines.append(f"**Baseline input:** `{args.baseline}`\n")
    lines.append("")
    lines.append("## Grid")
    lines.append("")
    lines.append("Threshold pair sweep over a 3x3 grid:")
    lines.append("")
    lines.append("- `accuracy_cutoff in {1e-10, 1e-12, 1e-14}`")
    lines.append("- `cv_cutoff in {1e-6, 1e-8, 1e-10}`")
    lines.append("")
    lines.append("## Promoted-count matrix")
    lines.append("")
    lines.append("| accuracy_cutoff \\\\ cv_cutoff | "
                 + " | ".join(f"{cv:.0e}" for cv in cv_axis) + " |")
    lines.append("|" + ("-" * 28) + "|"
                 + "|".join("-----------" for _ in cv_axis) + "|")
    for acc in accuracy_axis:
        row = f"| **{acc:.0e}** | "
        row += " | ".join(str(pair_matrix[(acc, cv)]) for cv in cv_axis) + " |"
        lines.append(row)
    lines.append("")
    lines.append("## Selection rationale")
    lines.append("")
    lines.append(rationale)
    lines.append("")
    lines.append("## Boundary diffs vs chosen pair (1e-12, 1e-8)")
    lines.append("")
    for entry in by_pair:
        key = f"acc={entry['accuracy_cutoff']:.0e}_cv={entry['cv_cutoff']:.0e}"
        b = boundary[key]
        if not b["vs_chosen_added"] and not b["vs_chosen_dropped"]:
            continue
        lines.append(f"### {key}")
        lines.append("")
        if b["vs_chosen_added"]:
            lines.append(f"**Added** (in this pair, not in chosen):")
            for row in b["vs_chosen_added"]:
                lines.append(f"- `{row[0]} / {row[1]} / {row[2]}`")
        if b["vs_chosen_dropped"]:
            lines.append(f"**Dropped** (in chosen, not in this pair):")
            for row in b["vs_chosen_dropped"]:
                lines.append(f"- `{row[0]} / {row[1]} / {row[2]}`")
        lines.append("")
    lines.append("## Chosen pair promoted set")
    lines.append("")
    lines.append(f"**Pair:** (accuracy_cutoff={chosen_acc:.0e}, "
                 f"cv_cutoff={chosen_cv:.0e})")
    lines.append(f"**Count:** {chosen_entry['promoted_count']} rows promoted "
                 f"from skip to pass.")
    lines.append("")
    if chosen_entry["rows"]:
        lines.append("| solver | problem | mode | max_accuracy | max_cv |")
        lines.append("|--------|---------|------|--------------|--------|")
        for r in chosen_entry["rows"]:
            lines.append(f"| `{r['solver']}` | `{r['problem']}` | "
                         f"`{r['mode']}` | {r['max_accuracy']:.3e} | "
                         f"{r['max_cv']:.3e} |")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n")

    print(f"sweep: emitted {args.output_json}", file=sys.stderr)
    print(f"sweep: emitted {args.output_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
