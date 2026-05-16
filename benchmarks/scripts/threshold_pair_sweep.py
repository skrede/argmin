#!/usr/bin/env python3
"""Sweep the (accuracy_cutoff, cv_cutoff) pair for regression_check's
status-aware accept-as-pass rule.

Runs regression_check N times across the 3x3 grid
(accuracy_cutoff in {1e-10, 1e-12, 1e-14}) x (cv_cutoff in {1e-6, 1e-8, 1e-10}),
reads the PROMOTE: stderr lines emitted under
ARGMIN_REGRESSION_CHECK_PROMOTE_LOG=1, and aggregates the per-pair promoted
set + boundary diffs. Emits a JSON record + markdown writeup at the paths
specified by --output-json and --output-md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


PROMOTE_LINE_RE = re.compile(
    r"^PROMOTE:\s+solver=(?P<solver>\S+)\s+problem=(?P<problem>\S+)\s+"
    r"mode=(?P<mode>\S+)\s+max_accuracy=(?P<max_acc>\S+)\s+"
    r"max_cv=(?P<max_cv>\S+)\s*$"
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--binary", required=True, type=Path)
    p.add_argument("--summary", required=True, type=Path)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--head-sha", default="HEAD", type=str)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--output-md", required=True, type=Path)
    args = p.parse_args()

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
    lines.append(f"# Plan-04 threshold-pair sweep\n")
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
