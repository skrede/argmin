#!/usr/bin/env python3
"""Coverage floor gate over the argmin header tree.

Reads a gcovr JSON report and a committed floors file, computes true
per-source-line and per-source-branch coverage (globally and per top-level
subtree under lib/argmin/include/argmin/), and fails if any measured value
drops below its committed floor.

Why a custom aggregator instead of `gcovr --fail-under-line/-branch`:
gcovr's own summary counts one entry per template instantiation, so a heavily
templated header contributes the same source line many times and the headline
percentage is skewed low. This script merges by (file, line) — a source line
is covered if it is hit in any instantiation, a branch is covered if it is
taken in any instantiation — which is the coverage a reader means. The floors
are stored just below the measured baseline and ratchet upward over time; the
build fails on any regression below a floor.

Usage:
    coverage_gate.py <gcovr.json> <floors.txt>

Floors file format (one key=value per line, '#' comments allowed):
    global_line   = 92.0
    global_branch = 48.0
    solver_line   = 91.0
    solver_branch = 50.0
    detail_line   = 91.0
    ...
A subtree floor is enforced only if a key for it is present; a directory that
appears in the report with no matching floor key is reported but not gated.
"""

import json
import sys
from collections import defaultdict

_MARKER = "include/argmin/"


def _subtree(path):
    rel = path.split(_MARKER, 1)[-1]
    return rel.split("/", 1)[0] if "/" in rel else "(root)"


def _load_floors(path):
    floors = {}
    with open(path) as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            key, _, value = line.partition("=")
            floors[key.strip()] = float(value.strip())
    return floors


def _measure(report_path):
    data = json.load(open(report_path))
    # dir -> {"line": {(file, no): covered}, "branch": {(file, no, idx): covered}}
    lines = defaultdict(dict)
    branches = defaultdict(dict)
    for entry in data.get("files", []):
        path = entry["file"]
        sub = _subtree(path)
        for ln in entry.get("lines", []):
            no = ln["line_number"]
            covered = ln.get("count", 0) > 0
            key = (path, no)
            lines[sub][key] = lines[sub].get(key, False) or covered
            for idx, br in enumerate(ln.get("branches", []) or []):
                bkey = (path, no, idx)
                bcov = br.get("count", 0) > 0
                branches[sub][bkey] = branches[sub].get(bkey, False) or bcov
    return lines, branches


def _pct(covered, total):
    return 100.0 * covered / total if total else 100.0


def main(argv):
    if len(argv) != 3:
        print("usage: coverage_gate.py <gcovr.json> <floors.txt>", file=sys.stderr)
        return 2
    report_path, floors_path = argv[1], argv[2]
    floors = _load_floors(floors_path)
    lines, branches = _measure(report_path)

    subtrees = sorted(set(lines) | set(branches))
    rows = []
    g_lc = g_lt = g_bc = g_bt = 0
    for sub in subtrees:
        lm = lines.get(sub, {})
        bm = branches.get(sub, {})
        lc = sum(1 for v in lm.values() if v)
        lt = len(lm)
        bc = sum(1 for v in bm.values() if v)
        bt = len(bm)
        g_lc += lc
        g_lt += lt
        g_bc += bc
        g_bt += bt
        rows.append((sub, lc, lt, bc, bt))

    failures = []

    def check(key, measured):
        floor = floors.get(key)
        if floor is not None and measured + 1e-9 < floor:
            failures.append((key, measured, floor))
        return floor

    print(f"{'scope':<18}{'line%':>8}{'floor':>8}{'branch%':>9}{'floor':>8}")
    gl = _pct(g_lc, g_lt)
    gb = _pct(g_bc, g_bt)
    lf = check("global_line", gl)
    bf = check("global_branch", gb)
    print(f"{'GLOBAL':<18}{gl:>7.1f}%{('-' if lf is None else f'{lf:.1f}'):>8}"
          f"{gb:>8.1f}%{('-' if bf is None else f'{bf:.1f}'):>8}")
    for sub, lc, lt, bc, bt in rows:
        pl = _pct(lc, lt)
        pb = _pct(bc, bt)
        lf = check(f"{sub}_line", pl)
        bf = check(f"{sub}_branch", pb)
        print(f"{sub:<18}{pl:>7.1f}%{('-' if lf is None else f'{lf:.1f}'):>8}"
              f"{pb:>8.1f}%{('-' if bf is None else f'{bf:.1f}'):>8}")

    if failures:
        print("\nCOVERAGE GATE FAILED:", file=sys.stderr)
        for key, measured, floor in failures:
            print(f"  {key}: {measured:.2f}% < floor {floor:.2f}%", file=sys.stderr)
        return 1
    print("\ncoverage gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
