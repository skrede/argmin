#!/usr/bin/env python3
"""Fixture repositories exercising every debt-baseline verdict, including growth against a git ref."""

import shutil
import tempfile
from pathlib import Path

from check_file_size_test import BASELINE, WHY, build, case_dir, check, git, run_gate, write_tree


def frow(path: str, lines: int) -> str:
    return f"file\t{path}\t-\t{lines}"


BIG = {"lib/big.h": 205}
ROW = [frow("lib/big.h", 205)]
SOLVE = "function\tlib/big.h\tsolve\t40"
CASES = [
    ("an exact row passes", BIG, [], ROW, 0, "1 baseline file rows exact"),
    ("one line over the row is growth", {"lib/big.h": 206}, [], ROW, 4, "grown to 206 lines from the 205"),
    ("one line under the row is inflated", {"lib/big.h": 204}, [], [frow("lib/big.h", 206)],
     7, "records 206 lines but the unit is 204"),
    ("a row for a file at the ceiling is stale", {"lib/big.h": 200}, [], [frow("lib/big.h", 200)],
     3, "is stale"),
    ("a row for a missing file is stale", BIG, [], ROW + [frow("lib/gone.h", 300)], 3, "lib/gone.h"),
    ("an overage in neither register", BIG, [], [], 1, "in neither EXCEPTIONS.md nor the baseline"),
    ("function rows do not affect file verdicts", BIG, [], ROW + [SOLVE], 0, "1 baseline file"),
    ("a row of the wrong kind", BIG, [], ROW + ["class\tlib/big.h\t-\t205"], 5, "neither 'file' nor"),
    ("a row missing a column", BIG, [], ["file\tlib/big.h\t205"], 5, "four tab-separated columns"),
    ("a non-numeric count", BIG, [], ["file\tlib/big.h\t-\tmany"], 5, "is not a number"),
    ("a file row carrying a name", BIG, [], ["file\tlib/big.h\tsolve\t205"], 5, "as its name"),
    ("a function row with no name", BIG, [], ROW + ["function\tlib/big.h\t-\t40"], 5, "as its name"),
    ("a path outside the scope", BIG, [], ROW + [frow("tests/big.cpp", 300)], 5, "outside the checked"),
    ("rows out of order", {"lib/a.h": 205, "lib/b.h": 205}, [],
     [frow("lib/b.h", 205), frow("lib/a.h", 205)], 5, "out of order"),
    ("a file with two rows", BIG, [], ROW + ROW, 5, "a second row"),
    ("a path in both registers", BIG, [("lib/big.h", 205, WHY)], ROW, 5, "in both EXCEPTIONS.md"),
]

HISTORY_CASES = [
    ("a new row since the ref", [({}, []), (BIG, ROW)], "HEAD", 8, "gained a row for lib/big.h"),
    ("a raised count since the ref", [(BIG, ROW), ({"lib/big.h": 206}, [frow("lib/big.h", 206)])],
     "HEAD", 8, "rose from 205 to 206"),
    ("a raised function row since the ref", [(BIG, ROW + [SOLVE]), (BIG, ROW + [SOLVE + "1"])],
     "HEAD", 8, "lib/big.h solve rose from 40 to 401"),
    ("a lowered row passes", [({"lib/big.h": 210}, [frow("lib/big.h", 210)]), (BIG, ROW)],
     "HEAD", 0, "1 baseline file rows exact"),
    ("a deleted row passes", [(BIG, ROW), ({"lib/big.h": 150}, [])], "HEAD", 0, "0 baseline file"),
    ("a ref older than the baseline", [({}, None), (BIG, ROW), (BIG, ROW)], "HEAD~1", 0,
     "baseline absent at HEAD~1: first recording"),
    ("a baseline not yet committed", [({}, None), (BIG, ROW)], "HEAD", 0,
     "baseline absent at HEAD: first recording"),
    ("a baseline removed after its introduction", [(BIG, ROW), (BIG, None), (BIG, ROW)], "HEAD", 8,
     "absent at HEAD, after it was introduced"),
    ("a ref that names no commit", [(BIG, ROW), (BIG, ROW)], "no-such-ref", 6, "does not name"),
]


def snapshot(root: Path, sources: dict[str, int], baseline: list[str] | None) -> None:
    shutil.rmtree(root / "lib", ignore_errors=True)
    (root / BASELINE).unlink(missing_ok=True)
    write_tree(root, sources, [], "", baseline)
    git(root, "add", "-A")


def build_history(root: Path, history: list) -> None:
    git(root, "init", "-q")
    for sources, baseline in history[:-1]:
        snapshot(root, sources, baseline)
        git(root, "commit", "-q", "--allow-empty", "-m", "snapshot")
    snapshot(root, *history[-1])


def run_cases(root: Path) -> int:
    failures = 0
    for label, sources, rows, baseline, want_code, want_text in CASES:
        case = case_dir(root, label)
        build(case, sources, rows, "", baseline)
        failures += not check(label, *run_gate(case), want_code, want_text)
    for label, history, ref, want_code, want_text in HISTORY_CASES:
        case = case_dir(root, label)
        build_history(case, history)
        failures += not check(label, *run_gate(case, "--baseline-ref", ref), want_code, want_text)
    return failures


def run_edges(root: Path) -> int:
    case = case_dir(root, "edges")
    build_history(case, [(BIG, ROW), (BIG, ROW)])
    failures = not check("an empty baseline ref", *run_gate(case, "--baseline-ref", ""), 2, "needs a git")
    before = git_state(case)
    code, output = run_gate(case, "--print-baseline")
    unchanged = git_state(case) == before
    failures += not check("the print flag writes nothing", code, output if unchanged else "wrote",
                          0, frow("lib/big.h", 205))
    (case / BASELINE).unlink()
    return failures + (not check("the baseline itself missing", *run_gate(case), 5, "does not exist"))


def git_state(root: Path) -> tuple[str, str]:
    status = git(root, "status", "--porcelain")
    return status, (root / BASELINE).read_text()


def main() -> int:
    with tempfile.TemporaryDirectory() as workspace:
        root = Path(workspace)
        failures = run_cases(root) + run_edges(root)
    total = len(CASES) + len(HISTORY_CASES) + 3
    if failures:
        print(f"{failures} of {total} cases failed")
        return 1
    print(f"{total} cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
