#!/usr/bin/env python3
"""Fixture repositories and canned clang-tidy output exercising every verdict the function-size gate can reach."""

import subprocess
import sys
import tempfile
from pathlib import Path

GATE = Path(__file__).resolve().parent / "check_function_size.py"
WHY = "one cohesive routine whose shared locals splitting would scatter across helpers"
FILES = "# register\n\n## Files\n\n| Path | Lines at registration | Why |\n| --- | --- | --- |\n"
FUNCTIONS = "\n## Functions\n\n| Path | Function | Lines at registration | Why |\n| --- | --- | --- | --- |\n"
BASELINE = "scripts/size_baseline.tsv"
CHECK = "[readability-function-size]"
HEADER = "lib/argmin/include/argmin/solver.h"
OTHER = "lib/argmin/include/argmin/detail/kernel.h"
ERROR = "/x/tu.cpp:1:10: error: 'missing.h' file not found [clang-diagnostic-error]\n"
ORPHAN = f"/x/{HEADER}:9:5: warning: function 'orphan' exceeds recommended size/complexity thresholds {CHECK}\n"


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=fixture", "-c", "user.email=f@x",
                           "-c", "commit.gpgsign=false", *args], check=True, capture_output=True, text=True).stdout


def diagnostic(path: str, line: int, name: str, lines: int) -> str:
    at = f"{path}:{line}:5"
    return (f"{at}: warning: function '{name}' exceeds recommended size/complexity thresholds {CHECK}\n"
            f"    void {name}()\n    ^\n"
            f"{at}: note: {lines} lines including whitespace and comments (threshold 25)\n")


def tidy_output(root: Path, findings: list, canary: bool, extra: str) -> str:
    text = "".join(diagnostic(f"{root}/{path}", 10 + i, name, n) for i, (path, name, n) in enumerate(findings))
    canary_text = diagnostic("/tmp/argmin-function-size.x/all_headers.cpp", 170, "argmin_size_gate_canary", 30)
    return text + (canary_text if canary else "") + extra


def baseline_text(rows: list) -> str:
    ordered = sorted(rows)
    return "# baseline\n" + "".join(f"function\t{path}\t{name}\t{n}\n" for path, name, n in ordered)


def build(root: Path, findings: list, rows: list, registered: list, canary: bool = True,
          extra: str = "") -> None:
    git(root, "init", "-q")
    (root / "scripts").mkdir()
    (root / BASELINE).write_text(baseline_text(rows))
    table = "".join(f"| {path} | {name} | {n} | {WHY} |\n" for path, name, n in registered)
    (root / "EXCEPTIONS.md").write_text(FILES + FUNCTIONS + table)
    (root / "tidy.txt").write_text(tidy_output(root.resolve(), findings, canary, extra))
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "fixture")


def run_gate(root: Path, *args: str) -> tuple[int, str]:
    done = subprocess.run([sys.executable, str(GATE), "--source-root", str(root),
                           "--tidy-output", str(root / "tidy.txt"), *args], capture_output=True, text=True)
    return done.returncode, done.stdout + done.stderr


SOLVE = (HEADER, "solve", 40)
CASES = [
    ("every finding recorded", [SOLVE], [SOLVE], [], {}, 0, "1 baseline function rows exact"),
    ("an unlisted finding", [SOLVE, (OTHER, "kernel", 30)], [SOLVE], [], {}, 1, "in neither"),
    ("a function one line past its baseline row", [(HEADER, "solve", 41)], [SOLVE], [], {}, 4,
     "has grown to 41 lines from the 40"),
    ("a baseline row longer than its function", [(HEADER, "solve", 39)], [SOLVE], [], {}, 7,
     "records 40 lines but the unit is 39"),
    ("a baseline row whose function is gone", [SOLVE], [SOLVE, (OTHER, "kernel", 30)], [], {}, 3,
     "kernel recording 30 lines is stale"),
    ("a registered function within its allowance", [(HEADER, "solve", 44)], [], [SOLVE], {}, 0,
     "1 registered exceptions"),
    ("a registered function past its allowance", [(HEADER, "solve", 45)], [], [SOLVE], {}, 4,
     "past its allowance of 44"),
    ("a registered row longer than its function", [(HEADER, "solve", 39)], [], [SOLVE], {}, 7,
     "records 40 lines but the function is 39"),
    ("a registered row whose function is gone", [], [], [SOLVE], {}, 3, "is stale"),
    ("overloads pair by sorted counts", [(HEADER, "solve", 40), (HEADER, "solve", 30)],
     [(HEADER, "solve", 30), SOLVE], [], {}, 0, "2 baseline function rows exact"),
    ("overloads with one grown", [(HEADER, "solve", 40), (HEADER, "solve", 31)],
     [(HEADER, "solve", 30), SOLVE], [], {}, 4, "has grown to 31 lines from the 30"),
    ("a function in both registers", [SOLVE], [SOLVE], [SOLVE], {}, 5, "in both EXCEPTIONS.md"),
    ("no canary in the output", [SOLVE], [SOLVE], [], {"canary": False}, 6, "canary function"),
    ("a clang-diagnostic-error line", [SOLVE], [SOLVE], [], {"extra": ERROR}, 6, "could not parse"),
    ("a warning with no line count", [SOLVE], [SOLVE], [], {"extra": ORPHAN}, 6, "carries no line count"),
    ("a finding outside the source root", [SOLVE], [SOLVE], [], {"extra": diagnostic("/elsewhere/x.h", 3, "f", 30)},
     6, "outside the checked scope"),
]


def check(label: str, code: int, output: str, want_code: int, want_text: str) -> bool:
    if code == want_code and want_text in output:
        print(f"PASS {label}")
        return True
    print(f"FAIL {label}: exit {code} (wanted {want_code})\n{output.strip()}")
    return False


def case_dir(root: Path, label: str) -> Path:
    case = root / "".join(c if c.isalnum() else "_" for c in label)
    case.mkdir()
    return case


def run_table(root: Path) -> int:
    failures = 0
    for label, findings, rows, registered, options, want_code, want_text in CASES:
        case = case_dir(root, label)
        build(case, findings, rows, registered, **options)
        failures += not check(label, *run_gate(case), want_code, want_text)
    return failures


def run_growth(root: Path) -> int:
    kernel = (OTHER, "kernel", 30)
    case = case_dir(root, "baseline growth")
    build(case, [SOLVE, kernel], [SOLVE], [])
    (case / BASELINE).write_text(baseline_text([SOLVE, kernel]))
    failures = not check("a new baseline row against the ref", *run_gate(case, "--baseline-ref", "HEAD"),
                         8, "gained a row")
    (case / BASELINE).write_text(baseline_text([(HEADER, "solve", 41)]))
    (case / "tidy.txt").write_text(tidy_output(case.resolve(), [(HEADER, "solve", 41)], True, ""))
    return failures + (not check("a raised baseline row against the ref",
                                 *run_gate(case, "--baseline-ref", "HEAD"), 8, "rose from 40 to 41"))


def run_printer(root: Path) -> int:
    case = case_dir(root, "printer")
    findings = [(OTHER, "kernel", 30), (HEADER, "solve", 40), (HEADER, "solve", 26), (HEADER, "step", 90)]
    build(case, findings, [], [(HEADER, "step", 90)])
    code, output = run_gate(case, "--print-baseline")
    want = baseline_text(findings[:3]).split("\n", 1)[1]
    return not check("the printer sorts rows and leaves registered functions out", code, output, 0, want)


def main() -> int:
    with tempfile.TemporaryDirectory() as workspace:
        root = Path(workspace)
        failures = run_table(root) + run_growth(root) + run_printer(root)
    total = len(CASES) + 3
    if failures:
        print(f"{failures} of {total} cases failed")
        return 1
    print(f"{total} cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
