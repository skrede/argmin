#!/usr/bin/env python3
"""Fixture repositories exercising every register verdict the file-size gate can reach."""

import subprocess
import sys
import tempfile
from pathlib import Path

GATE = Path(__file__).resolve().parent / "check_file_size.py"
WHY = "one cohesive unit whose shared state splitting would scatter across files"
FILES = "# register\n\n## Files\n\n| Path | Lines at registration | Why |\n| --- | --- | --- |\n"
FUNCTIONS = "\n## Functions\n\n| Path | Function | Lines at registration | Why |\n| --- | --- | --- | --- |\n"
BASELINE = "scripts/size_baseline.tsv"


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=fixture", "-c", "user.email=f@x",
                           *args], check=True, capture_output=True, text=True).stdout


def write_tree(root: Path, sources: dict[str, int], rows: list, extra: str = "",
               baseline: list[str] | None = None) -> None:
    for name, lines in sources.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// x\n" * lines)
    table = "".join(f"| {p} | {n} | {why} |\n" for p, n, why in rows)
    (root / "EXCEPTIONS.md").write_text(FILES + table + extra + FUNCTIONS)
    if baseline is not None:
        (root / BASELINE).parent.mkdir(parents=True, exist_ok=True)
        (root / BASELINE).write_text("# baseline\n" + "".join(f"{row}\n" for row in baseline))


def build(root: Path, sources: dict[str, int], rows: list, extra: str = "",
          baseline: list[str] | None = None) -> None:
    git(root, "init", "-q")
    write_tree(root, sources, rows, extra, [] if baseline is None else baseline)
    git(root, "add", "-A")


def run_gate(root: Path, *args: str) -> tuple[int, str]:
    done = subprocess.run([sys.executable, str(GATE), "--source-root", str(root), *args],
                          capture_output=True, text=True)
    return done.returncode, done.stdout + done.stderr


BIG = {"lib/big.h": 205}
CASES = [
    ("every overage registered", BIG, [("lib/big.h", 205, WHY)], "", 0, "1 registered"),
    ("unregistered overage", BIG, [], "", 1, "in neither EXCEPTIONS.md nor the baseline"),
    ("a .cpp overage is seen too", {"lib/big.cpp": 205}, [], "", 1, "lib/big.cpp"),
    ("files outside lib are out of scope",
     {"tests/big.cpp": 900, "benchmarks/big.h": 900, "lib/small.h": 10}, [], "", 0, "checked 1 "),
    ("boundary: the ceiling itself passes", {"lib/edge.h": 200}, [], "", 0, "0 registered"),
    ("boundary: one line over fails", {"lib/edge.h": 201}, [], "", 1, "is 201 lines"),
    ("stale row, file under the ceiling",
     {"lib/small.h": 10, "lib/big.h": 205}, [("lib/small.h", 205, WHY), ("lib/big.h", 205, WHY)],
     "", 3, "at or under the"),
    ("stale row, no such file", BIG, [("lib/big.h", 205, WHY), ("lib/gone.h", 205, WHY)], "",
     3, "lib/gone.h is stale"),
    ("grown past the allowance", {"lib/big.h": 227}, [("lib/big.h", 205, WHY)], "", 4, "has grown to"),
    ("at exactly the allowance passes", {"lib/big.h": 226}, [("lib/big.h", 205, WHY)], "", 0, "1 reg"),
    ("recorded count above the file's length", BIG, [("lib/big.h", 300, WHY)], "",
     7, "records 300 lines but the file is 205"),
    ("a row with no count column", BIG, [], "| lib/big.h | a reason |\n", 5,
     "needs a path, a recorded line count and a reason"),
    ("a row with an empty reason", BIG, [("lib/big.h", 205, "")], "", 5, "reason is empty"),
    ("a reason restating the count", BIG, [("lib/big.h", 205, "it is 205 lines")],
     "", 5, "restates the unit's line count"),
    ("a reason naming no split harm", BIG, [("lib/big.h", 205, "a large file")],
     "", 5, "must name what splitting"),
    ("a reason promising a future split", BIG,
     [("lib/big.h", 205, "cohesive today; splitting is planned")], "", 5, "promises future work"),
    ("a path outside the scope", BIG,
     [("lib/big.h", 205, WHY), ("tests/big.cpp", 205, WHY)], "", 5, "outside the checked scope"),
    ("a path registered twice", BIG, [("lib/big.h", 205, WHY), ("lib/big.h", 205, WHY)], "", 5,
     "registered twice"),
    ("a row under an unknown heading", BIG, [("lib/big.h", 205, WHY)],
     f"\n## Notes\n\n| lib/other.h | 205 | {WHY} |\n", 5, "neither 'Files' nor 'Functions'"),
    ("a path in backticks", BIG, [("`lib/big.h`", 205, WHY)], "", 0, "1 registered"),
    ("a path carrying a space", {"lib/my file.h": 205}, [("lib/my file.h", 205, WHY)], "", 0, "1 reg"),
    ("no lib C++ file anywhere", {"README.md": 5, "src/a.h": 300}, [], "", 6, "no tracked lib/ C++"),
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
    for label, sources, rows, extra, want_code, want_text in CASES:
        case = case_dir(root, label)
        build(case, sources, rows, extra)
        failures += not check(label, *run_gate(case), want_code, want_text)
    return failures


def function_rows(root: Path) -> int:
    valid = f"| lib/big.h | solve | 40 | {WHY} |\n"
    case = case_dir(root, "function rows")
    build(case, {"lib/small.h": 10}, [])
    (case / "EXCEPTIONS.md").write_text(FILES + FUNCTIONS + valid)
    failures = not check("a function row under its heading parses", *run_gate(case), 0, "checked 1")
    (case / "EXCEPTIONS.md").write_text(FILES + FUNCTIONS + "| lib/big.h | solve | a reason |\n")
    return failures + (not check("a function row with no count", *run_gate(case), 5,
                                 "needs a path, a function, a recorded line count and a reason"))


def run_tree_faults(root: Path) -> int:
    absent = case_dir(root, "absent")
    build(absent, {"lib/big.h": 205, "lib/kept.h": 10}, [("lib/big.h", 205, WHY)])
    (absent / "lib/big.h").unlink()
    failures = not check("a tracked file missing from the working tree", *run_gate(absent),
                         6, "tracked but absent from the working tree")
    bare = case_dir(root, "no register")
    build(bare, BIG, [])
    (bare / "EXCEPTIONS.md").unlink()
    return failures + (not check("the register itself missing", *run_gate(bare), 5, "does not exist"))


def run_counting(root: Path) -> int:
    tail = case_dir(root, "no trailing newline")
    build(tail, {"lib/big.h": 200}, [])
    (tail / "lib/big.h").write_text("// x\n" * 200 + "// last")
    return not check("a final line with no newline still counts", *run_gate(tail), 1, "is 201 lines")


def main() -> int:
    with tempfile.TemporaryDirectory() as workspace:
        root = Path(workspace)
        failures = run_table(root) + function_rows(root) + run_tree_faults(root) + run_counting(root)
    total = len(CASES) + 5
    if failures:
        print(f"{failures} of {total} cases failed")
        return 1
    print(f"{total} cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
