#!/usr/bin/env python3
"""Check every tracked lib/ C++ file against the file-size ceiling, its register and the debt baseline."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from size_baseline import (BASELINE_NAME, FILE_KIND, Baseline, check_against_ref, file_key,
                           parse_baseline, ratchet)
from size_rules import (CEILING, EXIT_BASELINE_GREW, EXIT_GROWTH, EXIT_INFLATED, EXIT_NO_INPUT,
                        EXIT_REGISTRY, EXIT_STALE, EXIT_UNLISTED, REGISTRY_NAME, Register, allowance,
                        in_scope, parse_registry)

PRECEDENCE = (EXIT_BASELINE_GREW, EXIT_UNLISTED, EXIT_STALE, EXIT_GROWTH, EXIT_INFLATED)

EXIT_HELP = f"""exit codes:
  0  every overage is registered or recorded, and every row still describes its file
  1  a file is over the ceiling and in neither register -- decompose it, or register it with a reason
  2  the arguments are wrong (argparse)
  3  a row is stale -- its file is under the ceiling or no longer exists; delete the row
  4  a file has grown past its {REGISTRY_NAME} allowance or past its exact baseline count
  5  the register or the baseline is missing or malformed, or a path sits in both
  6  the scan could not read what it was asked to read
  7  a row records more lines than its file has -- re-record it down
  8  the baseline gained a row or a count since the comparison ref; it may only shrink
"""


def tracked_sources(root: Path) -> list[str]:
    listing = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    names = listing.decode("utf-8", "surrogateescape").split("\0")
    return sorted(name for name in names if name and in_scope(name))


def line_count(path: Path) -> int:
    text = path.read_bytes().decode("utf-8", "surrogateescape")
    return text.count("\n") + (1 if text and not text.endswith("\n") else 0)


def measure(root: Path, names: list[str]) -> tuple[dict[str, int], list[str]]:
    counts, absent = {}, []
    for name in names:
        path = root / name
        if path.is_file():
            counts[name] = line_count(path)
        else:
            absent.append(name)
    return counts, absent


def report(messages) -> None:
    for message in messages:
        print(f"error: {message}", file=sys.stderr)


def scan(root: Path) -> tuple[dict[str, int], int]:
    try:
        names = tracked_sources(root)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"error: cannot list the tracked files under {root}: {exc}", file=sys.stderr)
        return {}, EXIT_NO_INPUT
    counts, absent = measure(root, names)
    report(f"{path} is tracked but absent from the working tree, so its length cannot be measured; "
           "restore it, or remove it from the index" for path in absent)
    if not counts:
        print(f"error: no tracked lib/ C++ file was found under {root}", file=sys.stderr)
    return counts, EXIT_NO_INPUT if absent or not counts else 0


def load_registry(path: Path, counts: dict[str, int]) -> tuple[Register, int]:
    if not path.is_file():
        print(f"error: the register {path} does not exist", file=sys.stderr)
        return Register(), EXIT_REGISTRY
    register = parse_registry(path.read_text(encoding="utf-8"), counts)
    report(register.faults)
    return register, EXIT_REGISTRY if register.faults else 0


def load_baseline(path: Path, register: Register) -> tuple[Baseline, int]:
    if not path.is_file():
        print(f"error: the baseline {path} does not exist", file=sys.stderr)
        return Baseline(), EXIT_REGISTRY
    baseline = parse_baseline(path.read_text(encoding="utf-8"))
    baseline.faults += [f"{p} is in both {REGISTRY_NAME} and the baseline; it is either a permanent "
                        "exception or debt, not both" for p in sorted(register.files)
                        if file_key(p) in baseline.rows]
    report(baseline.faults)
    return baseline, EXIT_REGISTRY if baseline.faults else 0


def exception_findings(counts: dict[str, int], recorded: dict[str, int]) -> dict[int, list[str]]:
    live = {p: counts[p] for p in recorded if counts.get(p, 0) > CEILING}
    return {
        EXIT_STALE: [f"the {REGISTRY_NAME} row for {p} is stale: the file is at or under the "
                     f"{CEILING}-line ceiling or gone" for p in sorted(recorded) if p not in live],
        EXIT_GROWTH: [f"{p} has grown to {n} lines from the {recorded[p]} recorded in {REGISTRY_NAME}, "
                      f"past its allowance of {allowance(recorded[p])}"
                      for p, n in sorted(live.items()) if n > allowance(recorded[p])],
        EXIT_INFLATED: [f"the {REGISTRY_NAME} row for {p} records {recorded[p]} lines but the file is "
                        f"{n}; re-record it down" for p, n in sorted(live.items()) if recorded[p] > n],
    }


def verdicts(counts: dict[str, int], register: Register, baseline: Baseline) -> dict[int, list[str]]:
    debt = {file_key(p): [n] for p, n in counts.items() if n > CEILING and p not in register.files}
    findings = ratchet(debt, baseline.of_kind(FILE_KIND))
    for code, messages in exception_findings(counts, register.files).items():
        findings[code] += messages
    findings[EXIT_BASELINE_GREW] = []
    return findings


def against_ref(root: Path, ref: str, path: Path, baseline: Baseline) -> tuple[int, list[str]]:
    relpath = Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()
    if relpath.startswith("../"):
        return EXIT_NO_INPUT, [f"the baseline {path} lies outside the source root {root}"]
    return check_against_ref(root, ref, relpath, baseline)


def evaluate(args: argparse.Namespace, counts: dict[str, int], register: Register) -> int:
    path = args.baseline or args.source_root / BASELINE_NAME
    baseline, failure = load_baseline(path, register)
    if failure:
        return failure
    findings = verdicts(counts, register, baseline)
    if args.baseline_ref is not None:
        code, messages = against_ref(args.source_root, args.baseline_ref, path, baseline)
        if code in (EXIT_REGISTRY, EXIT_NO_INPUT):
            report(messages)
            return code
        findings[EXIT_BASELINE_GREW] = messages
    for code in PRECEDENCE:
        report(findings[code])
    status = next((code for code in PRECEDENCE if findings[code]), 0)
    if status == 0:
        print(f"checked {len(counts)} tracked lib/ C++ files; {len(register.files)} registered "
              f"exceptions within allowance, {len(baseline.of_kind(FILE_KIND))} baseline file rows exact")
    return status


def run(args: argparse.Namespace) -> int:
    counts, failure = scan(args.source_root)
    if failure:
        return failure
    register, failure = load_registry(args.registry or args.source_root / REGISTRY_NAME, counts)
    if failure:
        return failure
    if args.print_baseline:
        for path, n in sorted(counts.items()):
            if n > CEILING and path not in register.files:
                print(f"file\t{path}\t-\t{n}")
        return 0
    return evaluate(args, counts, register)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, epilog=EXIT_HELP,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=Path("."),
                        help="repository root to check (default: the current directory)")
    parser.add_argument("--registry", type=Path, default=None,
                        help=f"register to read (default: {REGISTRY_NAME} under the source root)")
    parser.add_argument("--baseline", type=Path, default=None,
                        help=f"debt baseline to read (default: {BASELINE_NAME} under the source root)")
    parser.add_argument("--baseline-ref", default=None,
                        help="git ref whose baseline the working copy may not exceed")
    parser.add_argument("--print-baseline", action="store_true",
                        help="print the over-limit files outside the register as baseline rows, "
                             "write nothing, exit 0")
    args = parser.parse_args(argv)
    if args.baseline_ref is not None and not args.baseline_ref.strip():
        parser.error("--baseline-ref needs a git ref")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args(sys.argv[1:])))
