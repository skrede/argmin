#!/usr/bin/env python3
"""Check every function in the tracked lib/ headers against the function-size ceiling, its register and the
debt baseline, as clang-tidy's readability-function-size measures them."""

import argparse
import os
import sys
from pathlib import Path

from check_file_size import PRECEDENCE, against_ref, load_registry, report
from size_baseline import BASELINE_NAME, FUNCTION_KIND, Baseline, pair, parse_baseline, ratchet
from size_rules import (EXIT_BASELINE_GREW, EXIT_GROWTH, EXIT_INFLATED, EXIT_NO_INPUT, EXIT_REGISTRY,
                        EXIT_STALE, EXIT_UNLISTED, REGISTRY_NAME, Register, allowance)
from size_tidy import THRESHOLD, Finding, measure

EXIT_HELP = f"""exit codes:
  0  every function over {THRESHOLD} body lines is registered or recorded, and every row still describes it
  1  a function is over the ceiling and in neither register -- decompose it, or register it with a reason
  2  the arguments are wrong (argparse)
  3  a row is stale -- its function is at or under the ceiling or no longer exists; delete the row
  4  a function has grown past its {REGISTRY_NAME} allowance or past its exact baseline count
  5  the register or the baseline is missing or malformed, or a function sits in both
  6  clang-tidy failed, could not parse the headers, or did not report the generated canary function
  7  a row records more lines than its function has -- re-record it down
  8  the baseline gained a row or a count since the comparison ref; it may only shrink
"""


def keyed(findings: list[Finding]) -> dict[tuple[str, str], list[int]]:
    grouped = {}
    for path, name, lines in findings:
        grouped.setdefault((path, name), []).append(lines)
    return grouped


def exception_findings(measured: dict, recorded: dict) -> dict[int, list[str]]:
    found = {EXIT_UNLISTED: [], EXIT_STALE: [], EXIT_GROWTH: [], EXIT_INFLATED: []}
    for key in sorted(recorded):
        pairs, unlisted, stale = pair(measured.get(key, []), recorded[key])
        unit = f"{key[0]} {key[1]}"
        found[EXIT_UNLISTED] += [f"{unit} is {n} lines, an overload beyond its {REGISTRY_NAME} rows; "
                                 "decompose it, or register it with a reason" for n in unlisted]
        found[EXIT_STALE] += [f"the {REGISTRY_NAME} row for {unit} recording {n} lines is stale: the "
                              "function is at or under the ceiling or gone" for n in stale]
        found[EXIT_GROWTH] += [f"{unit} has grown to {m} lines from the {r} recorded in {REGISTRY_NAME}, "
                               f"past its allowance of {allowance(r)}" for m, r in pairs if m > allowance(r)]
        found[EXIT_INFLATED] += [f"the {REGISTRY_NAME} row for {unit} records {r} lines but the function "
                                 f"is {m}; re-record it down" for m, r in pairs if r > m]
    return found


def verdicts(findings: list[Finding], register: Register, baseline: Baseline) -> dict[int, list[str]]:
    measured = keyed(findings)
    debt = {(FUNCTION_KIND, *key): n for key, n in measured.items() if key not in register.functions}
    result = ratchet(debt, baseline.of_kind(FUNCTION_KIND))
    for code, messages in exception_findings(measured, register.functions).items():
        result[code] += messages
    result[EXIT_BASELINE_GREW] = []
    return result


def load_baseline(path: Path, register: Register) -> tuple[Baseline, int]:
    if not path.is_file():
        print(f"error: the baseline {path} does not exist", file=sys.stderr)
        return Baseline(), EXIT_REGISTRY
    baseline = parse_baseline(path.read_text(encoding="utf-8"))
    baseline.faults += [f"{p} {n} is in both {REGISTRY_NAME} and the baseline; it is either a permanent "
                        "exception or debt, not both" for p, n in sorted(register.functions)
                        if (FUNCTION_KIND, p, n) in baseline.rows]
    report(baseline.faults)
    return baseline, EXIT_REGISTRY if baseline.faults else 0


def evaluate(args: argparse.Namespace, findings: list[Finding], source: str, register: Register) -> int:
    path = args.baseline or args.source_root / BASELINE_NAME
    baseline, failure = load_baseline(path, register)
    if failure:
        return failure
    result = verdicts(findings, register, baseline)
    if args.baseline_ref is not None:
        code, messages = against_ref(args.source_root, args.baseline_ref, path, baseline)
        if code in (EXIT_REGISTRY, EXIT_NO_INPUT):
            report(messages)
            return code
        result[EXIT_BASELINE_GREW] = messages
    for code in PRECEDENCE:
        report(result[code])
    status = next((code for code in PRECEDENCE if result[code]), 0)
    if status == 0:
        rows = sum(len(counts) for counts in baseline.of_kind(FUNCTION_KIND).values())
        print(f"checked {source}: {len(findings)} functions over {THRESHOLD} lines; "
              f"{len(register.functions)} registered exceptions within allowance, "
              f"{rows} baseline function rows exact")
    return status


def run(args: argparse.Namespace) -> int:
    findings, source, failure = measure(args)
    if failure:
        return failure
    register, failure = load_registry(args.registry or args.source_root / REGISTRY_NAME, {})
    if failure:
        return failure
    if args.print_baseline:
        for path, name, lines in findings:
            if (path, name) not in register.functions:
                print(f"{FUNCTION_KIND}\t{path}\t{name}\t{lines}")
        return 0
    return evaluate(args, findings, source, register)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, epilog=EXIT_HELP,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add = parser.add_argument
    add("--source-root", type=Path, default=Path("."), help="repository root (default: the current directory)")
    add("--clang-tidy", default=os.environ.get("CLANG_TIDY") or "clang-tidy-18",
        help="clang-tidy binary (default: $CLANG_TIDY, else clang-tidy-18)")
    add("--eigen-include", type=Path, default=Path("/usr/include/eigen3"), help="Eigen include directory")
    add("--extra-arg", action="append", default=[], help="extra compiler argument (repeatable)")
    add("--registry", type=Path, default=None, help=f"register (default: {REGISTRY_NAME} under the root)")
    add("--baseline", type=Path, default=None, help=f"debt baseline (default: {BASELINE_NAME} under the root)")
    add("--baseline-ref", default=None, help="git ref whose baseline the working copy may not exceed")
    add("--print-baseline", action="store_true",
        help="print the over-limit functions outside the register as baseline rows, write nothing, exit 0")
    add("--tidy-output", type=Path, default=None, help="read a saved clang-tidy output instead of running it")
    args = parser.parse_args(argv)
    if args.baseline_ref is not None and not args.baseline_ref.strip():
        parser.error("--baseline-ref needs a git ref")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args(sys.argv[1:])))
