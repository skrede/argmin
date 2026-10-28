#!/usr/bin/env python3
"""Run clang-tidy's readability-function-size over every tracked lib/ header and read back its findings."""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from check_file_size import report, tracked_sources
from size_rules import EXIT_NO_INPUT, in_scope

THRESHOLD = 25
INCLUDE_ROOT = "lib/argmin/include"
HEADER_FILTER = ".*/lib/argmin/include/.*"
CANARY = "argmin_size_gate_canary"
CANARY_LINES = 30
CONFIG = ("{Checks: '-*,readability-function-size', CheckOptions: {"
          f"readability-function-size.LineThreshold: {THRESHOLD}, "
          "readability-function-size.StatementThreshold: -1, readability-function-size.BranchThreshold: -1, "
          "readability-function-size.NestingThreshold: -1}}")
WARNING = re.compile(r"^(/.+?):(\d+):\d+: warning: function '(.*)' exceeds recommended size/complexity "
                     r"thresholds \[readability-function-size\]$")
NOTE = re.compile(r"^(/.+?):(\d+):\d+: note: (\d+) lines including whitespace and comments \(threshold \d+\)$")


Finding = tuple[str, str, int]


def headers(root: Path) -> list[str]:
    return [name for name in tracked_sources(root) if name.startswith(f"{INCLUDE_ROOT}/") and name.endswith(".h")]


def translation_unit(names: list[str]) -> str:
    includes = "".join(f'#include "{name[len(INCLUDE_ROOT) + 1:]}"\n' for name in names)
    body = "    static_cast<void>(0);\n" * (CANARY_LINES - 1)
    return f"{includes}\nvoid {CANARY}()\n{{\n{body}}}\n"


def command(args: argparse.Namespace, tu: Path) -> list[str]:
    include = (args.source_root / INCLUDE_ROOT).resolve()
    return [args.clang_tidy, str(tu), "--quiet", f"--config={CONFIG}", f"--header-filter={HEADER_FILTER}",
            "--", "-std=c++20", "-I", str(include), "-isystem", str(args.eigen_include), *args.extra_arg]


def run_tidy(args: argparse.Namespace, names: list[str]) -> tuple[str, int]:
    with tempfile.TemporaryDirectory(prefix="argmin-function-size.") as work:
        tu = Path(work) / "all_headers.cpp"
        tu.write_text(translation_unit(names), encoding="utf-8")
        try:
            done = subprocess.run(command(args, tu), capture_output=True, text=True)
        except OSError as exc:
            print(f"error: cannot run {args.clang_tidy}: {exc}", file=sys.stderr)
            return "", EXIT_NO_INPUT
    if done.returncode != 0:
        print(done.stdout + done.stderr, file=sys.stderr)
        print(f"error: {args.clang_tidy} exited {done.returncode}", file=sys.stderr)
        return "", EXIT_NO_INPUT
    return done.stdout + done.stderr, 0


def parse(text: str) -> tuple[list[tuple[str, str, int]], list[str]]:
    raw, faults, pending = [], [], None
    for line in text.splitlines():
        if "clang-diagnostic-error" in line:
            faults.append(f"clang-tidy could not parse the headers: {line}")
        warning, note = WARNING.match(line), NOTE.match(line)
        if warning:
            faults += [f"the warning for {pending[2]} carries no line count"] if pending else []
            pending = warning.groups()
        elif note and pending and note.group(1, 2) == pending[:2]:
            raw.append((pending[0], pending[2], int(note.group(3))))
            pending = None
    faults += [f"the warning for {pending[2]} carries no line count"] if pending else []
    return raw, faults


def localize(raw: list[tuple[str, str, int]], root: Path) -> tuple[list[Finding], list[str]]:
    base, findings, faults, canary = root.resolve(), [], [], []
    for path, name, lines in raw:
        relpath = Path(os.path.relpath(Path(path).resolve(), base)).as_posix()
        if name == CANARY:
            canary.append(lines)
        elif relpath.startswith("../") or not in_scope(relpath):
            faults.append(f"{path} {name} lies outside the checked scope")
        else:
            findings.append((relpath, name, lines))
    if canary != [CANARY_LINES]:
        faults.append(f"clang-tidy did not report the generated {CANARY_LINES}-line canary function "
                      f"{CANARY} exactly once, so the run cannot be trusted to have measured anything")
    return sorted(findings), faults


def acquire(args: argparse.Namespace) -> tuple[str, str, int]:
    try:
        if args.tidy_output is not None:
            return args.tidy_output.read_text(encoding="utf-8"), str(args.tidy_output), 0
        names = headers(args.source_root)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"error: cannot read the input: {exc}", file=sys.stderr)
        return "", "", EXIT_NO_INPUT
    if not names:
        print(f"error: no tracked header was found under {args.source_root / INCLUDE_ROOT}", file=sys.stderr)
        return "", "", EXIT_NO_INPUT
    text, failure = run_tidy(args, names)
    return text, f"{len(names)} tracked lib/ headers", failure


def measure(args: argparse.Namespace) -> tuple[list[Finding], str, int]:
    text, source, failure = acquire(args)
    if failure:
        return [], source, failure
    raw, faults = parse(text)
    findings, more = localize(raw, args.source_root)
    report(faults + more)
    return findings, source, EXIT_NO_INPUT if faults or more else 0
