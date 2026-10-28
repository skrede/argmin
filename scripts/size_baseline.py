#!/usr/bin/env python3
"""The shrink-only debt baseline: its format, its exact ratchet, and its check against a git ref."""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from size_rules import (EXIT_BASELINE_GREW, EXIT_GROWTH, EXIT_INFLATED, EXIT_NO_INPUT,
                        EXIT_REGISTRY, EXIT_STALE, EXIT_UNLISTED, REGISTRY_NAME, in_scope)

BASELINE_NAME = "scripts/size_baseline.tsv"
FILE_KIND = "file"
FUNCTION_KIND = "function"
NO_NAME = "-"

Key = tuple[str, str, str]


@dataclass
class Baseline:
    rows: dict[Key, list[int]] = field(default_factory=dict)
    faults: list[str] = field(default_factory=list)

    def of_kind(self, kind: str) -> dict[Key, list[int]]:
        return {key: counts for key, counts in self.rows.items() if key[0] == kind}


def file_key(path: str) -> Key:
    return (FILE_KIND, path, NO_NAME)


def describe(key: Key) -> str:
    return key[1] if key[0] == FILE_KIND else f"{key[1]} {key[2]}"


def row_fault(fields: list[str]) -> str | None:
    if len(fields) != 4:
        return "a row needs exactly four tab-separated columns: kind, path, name and lines"
    kind, path, name, lines = fields
    if kind not in (FILE_KIND, FUNCTION_KIND):
        return f"the kind {kind!r} is neither {FILE_KIND!r} nor {FUNCTION_KIND!r}"
    if not in_scope(path):
        return f"the path {path!r} is outside the checked scope"
    if not name or (kind == FILE_KIND) != (name == NO_NAME):
        return f"a file row carries {NO_NAME!r} as its name, and a function row the function's name"
    if not lines.isdigit():
        return f"the line count {lines!r} is not a number"
    return None


def sort_key(fields: list[str]) -> tuple[str, str, str, int]:
    return (fields[0], fields[1], fields[2], int(fields[3]))


def placement_fault(fields: list[str], baseline: Baseline, previous: tuple | None) -> str | None:
    if previous is not None and sort_key(fields) < previous:
        return "the row is out of order; rows sort by kind, path, name, then line count"
    if fields[0] == FILE_KIND and file_key(fields[1]) in baseline.rows:
        return "the file has a second row"
    return None


def parse_baseline(text: str, origin: str = BASELINE_NAME) -> Baseline:
    baseline, previous = Baseline(), None
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        fault = row_fault(fields) or placement_fault(fields, baseline, previous)
        if fault is not None:
            baseline.faults.append(f"{origin} line {number}: {fault}")
            continue
        previous = sort_key(fields)
        baseline.rows.setdefault((fields[0], fields[1], fields[2]), []).append(int(fields[3]))
    return baseline


def pair(measured: list[int], recorded: list[int]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Pair two multisets largest to largest; overloads share a key, so no row carries a line number."""
    ours, theirs = sorted(measured, reverse=True), sorted(recorded, reverse=True)
    n = min(len(ours), len(theirs))
    return list(zip(ours[:n], theirs[:n])), ours[n:], theirs[n:]


def ratchet(over: dict[Key, list[int]], rows: dict[Key, list[int]]) -> dict[int, list[str]]:
    """Exact verdicts; `over` holds only measured counts past the ceiling and not otherwise registered."""
    findings = {EXIT_UNLISTED: [], EXIT_STALE: [], EXIT_GROWTH: [], EXIT_INFLATED: []}
    for key in sorted(set(over) | set(rows)):
        pairs, unlisted, stale = pair(over.get(key, []), rows.get(key, []))
        unit = describe(key)
        findings[EXIT_UNLISTED] += [
            f"{unit} is {n} lines, over its ceiling, and is in neither {REGISTRY_NAME} nor the "
            f"baseline; decompose it, or register it in {REGISTRY_NAME} with a reason" for n in unlisted]
        findings[EXIT_STALE] += [
            f"the baseline row for {unit} recording {n} lines is stale: the unit is at or under its "
            "ceiling or gone; delete the row" for n in stale]
        findings[EXIT_GROWTH] += [
            f"{unit} has grown to {measured} lines from the {recorded} recorded in the baseline, which "
            "only shrinks; decompose it" for measured, recorded in pairs if measured > recorded]
        findings[EXIT_INFLATED] += [
            f"the baseline row for {unit} records {recorded} lines but the unit is {measured}; "
            "re-record it down" for measured, recorded in pairs if recorded > measured]
    return findings


def grown_rows(previous: Baseline, current: Baseline, ref: str) -> list[str]:
    messages = []
    for key in sorted(current.rows):
        pairs, added, _ = pair(current.rows[key], previous.rows.get(key, []))
        unit = describe(key)
        messages += [f"the baseline gained a row for {unit} ({n} lines) absent at {ref}" for n in added]
        messages += [f"the baseline row for {unit} rose from {old} to {new} lines since {ref}"
                     for new, old in pairs if new > old]
    return messages


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def introduced_before(root: Path, ref: str, relpath: str) -> bool:
    added = git(root, "log", "--diff-filter=A", "--format=%H", "--", relpath).stdout.split()
    return any(git(root, "merge-base", "--is-ancestor", commit, ref).returncode == 0
               for commit in added)


def check_against_ref(root: Path, ref: str, relpath: str, current: Baseline) -> tuple[int, list[str]]:
    if git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").returncode != 0:
        return EXIT_NO_INPUT, [f"the comparison ref {ref!r} does not name a commit"]
    shown = git(root, "show", f"{ref}:{relpath}")
    if shown.returncode != 0:
        if introduced_before(root, ref, relpath):
            return EXIT_BASELINE_GREW, [f"{relpath} is absent at {ref}, after it was introduced; "
                                        "a removed baseline cannot be recreated"]
        print(f"baseline absent at {ref}: first recording")
        return 0, []
    previous = parse_baseline(shown.stdout, f"{relpath} at {ref}")
    if previous.faults:
        return EXIT_REGISTRY, previous.faults
    grown = grown_rows(previous, current, ref)
    return (EXIT_BASELINE_GREW if grown else 0), grown
