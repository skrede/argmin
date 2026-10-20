"""Loading one recorded run of the head-to-head QP benchmark.

Checks run in a fixed order: the manifest first, then the CSV hashes, then every row while it is
parsed, then completeness last, so a defect is refused for itself rather than for a consequence of
it (a small run carrying one bad row is refused for that row, not for being incomplete).
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from . import preregistration as prereg
from .manifest_checks import RefusedInput, check_manifest
from .rows import COHORT_CELLS, Tick, check_row, dimensions, parse_row, polish_flag

__all__ = ["RefusedInput", "Tick", "RunData", "expected_combinations", "load_run"]


@dataclass(frozen=True)
class RunData:
    run_id: str
    manifest: dict
    ticks: tuple[Tick, ...]
    inputs: tuple[dict, ...]
    refused_families: tuple[str, ...]
    recorded: bool


def expected_combinations(refused_families=()) -> list[tuple[str, str, str, str, str, str]]:
    combinations = []
    for cohort in prereg.COHORTS:
        for cell, (nx, nu, N) in COHORT_CELLS[cohort].items():
            dense_only = cell == prereg.DENSE_ONLY_PROBLEM
            for variant in ("nominal",) if dense_only else prereg.VARIANTS:
                if variant == "tight" and prereg.family_id(nx, nu) in refused_families:
                    continue
                for regime in ("fixed",) if dense_only else prereg.REGIMES:
                    for tier in prereg.TIERS:
                        combinations.extend(
                            (cohort, cell, variant, regime, tier, arm.name)
                            for arm in prereg.cohort_arms(cohort)
                            if tier in arm.tiers and prereg.expressible(arm.name, N))
    return combinations


def _combination_table(refused_families: tuple[str, ...]) -> dict:
    duals = {(arm.cohort, arm.name): arm.dual_kind for arm in prereg.ARMS}
    return {key: (dimensions(key[0], key[1], key[2]), polish_flag(key[5], key[4]), duals[(key[0], key[5])])
            for key in expected_combinations(refused_families)}


def _read_manifest(run_dir: Path) -> object:
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise RefusedInput(f"{run_dir.name}: no manifest.json")

    def no_constants(name: str) -> None:
        raise RefusedInput(f"manifest.json: non-finite constant {name}")

    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=no_constants)
    except (ValueError, UnicodeDecodeError) as error:
        raise RefusedInput(f"manifest.json: not valid JSON ({error})") from error


def _verified_files(run_dir: Path, files: dict) -> list[tuple[str, Path, str, int]]:
    verified = []
    for name in sorted(files):
        match = prereg.CSV_FILE_PATTERN.match(name)
        if match is None or int(match.group(2)) >= prereg.REPETITIONS:
            raise RefusedInput(f"manifest.json: csv_files: {name!r} is not <cohort>_rep<r>.csv[.gz]")
        path = run_dir / name
        if not path.is_file():
            raise RefusedInput(f"{name}: listed in the manifest but missing")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != files[name]:
            raise RefusedInput(f"{name}: SHA-256 {digest} differs from the manifest's {files[name]}")
        verified.append((name, path, match.group(1), int(match.group(2))))
    return verified


def _check_unique(tick: Tick, where: str, seen: set) -> None:
    key = (tick.cohort, tick.arm, tick.cell, tick.variant, tick.regime, tick.tier, tick.rep, tick.step)
    if key in seen:
        raise RefusedInput(f"{where}: duplicate row for {' '.join(map(str, key[1:]))}")
    seen.add(key)


def _read_ticks(name: str, path: Path, context: tuple, seen: set) -> list[Tick]:
    opener = gzip.open if name.endswith(".gz") else open
    ticks = []
    try:
        with opener(path, "rt", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            if header is None or tuple(header) != prereg.CSV_COLUMNS:
                raise RefusedInput(f"{name}: header differs from the pre-registered columns")
            for row in reader:
                where = f"{name}: line {reader.line_num}"
                tick = parse_row(row, where)
                check_row(tick, where, *context)
                _check_unique(tick, where, seen)
                ticks.append(tick)
    except (OSError, EOFError, UnicodeDecodeError, csv.Error) as error:
        raise RefusedInput(f"{name}: unreadable ({error})") from error
    if not ticks:
        raise RefusedInput(f"{name}: no data rows")
    return ticks


def _check_complete(ticks: list[Tick], combinations: dict) -> None:
    counts = dict.fromkeys(combinations, 0)
    for tick in ticks:
        counts[(tick.cohort, tick.cell, tick.variant, tick.regime, tick.tier, tick.arm)] += 1
    rows = prereg.REPETITIONS * prereg.STEPS_PER_SEQUENCE
    for key, count in counts.items():
        if 0 < count != rows:
            raise RefusedInput(f"{' '.join(key)}: {count} rows, expected {rows}")
    for key, count in counts.items():
        if count == 0:
            raise RefusedInput(f"{' '.join(key)}: missing pre-registered combination")


def load_run(run_dir: Path, *, advisory: bool = False, schema_only: bool = False,
             expect_preregistration_commit: str | None = None) -> RunData:
    run_dir = Path(run_dir)
    manifest = _read_manifest(run_dir)
    refused = check_manifest(manifest, advisory=advisory, schema_only=schema_only,
                             expect_preregistration_commit=expect_preregistration_commit)
    files = _verified_files(run_dir, manifest["csv_files"])
    combinations = _combination_table(refused)
    ticks, inputs, seen = [], [], set()
    for name, path, cohort, rep in files:
        file_ticks = _read_ticks(name, path, (combinations, manifest["run_id"], cohort, rep), seen)
        ticks.extend(file_ticks)
        inputs.append({"name": name, "sha256": manifest["csv_files"][name], "rows": len(file_ticks)})
    if not schema_only:
        _check_complete(ticks, combinations)
    return RunData(run_id=manifest["run_id"], manifest=manifest, ticks=tuple(ticks), inputs=tuple(inputs),
                   refused_families=refused, recorded=not advisory)
