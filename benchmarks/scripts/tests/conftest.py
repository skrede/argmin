import csv
import gzip
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import types

import pytest

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "benchmarks" / "scripts"))

from qp_head_to_head import preregistration as prereg  # noqa: E402
from qp_head_to_head.records import expected_combinations, load_run  # noqa: E402
from qp_head_to_head.rows import dimensions, polish_flag  # noqa: E402

RUN_ID = "20261004T120000Z-0123456789"
DEFAULT_TICK_NS = {
    "argmin_dense_admm": 1500, "argmin_dense_active_set": 1400, "qpoases": 1100, "daqp": 900,
    "proxqp_dense": 2000, "osqp": 1000, "argmin_dense_admm_polished": 1600, "argmin_sparse_admm": 1200,
    "proxqp_sparse": 2500, "hpipm": 600, "tinympc": 500, "argmin_sparse_admm_polished": 1300,
    "osqp_polished": 1100,
}


def default_tick_ns(arm, cell, variant, regime, tier, rep, step):
    return DEFAULT_TICK_NS[arm]


def default_kkt(arm, cell, variant, regime, tier, rep, step):
    return 1e-9


def _binding_gate(refused_families):
    gate = {}
    for nx, nu in prereg.families():
        family = prereg.family_id(nx, nu)
        figures = (0.05, 0.2) if family in refused_families else (0.3, 0.8)
        cells = {prereg.cell_id(*cell): {"median_binding_fraction": figures[0],
                                         "max_state_row_step_fraction": figures[1]}
                 for cell in prereg.mpc_cells() if cell[:2] == (nx, nu)}
        gate[family] = {"cells": cells, "family_passed": family not in refused_families}
    return gate


def base_manifest(refused_families=(), clock_state="fixed"):
    manifest = prereg.manifest_template()
    manifest.update({
        "run_id": RUN_ID, "started_utc": "2026-10-04T12:00:00Z", "finished_utc": "2026-10-04T13:00:00Z",
        "preregistration_commit": "a" * 40, "ctrlpp_harness_sha": "b" * 40, "ctrlpp_harness_dirty": False,
        "host": {"cpu": "synthetic", "pinned_cpu": 3}, "clock_state": clock_state,
        "update_api": {arm.name: "separate" for arm in prereg.ARMS},
        "binding_gate": _binding_gate(refused_families),
        "probes": {arm.name: {"passed": True, "max_dual_error": 1e-12}
                   for arm in prereg.ARMS if arm.role != "report_only" and arm.dual_kind == "native"},
        "terminal_weights": {prereg.cell_id(*cell): "c" * 64 for cell in prereg.LONG_HORIZON_CELLS},
        "csv_files": {},
    })
    manifest["build"]["compiler_version"] = "14.2.1"
    return manifest


def _rows(cohort, rep, combinations, steps, tick_ns, kkt):
    duals = {(arm.cohort, arm.name): arm.dual_kind for arm in prereg.ARMS}
    rows = []
    for key_cohort, cell, variant, regime, tier, arm in combinations:
        if key_cohort != cohort:
            continue
        fixed = [*dimensions(cohort, cell, variant), regime, tier, arm, polish_flag(arm, tier), rep,
                 rep if regime == "churning" else -1]
        for step in steps:
            ticks = tick_ns(arm, cell, variant, regime, tier, rep, step)
            residual = kkt(arm, cell, variant, regime, tier, rep, step)
            passed = residual <= prereg.TIERS[tier]
            rows.append([RUN_ID, cohort, cell, variant, *fixed, step, 50000 if step == 0 else -1,
                         ticks // 4, ticks - ticks // 4, ticks, 100 if cohort == "dense" else -1,
                         -1, -1, -1, -1, 7, residual, residual, residual, residual, "solved", int(passed),
                         "none" if passed else "inaccurate", 0, duals[(cohort, arm)], 0, 0.25,
                         0.1 if regime == "churning" and step > 0 else -1.0])
    return rows


def write_csv(path, rows, header=prereg.CSV_COLUMNS):
    with gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=1) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_run(tmp_path, *, tick_ns=default_tick_ns, kkt=default_kkt, refused_families=(), clock_state="fixed",
             mutate=None, cells=None, smoke=False):
    run_dir = pathlib.Path(tmp_path) / "run"
    run_dir.mkdir(parents=True)
    combinations = [key for key in expected_combinations(refused_families) if cells is None or key[1] in cells]
    reps = range(1) if smoke else range(prereg.REPETITIONS)
    steps = range(2) if smoke else range(prereg.STEPS_PER_SEQUENCE)
    rows = {f"{cohort}_rep{rep}.csv.gz": _rows(cohort, rep, combinations, steps, tick_ns, kkt)
            for cohort in prereg.COHORTS for rep in reps}
    manifest = base_manifest(refused_families, clock_state)
    if mutate is not None:
        mutate(rows, manifest)
    for name, file_rows in rows.items():
        manifest["csv_files"][name] = write_csv(run_dir / name, file_rows)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return run_dir


def invoke(*arguments):
    command = [sys.executable, "benchmarks/scripts/qp_head_to_head_verdicts.py", *map(str, arguments)]
    environment = {**os.environ, "PYTHONHASHSEED": "random"}
    return subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True, text=True, env=environment)


def verdicts_for(run_dir, output_dir, *extra):
    output_dir = pathlib.Path(output_dir)
    json_path, md_path = output_dir / "verdicts.json", output_dir / "verdicts.md"
    process = invoke("--run-dir", run_dir, "--output-json", json_path, "--output-md", md_path, *extra)
    return types.SimpleNamespace(process=process, json_path=json_path, md_path=md_path)


@pytest.fixture(scope="session")
def build_run():
    return make_run


@pytest.fixture(scope="session")
def run_command():
    return types.SimpleNamespace(invoke=invoke, verdicts_for=verdicts_for, base_manifest=base_manifest,
                                 write_csv=write_csv, run_id=RUN_ID,
                                 default_tick_ns=default_tick_ns, default_kkt=default_kkt)


@pytest.fixture(scope="session")
def full_run(tmp_path_factory):
    return make_run(tmp_path_factory.mktemp("full"))


@pytest.fixture(scope="session")
def full_run_outputs(full_run, tmp_path_factory):
    return verdicts_for(full_run, tmp_path_factory.mktemp("full_outputs"))


@pytest.fixture(scope="session")
def full_run_data(full_run):
    return load_run(full_run)
