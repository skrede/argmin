import json
import shutil

import pytest

from qp_head_to_head import preregistration as prereg
from qp_head_to_head.records import RefusedInput, expected_combinations, load_run
from qp_head_to_head.rows import dimensions

COLUMN = {name: index for index, name in enumerate(prereg.CSV_COLUMNS)}
SMALL = {"nx2_nu1_N10"}


def small_run(build_run, tmp_path, mutate=None, **options):
    return build_run(tmp_path, smoke=True, cells=SMALL, mutate=mutate, **options)


def refused(run_dir, match, **options):
    with pytest.raises(RefusedInput, match=match):
        load_run(run_dir, **options)


def row_where(rows, cohort="sparse", **values):
    for row in rows[f"{cohort}_rep0.csv.gz"]:
        if all(row[COLUMN[name]] == value for name, value in values.items()):
            return row
    raise AssertionError(f"no row with {values}")


def edit_row(cohort="sparse", where=None, **changes):
    def mutate(rows, manifest):
        row = row_where(rows, cohort, **(where or {}))
        for name, value in changes.items():
            row[COLUMN[name]] = value
    return mutate


def edit_manifest(change):
    def mutate(rows, manifest):
        change(manifest)
    return mutate


def test_expected_combinations_match_the_hand_computed_totals():
    combinations = expected_combinations(())
    assert len(combinations) == 1441
    assert sum(key[0] == "sparse" for key in combinations) == 648
    assert sum(key[0] == "dense" for key in combinations) == 793
    assert len(set(combinations)) == 1441


def test_refused_family_drops_its_tight_variant_from_both_cohorts():
    combinations = expected_combinations(("nx4_nu2",))
    assert len(combinations) == 1441 - 132 - 156
    assert not [key for key in combinations if key[1].startswith("nx4_nu2_") and key[2] == "tight"]


def test_dimensions_match_the_hand_computed_layouts():
    assert dimensions("sparse", "nx4_nu2_N50", "tight") == (4, 2, 50, 304, 404)
    assert dimensions("sparse", "nx4_nu2_N50", "nominal") == (4, 2, 50, 304, 304)
    assert dimensions("dense", "nx4_nu2_N50", "tight") == (4, 2, 50, 100, 200)
    assert dimensions("dense", "dense_mpc_shaped", "nominal") == (0, 0, 0, 10, 5)


def test_complete_run_is_accepted_as_recorded(full_run_data):
    assert len(full_run_data.ticks) == 1441 * 300
    assert full_run_data.recorded is True
    assert full_run_data.refused_families == ()
    assert [entry["rows"] for entry in full_run_data.inputs] == [793 * 60] * 5 + [648 * 60] * 5


def test_smoke_run_passes_every_schema_check(build_run, tmp_path):
    run = load_run(small_run(build_run, tmp_path), schema_only=True)
    assert len(run.ticks) == 138 * 2


def test_missing_manifest_is_refused(build_run, tmp_path):
    run_dir = small_run(build_run, tmp_path)
    (run_dir / "manifest.json").unlink()
    refused(run_dir, "no manifest.json", schema_only=True)


def test_header_only_csv_is_refused(build_run, tmp_path):
    def empty(rows, manifest):
        rows["dense_rep0.csv.gz"] = []
    refused(small_run(build_run, tmp_path, empty), "dense_rep0.csv.gz: no data rows", schema_only=True)


def test_csv_whose_bytes_differ_from_the_manifest_hash_is_refused(build_run, tmp_path):
    run_dir = small_run(build_run, tmp_path)
    path = run_dir / "sparse_rep0.csv.gz"
    path.write_bytes(path.read_bytes() + b"\0")
    refused(run_dir, "sparse_rep0.csv.gz: SHA-256", schema_only=True)


@pytest.mark.parametrize("header", [
    tuple(reversed(prereg.CSV_COLUMNS)),
    prereg.CSV_COLUMNS[:-1],
    (*prereg.CSV_COLUMNS, "extra"),
    tuple("passed" if name == "pass" else name for name in prereg.CSV_COLUMNS),
], ids=["reordered", "missing", "unknown", "renamed"])
def test_header_other_than_the_registered_columns_is_refused(build_run, run_command, tmp_path, header):
    run_dir = small_run(build_run, tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = [[0] * len(header)]
    manifest["csv_files"]["dense_rep0.csv.gz"] = run_command.write_csv(run_dir / "dense_rep0.csv.gz", rows, header)
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    refused(run_dir, "header differs", schema_only=True)


def test_mixed_run_ids_are_refused(build_run, tmp_path):
    mutate = edit_row(run_id="20261004T120000Z-ffffffffff")
    refused(small_run(build_run, tmp_path, mutate), "run_id .* differs from the manifest")


def test_malformed_run_id_is_refused(build_run, tmp_path):
    mutate = edit_manifest(lambda manifest: manifest.update(run_id="2026-10-04-run"))
    refused(small_run(build_run, tmp_path, mutate), "run_id: .* does not match")


def test_duplicate_tick_key_is_refused(build_run, tmp_path):
    def duplicate(rows, manifest):
        rows["sparse_rep0.csv.gz"].append(list(rows["sparse_rep0.csv.gz"][0]))
    refused(small_run(build_run, tmp_path, duplicate), "duplicate row")


def test_pass_flag_disagreeing_with_the_residual_is_refused(build_run, tmp_path):
    mutate = edit_row(where={"tier": "high"}, kkt_max=1e-5)
    refused(small_run(build_run, tmp_path, mutate), "pass 1 disagrees with kkt_max")


def test_residual_equal_to_the_threshold_must_pass(build_run, tmp_path):
    failing = edit_row(where={"tier": "mpc"}, kkt_max=1e-3, **{"pass": 0, "fail_class": "inaccurate"})
    refused(small_run(build_run, tmp_path / "a", failing), "pass 0 disagrees with kkt_max", schema_only=True)
    passing = edit_row(where={"tier": "mpc"}, kkt_max=1e-3)
    assert load_run(small_run(build_run, tmp_path / "b", passing), schema_only=True)


def test_nan_residual_is_a_fail(build_run, tmp_path):
    refused(small_run(build_run, tmp_path, edit_row(kkt_max=float("nan"))), "pass 1 disagrees with kkt_max nan")


def test_fail_class_inconsistent_with_the_pass_flag_is_refused(build_run, tmp_path):
    refused(small_run(build_run, tmp_path, edit_row(fail_class="max_iter")), "fail_class 'max_iter' disagrees")


def test_unknown_fail_class_is_refused(build_run, tmp_path):
    refused(small_run(build_run, tmp_path, edit_row(fail_class="timeout")), "fail_class 'timeout' is not one of")


def test_row_for_a_not_expressible_combination_is_refused(build_run, tmp_path):
    mutate = edit_row(where={"arm": "hpipm"}, arm="tinympc", dual_kind="recovered")
    refused(small_run(build_run, tmp_path, mutate), "no pre-registered rows")


def test_high_tier_row_for_a_report_only_arm_is_refused(build_run, tmp_path):
    mutate = edit_row(where={"arm": "osqp_polished"}, tier="high")
    refused(small_run(build_run, tmp_path, mutate), "no pre-registered rows")


def test_unregistered_arm_is_refused(build_run, tmp_path):
    refused(small_run(build_run, tmp_path, edit_row(arm="scs")), "arm 'scs' is not registered")


def test_fallback_is_legal_only_on_the_dense_active_set_arm(build_run, tmp_path):
    allowed = edit_row("dense", where={"arm": "argmin_dense_active_set"}, fallback=1)
    assert load_run(small_run(build_run, tmp_path / "a", allowed), schema_only=True)
    illegal = edit_row("dense", where={"arm": "argmin_dense_admm"}, fallback=1)
    refused(small_run(build_run, tmp_path / "b", illegal), "fallback 1 is legal only")


@pytest.mark.parametrize("column, value, message", [
    ("m", 999, "dimensions"),
    ("polish", 1, "polish 1 differs"),
    ("seed", 3, "seed 3 must be"),
    ("setup_ns", 10, "setup_ns is recorded on step 0 only"),
    ("condense_ns", 5, "condense_ns is recorded in the dense cohort only"),
    ("tick_ns", 0, "tick_ns must be positive"),
    ("dual_kind", "recovered", "dual_kind 'recovered' differs"),
    ("churn_frac", 0.5, "churn_frac"),
    ("step", 60, "step 60 is outside"),
    ("iterations", "7.0", "iterations '7.0' is not a valid value"),
])
def test_row_value_outside_its_domain_is_refused(build_run, tmp_path, column, value, message):
    mutate = edit_row(where={"arm": "argmin_sparse_admm", "tier": "mpc", "regime": "fixed", "step": 1},
                      **{column: value})
    refused(small_run(build_run, tmp_path, mutate), message)


def test_combination_with_the_wrong_row_count_is_refused(build_run, tmp_path):
    def drop(rows, manifest):
        rows["sparse_rep2.csv.gz"].pop()
    refused(build_run(tmp_path, cells=SMALL, mutate=drop), "sparse nx2_nu1_N10 tight churning high hpipm: 299 rows")


def test_missing_combination_is_refused(build_run, tmp_path):
    message = "dense nx4_nu2_N10 nominal fixed mpc argmin_dense_admm: missing pre-registered combination"
    refused(build_run(tmp_path, cells=SMALL), message)


@pytest.mark.parametrize("change, message", [
    (lambda m: m["competitor_pins"].update(osqp="0" * 40), "competitor_pins"),
    (lambda m: m.update(argmin_sha="0" * 40), "argmin_sha"),
    (lambda m: m["build"].update(cxx_flags="-O3 -DNDEBUG"), "cxx_flags lacks -march=x86-64-v3"),
    (lambda m: m["build"].update(compiler="clang"), "compiler is 'clang'"),
    (lambda m: m["build"].update(daqp_faster_math=True), "daqp_faster_math"),
    (lambda m: m["constants"].update(repetitions=3), "constants"),
    (lambda m: m["arm_settings"]["osqp"]["mpc"].update(adaptive_rho_interval=0), "osqp mpc adaptive_rho_interval"),
    (lambda m: m["expressibility"].pop(), "expressibility"),
    (lambda m: m.update(ctrlpp_harness_dirty=True), "dirty"),
    (lambda m: m["probes"]["daqp"].update(passed=False), "daqp: no passed convention probe"),
    (lambda m: m["probes"].pop("hpipm"), "hpipm: no passed convention probe"),
    (lambda m: m["update_api"].update(qpoases="hotstart"), "update_api"),
    (lambda m: m["terminal_weights"].pop("nx8_nu3_N100"), "terminal_weights"),
    (lambda m: m["binding_gate"]["nx4_nu2"].update(family_passed=False), "nx4_nu2: family_passed disagrees"),
    (lambda m: m.pop("host"), "missing host"),
    (lambda m: m.update(schema="qp-head-to-head-manifest/0"), "schema"),
], ids=["pins", "argmin", "march", "compiler", "daqp", "constants", "settings", "expressibility", "dirty",
        "probe", "probe-missing", "update-api", "terminal-weights", "binding-gate", "missing-key", "schema"])
def test_manifest_differing_from_the_preregistration_is_refused(build_run, tmp_path, change, message):
    refused(small_run(build_run, tmp_path, edit_manifest(change)), message, schema_only=True)


def test_binding_gate_failure_refuses_the_family_tight_variant(build_run, tmp_path):
    run = load_run(small_run(build_run, tmp_path, refused_families=("nx2_nu1",)), schema_only=True)
    assert run.refused_families == ("nx2_nu1",)
    assert {tick.variant for tick in run.ticks} == {"nominal"}


def test_tight_row_of_a_refused_family_is_refused(build_run, tmp_path):
    mutate = edit_manifest(lambda m: m["binding_gate"]["nx2_nu1"].update(
        family_passed=False, cells={"nx2_nu1_N10": {"median_binding_fraction": 0.05,
                                                    "max_state_row_step_fraction": 0.9}}))
    refused(small_run(build_run, tmp_path, mutate), "no pre-registered rows")


def test_unfixed_clock_is_refused_without_advisory(build_run, run_command, tmp_path):
    run_dir = small_run(build_run, tmp_path, clock_state="unfixed")
    result = run_command.verdicts_for(run_dir, tmp_path)
    assert result.process.returncode == 3
    assert result.process.stderr.startswith("REFUSED: manifest.json: clock_state")
    assert not result.json_path.exists()


def test_unfixed_clock_is_accepted_with_advisory_and_not_recorded(full_run, run_command, tmp_path):
    run_dir = tmp_path / "advisory"
    shutil.copytree(full_run, run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["clock_state"] = "unfixed"
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    result = run_command.verdicts_for(run_dir, tmp_path, "--advisory")
    assert result.process.returncode == 0, result.process.stderr
    assert json.loads(result.json_path.read_text())["recorded"] is False
    assert result.md_path.read_text().startswith("> Advisory output")


def test_schema_only_accepts_a_two_row_smoke_run(build_run, run_command, tmp_path):
    result = run_command.invoke("--run-dir", build_run(tmp_path, smoke=True), "--schema-only")
    assert (result.returncode, result.stdout) == (0, "SCHEMA OK\n")


def test_schema_only_still_refuses_a_pin_mismatch(build_run, run_command, tmp_path):
    mutate = edit_manifest(lambda m: m["competitor_pins"].update(hpipm="0" * 40))
    result = run_command.invoke("--run-dir", build_run(tmp_path, smoke=True, mutate=mutate), "--schema-only")
    assert result.returncode == 3 and result.stderr.startswith("REFUSED: manifest.json: competitor_pins")


def test_schema_only_still_refuses_a_column_mismatch(build_run, run_command, tmp_path):
    run_dir = small_run(build_run, tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    header = prereg.CSV_COLUMNS[1:] + prereg.CSV_COLUMNS[:1]
    manifest["csv_files"]["dense_rep0.csv.gz"] = run_command.write_csv(run_dir / "dense_rep0.csv.gz", [], header)
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    result = run_command.invoke("--run-dir", run_dir, "--schema-only")
    assert result.returncode == 3 and "header differs" in result.stderr


def test_expected_preregistration_commit_is_enforced(build_run, tmp_path):
    run_dir = small_run(build_run, tmp_path)
    refused(run_dir, "preregistration_commit", schema_only=True, expect_preregistration_commit="f" * 40)
    assert load_run(run_dir, schema_only=True, expect_preregistration_commit="a" * 40)


def test_manifest_template_carries_the_preregistered_schema(run_command):
    result = run_command.invoke("--print-manifest-template")
    assert result.returncode == 0
    template = json.loads(result.stdout)
    assert list(template) == sorted(prereg.MANIFEST_KEYS)
    assert template["competitor_pins"] == prereg.COMPETITOR_PINS
    assert template["argmin_sha"] == prereg.ARGMIN_SHA


def test_manifest_built_from_the_printed_template_validates(build_run, run_command, tmp_path):
    template = json.loads(run_command.invoke("--print-manifest-template").stdout)

    def from_template(rows, manifest):
        run_values = {key: value for key, value in manifest.items() if template[key] is None}
        manifest.clear()
        manifest.update(template, **run_values)
        manifest["build"]["compiler_version"] = "14.2.1"
    run = load_run(small_run(build_run, tmp_path, from_template), schema_only=True)
    assert run.run_id == run_command.run_id
