import dataclasses
import json
import random

from qp_head_to_head import preregistration as prereg
from qp_head_to_head_verdicts import compute_outputs


def test_tracer_replace(full_run_outputs):
    assert full_run_outputs.process.returncode == 0, full_run_outputs.process.stderr
    verdicts = json.loads(full_run_outputs.json_path.read_text())
    assert verdicts["osqp_replacement"]["outcome"] == "replace"
    assert verdicts["osqp_replacement"]["failing_cells"] == []
    assert {cell["ratio"] for cell in verdicts["osqp_replacement"]["cells"]} == {"6/5"}
    assert "## OSQP replacement" in full_run_outputs.md_path.read_text()


def test_tracer_keep_names_cell(build_run, run_command, tmp_path):
    def tick_ns(arm, cell, variant, regime, tier, rep, step):
        if arm == "argmin_sparse_admm" and cell == "nx8_nu3_N100" and variant == "tight":
            return 1300
        return run_command.default_tick_ns(arm, cell, variant, regime, tier, rep, step)

    run_dir = build_run(tmp_path, tick_ns=tick_ns)
    outputs = run_command.verdicts_for(run_dir, tmp_path)
    assert outputs.process.returncode == 0, outputs.process.stderr
    rule = json.loads(outputs.json_path.read_text())["osqp_replacement"]
    assert rule["outcome"] == "keep"
    assert [(cell["cell"], cell["variant"]) for cell in rule["failing_cells"]] == [("nx8_nu3_N100", "tight")]
    assert rule["failing_cells"][0]["reasons"] == ["ratio"]


def test_outputs_are_byte_identical_across_processes(full_run_outputs, full_run_data):
    json_text, markdown_text = compute_outputs(full_run_data)
    assert full_run_outputs.json_path.read_bytes() == json_text.encode("utf-8")
    assert full_run_outputs.md_path.read_bytes() == markdown_text.encode("utf-8")
    assert set(json.loads(json_text)) == {"schema", "run_id", "recorded", "osqp_replacement", "block_structure",
                                          "dense_default", "summary", "excluded", "inputs", "preregistration"}


def test_outputs_name_inputs_without_the_run_directory(full_run, full_run_outputs):
    for path in (full_run_outputs.json_path, full_run_outputs.md_path):
        text = path.read_text()
        assert str(full_run) not in text and str(full_run.parent) not in text
    names = {entry["name"] for entry in json.loads(full_run_outputs.json_path.read_text())["inputs"]}
    assert names == {f"{cohort}_rep{rep}.csv.gz" for cohort in prereg.COHORTS for rep in range(prereg.REPETITIONS)}


def test_row_order_does_not_change_the_outputs(full_run_data):
    ticks = list(full_run_data.ticks)
    random.Random(0).shuffle(ticks)
    shuffled = dataclasses.replace(full_run_data, ticks=tuple(ticks))
    assert compute_outputs(shuffled) == compute_outputs(full_run_data)


def test_print_preregistration(run_command):
    result = run_command.invoke("--print-preregistration")
    assert result.returncode == 0
    assert prereg.ARGMIN_SHA in result.stdout and "5/4" in result.stdout and "x86-64-v3" in result.stdout


def test_missing_run_directory_argument_is_a_usage_error(run_command):
    assert run_command.invoke("--output-json", "a.json", "--output-md", "a.md").returncode == 2
