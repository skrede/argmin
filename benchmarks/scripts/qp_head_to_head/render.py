"""Deterministic JSON and Markdown renderings of the verdicts and of the pre-registration."""

from __future__ import annotations

import json
from fractions import Fraction

from . import preregistration as prereg
from .aggregate import display_ratio, exact_text
from .preregistration_tables import markdown_table, preregistration_tables
from .records import RunData
from .report import summary_entry

__all__ = ["render_json", "render_markdown", "preregistration_tables", "report_sections"]

NO_PASSING_TICK = "no passing tick"


def _encode(value: object) -> str:
    if isinstance(value, Fraction):
        return exact_text(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _excluded(run: RunData) -> dict:
    families = [prereg.family_id(nx, nu) for nx, nu in prereg.families()]
    refused = [{"family": family, **run.manifest["binding_gate"][family]}
               for family in families if family in run.refused_families]
    not_expressible = [{name: entry[name] for name in ("cohort", "arm", "cell")}
                       for entry in prereg.expressibility_matrix() if not entry["expressible"]]
    return {"refused_tight_families": refused, "not_expressible": not_expressible}


def _preregistration() -> dict:
    thresholds = {"block_structure_ratio_at_least": prereg.BLOCK_STRUCTURE_RATIO_THRESHOLD,
                  "osqp_replacement_ratio_at_most": prereg.OSQP_REPLACEMENT_RATIO_LIMIT,
                  "qualifying_pass_rate": Fraction(1)}
    return {"argmin_sha": prereg.ARGMIN_SHA, "competitor_pins": dict(prereg.COMPETITOR_PINS),
            "build_policy": dict(prereg.BUILD_POLICY), "constants": prereg.manifest_constants(),
            "arm_settings": prereg.ARM_SETTINGS, "thresholds": thresholds}


def report_sections(run: RunData, summaries: dict, reports: dict) -> dict:
    return {"summary": [summary_entry(key, summary, reports[key]) for key, summary in summaries.items()],
            "excluded": _excluded(run), "inputs": [dict(entry) for entry in run.inputs],
            "preregistration": _preregistration()}


def render_json(verdicts: dict) -> str:
    return json.dumps(verdicts, sort_keys=True, indent=2, default=_encode) + "\n"


def _median(value: Fraction | None) -> str:
    return NO_PASSING_TICK if value is None else display_ratio(value)


def _ratio(entry: dict) -> str:
    return "undefined" if entry["ratio"] is None else f"{entry['ratio_display']} ({exact_text(entry['ratio'])})"


def _failures(entries: list[dict]) -> list[str]:
    return ["", *(f"- {entry['message']}" for entry in entries)] if entries else []


def _block_structure(rule: dict) -> list[str]:
    rows = []
    for cell in rule["cells"]:
        best = next((ref for ref in cell["references"] if ref["arm"] == cell["best_reference"]), None)
        rows.append([cell["cell"], cell["variant"], _median(cell["argmin_median_tick_ns"]),
                     exact_text(cell["argmin_pass_rate"]), cell["best_reference"] or "none qualifies",
                     _median(best["median_tick_ns"]) if best else NO_PASSING_TICK, _ratio(cell),
                     "yes" if cell["meets_threshold"] else "no"])
    headers = ["Cell", "Variant", "argmin median tick (ns)", "argmin pass rate", "Best reference",
               "Reference median tick (ns)", "Ratio", "At least 3"]
    return ["## Block-structured storage", "", f"Outcome: **{rule['outcome']}**.", "",
            *markdown_table(headers, rows), *_failures(rule["failing_conditions"])]


def _dense_default(outcomes: list[dict]) -> list[str]:
    lines = ["## Dense default"]
    for outcome in outcomes:
        arms = list(outcome["cells"][0]["arms"])
        headers = ["Cell", "Variant"]
        for arm in arms:
            headers += [f"{arm} median tick (ns)", f"{arm} pass rate"]
        rows = []
        for cell in outcome["cells"]:
            row = [cell["cell"], cell["variant"]]
            for arm in arms:
                row += [_median(cell["arms"][arm]["median_tick_ns"]), exact_text(cell["arms"][arm]["pass_rate"])]
            rows.append(row + [_ratio(cell), cell["winner"] or ("tie" if len(outcome["qualifying"]) == 2 else "-")])
        lines += ["", f"### {outcome['regime']}, {outcome['tier']} tier", "",
                  f"Outcome: **{outcome['outcome']}** - {outcome['message']}.", "",
                  *markdown_table(headers + ["Ratio", "Winner"], rows)]
    return lines


def _osqp_replacement(rule: dict) -> list[str]:
    rows = [[cell["cell"], cell["variant"], _median(cell["argmin_median_tick_ns"]),
             exact_text(cell["argmin_pass_rate"]), _median(cell["osqp_median_tick_ns"]),
             exact_text(cell["osqp_pass_rate"]), _ratio(cell), "yes" if cell["satisfied"] else "no"]
            for cell in rule["cells"]]
    headers = ["Cell", "Variant", "argmin median tick (ns)", "argmin pass rate", "OSQP median tick (ns)",
               "OSQP pass rate", "Ratio", "Satisfied"]
    return ["## OSQP replacement", "", f"Outcome: **{rule['outcome']}**.", "",
            *markdown_table(headers, rows), *_failures(rule["failing_cells"])]


def _spread(entry: dict) -> str:
    return NO_PASSING_TICK if entry["p25_tick_ns"] is None else f"{entry['p25_tick_us']} - {entry['p75_tick_us']}"


def _summary_row(entry: dict, dense: bool) -> list[str]:
    def joined(*names: str) -> str:
        return " / ".join(entry[name] for name in names)

    return [entry["cell"], entry["variant"], f"`{entry['arm']}`", entry["pass_rate"], entry["median_tick_us"],
            _spread(entry), joined("median_setup_us", "median_update_us", "median_solve_us"),
            *([entry["median_condense_us"]] if dense else []),
            joined(*(f"median_instr_{phase}_display" for phase in ("setup", "update", "solve", "tick"))),
            joined("iterations_min_display", "iterations_median_display", "iterations_max_display"),
            entry["worst_kkt_max_display"], entry["fail_classes_display"], entry["allocs_max_display"],
            *([str(entry["fallback_count"])] if dense else [])]


def _summary_tables(entries: list[dict]) -> list[str]:
    lines = ["## Group summary"]
    for cohort in prereg.COHORTS:
        dense = cohort == "dense"
        headers = ["Cell", "Variant", "Arm", "Pass rate", "Median tick (us)", "Tick p25 - p75 (us)",
                   "Median setup / update / solve (us)", *(["Median condense (us)"] if dense else []),
                   "Median instructions setup / update / solve / tick", "Iterations min / median / max",
                   "Worst KKT residual", "Failures", "Allocations max", *(["Fallbacks"] if dense else [])]
        for regime in prereg.REGIMES:
            for tier in prereg.TIERS:
                rows = [_summary_row(entry, dense) for entry in entries
                        if (entry["cohort"], entry["regime"], entry["tier"]) == (cohort, regime, tier)]
                lines += ["", f"### {cohort} cohort, {regime} regime, {tier} tier", "",
                          *markdown_table(headers, rows)]
    return lines


def render_markdown(verdicts: dict) -> str:
    lines = []
    if not verdicts["recorded"]:
        lines += ["> Advisory output: not a recorded run, not publishable.", ""]
    lines += ["# Head-to-head QP benchmark verdicts", "", f"Run `{verdicts['run_id']}`.", ""]
    for section in (_summary_tables(verdicts["summary"]), _block_structure(verdicts["block_structure"]), _dense_default(verdicts["dense_default"]),
                    _osqp_replacement(verdicts["osqp_replacement"])):
        lines += section + [""]
    return "\n".join(lines)
