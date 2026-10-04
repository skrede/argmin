"""Deterministic JSON and Markdown renderings of the verdicts and of the pre-registration."""

from __future__ import annotations

import json
from fractions import Fraction

from .aggregate import display_ratio, exact_text
from .preregistration_tables import markdown_table, preregistration_tables

__all__ = ["render_json", "render_markdown", "preregistration_tables"]

NO_PASSING_TICK = "no passing tick"


def _encode(value: object) -> str:
    if isinstance(value, Fraction):
        return exact_text(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


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


def render_markdown(verdicts: dict) -> str:
    lines = []
    if not verdicts["recorded"]:
        lines += ["> Advisory output: not a recorded run, not publishable.", ""]
    lines += ["# Head-to-head QP benchmark verdicts", "", f"Run `{verdicts['run_id']}`.", ""]
    for section in (_block_structure(verdicts["block_structure"]), _dense_default(verdicts["dense_default"]),
                    _osqp_replacement(verdicts["osqp_replacement"])):
        lines += section + [""]
    return "\n".join(lines)
