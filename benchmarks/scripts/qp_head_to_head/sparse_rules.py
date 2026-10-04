"""The two pre-registered rules judged on the sparse cohort's closed-loop, MPC-tier cell-variants.

block_structure decides whether block-structured (Riccati-class) storage is worth building;
osqp_replacement decides whether argmin's sparse ADMM can replace OSQP as ctrlpp's MPC backend.
"""

from __future__ import annotations

from fractions import Fraction

from . import preregistration as prereg
from .aggregate import GroupSummary, display_ratio


def _scope(cells, refused_families) -> list[tuple[tuple[int, int, int], str]]:
    return [(cell, variant) for cell in cells for variant in prereg.VARIANTS
            if not (variant == "tight" and prereg.family_id(cell[0], cell[1]) in refused_families)]


def _group(summaries: dict, arm: str, cell: str, variant: str) -> GroupSummary:
    return summaries[("sparse", cell, variant, "closed_loop", "mpc", arm)]


def _ratio(numerator: GroupSummary, denominator: GroupSummary | None) -> Fraction | None:
    if denominator is None or numerator.median_tick_ns is None or denominator.median_tick_ns is None:
        return None
    return numerator.median_tick_ns / denominator.median_tick_ns


def _ratio_entry(ratio: Fraction | None) -> dict:
    return {"ratio": ratio, "ratio_display": None if ratio is None else display_ratio(ratio)}


def _failure(reasons: list[str], message: str, **where) -> dict:
    return {**where, "reasons": reasons, "message": message}


def _replacement_cell(summaries: dict, cell: str, variant: str) -> tuple[dict, dict | None]:
    argmin = _group(summaries, prereg.SPARSE_CANDIDATE_ARM, cell, variant)
    osqp = _group(summaries, prereg.REPLACEMENT_INCUMBENT_ARM, cell, variant)
    ratio = _ratio(argmin, osqp)
    osqp_empty = osqp.n_pass == 0
    reasons = []
    if argmin.pass_rate != 1:
        reasons.append("pass_rate")
    if not osqp_empty and (ratio is None or ratio > prereg.OSQP_REPLACEMENT_RATIO_LIMIT):
        reasons.append("ratio")
    entry = {"cell": cell, "variant": variant, "argmin_pass_rate": argmin.pass_rate,
             "osqp_pass_rate": osqp.pass_rate, "argmin_median_tick_ns": argmin.median_tick_ns,
             "osqp_median_tick_ns": osqp.median_tick_ns, "osqp_no_passing_tick": osqp_empty,
             "satisfied": not reasons, **_ratio_entry(ratio)}
    if not reasons:
        return entry, None
    limit = str(prereg.OSQP_REPLACEMENT_RATIO_LIMIT)
    parts = {"pass_rate": f"argmin pass rate {argmin.pass_rate} is below 1",
             "ratio": f"ratio {'undefined' if ratio is None else ratio} is not within {limit}"}
    message = f"{cell} {variant}: " + "; ".join(parts[reason] for reason in reasons)
    return entry, _failure(reasons, message, cell=cell, variant=variant)


def osqp_replacement(summaries: dict, refused_families) -> dict:
    cells, failing = [], []
    for (nx, nu, N), variant in _scope(prereg.mpc_cells(), refused_families):
        entry, failure = _replacement_cell(summaries, prereg.cell_id(nx, nu, N), variant)
        cells.append(entry)
        if failure is not None:
            failing.append(failure)
    return {"outcome": "keep" if failing else "replace", "cells": cells, "failing_cells": failing}


def _structure_cell(summaries: dict, nx: int, nu: int, N: int, variant: str) -> tuple[dict, dict | None]:
    cell = prereg.cell_id(nx, nu, N)
    argmin = _group(summaries, prereg.SPARSE_CANDIDATE_ARM, cell, variant)
    references = [(arm, _group(summaries, arm, cell, variant))
                  for arm in prereg.STRUCTURE_REFERENCE_ARMS if prereg.expressible(arm, N)]
    qualifying = [(arm, group) for arm, group in references if group.pass_rate == 1]
    best = min(qualifying, key=lambda item: item[1].median_tick_ns) if qualifying else None
    ratio = _ratio(argmin, best[1] if best else None) if argmin.pass_rate == 1 else None
    entry = {"cell": cell, "variant": variant, "N": N, "argmin_pass_rate": argmin.pass_rate,
             "argmin_median_tick_ns": argmin.median_tick_ns, "best_reference": best[0] if best else None,
             "references": [{"arm": arm, "pass_rate": group.pass_rate, "median_tick_ns": group.median_tick_ns,
                             "qualifies": group.pass_rate == 1} for arm, group in references],
             "meets_threshold": ratio is not None and ratio >= prereg.BLOCK_STRUCTURE_RATIO_THRESHOLD,
             **_ratio_entry(ratio)}
    failures = {}
    if best is None:
        failures["no_qualifying_reference"] = "no structure-exploiting reference passes every tick"
    if argmin.pass_rate != 1:
        failures["argmin_pass_rate"] = f"argmin pass rate {argmin.pass_rate} is below 1"
    if ratio is not None and not entry["meets_threshold"]:
        threshold = str(prereg.BLOCK_STRUCTURE_RATIO_THRESHOLD)
        failures["ratio_below_threshold"] = f"ratio {ratio} ({display_ratio(ratio)}) is below {threshold}"
    if not failures:
        return entry, None
    message = f"{cell} {variant}: " + "; ".join(failures.values())
    return entry, _failure(list(failures), message, cell=cell, variant=variant)


def _growth(ratios: dict) -> tuple[list[dict], list[dict]]:
    horizons = sorted({N for _, _, N in prereg.LONG_HORIZON_CELLS})
    short, long = horizons[0], horizons[-1]
    families, failures = [], []
    for (nx, nu, variant), by_horizon in ratios.items():
        if short == long or short not in by_horizon or long not in by_horizon:
            continue
        low, high = by_horizon[short], by_horizon[long]
        grows = low is not None and high is not None and high > low
        family = prereg.family_id(nx, nu)
        families.append({"family": family, "variant": variant, "short_horizon": short, "long_horizon": long,
                         "short_ratio": low, "long_ratio": high, "grows": grows})
        if not grows:
            text = f"ratio at N = {long} does not exceed the ratio at N = {short}"
            failures.append(_failure(["no_growth"], f"{family} {variant}: {text}", family=family, variant=variant))
    if not families:
        failures.append(_failure(["growth_not_testable"],
                                 "growth not testable: no family has both horizons in scope"))
    return families, failures


def block_structure(summaries: dict, refused_families) -> dict:
    cells, failing, ratios = [], [], {}
    for (nx, nu, N), variant in _scope(prereg.LONG_HORIZON_CELLS, refused_families):
        entry, failure = _structure_cell(summaries, nx, nu, N, variant)
        cells.append(entry)
        failing.extend([failure] if failure else [])
        ratios.setdefault((nx, nu, variant), {})[N] = entry["ratio"]
    families, growth_failures = _growth(ratios)
    failing.extend(growth_failures)
    return {"outcome": "NO-GO" if failing else "GO", "cells": cells, "families": families,
            "failing_conditions": failing}
