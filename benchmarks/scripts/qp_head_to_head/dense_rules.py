"""The pre-registered dense-default rule, one outcome per regime and tier.

A dense candidate qualifies only by passing every tick of every in-scope cell-variant; two
qualifiers are compared cell by cell on exact medians, and an exact tie awards the cell to neither.
"""

from __future__ import annotations

from . import preregistration as prereg
from .aggregate import GroupSummary, display_ratio


def _scope(regime: str, refused_families) -> list[tuple[str, str]]:
    scope = [(prereg.cell_id(nx, nu, N), variant) for nx, nu, N in prereg.mpc_cells()
             for variant in prereg.VARIANTS
             if not (variant == "tight" and prereg.family_id(nx, nu) in refused_families)]
    if regime == "fixed":
        scope.append((prereg.DENSE_ONLY_PROBLEM, "nominal"))
    return scope


def _group(summaries: dict, arm: str, cell: str, variant: str, regime: str, tier: str) -> GroupSummary:
    return summaries[("dense", cell, variant, regime, tier, arm)]


def _qualification(summaries: dict, arm: str, scope: list, regime: str, tier: str) -> dict:
    failures = []
    for cell, variant in scope:
        group = _group(summaries, arm, cell, variant, regime, tier)
        if group.pass_rate != 1:
            failures.append({"cell": cell, "variant": variant, "pass_rate": group.pass_rate})
    return {"arm": arm, "qualifies": not failures, "pass_rate_failures": failures}


def _cell_entry(summaries: dict, cell: str, variant: str, regime: str, tier: str, compare: bool) -> dict:
    first, second = (_group(summaries, arm, cell, variant, regime, tier) for arm in prereg.DENSE_CANDIDATE_ARMS)
    defined = first.median_tick_ns is not None and second.median_tick_ns is not None
    ratio = first.median_tick_ns / second.median_tick_ns if defined else None
    winner = None
    if compare and defined and first.median_tick_ns != second.median_tick_ns:
        winner = prereg.DENSE_CANDIDATE_ARMS[0 if first.median_tick_ns < second.median_tick_ns else 1]
    arms = {arm: {"pass_rate": group.pass_rate, "median_tick_ns": group.median_tick_ns}
            for arm, group in zip(prereg.DENSE_CANDIDATE_ARMS, (first, second))}
    return {"cell": cell, "variant": variant, "arms": arms, "ratio": ratio,
            "ratio_display": None if ratio is None else display_ratio(ratio), "winner": winner}


def _verdict(qualifying: list[str], wins: dict, ties: list) -> tuple[str, str | None, str]:
    if not qualifying:
        return "no-default", None, "no default: neither candidate passes every tick on every in-scope cell-variant"
    if len(qualifying) == 1:
        return "default", qualifying[0], f"{qualifying[0]} is the default: the only candidate passing every tick"
    winners = [arm for arm in qualifying if wins[arm]]
    if len(winners) == 1:
        arm = winners[0]
        return "default", arm, f"{arm} is the default: it wins {len(wins[arm])} cell-variants, loses none, {len(ties)} tied"
    if not winners:
        return "portfolio", None, "portfolio: the candidates are indistinguishable (every cell-variant tied)"
    counts = ", ".join(f"{arm} wins {len(wins[arm])}" for arm in qualifying)
    return "portfolio", None, f"portfolio: {counts}, {len(ties)} tied"


def _decide(summaries: dict, regime: str, tier: str, refused_families) -> dict:
    scope = _scope(regime, refused_families)
    candidates = [_qualification(summaries, arm, scope, regime, tier) for arm in prereg.DENSE_CANDIDATE_ARMS]
    qualifying = [entry["arm"] for entry in candidates if entry["qualifies"]]
    compare = len(qualifying) == 2
    cells = [_cell_entry(summaries, cell, variant, regime, tier, compare) for cell, variant in scope]
    wins = {arm: [{"cell": c["cell"], "variant": c["variant"]} for c in cells if c["winner"] == arm]
            for arm in prereg.DENSE_CANDIDATE_ARMS}
    ties = [{"cell": c["cell"], "variant": c["variant"]} for c in cells if compare and c["winner"] is None]
    outcome, default_arm, message = _verdict(qualifying, wins, ties)
    return {"regime": regime, "tier": tier, "outcome": outcome, "default_arm": default_arm,
            "qualifying": qualifying, "indistinguishable": compare and not any(wins.values()),
            "candidates": candidates, "cells": cells, "wins": wins, "ties": ties, "message": message}


def dense_default(summaries: dict, refused_families) -> list[dict]:
    return [_decide(summaries, regime, tier, refused_families) for regime in prereg.REGIMES for tier in prereg.TIERS]
