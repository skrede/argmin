"""Exact per-group statistics every decision rule reads.

A group is one (cohort, cell, variant, regime, tier, arm) combination. Medians and pass rates are
exact rationals; nothing here rounds, and the display string never feeds a decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Context, Decimal
from fractions import Fraction

from . import preregistration as prereg
from .records import RunData, Tick, expected_combinations

GroupKey = tuple[str, str, str, str, str, str]


@dataclass(frozen=True)
class GroupSummary:
    n_ticks: int
    n_pass: int
    pass_rate: Fraction
    median_tick_ns: Fraction | None


def exact_median(values) -> Fraction | None:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return None
    if count % 2 == 1:
        return Fraction(ordered[count // 2])
    return Fraction(ordered[count // 2 - 1] + ordered[count // 2], 2)


def exact_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def display_ratio(value: Fraction) -> str:
    context = Context(prec=prereg.DISPLAY_SIGNIFICANT_FIGURES, rounding=ROUND_HALF_EVEN)
    rounded = context.divide(Decimal(value.numerator), Decimal(value.denominator))
    decimals = max(0, prereg.DISPLAY_SIGNIFICANT_FIGURES - 1 - rounded.adjusted())
    return f"{rounded:.{decimals}f}"


def group_ticks(ticks) -> dict[GroupKey, list[Tick]]:
    groups: dict[GroupKey, list[Tick]] = {}
    for tick in ticks:
        key = (tick.cohort, tick.cell, tick.variant, tick.regime, tick.tier, tick.arm)
        groups.setdefault(key, []).append(tick)
    return groups


def summarize(run: RunData) -> dict[GroupKey, GroupSummary]:
    groups = group_ticks(run.ticks)
    summaries = {}
    for key in expected_combinations(run.refused_families):
        ticks = groups.get(key, [])
        passing = [tick.tick_ns for tick in ticks if tick.passed == 1]
        rate = Fraction(len(passing), len(ticks)) if ticks else Fraction(0)
        summaries[key] = GroupSummary(len(ticks), len(passing), rate, exact_median(passing))
    return summaries
