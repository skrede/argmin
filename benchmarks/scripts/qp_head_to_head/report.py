"""Reporting-only statistics of every group, beside the summary the decision rules read.

No rule module imports this one: every figure here is shown to a reader and none is compared
against a pre-registered threshold. Times and instruction counts are taken over passing ticks;
iterations, residuals, fail classes, allocations and fallbacks over every tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Context, Decimal
from fractions import Fraction

from . import preregistration as prereg
from .aggregate import GroupKey, GroupSummary, display_ratio, exact_median, exact_text, group_ticks
from .records import RunData, expected_combinations

NO_PASSING_TICK = "no passing tick"
NOT_AVAILABLE = "n/a"
UNAVAILABLE = -1


@dataclass(frozen=True)
class GroupReport:
    median_setup_ns: Fraction | None
    median_update_ns: Fraction | None
    median_solve_ns: Fraction | None
    median_condense_ns: Fraction | None
    p25_tick_ns: Fraction | None
    p75_tick_ns: Fraction | None
    median_instr_setup: Fraction | None
    median_instr_update: Fraction | None
    median_instr_solve: Fraction | None
    median_instr_tick: Fraction | None
    iterations_min: int | None
    iterations_median: Fraction | None
    iterations_max: int | None
    worst_kkt_max: float | None
    fail_classes: dict[str, int]
    allocs_max: int | None
    fallback_count: int
    near_threshold_count: int


def percentile_linear(values, q: Fraction) -> Fraction | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * Fraction(q)
    lower = math.floor(position)
    if lower == position:
        return Fraction(ordered[lower])
    return ordered[lower] + (ordered[lower + 1] - ordered[lower]) * (position - lower)


def display_time_us(value_ns: Fraction) -> str:
    return display_ratio(Fraction(value_ns) / 1000)


def _available(values) -> list[int]:
    return [value for value in values if value != UNAVAILABLE]


def _worst(residuals: list[float]) -> float | None:
    if any(math.isnan(value) for value in residuals):
        return math.nan
    return max(residuals, default=None)


def _near_threshold(residuals: list[float], threshold: float) -> int:
    low, high = math.nextafter(threshold, -math.inf), math.nextafter(threshold, math.inf)
    return sum(1 for value in residuals if low <= value <= high)


def _fail_classes(ticks) -> dict[str, int]:
    counts = dict.fromkeys(prereg.FAIL_CLASSES[1:], 0)
    for tick in ticks:
        if tick.fail_class in counts:
            counts[tick.fail_class] += 1
    return counts


def _report(tier: str, ticks) -> GroupReport:
    passing = [tick for tick in ticks if tick.passed == 1]
    first = [tick for tick in passing if tick.step == 0]
    iterations = _available(tick.iterations for tick in ticks)
    residuals = [tick.kkt_max for tick in ticks]
    allocations = _available(tick.allocs_tick for tick in ticks)
    return GroupReport(
        exact_median(tick.setup_ns for tick in first), exact_median(tick.update_ns for tick in passing),
        exact_median(tick.solve_ns for tick in passing),
        exact_median(_available(tick.condense_ns for tick in passing)),
        percentile_linear([tick.tick_ns for tick in passing], Fraction(1, 4)),
        percentile_linear([tick.tick_ns for tick in passing], Fraction(3, 4)),
        exact_median(_available(tick.instr_setup for tick in first)),
        exact_median(_available(tick.instr_update for tick in passing)),
        exact_median(_available(tick.instr_solve for tick in passing)),
        exact_median(_available(tick.instr_tick for tick in passing)),
        min(iterations, default=None), exact_median(iterations), max(iterations, default=None),
        _worst(residuals), _fail_classes(ticks), max(allocations, default=None),
        sum(tick.fallback for tick in ticks), _near_threshold(residuals, prereg.TIERS[tier]))


def group_reports(run: RunData) -> dict[GroupKey, GroupReport]:
    groups = group_ticks(run.ticks)
    return {key: _report(key[4], groups.get(key, [])) for key in expected_combinations(run.refused_families)}


def _exact_count(value: Fraction | int | None) -> str:
    if value is None:
        return NOT_AVAILABLE
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else str(Decimal(value.numerator) / value.denominator)


def _residual(value: float | None) -> str:
    if value is None:
        return NOT_AVAILABLE
    if not math.isfinite(value):
        return repr(value)
    rounded = Context(prec=prereg.DISPLAY_SIGNIFICANT_FIGURES, rounding=ROUND_HALF_EVEN).plus(Decimal(value))
    return f"{rounded:.{prereg.DISPLAY_SIGNIFICANT_FIGURES - 1}e}"


def _failures(counts: dict[str, int]) -> str:
    return ", ".join(f"{name} {count}" for name, count in counts.items() if count) or "none"


def _displays(summary: GroupSummary, report: GroupReport) -> dict[str, str]:
    absent = NO_PASSING_TICK if summary.n_pass == 0 else NOT_AVAILABLE
    displays = {}
    for name, value in (*vars(summary).items(), *vars(report).items()):
        if name.endswith("_ns"):
            displays[name[:-3] + "_us"] = absent if value is None else display_time_us(value)
        elif name.startswith("median_instr"):
            displays[name + "_display"] = absent if value is None else display_ratio(value)
        elif name.startswith("iterations"):
            displays[name + "_display"] = _exact_count(value)
    displays.update(pass_rate_display=display_ratio(summary.pass_rate),
                    worst_kkt_max_display=_residual(report.worst_kkt_max),
                    fail_classes_display=_failures(report.fail_classes),
                    allocs_max_display=_exact_count(report.allocs_max))
    return displays


def summary_entry(key: GroupKey, summary: GroupSummary, report: GroupReport) -> dict:
    entry = {**dict(zip(("cohort", "cell", "variant", "regime", "tier", "arm"), key)), **vars(summary),
             **vars(report), **_displays(summary, report)}
    entry["worst_kkt_max"] = None if report.worst_kkt_max is None else repr(report.worst_kkt_max)
    return {name: exact_text(value) if isinstance(value, Fraction) else value for name, value in entry.items()}
