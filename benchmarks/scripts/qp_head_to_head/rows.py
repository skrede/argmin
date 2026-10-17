"""Parsing and validation of one row of a recorded run against the pre-registered record format."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields

from . import preregistration as prereg
from .manifest_checks import RefusedInput

_FLOAT = re.compile(r"^-?(?:[0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?|nan|inf)$")

COHORT_CELLS = {
    cohort: {prereg.cell_id(*cell): cell for cell in prereg.mpc_cells()} for cohort in prereg.COHORTS
}
COHORT_CELLS["dense"][prereg.DENSE_ONLY_PROBLEM] = (0, 0, 0)


@dataclass(frozen=True, slots=True)
class Tick:
    run_id: str
    cohort: str
    cell: str
    variant: str
    nx: int
    nu: int
    N: int
    n: int
    m: int
    regime: str
    tier: str
    arm: str
    polish: int
    rep: int
    seed: int
    step: int
    setup_ns: int
    update_ns: int
    solve_ns: int
    tick_ns: int
    condense_ns: int
    instr_setup: int
    instr_update: int
    instr_solve: int
    instr_tick: int
    iterations: int
    kkt_stat: float
    kkt_prim: float
    kkt_comp: float
    kkt_max: float
    native_status: str
    passed: int
    fail_class: str
    allocs_tick: int
    dual_kind: str
    fallback: int
    binding_frac: float
    churn_frac: float


def _int(value: str) -> int:
    parsed = int(value)
    if str(parsed) != value:
        raise ValueError(value)
    return parsed


def _float(value: str) -> float:
    if not _FLOAT.match(value):
        raise ValueError(value)
    return float(value)


_PARSERS = tuple({"str": str, "int": _int, "float": _float}[field.type] for field in fields(Tick))


def parse_row(row: list[str], where: str) -> Tick:
    if len(row) != len(prereg.CSV_COLUMNS):
        raise RefusedInput(f"{where}: {len(row)} fields, expected {len(prereg.CSV_COLUMNS)}")
    try:
        return Tick(*[parse(value) for parse, value in zip(_PARSERS, row)])
    except ValueError:
        for column, parse, value in zip(prereg.CSV_COLUMNS, _PARSERS, row):
            try:
                parse(value)
            except ValueError:
                raise RefusedInput(f"{where}: {column} {value!r} is not a valid value") from None
        raise


def dimensions(cohort: str, cell: str, variant: str) -> tuple[int, int, int, int, int]:
    if cell == prereg.DENSE_ONLY_PROBLEM:
        return (0, 0, 0, *prereg.DENSE_ONLY_DIMENSIONS)
    nx, nu, N = COHORT_CELLS[cohort][cell]
    state_rows = N * nu if variant == "tight" else 0
    if cohort == "sparse":
        n = (N + 1) * nx + N * nu
        return nx, nu, N, n, n + state_rows
    return nx, nu, N, N * nu, N * nu + state_rows


def polish_flag(arm: str, tier: str) -> int:
    settings = prereg.ARM_SETTINGS[arm][tier]
    for field in ("polish", "polishing"):
        if field in settings:
            return int(settings[field])
    return -1


def _refuse(where: str, message: str) -> None:
    raise RefusedInput(f"{where}: {message}")


def _check_domains(tick: Tick, where: str) -> None:
    arms = {arm.name for arm in prereg.cohort_arms(tick.cohort)}
    domains = (
        ("cell", tick.cell, COHORT_CELLS[tick.cohort]),
        ("variant", tick.variant, prereg.VARIANTS),
        ("regime", tick.regime, prereg.REGIMES),
        ("tier", tick.tier, prereg.TIERS),
        ("arm", tick.arm, arms),
    )
    for column, value, domain in domains:
        if value not in domain:
            _refuse(where, f"{column} {value!r} is not registered for the {tick.cohort} cohort")
    _refuse(where, "row for a combination with no pre-registered rows (not expressible, refused or not run)")


def _check_identity(tick: Tick, where: str, combinations: dict, run_id: str, cohort: str, rep: int) -> None:
    if tick.run_id != run_id:
        _refuse(where, f"run_id {tick.run_id!r} differs from the manifest's {run_id!r}")
    if tick.cohort != cohort or tick.rep != rep:
        _refuse(where, f"cohort {tick.cohort!r} rep {tick.rep} does not belong in this file")
    for column, value, domain in (("fail_class", tick.fail_class, prereg.FAIL_CLASSES),
                                  ("dual_kind", tick.dual_kind, prereg.DUAL_KINDS)):
        if value not in domain:
            _refuse(where, f"{column} {value!r} is not one of {domain}")
    key = (tick.cohort, tick.cell, tick.variant, tick.regime, tick.tier, tick.arm)
    expected = combinations.get(key)
    if expected is None:
        _check_domains(tick, where)
    dims, polish, dual_kind = expected
    if (tick.nx, tick.nu, tick.N, tick.n, tick.m) != dims:
        _refuse(where, f"dimensions (nx, nu, N, n, m) must be {dims}")
    if tick.polish != polish:
        _refuse(where, f"polish {tick.polish} differs from the arm's setting {polish}")
    if tick.dual_kind != dual_kind:
        _refuse(where, f"dual_kind {tick.dual_kind!r} differs from the arm's {dual_kind!r}")


def _check_counters(tick: Tick, where: str) -> None:
    if not 0 <= tick.step < prereg.STEPS_PER_SEQUENCE:
        _refuse(where, f"step {tick.step} is outside 0..{prereg.STEPS_PER_SEQUENCE - 1}")
    if tick.seed != (tick.rep if tick.regime == "churning" else -1):
        _refuse(where, f"seed {tick.seed} must be the repetition index when churning and -1 otherwise")
    if tick.tick_ns <= 0 or tick.update_ns < 0 or tick.solve_ns < 0:
        _refuse(where, "tick_ns must be positive and update_ns, solve_ns non-negative")
    if (tick.setup_ns >= 0) != (tick.step == 0) or tick.setup_ns < -1:
        _refuse(where, "setup_ns is recorded on step 0 only (-1 elsewhere)")
    if (tick.condense_ns >= 0) != (tick.cohort == "dense") or tick.condense_ns < -1:
        _refuse(where, "condense_ns is recorded in the dense cohort only (-1 in the sparse one)")
    optional = (tick.instr_setup, tick.instr_update, tick.instr_solve, tick.instr_tick,
                tick.iterations, tick.allocs_tick)
    if min(optional) < -1:
        _refuse(where, "instruction, iteration and allocation counts are non-negative or -1")
    if tick.fallback not in (0, 1) or (tick.fallback == 1 and tick.arm != prereg.FALLBACK_ARM):
        _refuse(where, f"fallback {tick.fallback} is legal only as 0, or 1 on {prereg.FALLBACK_ARM}")


def _check_quality(tick: Tick, where: str) -> None:
    residuals = (tick.kkt_stat, tick.kkt_prim, tick.kkt_comp, tick.kkt_max)
    if any(value < 0.0 for value in residuals):
        _refuse(where, "KKT residuals cannot be negative")
    passes = tick.kkt_max <= prereg.TIERS[tick.tier]
    if tick.passed not in (0, 1) or bool(tick.passed) is not passes:
        _refuse(where, f"pass {tick.passed} disagrees with kkt_max {tick.kkt_max!r} at the {tick.tier} threshold")
    if (tick.fail_class == "none") is not passes:
        _refuse(where, f"fail_class {tick.fail_class!r} disagrees with pass {tick.passed}")
    if not math.isfinite(tick.binding_frac) or not (tick.binding_frac == -1.0 or 0.0 <= tick.binding_frac <= 1.0):
        _refuse(where, f"binding_frac {tick.binding_frac!r} is neither a fraction nor -1")
    churned = tick.regime == "churning" and tick.step > 0
    valid = 0.0 <= tick.churn_frac <= 1.0 if churned else tick.churn_frac == -1.0
    if not valid:
        _refuse(where, "churn_frac is a fraction on churning steps after the first and -1 elsewhere")


def check_row(tick: Tick, where: str, combinations: dict, run_id: str, cohort: str, rep: int) -> None:
    _check_identity(tick, where, combinations, run_id, cohort, rep)
    _check_counters(tick, where)
    _check_quality(tick, where)
