"""The Bounds column of the bindings document is what each nonlinear solver does with a box.

A box that a solver silently ignores returns an answer outside it, so the
document's claim is checked by solving a problem whose unconstrained minimum lies
outside the box and classifying where the solve ends.
"""

import pathlib
import re

import numpy as np
import pytest

import argmin

DOCUMENT = pathlib.Path(__file__).resolve().parents[2] / "docs" / "python-bindings.md"

APPLIED = "applied"
REJECTED = "rejected"
IGNORED = "length-checked, not applied"
VALUES = (APPLIED, REJECTED, IGNORED)

LOWER = np.array([-1.0, -1.0])
UPPER = np.array([1.0, 1.0])

CONSTRAINED = ("KraftSlsqpSolver", "FilterSlsqpSolver", "TrSqpSolver")
SOLVERS = ("LbfgsbSolver",) + CONSTRAINED


def shifted_bowl(x):
    return float((x[0] - 3.0) ** 2 + (x[1] - 3.0) ** 2)


def shifted_bowl_gradient(x):
    return np.array([2.0 * (x[0] - 3.0), 2.0 * (x[1] - 3.0)])


def loose_cap(x):
    return np.array([10.0 - x[0] - x[1]])


def loose_cap_jacobian(x):
    return np.array([[-1.0, -1.0]])


def cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def documented_bounds(text):
    lines = text.splitlines()
    rows = {}
    for index, line in enumerate(lines):
        if not line.startswith("|") or "Bounds" not in cells(line):
            continue
        column = cells(line).index("Bounds")
        for row in lines[index + 2 :]:
            if not row.startswith("|"):
                break
            name = re.match(r"`([A-Za-z0-9]+)", cells(row)[0]).group(1)
            rows[name] = cells(row)[column]
    return rows


def observed_bounds(name):
    arguments = dict(
        objective=shifted_bowl,
        x0=np.array([0.0, 0.0]),
        gradient=shifted_bowl_gradient,
        lower_bounds=LOWER,
        upper_bounds=UPPER,
    )
    if name in CONSTRAINED:
        arguments.update(
            constraints=loose_cap,
            constraint_jacobian=loose_cap_jacobian,
            num_equality=0,
            num_inequality=1,
        )
    try:
        result = getattr(argmin, name)(**arguments).solve()
    except argmin.ArgminError as error:
        if error.kind == argmin.ErrorKind.invalid_problem:
            return REJECTED
        raise
    if np.allclose(result.x, UPPER, rtol=0.0, atol=1e-6):
        return APPLIED
    if np.any(result.x < LOWER) or np.any(result.x > UPPER):
        return IGNORED
    return f"unclassified: ended at {result.x} with {result.status}"


def mismatches(documented, observed):
    return {
        name: (documented.get(name), behavior)
        for name, behavior in observed.items()
        if documented.get(name) != behavior
    }


@pytest.fixture(scope="module")
def documented():
    return documented_bounds(DOCUMENT.read_text())


def test_every_nonlinear_solver_has_one_documented_bounds_value(documented):
    assert sorted(documented) == sorted(SOLVERS)
    assert set(documented.values()) <= set(VALUES)


@pytest.mark.parametrize("name", SOLVERS)
def test_the_documented_bounds_value_is_the_observed_behavior(documented, name):
    assert mismatches(documented, {name: observed_bounds(name)}) == {}


def test_a_documented_value_that_differs_from_the_behavior_is_reported():
    observed = {"TrSqpSolver": APPLIED}
    assert mismatches({"TrSqpSolver": REJECTED}, observed) == {
        "TrSqpSolver": (REJECTED, APPLIED)
    }
    assert mismatches({}, observed) == {"TrSqpSolver": (None, APPLIED)}
    assert mismatches({"TrSqpSolver": APPLIED}, observed) == {}


def test_the_table_parser_reads_the_bounds_column():
    table = "| name | method | Bounds |\n| --- | --- | --- |\n| `TrSqpSolver` | m | rejected |\n"
    assert documented_bounds(table) == {"TrSqpSolver": REJECTED}
