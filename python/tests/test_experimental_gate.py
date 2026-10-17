"""The solvers with a confirmed defect are reachable only through a deliberate import.

A plain ``import argmin`` must not load ``argmin.experimental`` or expose any of
its types, and each gated solver states its defect where a caller reaches it:
in its docstring and in machine-readable class attributes.
"""

import os
import re
import sys
import json
import inspect
import warnings
import subprocess

import pytest

import argmin
import argmin.experimental as experimental

from nlp_factories import FACTORIES

GATED_SOLVERS = {
    "BobyqaSolver": "bobyqa",
    "CmaesSolver": "cmaes",
    "CobylaSolver": "cobyla",
}
GATED_OPTIONS = (
    "BobyqaOptions",
    "CmaesDetectionOptions",
    "CmaesOptions",
    "CmaesRestartStrategy",
    "CobylaOptions",
    "TrustRegionOptions",
)
GATED_TYPES = tuple(GATED_SOLVERS) + GATED_OPTIONS
PLANNING_IDENTIFIER = re.compile(r"SEED-\d+|[Pp]hase \d")

PROBE = """
import sys, json, argmin
names = json.loads(sys.argv[1])
reachable = [name for name in names if hasattr(argmin, name) or name in argmin.__all__]
print(json.dumps({"loaded": "argmin.experimental" in sys.modules, "reachable": reachable}))
"""


@pytest.fixture(scope="module")
def plain_import(repository_root):
    environment = dict(
        os.environ, PYTHONPATH=str(repository_root / "python"), PYTHONNOUSERSITE="1"
    )
    completed = subprocess.run(
        [sys.executable, "-c", PROBE, json.dumps(GATED_TYPES)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_a_plain_import_leaves_the_submodule_unloaded(plain_import):
    assert plain_import["loaded"] is False


@pytest.mark.parametrize("name", GATED_TYPES)
def test_a_plain_import_cannot_reach_a_gated_type(plain_import, name):
    assert name not in plain_import["reachable"]
    with pytest.raises(AttributeError):
        getattr(argmin, name)


@pytest.mark.parametrize("name", GATED_TYPES)
def test_a_deliberate_import_reaches_a_gated_type(name):
    assert name in experimental.__all__
    assert inspect.isclass(getattr(experimental, name))


@pytest.mark.parametrize("name", sorted(GATED_SOLVERS))
def test_a_gated_solver_states_its_defect(name):
    solver = getattr(experimental, name)
    assert solver.maturity == "experimental"
    assert isinstance(solver.known_defect, str) and solver.known_defect.strip()
    assert solver.known_defect in solver.__doc__


@pytest.mark.parametrize("name", sorted(GATED_SOLVERS))
def test_the_bindings_document_states_the_same_defect(repository_root, name):
    document = (repository_root / "docs" / "python-bindings.md").read_text()
    rows = [line for line in document.splitlines() if line.startswith(f"| `{name}` |")]
    assert len(rows) == 1
    assert f"| {getattr(experimental, name).known_defect} |" in rows[0]


@pytest.mark.parametrize("name", sorted(GATED_SOLVERS))
def test_a_known_defect_names_no_planning_identifier(name):
    solver = getattr(experimental, name)
    assert not PLANNING_IDENTIFIER.search(solver.known_defect)
    assert not PLANNING_IDENTIFIER.search(solver.__doc__)


def test_no_top_level_type_claims_to_be_experimental():
    labeled = [
        name
        for name in argmin.__all__
        if getattr(getattr(argmin, name), "maturity", None) == "experimental"
    ]
    assert not labeled, labeled


def test_only_the_gated_solvers_carry_a_maturity_label():
    labeled = {name for name in GATED_TYPES if hasattr(getattr(experimental, name), "maturity")}
    assert labeled == set(GATED_SOLVERS)


@pytest.mark.parametrize("name", sorted(GATED_SOLVERS))
def test_constructing_a_gated_solver_emits_no_warning(name):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FACTORIES[GATED_SOLVERS[name]]()
    assert not caught, [str(warning.message) for warning in caught]


def test_the_submodule_exports_exactly_the_gated_surface():
    assert sorted(experimental.__all__) == sorted(GATED_TYPES + ("solve_native",))


TOP_LEVEL_PROBLEMS = {
    "bobyqa": lambda: argmin.Rosenbrock(2),
    "cmaes": lambda: argmin.Rosenbrock(2),
    "cobyla": lambda: argmin.ConstrainedTestProblem(),
}


@pytest.mark.parametrize("method", sorted(TOP_LEVEL_PROBLEMS))
def test_the_top_level_native_solve_points_a_gated_method_at_the_gate(method):
    with pytest.raises(argmin.ArgminError) as raised:
        TOP_LEVEL_PROBLEMS[method]().solve_native(method)
    assert raised.value.kind == argmin.ErrorKind.invalid_problem
    assert "argmin.experimental.solve_native" in str(raised.value)


@pytest.mark.parametrize("method", sorted(TOP_LEVEL_PROBLEMS))
def test_the_gated_native_solve_runs_each_gated_method(method):
    result = experimental.solve_native(TOP_LEVEL_PROBLEMS[method](), method, max_iterations=2000)
    assert isinstance(result, argmin.SolveResult)
    assert result.iterations > 0


def test_the_gated_native_solve_reaches_the_rosenbrock_optimum():
    result = experimental.solve_native(argmin.Rosenbrock(2), "bobyqa", max_iterations=2000)
    assert result.objective_value < 1e-6, result


def test_the_gated_native_solve_rejects_a_method_its_problem_does_not_take():
    with pytest.raises(argmin.ArgminError) as raised:
        experimental.solve_native(argmin.ConstrainedTestProblem(), "bobyqa")
    assert raised.value.kind == argmin.ErrorKind.invalid_problem
