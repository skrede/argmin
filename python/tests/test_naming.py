"""Every bound Python type is named after the C++ type it binds.

The rule: ``x_policy`` binds as ``XSolver``, ``x_policy::options_type`` as
``XOptions``, ``x_policy::foo`` as ``XFoo``, and any other ``foo_bar`` as
``FooBar``, with namespace qualifiers and template arguments dropped. No
runtime introspection recovers a C++ spelling from a bound class, so the
spellings live in a table, and the table is checked against the binding sources
so that a typo cannot agree with itself.
"""

import inspect
import pathlib
import re

import pytest

import argmin

BINDINGS = pathlib.Path(__file__).resolve().parents[1] / "src" / "bindings"

CPP_TYPE = {
    "ArgminError": "argmin_error",
    "BobyqaOptions": "bobyqa_policy::options_type",
    "BobyqaSolver": "bobyqa_policy",
    "CmaesDetectionOptions": "cmaes_detection_options",
    "CmaesOptions": "cmaes_policy::options_type",
    "CmaesRestartStrategy": "cmaes_policy::restart_strategy",
    "CmaesSolver": "cmaes_policy",
    "CobylaOptions": "cobyla_policy::options_type",
    "CobylaSolver": "cobyla_policy",
    "ConstrainedTestProblem": "constrained_test_problem",
    "DenseActiveSetQpOptions": "dense_active_set_qp_options",
    "DenseActiveSetQpSolver": "dense_active_set_qp_solver",
    "DenseAdmmQpSolver": "dense_admm_qp_solver",
    "DenseQpOptions": "dense_qp_options",
    "ErrorKind": "error_kind",
    "FilterSlsqpOptions": "filter_slsqp_policy::options_type",
    "FilterSlsqpSolver": "filter_slsqp_policy",
    "KraftSlsqpOptions": "kraft_slsqp_policy::options_type",
    "KraftSlsqpSolver": "kraft_slsqp_policy",
    "LbfgsbLineSearch": "lbfgsb_line_search",
    "LbfgsbOptions": "lbfgsb_policy::options_type",
    "LbfgsbSolver": "lbfgsb_policy",
    "LineSearchOptions": "line_search_options",
    "QpOptions": "qp_options",
    "QpResult": "qp_result",
    "QpSolveStatus": "qp_solve_status",
    "RestorationStrategy": "detail::restoration_strategy",
    "Rosenbrock": "rosenbrock",
    "SolveResult": "solve_result",
    "SolverOptions": "solver_options",
    "SolverStatus": "solver_status",
    "SparseAdmmQpSolver": "sparse_admm_qp_solver",
    "SparseQpOptions": "sparse_qp_options",
    "TrSqpOptions": "tr_sqp_policy::options_type",
    "TrSqpSolver": "tr_sqp_policy",
    "TrustRegionOptions": "trust_region_options",
}

RENAMED = {
    "SlsqpSolver": "KraftSlsqpSolver",
    "SlsqpOptions": "KraftSlsqpOptions",
    "QpSubproblemOptions": "QpOptions",
    "QpStatus": "QpSolveStatus",
    "CmaesRestart": "CmaesRestartStrategy",
}

NAMESPACE_QUALIFIERS = ("argmin", "python", "detail")


def camel_case(spelling):
    return "".join(word.capitalize() for word in spelling.split("_"))


def expected_name(cpp):
    parts = [part for part in cpp.split("::") if part not in NAMESPACE_QUALIFIERS]
    head, *nested = parts
    if not head.endswith("_policy"):
        return camel_case("_".join(parts))
    stem = camel_case(head.removesuffix("_policy"))
    if not nested:
        return stem + "Solver"
    if nested == ["options_type"]:
        return stem + "Options"
    return stem + camel_case("_".join(nested))


def binding_sources():
    return "\n".join(
        path.read_text() for path in sorted(BINDINGS.rglob("*")) if path.suffix in (".cpp", ".h")
    )


def quoted(name):
    return re.compile(r'"(?:argmin\.)?' + re.escape(name) + '"')


def spelled(cpp):
    parts = [re.escape(part) + r"(?:<[^<>]*>)?" for part in cpp.split("::")]
    return re.compile(r"\b" + "::".join(parts) + r"\b")


def exported_types():
    return {
        name
        for name in argmin.__all__
        if inspect.isclass(getattr(argmin, name, None))
    }


@pytest.mark.parametrize("name", sorted(CPP_TYPE))
def test_the_name_follows_the_rule(name):
    assert expected_name(CPP_TYPE[name]) == name


@pytest.mark.parametrize("name", sorted(CPP_TYPE))
def test_the_name_is_exported(name):
    assert name in argmin.__all__
    assert inspect.isclass(getattr(argmin, name, None))


@pytest.mark.parametrize("old", sorted(RENAMED))
def test_the_old_name_is_gone(old):
    assert old not in argmin.__all__
    assert not hasattr(argmin, old)
    assert RENAMED[old] in CPP_TYPE


@pytest.mark.parametrize("name", sorted(CPP_TYPE))
def test_both_spellings_occur_in_the_binding_sources(name):
    sources = binding_sources()
    assert quoted(name).search(sources), f'"{name}" is not registered under {BINDINGS}'
    assert spelled(CPP_TYPE[name]).search(sources), f"{CPP_TYPE[name]} is not bound"


def test_every_exported_type_has_an_entry():
    exported = exported_types()
    assert exported - set(CPP_TYPE) == set()
    assert set(CPP_TYPE) - exported == set()


def test_the_rule_discriminates():
    assert expected_name("kraft_slsqp_policy") == "KraftSlsqpSolver"
    assert expected_name("kraft_slsqp_policy::options_type") == "KraftSlsqpOptions"
    assert expected_name("cmaes_policy::restart_strategy") == "CmaesRestartStrategy"
    assert expected_name("detail::restoration_strategy") == "RestorationStrategy"
    assert expected_name("kraft_slsqp_policy") != "SlsqpSolver"
    assert expected_name("qp_solve_status") != "QpStatus"


def test_the_source_check_discriminates():
    sources = binding_sources()
    assert not quoted("AnUnboundTypeName").search(sources)
    assert not spelled("an_unbound_policy::options_type").search(sources)
    assert spelled("kraft_slsqp_policy::options_type").search("kraft_slsqp_policy<>::options_type")
