"""The bindings document names exactly the exported surface and the baseline's figures.

``docs/python-bindings.md`` is where a caller learns what is bound, what is gated
behind ``argmin.experimental`` and why. These checks fail when an export goes
undocumented, when the document names a type the extension does not export, when
a gated solver is listed as bound, or when a cited publication-cohort figure no
longer matches the committed baseline.
"""

import builtins
import collections
import csv
import pathlib
import re

import pytest

import argmin
import argmin.experimental as experimental

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCUMENT = REPOSITORY_ROOT / "docs" / "python-bindings.md"
BASELINE = REPOSITORY_ROOT / "benchmarks" / "baselines" / "publication-regression.csv"

BOUND_SECTION = "What is bound"
GATED_SECTION = "Gated: argmin.experimental"
SELECTION_ROWS = ("kraft_slsqp_accurate", "filter_slsqp_accurate", "tr_sqp_accurate", "tr_sqp_fast")
TYPE_SUFFIXES = ("Solver", "Options", "Status", "Strategy", "Result", "Error", "Kind", "LineSearch")
FIXTURE_TYPES = ("Rosenbrock", "ConstrainedTestProblem")

CODE_SPAN = re.compile(r"`([^`\n]+)`")
CAPITALIZED = re.compile(r"\b[A-Z][A-Za-z0-9]*\b")
FIGURE = re.compile(r"`([a-z0-9_]+)` (\d+) pass / (\d+) expected_fail(?: / (\d+) excluded)?")
BARE_FIGURE = re.compile(r"\d+ pass / \d+ expected_fail")
PLANNING_ID = re.compile(r"SEED-\d|[Pp]hase \d|\bD-\d\d\b|PYB-\d")

EXPORTED = sorted(
    name for name in set(argmin.__all__) | set(experimental.__all__) if not name.startswith("__")
)
TOP_LEVEL_SOLVERS = {name for name in argmin.__all__ if name.endswith("Solver")}
GATED_SOLVERS = {name for name in experimental.__all__ if name.endswith("Solver")}


def sections(text):
    parts = re.split(r"^## (.+)$", text, flags=re.MULTILINE)
    return dict(zip(parts[1::2], parts[2::2]))


def code_identifiers(text):
    return {word for span in CODE_SPAN.findall(text) for word in CAPITALIZED.findall(span)}


def missing_exports(text, exported):
    spans = " ".join(CODE_SPAN.findall(text))
    return [name for name in exported if not re.search(rf"\b{re.escape(name)}\b", spans)]


def unknown_types(text, exported):
    candidates = {
        word
        for word in code_identifiers(text)
        if (word.endswith(TYPE_SUFFIXES) or word in FIXTURE_TYPES) and not hasattr(builtins, word)
    }
    return sorted(candidates - set(exported))


def table_names(section):
    names = set()
    for line in section.splitlines():
        match = re.match(r"\| `([A-Za-z0-9]+)", line)
        if match:
            names.add(match.group(1))
    return names


def tally(path):
    lines = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    counts = collections.defaultdict(collections.Counter)
    for row in csv.DictReader(lines):
        counts[row["solver"]][row["disposition"]] += 1
    return counts


def figure_errors(text, counts):
    errors = []
    figures = list(FIGURE.finditer(text))
    if len(figures) != len(BARE_FIGURE.findall(text)):
        errors.append("a cohort figure is not in the form `<row>` N pass / M expected_fail")
    for match in figures:
        row, passed, failed, excluded = match.groups()
        if row not in counts:
            errors.append(f"{row} is not a solver in the baseline")
            continue
        actual = counts[row]
        if (int(passed), int(failed)) != (actual["pass"], actual["expected_fail"]):
            errors.append(f"{match.group(0)} disagrees with the baseline {dict(actual)}")
        if excluded is None and actual["excluded"]:
            errors.append(f"{match.group(0)} omits its {actual['excluded']} excluded cells")
        if excluded is not None and int(excluded) != actual["excluded"]:
            errors.append(f"{match.group(0)} disagrees with the baseline {dict(actual)}")
    return errors


@pytest.fixture(scope="module")
def document():
    return DOCUMENT.read_text()


@pytest.fixture(scope="module")
def parts(document):
    return sections(document)


@pytest.fixture(scope="module")
def counts():
    return tally(BASELINE)


def test_every_export_is_named_in_the_document(document):
    assert missing_exports(document, EXPORTED) == []


def test_every_documented_type_is_exported(document):
    assert unknown_types(document, EXPORTED) == []


def test_the_bound_tables_name_exactly_the_top_level_solvers(parts):
    assert table_names(parts[BOUND_SECTION]) & GATED_SOLVERS == set()
    assert {name for name in table_names(parts[BOUND_SECTION]) if name.endswith("Solver")} == (
        TOP_LEVEL_SOLVERS
    )


def test_the_gated_section_documents_exactly_the_gated_solvers(parts):
    gated = parts[GATED_SECTION]
    assert "argmin.experimental" in gated
    assert table_names(gated) == GATED_SOLVERS
    assert GATED_SOLVERS <= code_identifiers(gated)


def test_every_cited_cohort_figure_matches_the_baseline(document, counts):
    assert figure_errors(document, counts) == []


@pytest.mark.parametrize("row", SELECTION_ROWS)
def test_the_selection_evidence_is_cited_where_the_solvers_are_listed(parts, row):
    cited = {match.group(1) for match in FIGURE.finditer(parts[BOUND_SECTION])}
    assert row in cited


def test_the_trust_region_entry_states_its_bound_aware_residual(parts):
    paragraphs = parts[BOUND_SECTION].split("\n\n")
    entries = [paragraph for paragraph in paragraphs if "`TrSqpSolver`" in paragraph]
    assert entries
    assert any("bound-aware" in paragraph for paragraph in entries)


def test_no_planning_identifier_appears(document):
    hits = [line for line in document.splitlines() if PLANNING_ID.search(line)]
    assert hits == []


def test_a_missing_export_is_reported(document):
    name = EXPORTED[0]
    assert missing_exports(re.sub(rf"\b{name}\b", "Removed", document), EXPORTED) == [name]


def test_an_unexported_type_is_reported(document):
    assert unknown_types(document + "\nAlso `FooSolver(n)`.\n", EXPORTED) == ["FooSolver"]
    assert unknown_types("`ValueError` and `KeyError`", EXPORTED) == []


def test_a_gated_solver_in_the_bound_table_is_reported():
    gated = sorted(GATED_SOLVERS)[0]
    assert table_names(f"| name | method |\n| --- | --- |\n| `{gated}` | x |\n") & GATED_SOLVERS


def test_a_wrong_cohort_figure_is_reported(counts):
    actual = counts["filter_slsqp_accurate"]
    passed, failed = actual["pass"] + 1, actual["expected_fail"]
    wrong = f"`filter_slsqp_accurate` {passed} pass / {failed} expected_fail"
    assert figure_errors(wrong, counts)


def test_an_excluded_count_left_out_is_reported(counts):
    row = next(name for name, actual in sorted(counts.items()) if actual["excluded"])
    actual = counts[row]
    cited = f"`{row}` {actual['pass']} pass / {actual['expected_fail']} expected_fail"
    assert figure_errors(cited, counts)
    assert figure_errors(f"{cited} / {actual['excluded']} excluded", counts) == []


def test_a_figure_off_the_canonical_form_is_reported(counts):
    assert figure_errors("`cobyla`\n23 pass / 4 expected_fail", counts)
