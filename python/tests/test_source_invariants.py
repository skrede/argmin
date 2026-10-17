"""Source-level invariants that no runtime test can enforce.

Two properties of this binding tree are enforced by inspecting the sources rather
than by exercising the module, because in both cases a runtime test provably
cannot see the regression.

The formatting invariant is the sharper of the two. A sibling project shipped a
crash in which locale-dependent stream formatting segfaulted numpy when two
standard-library runtimes were loaded in one process. A runtime test cannot catch
its reintroduction here: the interpreter's ``locale.setlocale`` moves only the C
locale, while C++ streams format through the C++ global locale, which stays
classic -- so a reinstated ``ostringstream`` still renders a decimal point and a
runtime assertion still passes. This scan is the enforcement.
"""

from __future__ import annotations

import pathlib
import re

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

# Every construct that would route a number through the C++ locale facets, or
# that would open a second formatting path outside the single approved helper.
BANNED_FORMATTING = re.compile(
    r"\bstd::(?:ostringstream|istringstream|stringstream|ostream|locale|num_put|setprecision)\b"
    r"|\bimbue\b"
    r"|\bstd::cout\b|\bstd::cerr\b"
    r"|\{:[^}]*L[^}]*\}"
)

FORMATTER = "format_number.h"


def _binding_sources() -> list[pathlib.Path]:
    return sorted(p for p in SRC.rglob("*") if p.suffix in {".h", ".cpp"})


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def test_the_scan_sees_a_populated_tree() -> None:
    """A scan over an empty tree would pass vacuously and prove nothing."""
    sources = _binding_sources()
    assert len(sources) >= 8, f"expected a populated binding tree, found {sources}"
    assert any(p.name == FORMATTER for p in sources)


@pytest.mark.parametrize("path", _binding_sources(), ids=lambda p: p.name)
def test_no_locale_dependent_formatting(path: pathlib.Path) -> None:
    body = _strip_comments(path.read_text(encoding="utf-8"))
    hits = sorted({m.group(0) for m in BANNED_FORMATTING.finditer(body)})
    assert not hits, (
        f"{path.name} reintroduces locale-dependent or unapproved formatting: {hits}. "
        f"Numbers must go through {FORMATTER}; a runtime locale test cannot catch this."
    )


def test_the_formatting_scan_would_catch_a_regression() -> None:
    """The scan is only worth having if it rejects the thing it forbids."""
    planted = (
        "#include <sstream>\n"
        "std::string render(double v)\n"
        "{\n"
        "    std::ostringstream out;\n"
        "    out << v;\n"
        "    return out.str();\n"
        "}\n"
    )
    assert BANNED_FORMATTING.search(_strip_comments(planted)), (
        "the scan fails to reject a plain ostringstream helper, so it enforces nothing"
    )


def test_a_single_formatting_helper_exists() -> None:
    helpers = [p for p in _binding_sources() if p.name == FORMATTER]
    assert len(helpers) == 1, f"expected exactly one formatting helper, found {helpers}"


def test_the_adapter_concept_assertions_are_compiled_in() -> None:
    """The adapter tiers must not over-satisfy the concept ladder.

    The assertions live in the adapter header itself rather than in a verify
    command, so a tier that silently grows a capability fails the build instead
    of passing unnoticed until someone reruns a checklist.
    """
    adapter = SRC / "bindings" / "detail" / "problem_adapter.h"
    body = adapter.read_text(encoding="utf-8")
    positive = body.count("static_assert(argmin::")
    negative = body.count("static_assert(!argmin::")
    assert negative >= 5, (
        f"expected at least five non-satisfaction assertions pinning the tiers, found {negative}. "
        "The negative half is the load-bearing one: it is what stops an adapter advertising a "
        "capability the caller never supplied."
    )
    assert positive >= 5, (
        f"expected each of the five tiers to assert what it does satisfy, found {positive}"
    )
