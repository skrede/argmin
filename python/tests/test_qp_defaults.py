"""Default parity, proven against the committed headers rather than the binding.

A test that read the exposed default and compared it against itself would prove
nothing. Every expected value here is recovered by parsing the library's own
option headers, so a default that drifts away from the struct it is supposed to
mirror fails the suite. The recovered-field-count guards exist because a parser
that silently matched nothing would otherwise pass vacuously, which is the only
way this module can lie.
"""

import re
import pathlib

import pytest

HEADERS = {
    "DenseActiveSetQpOptions": (
        "options/dense_active_set_qp_options.h",
        "dense_active_set_qp_options",
        3,
    ),
    "DenseQpOptions": ("options/dense_qp_options.h", "dense_qp_options", 18),
    "SparseQpOptions": ("options/sparse_qp_options.h", "sparse_qp_options", 18),
}


# A brace default may name a per-scalar table member or wrap a literal in a
# scalar cast. Both resolve to the double row, which is what the double-only
# binding exposes; the float row is never read.
TABLE_REFERENCE = re.compile(r"(\w+)<(?:Scalar|scalar_type|S|double)>::(\w+)")
SCALAR_CAST = re.compile(r"(?:Scalar|scalar_type|S)\((.*)\)")


def double_row(include_root: pathlib.Path, table: str, member: str) -> str:
    specialization = re.compile(r"\bstruct\s+" + re.escape(table) + r"<double>\s*\{(.*?)\n\s*\};", re.DOTALL)
    for path in sorted(include_root.rglob("*.h")):
        body = specialization.search(path.read_text())
        if body:
            row = re.search(
                r"static\s+constexpr\s+[\w:]+\s+" + re.escape(member) + r"\s*=\s*([^;]+);", body.group(1)
            )
            assert row, f"{table}<double> in {path} declares no {member}"
            return row.group(1)
    raise AssertionError(f"no header under {include_root} specializes {table}<double>")


def literal_value(text: str, include_root: pathlib.Path):
    stripped = text.strip()
    reference = TABLE_REFERENCE.fullmatch(stripped)
    if reference:
        return literal_value(double_row(include_root, *reference.groups()), include_root)
    cast = SCALAR_CAST.fullmatch(stripped)
    if cast:
        return literal_value(cast.group(1), include_root)
    if stripped == "true":
        return True
    if stripped == "false":
        return False
    if any(character in stripped for character in ".eE"):
        return float(stripped)
    return int(stripped)


def committed_defaults(repository_root: pathlib.Path, binding_name: str) -> dict:
    relative, struct, _ = HEADERS[binding_name]
    include_root = repository_root / "lib" / "argmin" / "include" / "argmin"
    path = include_root / relative
    text = path.read_text()
    assert f"struct {struct}" in text, f"{path} declares no {struct}"
    # The body terminator has to be anchored to a line start: an unanchored
    # "};" first matches the end of "rho{0.1};" and truncates the body to
    # nothing, which the recovered-count guard below catches.
    body = text.split(f"struct {struct}", 1)[1].split("\n};", 1)[0]
    return {
        name: literal_value(literal, include_root)
        for name, literal in re.findall(r"([a-z_][a-z_0-9]*)\{([^}]*)\}", body)
    }


@pytest.mark.parametrize("binding_name", sorted(HEADERS))
def test_the_parser_recovers_the_whole_roster(repository_root, binding_name):
    recovered = committed_defaults(repository_root, binding_name)
    assert len(recovered) == HEADERS[binding_name][2], recovered


@pytest.mark.parametrize("binding_name", sorted(HEADERS))
def test_every_committed_default_is_the_exposed_default(argmin, repository_root, binding_name):
    recovered = committed_defaults(repository_root, binding_name)
    assert len(recovered) == HEADERS[binding_name][2], recovered

    options = getattr(argmin, binding_name)()
    for name, expected in recovered.items():
        assert hasattr(options, name), f"{binding_name} exposes no {name}"
        actual = getattr(options, name)
        assert actual == expected, f"{binding_name}.{name} is {actual!r}, header says {expected!r}"
        assert type(actual) is type(expected), (name, type(actual), type(expected))


@pytest.mark.parametrize("binding_name", sorted(HEADERS))
def test_a_misspelled_field_is_not_found(argmin, repository_root, binding_name):
    recovered = committed_defaults(repository_root, binding_name)
    options = getattr(argmin, binding_name)()
    for name in recovered:
        misspelled = name + "_"
        assert not hasattr(options, misspelled), (
            f"{binding_name} answers to {misspelled}, so the attribute lookup this "
            "module relies on cannot discriminate"
        )


def changed(value):
    if isinstance(value, bool):
        return not value
    return 7 if isinstance(value, int) else 0.5


@pytest.mark.parametrize("binding_name", sorted(HEADERS))
def test_a_written_field_reads_back(argmin, repository_root, binding_name):
    recovered = committed_defaults(repository_root, binding_name)
    options = getattr(argmin, binding_name)()
    for name, default in recovered.items():
        setattr(options, name, changed(default))
    for name, default in recovered.items():
        assert getattr(options, name) == changed(default), name
        assert getattr(getattr(argmin, binding_name)(), name) == default, name


@pytest.mark.parametrize("binding_name", sorted(HEADERS))
def test_the_repr_carries_every_field(argmin, repository_root, binding_name):
    recovered = committed_defaults(repository_root, binding_name)
    text = repr(getattr(argmin, binding_name)())
    assert text.startswith(binding_name + "(")
    for name in recovered:
        assert name + "=" in text, f"{name} is missing from {text}"
