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

MINIMUM_RECOVERED_FIELDS = 17

HEADERS = {
    "DenseQpOptions": ("options/dense_qp_options.h", "dense_qp_options"),
    "SparseQpOptions": ("options/sparse_qp_options.h", "sparse_qp_options"),
}


def literal_value(text: str):
    stripped = text.strip()
    if stripped == "true":
        return True
    if stripped == "false":
        return False
    if any(character in stripped for character in ".eE"):
        return float(stripped)
    return int(stripped)


def committed_defaults(repository_root: pathlib.Path, binding_name: str) -> dict:
    relative, struct = HEADERS[binding_name]
    path = repository_root / "lib" / "argmin" / "include" / "argmin" / relative
    text = path.read_text()
    assert f"struct {struct}" in text, f"{path} declares no {struct}"
    # The body terminator has to be anchored to a line start: an unanchored
    # "};" first matches the end of "rho{0.1};" and truncates the body to
    # nothing, which the recovered-count guard below catches.
    body = text.split(f"struct {struct}", 1)[1].split("\n};", 1)[0]
    return {
        name: literal_value(literal)
        for name, literal in re.findall(r"([a-z_][a-z_0-9]*)\{([^}]*)\}", body)
    }


@pytest.mark.parametrize("binding_name", sorted(HEADERS))
def test_the_parser_recovers_the_whole_roster(repository_root, binding_name):
    recovered = committed_defaults(repository_root, binding_name)
    assert len(recovered) >= MINIMUM_RECOVERED_FIELDS, recovered


@pytest.mark.parametrize("binding_name", sorted(HEADERS))
def test_every_committed_default_is_the_exposed_default(argmin, repository_root, binding_name):
    recovered = committed_defaults(repository_root, binding_name)
    assert len(recovered) >= MINIMUM_RECOVERED_FIELDS, recovered

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


@pytest.mark.parametrize("binding_name", sorted(HEADERS))
def test_a_written_field_reads_back(argmin, binding_name):
    options = getattr(argmin, binding_name)()
    options.max_iterations = 7
    options.eps_abs = 0.5
    options.polish = not options.polish
    assert options.max_iterations == 7
    assert options.eps_abs == 0.5
    assert getattr(argmin, binding_name)().polish is not options.polish


@pytest.mark.parametrize("binding_name", sorted(HEADERS))
def test_the_repr_carries_every_field(argmin, repository_root, binding_name):
    recovered = committed_defaults(repository_root, binding_name)
    text = repr(getattr(argmin, binding_name)())
    assert text.startswith(binding_name + "(")
    for name in recovered:
        assert name + "=" in text, f"{name} is missing from {text}"
