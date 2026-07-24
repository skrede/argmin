"""Every exported name is reachable, documented, and covered by a test.

The discipline this phase set out to keep is that no surface ships untested. A
discipline nobody can fail is a good intention, so this module makes it a
build-breaking rule: an export that no other test module mentions fails here,
and so does an export with no docstring for a caller to read.

The allowance list is the escape hatch, and it is deliberately visible: an
entry needs a one-line reason, and the reason is asserted to be there.
"""

import inspect
import pathlib

import pytest

import argmin

THIS_MODULE = pathlib.Path(__file__).name

# Names that legitimately need no behavioral test of their own. One line of
# reason each, so an unjustified entry is obvious on sight.
NO_BEHAVIORAL_TEST = {
    "__version__": "a string the build injects; the import module already asserts it "
    "matches the version in the top-level build file",
}


def test_dependencies():
    assert argmin.__all__, "the package exports nothing"


@pytest.mark.parametrize("name", sorted(argmin.__all__))
def test_every_exported_name_resolves(name):
    assert hasattr(argmin, name), f"{name} is exported but not importable"
    assert getattr(argmin, name) is not None


@pytest.mark.parametrize("name", sorted(argmin.__all__))
def test_every_exported_type_and_function_is_documented(name):
    value = getattr(argmin, name)
    if not (inspect.isclass(value) or inspect.isroutine(value)):
        assert name in NO_BEHAVIORAL_TEST, (
            f"{name} is neither a class nor a function and is not on the allowance list"
        )
        return
    documentation = (value.__doc__ or "").strip()
    assert documentation, f"{name} carries no docstring"
    assert documentation != name, f"{name}'s docstring only restates its name"


def other_test_sources():
    directory = pathlib.Path(__file__).parent
    return {
        path.name: path.read_text()
        for path in sorted(directory.glob("test_*.py"))
        if path.name != THIS_MODULE
    }


@pytest.mark.parametrize("name", sorted(argmin.__all__))
def test_every_exported_name_is_referenced_by_a_test(name):
    if name in NO_BEHAVIORAL_TEST:
        pytest.skip(f"allowed: {NO_BEHAVIORAL_TEST[name]}")
    mentions = [
        module for module, text in other_test_sources().items() if name in text
    ]
    assert mentions, (
        f"{name} is exported but no other test module mentions it. Add a test, or "
        "add it to the allowance list with a reason."
    )


def test_every_allowance_carries_a_reason():
    for name, reason in NO_BEHAVIORAL_TEST.items():
        assert name in argmin.__all__, f"{name} is allowed but no longer exported"
        assert reason.strip(), f"{name} is allowed with no reason"


# The reference check is only worth something if it can fail, and it has only
# ever been evaluated on names that pass.
def test_the_reference_check_is_discriminating():
    sources = other_test_sources()
    assert sources, "no other test modules were found to search"
    absent = "AnExportNobodyHasTestedYet"
    assert not [module for module, text in sources.items() if absent in text]


def test_the_export_list_matches_the_module_namespace():
    exported = set(argmin.__all__)
    public = {
        name
        for name in dir(argmin)
        if not name.startswith("_") and name != "annotations"
    }
    public.discard("argmin")
    assert public <= exported, sorted(public - exported)
