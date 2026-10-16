"""The defaults parser's own extensions, each shown to discriminate.

The sweep proves a default equal to its header only as far as the parser reads
the header correctly. A resolver that quietly mapped an initializer to the wrong
value would let the sweep agree with itself, so each extension is checked here
on its own input and against a perturbed bound value it must refuse.
"""

import pytest

import argmin

from defaults_table import TABLE
from defaults_parser import EnumMember, Unresolved, check_source, resolve


class Perturbed:
    def __init__(self, bound, **replaced):
        self._bound = bound
        self._replaced = replaced

    def __getattr__(self, name):
        if name in self._replaced:
            return self._replaced[name]
        return getattr(self._bound, name)


def filter_source():
    (source,) = TABLE["FilterSlsqpOptions"].sources
    return source


def test_a_multiply_qualified_enum_default_resolves_to_its_member():
    assert resolve("detail::restoration_strategy::hybrid", {}) == EnumMember("hybrid")
    assert resolve("restoration_strategy::hybrid", {}) == EnumMember("hybrid")


def test_a_qualified_name_that_is_not_an_enumerator_stays_unresolved():
    assert isinstance(resolve("detail::restoration_strategy::hybrid()", {}), Unresolved)


def test_the_unperturbed_enum_default_passes_the_sweep_check(repository_root):
    bound = Perturbed(argmin.FilterSlsqpOptions())
    check_source(repository_root, bound, filter_source(), (), "FilterSlsqpOptions.")


def test_a_perturbed_enum_default_is_reported_as_a_mismatch(repository_root):
    bound = Perturbed(
        argmin.FilterSlsqpOptions(), restoration=argmin.RestorationStrategy.l1_penalty
    )
    with pytest.raises(AssertionError, match="restoration is l1_penalty, header says hybrid"):
        check_source(repository_root, bound, filter_source(), (), "FilterSlsqpOptions.")
