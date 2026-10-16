"""The defaults parser's own extensions, each shown to discriminate.

The sweep proves a default equal to its header only as far as the parser reads
the header correctly. A resolver that quietly mapped an initializer to the wrong
value would let the sweep agree with itself, so each extension is checked here
on its own input and against a perturbed bound value it must refuse.
"""

import pytest

import argmin

from defaults_table import TABLE
from defaults_parser import EnumMember, Unresolved, check_source, recover, resolve


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


def tr_sqp_source():
    (source,) = TABLE["TrSqpOptions"].sources
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


def test_a_fixed_width_brace_cast_resolves_to_its_literal():
    assert resolve("std::size_t{0}", {}) == 0
    assert resolve("std::uint16_t{50}", {}) == 50


def test_a_mode_conditional_default_resolves_to_its_accurate_branch(repository_root):
    _, constants = recover(repository_root, tr_sqp_source())
    assert resolve(constants["default_multiplier_reest_every_k"], constants) == 1
    assert resolve(constants["default_max_cg_iterations_multiplier"], constants) == 2
    assert resolve(constants["default_soc_max_iterations"], constants) == 0


@pytest.mark.parametrize(
    "expression",
    [
        "(Mode == sqp_mode::accurate) ? std::size_t{1} : std::size_t{2}",
        "(N == dynamic_dimension) ? std::size_t{1} : std::size_t{2}",
        "(Mode == sqp_mode::fast) ? std::size_t{1}",
    ],
)
def test_any_other_conditional_stays_unresolved(expression):
    assert isinstance(resolve(expression, {}), Unresolved)


def test_the_unperturbed_mode_conditional_defaults_pass_the_sweep_check(repository_root):
    bound = Perturbed(argmin.TrSqpOptions())
    check_source(repository_root, bound, tr_sqp_source(), (), "TrSqpOptions.")


def test_a_default_perturbed_to_the_fast_branch_is_reported_as_a_mismatch(repository_root):
    bound = Perturbed(argmin.TrSqpOptions(), max_cg_iterations_multiplier=1)
    with pytest.raises(AssertionError, match="max_cg_iterations_multiplier is 1, header says 2"):
        check_source(repository_root, bound, tr_sqp_source(), (), "TrSqpOptions.")
