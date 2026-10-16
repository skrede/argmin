"""Integer count options reach unsigned C++ fields; an out-of-range value must be
refused with a typed error rather than wrapped into a different count."""

import pytest

import argmin
import argmin.experimental as experimental

from nlp_factories import FACTORIES, make_cmaes, make_lbfgsb

UINT16_MAX = 65535
INT_MAX = 2**31 - 1

ALL_METHODS = pytest.mark.parametrize("method", sorted(FACTORIES))
STALL_KEYWORDS = pytest.mark.parametrize("keyword", ["stall_window", "policy_stall_window"])
OUT_OF_UINT16 = pytest.mark.parametrize("value", [UINT16_MAX + 1, UINT16_MAX + 10, -1, 0])

NATIVE_SOLVES = {
    "rosenbrock": lambda **kw: argmin.Rosenbrock(2).solve_native("lbfgsb", **kw),
    "constrained": lambda **kw: argmin.ConstrainedTestProblem().solve_native("kraft_slsqp", **kw),
    "experimental": lambda **kw: experimental.solve_native(argmin.Rosenbrock(2), "bobyqa", **kw),
}
NATIVE = pytest.mark.parametrize("entry", sorted(NATIVE_SOLVES))


def assert_rejected(raised, keyword):
    assert raised.value.kind == argmin.ErrorKind.dimension_mismatch
    assert keyword in str(raised.value)


@ALL_METHODS
def test_the_largest_policy_stall_window_is_kept_exactly(method):
    solver = FACTORIES[method](policy_stall_window=UINT16_MAX)
    assert solver.policy_options().stall_window == UINT16_MAX
    assert solver.options().stall_window == UINT16_MAX


@ALL_METHODS
def test_the_largest_driver_stall_window_is_accepted(method):
    assert FACTORIES[method](stall_window=UINT16_MAX).n == 2


@ALL_METHODS
@STALL_KEYWORDS
@OUT_OF_UINT16
def test_a_stall_window_outside_the_field_range_is_rejected(method, keyword, value):
    with pytest.raises(argmin.ArgminError) as raised:
        FACTORIES[method](**{keyword: value})
    assert_rejected(raised, keyword)


@NATIVE
def test_a_native_solve_accepts_the_largest_stall_window(entry):
    result = NATIVE_SOLVES[entry](stall_window=UINT16_MAX, max_iterations=5)
    assert result.iterations <= 5


@NATIVE
@OUT_OF_UINT16
def test_a_native_solve_rejects_a_stall_window_outside_the_field_range(entry, value):
    with pytest.raises(argmin.ArgminError) as raised:
        NATIVE_SOLVES[entry](stall_window=value)
    assert_rejected(raised, "stall_window")


# The 32-bit counts arrive through a C int, so their own maximum is not reachable;
# the binding refuses anything past the int range before the check runs.
@pytest.mark.parametrize("value", [-1, 0])
@pytest.mark.parametrize(
    "keyword, build",
    [
        ("max_iterations", lambda v: make_lbfgsb(max_iterations=v)),
        ("population_size", lambda v: make_cmaes(population_size=v)),
        ("budget", lambda v: make_lbfgsb().step_n(v)),
    ],
)
def test_a_32_bit_count_that_is_not_positive_is_rejected(keyword, build, value):
    with pytest.raises(argmin.ArgminError) as raised:
        build(value)
    assert_rejected(raised, keyword)


@pytest.mark.parametrize("keyword", ["max_iterations", "population_size"])
def test_a_32_bit_count_past_the_int_range_is_refused_not_wrapped(keyword):
    with pytest.raises(TypeError):
        (make_lbfgsb if keyword == "max_iterations" else make_cmaes)(**{keyword: INT_MAX + 1})


def test_the_largest_int_iteration_count_is_kept_exactly():
    assert make_lbfgsb(max_iterations=INT_MAX).options().max_iterations == INT_MAX
