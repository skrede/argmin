"""A bounds keyword the constrained solver cannot apply is refused, not ignored."""

import numpy as np
import pytest

import argmin

from nlp_factories import FACTORIES

REJECTS_BOUNDS = ("filter_slsqp",)

LOWER = np.array([-1.0, -1.0])
UPPER = np.array([1.0, 1.0])

BOUND_KEYWORDS = pytest.mark.parametrize(
    "bounds",
    [
        {"lower_bounds": LOWER},
        {"upper_bounds": UPPER},
        {"lower_bounds": lambda: LOWER, "upper_bounds": lambda: UPPER},
    ],
    ids=["lower_array", "upper_array", "callable_pair"],
)


@pytest.mark.parametrize("method", REJECTS_BOUNDS)
@BOUND_KEYWORDS
def test_a_bounds_keyword_is_rejected_with_a_typed_error(method, bounds):
    with pytest.raises(argmin.ArgminError) as raised:
        FACTORIES[method](**bounds)
    assert raised.value.kind == argmin.ErrorKind.invalid_problem
    assert "not supported" in str(raised.value)


@pytest.mark.parametrize("method", REJECTS_BOUNDS)
def test_unset_bounds_construct_normally(method):
    assert FACTORIES[method](lower_bounds=None, upper_bounds=None).n == 2
