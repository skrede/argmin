"""The trust-region solver's own keywords: refused when unusable, visible when accepted."""

import math

import pytest

import argmin

from nlp_factories import make_tr_sqp

KEYWORDS = pytest.mark.parametrize("keyword", ["initial_trust_radius", "initial_penalty"])


@KEYWORDS
@pytest.mark.parametrize("value", [0.0, -1.0], ids=["zero", "negative"])
def test_a_non_positive_value_is_rejected(keyword, value):
    with pytest.raises(argmin.ArgminError) as raised:
        make_tr_sqp(**{keyword: value})
    assert raised.value.kind == argmin.ErrorKind.invalid_problem
    assert keyword in str(raised.value)


@KEYWORDS
@pytest.mark.parametrize("value", [math.nan, math.inf], ids=["nan", "inf"])
def test_a_non_finite_value_is_rejected(keyword, value):
    with pytest.raises(argmin.ArgminError) as raised:
        make_tr_sqp(**{keyword: value})
    assert raised.value.kind == argmin.ErrorKind.non_finite_input
    assert keyword in str(raised.value)


@KEYWORDS
def test_an_accepted_value_is_visible_in_the_policy_snapshot(keyword):
    default = getattr(argmin.TrSqpOptions(), keyword)
    chosen = default * 0.25
    assert getattr(make_tr_sqp(**{keyword: chosen}).policy_options(), keyword) == chosen


@KEYWORDS
def test_an_unset_keyword_keeps_the_library_default(keyword):
    snapshot = make_tr_sqp().policy_options()
    assert getattr(snapshot, keyword) == getattr(argmin.TrSqpOptions(), keyword)

