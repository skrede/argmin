import re
import locale
import subprocess

import numpy as np
import pytest

# A locale is only useful here if the platform actually has it installed, and
# only interesting if its numeric category uses a decimal comma.
DECIMAL_COMMA_CANDIDATES = (
    "de_DE.UTF-8",
    "de_DE.utf8",
    "fr_FR.UTF-8",
    "fr_FR.utf8",
    "nl_NL.UTF-8",
    "es_ES.UTF-8",
    "it_IT.UTF-8",
    "pt_BR.UTF-8",
    "nb_NO.UTF-8",
    "nb_NO.utf8",
    "de_DE",
    "fr_FR",
)


def installed_locales() -> tuple[str, ...]:
    """Every locale the platform will actually accept, best effort.

    The curated list above is a fast path, not an inventory: a host that has
    exactly one decimal-comma locale outside the list would skip this whole
    family and report green, which is how the guarded crash class goes
    untested. Asking the platform closes that hole wherever it can be asked.
    """
    try:
        listing = subprocess.run(
            ["locale", "-a"], capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    return tuple(line.strip() for line in listing.stdout.splitlines() if line.strip())

NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
DIGIT_COMMA_DIGIT = re.compile(r"\d,\d")


@pytest.fixture
def decimal_comma_locale():
    previous = locale.setlocale(locale.LC_ALL)
    for candidate in DECIMAL_COMMA_CANDIDATES + installed_locales():
        try:
            locale.setlocale(locale.LC_ALL, candidate)
        except locale.Error:
            continue
        if locale.localeconv()["decimal_point"] == ",":
            try:
                yield candidate
            finally:
                locale.setlocale(locale.LC_ALL, previous)
            return
        locale.setlocale(locale.LC_ALL, previous)
    pytest.skip("no decimal-comma locale is installed on this platform")


def sample_result(argmin):
    return argmin.QpResult(
        x=np.array([1.5, -2.25, 3.0]),
        y=np.array([0.125]),
        status=argmin.QpStatus.solved,
        iterations=12,
        polished=True,
        primal_residual=1.25e-09,
        dual_residual=2.5e-10,
        objective_value=-1.25,
    )


def test_repr_is_unchanged_under_a_decimal_comma_locale(argmin, decimal_comma_locale):
    assert locale.localeconv()["decimal_point"] == ","

    text = repr(sample_result(argmin))

    assert "objective_value=-1.25" in text
    assert DIGIT_COMMA_DIGIT.search(text) is None
    for number in NUMBER.findall(text):
        float(number)


def test_error_message_numbers_parse_with_the_neutral_parser(argmin, decimal_comma_locale):
    assert locale.localeconv()["decimal_point"] == ","

    with pytest.raises(argmin.ArgminError) as raised:
        argmin.SolveResult(gradient_norm=-0.5)

    message = str(raised.value)
    assert "-0.5" in message
    numbers = NUMBER.findall(message)
    assert numbers
    for number in numbers:
        float(number)


# The recovered failure was a crash inside the array library, not wrong text, so
# the array library has to be driven after the switch rather than merely present.
def test_the_array_library_survives_the_locale_change(decimal_comma_locale):
    values = np.array([1.5, 2.5, 3.0])

    assert "1.5" in repr(values)
    assert "1.5" in str(values)
    assert float(values.sum()) == 7.0
    assert (values * 2.0).tolist() == [3.0, 5.0, 6.0]
