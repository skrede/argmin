import re
import pathlib

import numpy as np
import pytest

QP_ERROR_BODY = re.compile(r"enum\s+class\s+qp_error\s*:[^{]*\{(.*?)\}", re.DOTALL)


def library_qp_error_names(repository_root: pathlib.Path) -> list[str]:
    header = repository_root / "lib" / "argmin" / "include" / "argmin" / "qp" / "qp_types.h"
    match = QP_ERROR_BODY.search(header.read_text())
    assert match, "the library header declares no qp_error enumeration"
    return [name.strip() for name in match.group(1).split(",") if name.strip()]


def test_the_exception_type_derives_from_the_base_exception(argmin):
    assert issubclass(argmin.ArgminError, Exception)


def test_a_raised_instance_carries_a_bound_kind(argmin):
    with pytest.raises(argmin.ArgminError) as raised:
        argmin.QpResult(x=np.array([1.0, np.nan]))

    assert isinstance(raised.value.kind, argmin.ErrorKind)
    assert raised.value.kind == argmin.ErrorKind.non_finite_input


# A value added to the library's error channel and forgotten here would
# otherwise degrade silently into a generic message.
def test_the_kind_enumeration_covers_the_library_error_channel(argmin, repository_root):
    bound = set(argmin.ErrorKind.__members__)
    library = library_qp_error_names(repository_root)

    assert library
    assert set(library) <= bound


def test_the_answer_channel_is_not_the_failure_channel(argmin):
    for enumeration in (argmin.QpStatus, argmin.SolverStatus):
        for member in enumeration.__members__.values():
            assert not isinstance(member, BaseException)
            assert not (isinstance(member, type) and issubclass(member, BaseException))
