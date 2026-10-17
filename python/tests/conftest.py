import pathlib

import pytest

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: a timing measurement, deselectable locally with -m 'not slow'"
    )


@pytest.fixture(scope="session")
def repository_root() -> pathlib.Path:
    return REPOSITORY_ROOT


@pytest.fixture(scope="session")
def argmin():
    import argmin

    return argmin


# A suite that silently imported an installed or stale copy of the package would
# report on something this build never produced, so the copy under test is
# pinned to this repository before any test runs.
@pytest.fixture(scope="session", autouse=True)
def _extension_under_test_is_ours():
    import argmin

    extension = pathlib.Path(argmin._argmin.__file__).resolve()
    assert extension.is_relative_to(REPOSITORY_ROOT), (
        f"the imported extension is {extension}, which is outside {REPOSITORY_ROOT}"
    )
