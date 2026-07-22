import re
import pathlib


def project_version(repository_root: pathlib.Path) -> str:
    text = (repository_root / "CMakeLists.txt").read_text()
    match = re.search(r"^project\(argmin\s+VERSION\s+([0-9][0-9.]*)\)", text, re.MULTILINE)
    assert match, "the top-level build file declares no project version"
    return match.group(1)


def test_package_imports(argmin):
    assert "__version__" in argmin.__all__
    for name in argmin.__all__:
        assert hasattr(argmin, name), f"{name} is exported but not importable"


def test_version_matches_the_build(argmin, repository_root):
    assert argmin.__version__
    assert argmin.__version__ == project_version(repository_root)


def test_extension_resolves_inside_the_repository(argmin, repository_root):
    extension = pathlib.Path(argmin._argmin.__file__).resolve()
    assert extension.is_relative_to(repository_root)
