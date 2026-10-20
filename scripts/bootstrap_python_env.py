#!/usr/bin/env python3
"""Create the repository-local interpreter environment for the binding suite.

Contract, and the reason this script exists at all: the package installer is
only ever invoked through the interpreter this script just created inside the
target directory. No host interpreter is written to, on any path through this
code, including the failure paths -- the checks below exit before an environment
is created rather than falling back to whatever interpreter is on PATH.

Distributions are installed binary-only, so no dependency's build script runs on
this machine.

The last line printed is the absolute path of the created interpreter; the build
configures against exactly that path.
"""

import sys
import venv
import argparse
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENVIRONMENT = REPOSITORY_ROOT / ".venv"
REQUIREMENTS = REPOSITORY_ROOT / "python" / "requirements.txt"


def interpreter_in(environment: Path) -> Path:
    if sys.platform == "win32":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def refuse(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "environment",
        nargs="?",
        type=Path,
        default=DEFAULT_ENVIRONMENT,
        help=f"directory to create the environment in (default: {DEFAULT_ENVIRONMENT})",
    )
    arguments = parser.parse_args()

    environment = arguments.environment.resolve()
    launcher = Path(sys.executable).resolve()

    if launcher.is_relative_to(environment):
        refuse(
            f"this script is running from inside {environment}; "
            "run it with an interpreter outside the target environment"
        )

    if environment.exists() and not (environment / "pyvenv.cfg").is_file():
        refuse(
            f"{environment} exists and is not an environment "
            "(no pyvenv.cfg); refusing to write into it"
        )

    if not REQUIREMENTS.is_file():
        refuse(f"{REQUIREMENTS} is missing")

    venv.EnvBuilder(
        system_site_packages=False,
        with_pip=True,
        clear=False,
    ).create(environment)

    python = interpreter_in(environment)
    if not python.is_file():
        refuse(f"the environment was created but {python} does not exist")

    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--only-binary=:all:",
            "--requirement",
            str(REQUIREMENTS),
        ],
        check=True,
    )

    print(python)


if __name__ == "__main__":
    main()
