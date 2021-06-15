"""Module for resolving a runfiles path."""
import os
from pathlib import Path

# NOTE(cummins): Moving this file may require updating this relative path.
_PACKAGE_ROOT = Path(os.path.join(os.path.dirname(__file__), "../../../")).resolve(
    strict=True
)

_TESTS_ROOT = Path("/tmp/programl/install_tests")


def runfiles_path(relpath: str) -> Path:  # pragma: no cover
    """Resolve the path to a runfiles data path.

    Use environment variable PROGRAML_RUNFILES=/path/to/runfiles if running
    outside of bazel.
    """
    runfiles_path = os.environ.get("PROGRAML_RUNFILES")
    if runfiles_path:
        return Path(runfiles_path) / relpath
    else:
        try:
            from rules_python.python.runfiles import runfiles

            return Path(
                runfiles.Create().Rlocation(
                    "programl" if relpath == "." else f"programl/{relpath}"
                )
            )
        except (ModuleNotFoundError, TypeError):
            # Special handler for paths to test data.
            if relpath.startswith("tests/"):
                return _TESTS_ROOT / relpath
            return _PACKAGE_ROOT / relpath
