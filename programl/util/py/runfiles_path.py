"""Module for resolving a runfiles path."""
import os
from pathlib import Path


def runfiles_path(relpath: str) -> Path:
    """Resolve the path to a runfiles data path.

    Use environment variable PROGRAML_RUNFILES=/path/to/runfiles if running
    outside of bazel.
    """
    runfiles_path = os.environ.get("PROGRAML_RUNFILES")
    if runfiles_path:
        return Path(runfiles_path) / relpath
    else:
        # Defer importing this module so that if we have set
        # $COMPILER_GYM_RUNFILES we do not need any bazel dependencies.
        from rules_python.python.runfiles import runfiles

        return Path(runfiles.Create().Rlocation(relpath))
