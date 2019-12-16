"""Module for generating filesystem paths.

We occasionally want to create or read files. When doing so, use this module to
generate the path. This module contains a default hardcoded location for files
which can be overridden by setting the ${ML4PL_TMP_ROOT} environment variable.
"""
import os
import pathlib
from typing import Union

from labm8.py import app

FLAGS = app.FLAGS

# The root directory for storing temporary files.
TMP_ROOT = pathlib.Path(
  os.environ.get(
    "ML4PL_TMP_ROOT", f"/tmp/ml4pl/{os.environ.get('USER', 'anon')}"
  )
).absolute()
os.environ["ML4PL_TMP_ROOT"] = str(TMP_ROOT)


def TemporaryFilePath(relpath: Union[str, pathlib.Path]):
  """Generate an absolute path for a temporary file.

  Args:
    relpath: A relative path.

  Returns:
    A concatenation of the ${ML4PL_TMP_ROOT} directory and the relative path. No
    assumption is made on the type of path, or whether it (or any parent
    directories) exist.
  """
  return TMP_ROOT / relpath
