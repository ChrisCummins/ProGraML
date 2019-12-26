# Copyright 2019 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
TMP_ROOT = pathlib.Path("~/.cache/phd/ml4pl").expanduser()
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
