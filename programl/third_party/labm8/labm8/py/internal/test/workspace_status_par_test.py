# Copyright 2014-2020 Chris Cummins <chrisc.101@gmail.com>.
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
"""Unit tests for //labm8/py/internal:workspace_status."""
import subprocess

from labm8.py import bazelutil, test

PRINT_WORKSPACE_STATUS = bazelutil.DataPath(
    "phd/labm8/py/internal/test/print_workspace_status.par"
)

FLAGS = test.FLAGS


def test_print_workspace_status():
    """Test for the presence of bazel builtin workspace status variables."""
    assert subprocess.check_output([str(PRINT_WORKSPACE_STATUS)])


if __name__ == "__main__":
    test.Main()
