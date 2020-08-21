# Copyright 2019-2020 the ProGraML authors.
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
"""End-to-end test that --strict flag on llvm2graph rejects invalid graphs."""
import subprocess

from labm8.py import bazelutil
from labm8.py import test


LLVM2GRAPH = bazelutil.DataPath("programl/programl/cmd/llvm2graph")


# This IR file contains unreachable instructions, which will be rejected if
# --strict mode is enabled.
INVALID_MODULE = bazelutil.DataPath(
    "programl/programl/test/data/module_with_unreachable_instructions.ll"
)


def test_invalid_module():
  subprocess.check_call([str(LLVM2GRAPH), str(INVALID_MODULE)])


def test_invalid_module_strict():
  with test.Raises(subprocess.CalledProcessError):
    subprocess.check_call([str(LLVM2GRAPH), str(INVALID_MODULE), "--strict"])


if __name__ == '__main__':
  test.Main()
