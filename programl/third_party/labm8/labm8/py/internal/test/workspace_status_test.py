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
from labm8.py import test
from labm8.py.internal import workspace_status

FLAGS = test.FLAGS


def test_bazel_builtins_are_set():
    """Test for the presence of bazel builtin workspace status variables."""
    assert workspace_status.BUILD_HOST
    assert workspace_status.BUILD_USER
    assert workspace_status.BUILD_TIMESTAMP


def test_expected_constants_are_set():
    """Test for custom workspace status variables."""
    assert workspace_status.STABLE_ARCH
    assert workspace_status.STABLE_GIT_URL
    assert workspace_status.STABLE_GIT_COMMIT
    assert workspace_status.STABLE_GIT_DIRTY
    assert workspace_status.STABLE_VERSION


if __name__ == "__main__":
    test.Main()
