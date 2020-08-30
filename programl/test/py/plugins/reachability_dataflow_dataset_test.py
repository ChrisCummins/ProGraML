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
"""Unit tests for //programl/test/py/plugins:reachability_dataflow_dataset."""
from pathlib import Path

from labm8.py import test

pytest_plugins = ["programl.test.py.plugins.reachability_dataflow_dataset"]


def test_reachability_dataflow_dataset(reachability_dataflow_dataset: Path):
    assert (reachability_dataflow_dataset / "ir").is_dir()
    assert (reachability_dataflow_dataset / "graphs").is_dir()
    assert (reachability_dataflow_dataset / "train").is_dir()
    assert (reachability_dataflow_dataset / "val").is_dir()
    assert (reachability_dataflow_dataset / "test").is_dir()


if __name__ == "__main__":
    test.Main()
