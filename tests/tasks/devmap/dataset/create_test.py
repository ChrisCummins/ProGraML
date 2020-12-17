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
"""Smoke test for //tasks/devmap/dataset:create."""
from pathlib import Path

from tasks.devmap.dataset.create import create_devmap_dataset
from tests.test_main import main

pytest_plugins = ["tests.plugins.tempdir"]


def test_create_devmap_dataset(tempdir: Path):
    """Test dataset creation."""
    create_devmap_dataset(tempdir)
    assert (tempdir / "ir").is_dir()
    assert (tempdir / "src").is_dir()
    assert (tempdir / "graphs_amd").is_dir()
    assert (tempdir / "graphs_nvidia").is_dir()

    assert len(list((tempdir / "graphs_amd").iterdir())) == 680
    assert len(list((tempdir / "graphs_nvidia").iterdir())) == 680


if __name__ == "__main__":
    main()
