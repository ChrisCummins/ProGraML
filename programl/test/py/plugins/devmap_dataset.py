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
from pathlib import Path

from labm8.py import bazelutil

DEVMAP_DATASET = bazelutil.DataArchive("programl/test/data/devmap_dataset.tar.bz2")


@test.Fixture(scope="function")
def devmap_dataset() -> Path:
    """A test fixture which yields the root of the devmap dataset."""
    with DEVMAP_DATASET as d:
        yield d
