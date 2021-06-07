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
from typing import Iterable, List, Optional

from programl.proto import ProgramGraph

# -> Serialization to file


def save_graphs(path: Path, graphs: Iterable[ProgramGraph]) -> None:
    pass


def load_graphs(path: Path, idx_list: Optional[List[int]] = None) -> List[ProgramGraph]:
    pass


# -> Serialization to string


def to_string(graphs: Iterable[ProgramGraph]) -> str:
    pass


def from_string(string: str) -> List[ProgramGraph]:
    pass
