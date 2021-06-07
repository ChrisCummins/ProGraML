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
from typing import Iterable, Tuple

import pytest

from programl.proto import ProgramGraph
from programl.util.py import pbutil
from programl.util.py.runfiles_path import runfiles_path

LLVM_IR_GRAPHS = runfiles_path("tests/data/llvm_ir_graphs")


def EnumerateLlvmProgramGraphs() -> Iterable[Tuple[str, ProgramGraph]]:
    """Enumerate a test set of LLVM IR file paths."""
    for path in LLVM_IR_GRAPHS.iterdir():
        yield path.name, pbutil.FromFile(path, ProgramGraph())


_PROGRAM_GRAPHS = list(EnumerateLlvmProgramGraphs())


@pytest.fixture(
    scope="session",
    params=_PROGRAM_GRAPHS,
    ids=[s[0] for s in _PROGRAM_GRAPHS],
)
def llvm_program_graph(request) -> str:
    """A test fixture which yields an LLVM-IR string."""
    return request.param[1]
