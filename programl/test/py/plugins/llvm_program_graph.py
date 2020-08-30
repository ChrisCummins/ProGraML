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
from typing import Iterable, Tuple

from labm8.py import bazelutil, pbutil, test

from programl.proto import program_graph_pb2

LLVM_IR_GRAPHS = bazelutil.DataPath("programl/programl/test/data/llvm_ir_graphs")


def EnumerateLlvmProgramGraphs() -> Iterable[
    Tuple[str, program_graph_pb2.ProgramGraph]
]:
    """Enumerate a test set of LLVM IR file paths."""
    for path in LLVM_IR_GRAPHS.iterdir():
        yield path.name, pbutil.FromFile(path, program_graph_pb2.ProgramGraph())


@test.Fixture(
    scope="session",
    params=list(EnumerateLlvmProgramGraphs()),
    namer=lambda s: s[0],
)
def llvm_program_graph(request) -> str:
    """A test fixture which yields an LLVM-IR string."""
    return request.param[1]


@test.Fixture(
    scope="session",
    params=list(LLVM_IR_GRAPHS.iterdir()),
    namer=lambda path: path.name,
)
def llvm_program_graph_path(request) -> Path:
    """A test fixture which yields the path of an LLVM ProGraML file."""
    return request.param
