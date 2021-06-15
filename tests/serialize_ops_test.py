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
from itertools import islice
from pathlib import Path
from typing import List

import pytest

import programl as pg
from tests.plugins.llvm_program_graph import EnumerateLlvmProgramGraphs
from tests.test_main import main

pytest_plugins = ["tests.plugins.llvm_program_graph", "tests.plugins.tempdir"]


@pytest.fixture(scope="session")
def llvm_program_graphs() -> List[pg.ProgramGraph]:
    return list(islice(EnumerateLlvmProgramGraphs(), 100))


@pytest.mark.parametrize("compression", ["gz", None])
def test_save_load_graphs_smoke_test(
    llvm_program_graphs: List[pg.ProgramGraph], tempdir: Path, compression: str
):
    """Save/load graph equivalence test."""
    pg.save_graphs(tempdir / "graphs.pb", llvm_program_graphs, compression=compression)
    assert (tempdir / "graphs.pb").is_file()
    graphs = pg.load_graphs(tempdir / "graphs.pb", compression=compression)
    assert llvm_program_graphs == graphs

    # Again with an index list.
    some_graphs = pg.load_graphs(
        tempdir / "graphs.pb", idx_list=[0, 2], compression=compression
    )
    assert some_graphs == [llvm_program_graphs[0], llvm_program_graphs[2]]


def test_file_invalid_compression(
    llvm_program_graphs: List[pg.ProgramGraph], tempdir: Path
):
    error_message = (
        "Invalid compression argument: txt. Supported compressions: [gz, None]"
    )
    with pytest.raises(TypeError, match=error_message):
        pg.save_graphs(tempdir / "graphs.pb", llvm_program_graphs, compression="txt")

    (tempdir / "graphs.pb").touch()
    with pytest.raises(TypeError, match=error_message):
        pg.load_graphs(tempdir / "graphs.pb", compression="txt")


def test_bytes_invalid_compression(llvm_program_graphs: List[pg.ProgramGraph]):
    error_message = (
        "Invalid compression argument: txt. Supported compressions: [gz, None]"
    )
    with pytest.raises(TypeError, match=error_message):
        pg.to_bytes(llvm_program_graphs, compression="txt")

    with pytest.raises(TypeError, match=error_message):
        pg.from_bytes(b"", compression="txt")


@pytest.mark.parametrize("compression", ["gz", None])
def test_to_bytes_from_bytes_smoke_test(
    llvm_program_graphs: List[pg.ProgramGraph], compression: str
):
    serialized = pg.to_bytes(llvm_program_graphs, compression=compression)
    assert isinstance(serialized, bytes)
    graphs = pg.from_bytes(serialized, compression=compression)
    assert llvm_program_graphs == graphs

    # Again with an index list.
    some_graphs = pg.from_bytes(serialized, idx_list=[0, 2], compression=compression)
    assert some_graphs == [llvm_program_graphs[0], llvm_program_graphs[2]]


def test_to_string_from_string_smoke_test(llvm_program_graphs: List[pg.ProgramGraph]):
    # Use a smaller set of graphs for string serialization as it is quite slow.
    llvm_program_graphs = llvm_program_graphs[:10]

    serialized = pg.to_string(llvm_program_graphs)
    assert isinstance(serialized, str)
    graphs = pg.from_string(serialized)
    assert llvm_program_graphs == graphs

    # Again with an index list.
    some_graphs = pg.from_string(serialized, idx_list=[0, 2])
    assert some_graphs == [llvm_program_graphs[0], llvm_program_graphs[2]]


if __name__ == "__main__":
    main()
