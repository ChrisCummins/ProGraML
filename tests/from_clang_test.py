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

import pytest

import programl as pg
from programl.exceptions import GraphCreationError
from tests.test_main import main

pytest_plugins = ["tests.plugins.llvm_ir", "tests.plugins.tempdir"]


def test_from_invalid_c(tempdir: Path):
    with open(tempdir / "foo.c", "w") as f:
        f.write("int main() syntax error!")

    with pytest.raises(
        pg.GraphCreationError,
        match="error: expected function body after function declarator",
    ):
        pg.from_clang([str(tempdir / "foo.c")])


def test_c_input(tempdir: Path):
    with open(tempdir / "a.c", "w") as f:
        f.write("int A() { return 0; }")
    graph = pg.from_clang([str(tempdir / "a.c")])
    assert isinstance(graph, pg.ProgramGraph)


def test_multiple_inputs(tempdir: Path):
    with open(tempdir / "a.c", "w") as f:
        f.write("int A() { return 0; }")
    graphs = list(pg.from_clang([[str(tempdir / "a.c")]] * 10))
    assert len(graphs) == 10
    for graph in graphs:
        assert isinstance(graph, pg.ProgramGraph)


def test_c_input_with_std_header(tempdir: Path):
    with open(tempdir / "a.c", "w") as f:
        f.write(
            """
#include <stdio.h>
void A() {
    printf("Hello, world\\n");
}
"""
        )
    graph = pg.from_clang([str(tempdir / "a.c")])
    assert isinstance(graph, pg.ProgramGraph)


def test_cxx_input_with_std_header(tempdir: Path):
    with open(tempdir / "a.cpp", "w") as f:
        f.write(
            """
#include <iostream>
void A() {
    std::cout << "Hello, world" << std::endl;
}
"""
        )
    graph = pg.from_clang([str(tempdir / "a.cpp")])
    assert isinstance(graph, pg.ProgramGraph)


def test_c_input_with_undefined_identifier(tempdir: Path):
    """Implicit use is allowed in C."""
    with open(tempdir / "a.c", "w") as f:
        f.write(
            """
void A() {
    B();
}
"""
        )
    graph = pg.from_clang([str(tempdir / "a.c")])
    assert isinstance(graph, pg.ProgramGraph)


def test_cxx_input_with_undefined_identifier(tempdir: Path):
    """Implicit use is an error in C++."""
    with open(tempdir / "a.cpp", "w") as f:
        f.write(
            """
void A() {
    B();
}
"""
        )
    with pytest.raises(GraphCreationError, match="error: use of undeclared identifier"):
        pg.from_clang([str(tempdir / "a.cpp")])


def test_from_multifile_c(tempdir: Path):
    """Multi-file inputs are not yet supported."""
    with open(tempdir / "a.c", "w") as f:
        f.write("int A() { return 0; }")
    with open(tempdir / "b.c", "w") as f:
        f.write("int B() { return 0; }")

    # TODO(https://github.com/ChrisCummins/ProGraML/issues/168): Add support
    # for multi-file inputs.
    with pytest.raises(
        pg.GraphCreationError,
        match="error: unable to handle compilation, expected exactly one compiler job",
    ):
        pg.from_clang([str(tempdir / "a.c"), str(tempdir / "b.c")])


def test_from_clang_smoke_test(llvm_ir_path: Path):
    """Smoke test on real IRs."""
    graph = pg.from_clang([str(llvm_ir_path)])
    assert isinstance(graph, pg.ProgramGraph)


if __name__ == "__main__":
    main()
