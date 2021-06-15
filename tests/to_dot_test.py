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
from concurrent.futures.thread import ThreadPoolExecutor

import pydot
import pytest

import programl as pg
from tests.test_main import main

pytest_plugins = ["tests.plugins.llvm_program_graph"]


@pytest.fixture(scope="session")
def graph() -> pg.ProgramGraph:
    return pg.from_cpp("int A() { return 0; }")


def assert_valid_graph(dot: str):
    return pydot.graph_from_dot_data(dot)


def test_to_dot_simple_graph(graph: pg.ProgramGraph):
    dots = list(pg.to_dot([graph]))
    assert len(dots) == 1
    assert_valid_graph(dots[0])


def test_to_dot_simple_graph_single_input(graph: pg.ProgramGraph):
    dot = pg.to_dot(graph)
    assert_valid_graph(dot)


def test_to_dot_two_inputs(graph: pg.ProgramGraph):
    dots = list(pg.to_dot([graph, graph]))
    assert len(dots) == 2
    # NOTE graph2dot output is not stable so we can't do an equivalent check as
    # things like parameter orders can change from one run to the next.
    assert_valid_graph(dots[0])
    assert_valid_graph(dots[1])


def test_to_dot_generator(graph: pg.ProgramGraph):
    dots = list(pg.to_dot((graph for _ in range(10)), chunksize=3))
    assert len(dots) == 10


def test_to_dot_generator_parallel_executor(graph: pg.ProgramGraph):
    with ThreadPoolExecutor() as executor:
        dots = list(
            pg.to_dot((graph for _ in range(10)), chunksize=3, executor=executor)
        )
    assert len(dots) == 10


def test_to_dot_smoke_test(llvm_program_graph: pg.ProgramGraph):
    dots = list(pg.to_dot([llvm_program_graph]))
    assert len(dots) == 1
    assert isinstance(dots[0], str)


if __name__ == "__main__":
    main()
