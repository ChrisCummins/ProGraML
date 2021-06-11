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

import networkx as nx
import pytest

import programl as pg
from tests.test_main import main

pytest_plugins = ["tests.plugins.llvm_program_graph"]


@pytest.fixture(scope="session")
def graph() -> pg.ProgramGraph:
    return pg.from_cpp("int A() { return 0; }")


@pytest.fixture(scope="session")
def graph2() -> pg.ProgramGraph:
    return pg.from_cpp("int B(int x) { return x + 1; }")


def test_to_networkx_simple_graph(graph: pg.ProgramGraph):
    graphs = list(pg.to_networkx([graph]))
    assert len(graphs) == 1
    assert isinstance(graphs[0], nx.MultiDiGraph)


def test_to_networkx_two_inputs(graph: pg.ProgramGraph):
    graphs = list(pg.to_networkx([graph, graph]))
    assert len(graphs) == 2
    assert nx.is_isomorphic(graphs[0], graphs[1])


def test_to_networkx_two_different_inputs(
    graph: pg.ProgramGraph, graph2: pg.ProgramGraph
):
    graphs = list(pg.to_networkx([graph, graph2]))
    assert len(graphs) == 2
    assert not nx.is_isomorphic(graphs[0], graphs[1])


def test_to_networkx_generator(graph: pg.ProgramGraph):
    graphs = list(pg.to_networkx((graph for _ in range(10)), chunksize=3))
    assert len(graphs) == 10
    for x in graphs[1:]:
        assert nx.is_isomorphic(graphs[0], x)


def test_to_networkx_generator_parallel_executor(graph: pg.ProgramGraph):
    with ThreadPoolExecutor() as executor:
        graphs = list(
            pg.to_networkx((graph for _ in range(10)), chunksize=3, executor=executor)
        )
    assert len(graphs) == 10
    for x in graphs[1:]:
        assert nx.is_isomorphic(graphs[0], x)


def test_to_networkx_smoke_test(llvm_program_graph: pg.ProgramGraph):
    graphs = list(pg.to_networkx([llvm_program_graph]))
    assert len(graphs) == 1
    assert isinstance(graphs[0], nx.MultiDiGraph)
    assert graphs[0].number_of_nodes() == len(llvm_program_graph.node)
    assert graphs[0].number_of_edges() <= len(llvm_program_graph.edge)


if __name__ == "__main__":
    main()
