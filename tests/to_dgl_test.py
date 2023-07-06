import pytest
from dgl import DGLGraph

import programl as pg
from tests.test_main import main

pytest_plugins = ["tests.plugins.llvm_program_graph"]


@pytest.fixture(scope="session")
def graph() -> pg.ProgramGraph:
    return pg.from_cpp("int A() { return 0; }")


def test_to_dgl_simple_graph(graph: pg.ProgramGraph):
    graphs = list(pg.to_dgl([graph]))
    assert len(graphs) == 1
    assert isinstance(graphs[0], DGLGraph)


def test_to_dgl_simple_graph_single_input(graph: pg.ProgramGraph):
    dgl_graph = pg.to_dgl(graph)
    assert isinstance(dgl_graph, DGLGraph)


def test_to_dgl_two_inputs(graph: pg.ProgramGraph):
    graphs = list(pg.to_dgl([graph, graph]))
    assert len(graphs) == 2


if __name__ == "__main__":
    main()
