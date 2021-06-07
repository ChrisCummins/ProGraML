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
"""Unit tests for //program/graph/format/py:graph_tuple."""
import numpy as np
import pytest

from programl.graph.format.py.graph_tuple_builder import GraphTupleBuilder
from programl.proto import Edge, Node, ProgramGraph
from tests.test_main import main


def test_GraphTuple_empty_shapes():
    """Build from an empty proto."""
    builder = GraphTupleBuilder()
    with pytest.raises(ValueError) as e_ctx:
        builder.Build()
    assert "contains no graphs" in str(e_ctx.value)


def test_GraphTuple_one_graph():
    graph = ProgramGraph(
        node=[
            Node(),
        ],
        edge=[
            Edge(
                source=0,
                target=1,
            ),
            Edge(
                source=0,
                target=2,
                position=1,
            ),
            Edge(
                source=1,
                target=0,
                position=10,
                flow=Edge.CALL,
            ),
        ],
    )

    builder = GraphTupleBuilder()
    builder.AddProgramGraph(graph)
    gt = builder.Build()

    assert np.array_equal(gt.adjacencies[Edge.CONTROL], [(0, 1), (0, 2)])
    assert np.array_equal(gt.adjacencies[Edge.DATA], np.zeros((0, 2), dtype=np.int32))
    assert np.array_equal(gt.adjacencies[Edge.CALL], [(1, 0)])

    assert np.array_equal(gt.edge_positions[Edge.CONTROL], [0, 1])
    assert np.array_equal(gt.edge_positions[Edge.DATA], [])
    assert np.array_equal(gt.edge_positions[Edge.CALL], [10])


if __name__ == "__main__":
    main()
