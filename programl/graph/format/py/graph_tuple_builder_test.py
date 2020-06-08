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

from labm8.py import test
from programl.graph.format.py.graph_tuple_builder import GraphTupleBuilder
from programl.proto import edge_pb2
from programl.proto import node_pb2
from programl.proto import program_graph_pb2

FLAGS = test.FLAGS


def test_GraphTuple_empty_shapes():
  """Build from an empty proto."""
  builder = GraphTupleBuilder()
  with test.Raises(ValueError) as e_ctx:
    gt = builder.Build()
  assert "contains no graphs" in str(e_ctx.value)


def test_GraphTuple_one_graph():
  graph = program_graph_pb2.ProgramGraph(
    node=[node_pb2.Node(),],
    edge=[
      edge_pb2.Edge(source=0, target=1,),
      edge_pb2.Edge(source=0, target=2, position=1,),
      edge_pb2.Edge(source=1, target=0, position=10, flow=edge_pb2.Edge.CALL,),
    ],
  )

  builder = GraphTupleBuilder()
  builder.AddProgramGraph(graph)
  gt = builder.Build()

  assert np.array_equal(gt.adjacencies[edge_pb2.Edge.CONTROL], [(0, 1), (0, 2)])
  assert np.array_equal(
    gt.adjacencies[edge_pb2.Edge.DATA], np.zeros((0, 2), dtype=np.int32)
  )
  assert np.array_equal(gt.adjacencies[edge_pb2.Edge.CALL], [(1, 0)])

  assert np.array_equal(gt.edge_positions[edge_pb2.Edge.CONTROL], [0, 1])
  assert np.array_equal(gt.edge_positions[edge_pb2.Edge.DATA], [])
  assert np.array_equal(gt.edge_positions[edge_pb2.Edge.CALL], [10])


if __name__ == "__main__":
  test.Main()
