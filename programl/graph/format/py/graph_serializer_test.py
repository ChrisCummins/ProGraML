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
"""Unit tests for //program/graph/format/py:graph_serializer."""
from labm8.py import test
from programl.graph.format.py import graph_serializer
from programl.proto import edge_pb2
from programl.proto import node_pb2
from programl.proto import program_graph_pb2


FLAGS = test.FLAGS


def test_SerializeInstructionsInProgramGraph_empty_graph():
  proto = program_graph_pb2.ProgramGraph()
  n = graph_serializer.SerializeInstructionsInProgramGraph(
    proto, max_nodes=1000
  )
  assert n == []


def test_SerializeInstructionsInProgramGraph_root_node_only():
  proto = program_graph_pb2.ProgramGraph(
    node=[node_pb2.Node(type=node_pb2.Node.INSTRUCTION),]
  )
  n = graph_serializer.SerializeInstructionsInProgramGraph(
    proto, max_nodes=1000
  )
  assert n == []


def test_SerializeInstructionsInProgramGraph_single_function():
  proto = program_graph_pb2.ProgramGraph(
    node=[
      node_pb2.Node(type=node_pb2.Node.INSTRUCTION),
      node_pb2.Node(type=node_pb2.Node.INSTRUCTION),
      node_pb2.Node(type=node_pb2.Node.INSTRUCTION),
    ],
    edge=[
      edge_pb2.Edge(flow=edge_pb2.Edge.CALL, source=0, target=1),
      edge_pb2.Edge(flow=edge_pb2.Edge.CONTROL, source=1, target=2),
    ],
  )
  n = graph_serializer.SerializeInstructionsInProgramGraph(
    proto, max_nodes=1000
  )
  assert n == [1, 2]


if __name__ == "__main__":
  test.Main()
