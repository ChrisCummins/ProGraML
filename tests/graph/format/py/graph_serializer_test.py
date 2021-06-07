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
from programl.graph.format.py import graph_serializer
from programl.proto import Edge, Node, ProgramGraph
from tests.test_main import main


def test_SerializeInstructionsInProgramGraph_empty_graph():
    proto = ProgramGraph()
    n = graph_serializer.SerializeInstructionsInProgramGraph(proto, max_nodes=1000)
    assert n == []


def test_SerializeInstructionsInProgramGraph_root_node_only():
    proto = ProgramGraph(
        node=[
            Node(type=Node.INSTRUCTION),
        ]
    )
    n = graph_serializer.SerializeInstructionsInProgramGraph(proto, max_nodes=1000)
    assert n == []


def test_SerializeInstructionsInProgramGraph_single_function():
    proto = ProgramGraph(
        node=[
            Node(type=Node.INSTRUCTION),
            Node(type=Node.INSTRUCTION),
            Node(type=Node.INSTRUCTION),
        ],
        edge=[
            Edge(flow=Edge.CALL, source=0, target=1),
            Edge(flow=Edge.CONTROL, source=1, target=2),
        ],
    )
    n = graph_serializer.SerializeInstructionsInProgramGraph(proto, max_nodes=1000)
    assert n == [1, 2]


if __name__ == "__main__":
    main()
