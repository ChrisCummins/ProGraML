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
"""Unit tests for //programl/graph/py:program_graph_builder."""
import pytest
from programl.proto.program_graph_options_pb2 import ProgramGraphOptions

from programl.graph.py.program_graph_builder import ProgramGraphBuilder
from programl.proto import edge_pb2, node_pb2, program_graph_pb2
from tests.test_main import main


def test_empty_proto():
    builder = ProgramGraphBuilder()
    assert isinstance(builder.Build(), program_graph_pb2.ProgramGraph)


def test_empty_proto_strict():
    builder = ProgramGraphBuilder(ProgramGraphOptions(strict=True))
    with pytest.raises(ValueError) as e_ctx:
        builder.Build()
    assert "INSTRUCTION has no connections: `[external]`" == str(e_ctx.value)


def test_add_empty_module():
    builder = ProgramGraphBuilder()
    foo = builder.AddModule("foo")

    assert foo == 0
    assert isinstance(builder.Build(), program_graph_pb2.ProgramGraph)


def test_add_empty_module_strict():
    builder = ProgramGraphBuilder(ProgramGraphOptions(strict=True))
    foo = builder.AddModule("foo")

    assert foo == 0
    with pytest.raises(ValueError) as e_ctx:
        builder.Build()
    assert str(e_ctx.value) == "Module `foo` is empty"


def test_add_empty_function():
    builder = ProgramGraphBuilder()
    mod = builder.AddModule("foo")
    foo = builder.AddFunction("bar", mod)

    assert foo == 0
    assert isinstance(builder.Build(), program_graph_pb2.ProgramGraph)


def test_add_empty_function_strict():
    builder = ProgramGraphBuilder(ProgramGraphOptions(strict=True))
    mod = builder.AddModule("foo")
    foo = builder.AddFunction("bar", mod)

    assert foo == 0
    with pytest.raises(ValueError) as e_ctx:
        builder.Build()
    assert str(e_ctx.value) == "Function `bar` is empty"


def test_graph_with_unconnected_node():
    builder = ProgramGraphBuilder()
    mod = builder.AddModule("x")
    fn = builder.AddFunction("x", mod)
    builder.AddInstruction("x", fn)

    assert isinstance(builder.Build(), program_graph_pb2.ProgramGraph)


def test_graph_with_unconnected_node_strict():
    builder = ProgramGraphBuilder(ProgramGraphOptions(strict=True))
    mod = builder.AddModule("x")
    fn = builder.AddFunction("x", mod)
    builder.AddInstruction("x", fn)
    with pytest.raises(ValueError) as e_ctx:
        builder.Build()
    assert "INSTRUCTION has no connections: " in str(e_ctx.value)


def test_linear_statement_control_flow():
    """Test that graph construction doesn't set on fire."""
    builder = ProgramGraphBuilder()
    mod = builder.AddModule("x")
    fn = builder.AddFunction("x", mod)
    a = builder.AddInstruction("a", fn)
    b = builder.AddInstruction("b", fn)
    builder.AddControlEdge(builder.root, a, position=0)
    builder.AddControlEdge(a, b, position=0)

    assert isinstance(builder.Build(), program_graph_pb2.ProgramGraph)

    assert len(builder.Build().node) == 3

    assert builder.Build().node[builder.root].text == "[external]"
    assert builder.Build().node[builder.root].type == node_pb2.Node.INSTRUCTION

    assert builder.Build().node[a].text == "a"
    assert builder.Build().node[a].type == node_pb2.Node.INSTRUCTION
    assert builder.Build().node[a].function == fn

    assert builder.Build().node[b].text == "b"
    assert builder.Build().node[b].type == node_pb2.Node.INSTRUCTION
    assert builder.Build().node[b].function == fn

    assert len(builder.Build().edge) == 2
    assert builder.Build().edge[0].source == builder.root
    assert builder.Build().edge[0].target == a
    assert builder.Build().edge[0].position == 0
    assert builder.Build().edge[0].flow == edge_pb2.Edge.CONTROL

    assert builder.Build().edge[1].source == a
    assert builder.Build().edge[1].target == b
    assert builder.Build().edge[1].position == 0
    assert builder.Build().edge[1].flow == edge_pb2.Edge.CONTROL


def test_clear():
    builder = ProgramGraphBuilder()
    a = builder.AddModule("x")
    builder.Clear()
    b = builder.AddModule("x")
    assert a == b


if __name__ == "__main__":
    main()
