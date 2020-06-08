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
"""Unit tests for //programl/ir/llvm:inst2vec_encoder."""
from labm8.py import test
from programl.graph.py import program_graph_builder
from programl.ir.llvm import inst2vec_encoder
from programl.proto import node_pb2
from programl.proto import program_graph_pb2


FLAGS = test.FLAGS

pytest_plugins = ["programl.test.py.plugins.llvm_program_graph"]


@test.Fixture(scope="session")
def encoder() -> inst2vec_encoder.Inst2vecEncoder:
  """A session-level fixture to re-use a graph encoder instance."""
  return inst2vec_encoder.Inst2vecEncoder()


class Inst2vecGraphBuilder(program_graph_builder.ProgramGraphBuilder):
  """A program graph builder for conveniently creating inst2vec test graphs."""

  def __init__(self):
    super(Inst2vecGraphBuilder, self).__init__()
    mod = self.AddModule("mod")
    self.fn = self.AddFunction("fn", mod)
    # Store the full texts which will be assigned as features during Build().
    self.full_texts = {}

  def AddInstruction(self, full_text: str):
    """Add an instruction with the given full text."""
    node = super(Inst2vecGraphBuilder, self).AddInstruction(full_text, self.fn)
    self.full_texts[node] = full_text
    return node

  def AddVariable(self, full_text: str):
    """Add a variable with the given full text."""
    node = super(Inst2vecGraphBuilder, self).AddVariable(full_text, self.fn)
    self.full_texts[node] = full_text
    return node

  def Build(self):
    proto = super(Inst2vecGraphBuilder, self).Build()
    for node, full_text in self.full_texts.items():
      proto.node[node].features.feature["full_text"].bytes_list.value.append(
        full_text.encode("utf-8")
      )
    return proto


def test_Encode_equivalent_preprocessed_text(
  encoder: inst2vec_encoder.Inst2vecEncoder,
):
  """Test equivalence of nodes that pre-process to the same text."""
  builder = Inst2vecGraphBuilder()
  a = builder.AddInstruction("%7 = add nsw i32 %5, -1")
  b = builder.AddInstruction("%9 = add nsw i32 %5, -2")
  builder.AddControlEdge(builder.root, a, position=0)
  builder.AddControlEdge(a, b, position=0)

  proto = encoder.Encode(builder.Build())

  assert proto.node[a].features.feature[
    "inst2vec_preprocessed"
  ].bytes_list.value == [b"<%ID> = add nsw i32 <%ID>, <INT>"]
  assert proto.node[b].features.feature[
    "inst2vec_preprocessed"
  ].bytes_list.value == [b"<%ID> = add nsw i32 <%ID>, <INT>"]


def test_Encode_variable(encoder: inst2vec_encoder.Inst2vecEncoder):
  """Test the encoding of variable nodes."""
  builder = Inst2vecGraphBuilder()
  a = builder.AddVariable("abcd")
  builder.AddDataEdge(builder.root, a, position=0)

  proto = encoder.Encode(builder.Build())

  assert proto.node[a].features.feature[
    "inst2vec_embedding"
  ].int64_list.value == [encoder.dictionary["!IDENTIFIER"]]


def test_Encode_constant(encoder: inst2vec_encoder.Inst2vecEncoder):
  """Test the encoding of constant nodes."""
  builder = Inst2vecGraphBuilder()
  a = builder.AddInstruction("x")
  b = builder.AddConstant("abcd")
  builder.AddControlEdge(builder.root, a, position=0)
  builder.AddDataEdge(b, a, position=0)

  proto = encoder.Encode(builder.Build())

  assert proto.node[b].features.feature[
    "inst2vec_embedding"
  ].int64_list.value == [encoder.dictionary["!IMMEDIATE"]]


def test_Encode_encoded_values(encoder: inst2vec_encoder.Inst2vecEncoder):
  """Test that "x" attribute of a node matches dictionary value."""
  builder = Inst2vecGraphBuilder()
  a = builder.AddInstruction("br label %4")
  builder.AddControlEdge(builder.root, a, position=0)

  proto = encoder.Encode(builder.Build())

  assert proto.node[a].features.feature[
    "inst2vec_preprocessed"
  ].bytes_list.value == [b"br label <%ID>"]


def test_Encode_encoded_values_differ_between_instructions(
  encoder: inst2vec_encoder.Inst2vecEncoder,
):
  """Test that "x" attribute of nodes differ between different texts."""
  builder = Inst2vecGraphBuilder()
  a = builder.AddInstruction("%7 = add nsw i32 %5, -1")
  b = builder.AddInstruction("br label %4")
  builder.AddControlEdge(builder.root, a, position=0)
  builder.AddControlEdge(a, b, position=0)

  proto = encoder.Encode(builder.Build())

  assert (
    proto.node[a].features.feature["inst2vec_preprocessed"].bytes_list.value
    != proto.node[b].features.feature["inst2vec_preprocessed"].bytes_list.value
  )


def test_Encode_inlined_struct(encoder: inst2vec_encoder.Inst2vecEncoder):
  """Test that struct definition is inlined in pre-processed text.

  NOTE(github.com/ChrisCummins/ProGraML/issues/57): Regression test.
  """
  ir = """
%struct.foo = type { i32, i8* }

define i32 @A(%struct.foo*) #0 {
  %2 = alloca %struct.foo*, align 8
  store %struct.foo* %0, %struct.foo** %2, align 8
  %3 = load %struct.foo*, %struct.foo** %2, align 8
  %4 = getelementptr inbounds %struct.foo, %struct.foo* %3, i32 0, i32 0
  %5 = load i32, i32* %4, align 8
  ret i32 %5
}
"""
  builder = Inst2vecGraphBuilder()
  a = builder.AddInstruction("%2 = alloca %struct.foo*, align 8")
  builder.AddControlEdge(builder.root, a, position=0)

  proto = encoder.Encode(builder.Build(), ir=ir)

  assert proto.node[a].features.feature[
    "inst2vec_preprocessed"
  ].bytes_list.value == [b"<%ID> = alloca { i32, i8* }*, align 8"]


def test_Encode_inlined_nested_struct(
  encoder: inst2vec_encoder.Inst2vecEncoder,
):
  """Test that nested struct definitions are inlined in pre-processed text.

  NOTE(github.com/ChrisCummins/ProGraML/issues/57): Regression test.
  """
  ir = """
%struct.bar = type { %struct.foo* }
%struct.foo = type { i32 }

; Function Attrs: noinline nounwind optnone uwtable
define i32 @Foo(%struct.bar*) #0 {
  %2 = alloca %struct.bar*, align 8
  store %struct.bar* %0, %struct.bar** %2, align 8
  %3 = load %struct.bar*, %struct.bar** %2, align 8
  %4 = getelementptr inbounds %struct.bar, %struct.bar* %3, i32 0, i32 0
  %5 = load %struct.foo*, %struct.foo** %4, align 8
  %6 = getelementptr inbounds %struct.foo, %struct.foo* %5, i32 0, i32 0
  %7 = load i32, i32* %6, align 4
  ret i32 %7
}
"""
  builder = Inst2vecGraphBuilder()
  a = builder.AddInstruction("%2 = alloca %struct.bar*, align 8")
  b = builder.AddInstruction("store %struct.bar* %0, %struct.bar** %2, align 8")
  c = builder.AddInstruction(
    "%3 = load %struct.bar*, %struct.bar** %2, align 8"
  )
  d = builder.AddInstruction(
    "%4 = getelementptr inbounds %struct.bar, %struct.bar* %3, i32 0, i32 0"
  )
  e = builder.AddInstruction(
    "%5 = load %struct.foo*, %struct.foo** %4, align 8"
  )
  f = builder.AddInstruction(
    "%6 = getelementptr inbounds %struct.foo, %struct.foo* %5, i32 0, i32 0"
  )
  builder.AddControlEdge(builder.root, a, position=0)
  builder.AddControlEdge(a, b, position=0)
  builder.AddControlEdge(b, c, position=0)
  builder.AddControlEdge(c, d, position=0)
  builder.AddControlEdge(d, e, position=0)
  builder.AddControlEdge(e, f, position=0)

  proto = encoder.Encode(builder.Build(), ir=ir)

  assert proto.node[a].features.feature[
    "inst2vec_preprocessed"
  ].bytes_list.value == [b"<%ID> = alloca { { i32 }* }*, align 8"]
  assert proto.node[b].features.feature[
    "inst2vec_preprocessed"
  ].bytes_list.value == [
    b"store { { i32 }* }* <%ID>, { { i32 }* }** <%ID>, align 8"
  ]
  assert proto.node[c].features.feature[
    "inst2vec_preprocessed"
  ].bytes_list.value == [
    b"<%ID> = load { { i32 }* }*, { { i32 }* }** <%ID>, align 8"
  ]
  assert proto.node[d].features.feature[
    "inst2vec_preprocessed"
  ].bytes_list.value == [
    b"<%ID> = getelementptr inbounds { { i32 }* }, { { i32 }* }* <%ID>, i32 <INT>, i32 <INT>"
  ]
  assert proto.node[e].features.feature[
    "inst2vec_preprocessed"
  ].bytes_list.value == [b"<%ID> = load { i32 }*, { i32 }** <%ID>, align 8"]
  assert proto.node[f].features.feature[
    "inst2vec_preprocessed"
  ].bytes_list.value == [
    b"<%ID> = getelementptr inbounds { i32 }, { i32 }* <%ID>, i32 <INT>, i32 <INT>"
  ]


def test_Encode_llvm_program_graph(
  llvm_program_graph: program_graph_pb2.ProgramGraph,
  encoder: inst2vec_encoder.Inst2vecEncoder,
):
  """Black-box test encoding LLVM program graphs."""
  proto = program_graph_pb2.ProgramGraph()
  proto.CopyFrom(llvm_program_graph)
  encoder.Encode(proto)

  # This assumes that all of the test graphs have at least one instruction.
  num_instructions = sum(
    1 if node.type == node_pb2.Node.INSTRUCTION else 0 for node in proto.node
  )
  assert num_instructions >= 1

  # Check for the presence of expected node attributes.
  for node in proto.node:
    assert "inst2vec_embedding" in node.features.feature
    if node.type == node_pb2.Node.INSTRUCTION:
      assert "inst2vec_preprocessed" in node.features.feature


if __name__ == "__main__":
  test.Main()
