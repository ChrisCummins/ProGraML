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
"""Unit tests for //programl/ir/llvm/py:llvm."""
from labm8.py import test
from programl.ir.llvm.py import llvm
from programl.proto import node_pb2
from programl.proto import program_graph_options_pb2
from programl.proto import program_graph_pb2

SIMPLE_IR = """
source_filename = "foo.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

define i32 @A(i32, i32) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  store i32 %1, i32* %4, align 4
  %5 = load i32, i32* %3, align 4
  %6 = load i32, i32* %4, align 4
  %7 = add nsw i32 %5, %6
  ret i32 %7
}
"""


def GetStringScalar(proto, name):
  return proto.features.feature[name].bytes_list.value[0].decode("utf-8")


def test_simple_ir():
  """Test equivalence of nodes that pre-process to the same text."""
  options = program_graph_options_pb2.ProgramGraphOptions(opt_level=3)
  proto = llvm.BuildProgramGraph(SIMPLE_IR, options)
  assert isinstance(proto, program_graph_pb2.ProgramGraph)

  assert len(proto.module) == 1
  assert proto.module[0].name == "foo.c"

  assert len(proto.node) == 6
  assert proto.node[0].text == "<root>"
  assert proto.node[0].type == node_pb2.Node.INSTRUCTION

  assert proto.node[1].text == "add"
  assert proto.node[1].type == node_pb2.Node.INSTRUCTION
  assert (
    GetStringScalar(proto.node[1], "full_text") == "%3 = add nsw i32 %1, %0"
  )

  assert proto.node[2].text == "ret"
  assert proto.node[2].type == node_pb2.Node.INSTRUCTION
  assert GetStringScalar(proto.node[2], "full_text") == "ret i32 %3"

  assert proto.node[3].text == "i32"
  assert proto.node[3].type == node_pb2.Node.VARIABLE
  assert GetStringScalar(proto.node[3], "full_text") == "i32 %3"

  # Use startswith() to compare names for these last two variables as thier
  # order may differ.
  assert proto.node[4].text == "i32"
  assert proto.node[4].type == node_pb2.Node.VARIABLE
  assert GetStringScalar(proto.node[4], "full_text").startswith("i32 %")

  assert proto.node[5].text == "i32"
  assert proto.node[5].type == node_pb2.Node.VARIABLE
  assert GetStringScalar(proto.node[5], "full_text").startswith("i32 %")


def test_opt_level():
  """Test equivalence of nodes that pre-process to the same text."""
  options = program_graph_options_pb2.ProgramGraphOptions(opt_level=0,)
  unoptimized = llvm.BuildProgramGraph(SIMPLE_IR)
  options.opt_level = 3
  optimized = llvm.BuildProgramGraph(SIMPLE_IR, options)

  assert len(optimized.node) < len(unoptimized.node)


def test_invalid_ir():
  """Test equivalence of nodes that pre-process to the same text."""
  with test.Raises(ValueError) as e_ctx:
    llvm.BuildProgramGraph("foo bar")
  assert "expected top-level entity" in str(e_ctx.value)


if __name__ == "__main__":
  test.Main()
