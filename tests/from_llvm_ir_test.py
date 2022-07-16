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
import pytest

import programl as pg
from tests.test_main import main

pytest_plugins = ["tests.plugins.llvm_ir"]

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


@pytest.fixture(scope="module", params=["6", "10"])
def simple_ir_graph(request) -> pg.ProgramGraph:
    return pg.from_llvm_ir(
        SIMPLE_IR,
        version=request.param,
    )


def test_from_llvm_ir_multiple_inputs():
    graphs = list(pg.from_llvm_ir([SIMPLE_IR] * 10))
    assert len(graphs) == 10
    for graph in graphs:
        assert isinstance(graph, pg.ProgramGraph)


def test_graph_type(simple_ir_graph):
    assert isinstance(simple_ir_graph, pg.ProgramGraph)


def test_module(simple_ir_graph):
    assert len(simple_ir_graph.module) == 1
    assert simple_ir_graph.module[0].name == "foo.c"


def test_num_nodes(simple_ir_graph):
    assert len(simple_ir_graph.node) == 21


def test_node_types(simple_ir_graph):
    node_types = [n.type for n in simple_ir_graph.node]
    assert node_types == [
        pg.proto.Node.INSTRUCTION,
        pg.proto.Node.INSTRUCTION,
        pg.proto.Node.INSTRUCTION,
        pg.proto.Node.INSTRUCTION,
        pg.proto.Node.VARIABLE,
        pg.proto.Node.TYPE,
        pg.proto.Node.TYPE,
        pg.proto.Node.INSTRUCTION,
        pg.proto.Node.VARIABLE,
        pg.proto.Node.INSTRUCTION,
        pg.proto.Node.VARIABLE,
        pg.proto.Node.INSTRUCTION,
        pg.proto.Node.VARIABLE,
        pg.proto.Node.INSTRUCTION,
        pg.proto.Node.VARIABLE,
        pg.proto.Node.VARIABLE,
        pg.proto.Node.INSTRUCTION,
        pg.proto.Node.VARIABLE,
        pg.proto.Node.VARIABLE,
        pg.proto.Node.VARIABLE,
        pg.proto.Node.CONSTANT,
    ]


def test_node_texts(simple_ir_graph):
    node_texts = [n.text for n in simple_ir_graph.node]
    assert node_texts == [
        "[external]",
        "alloca",
        "alloca",
        "store",
        "var",
        "*",
        "i32",
        "store",
        "var",
        "load",
        "var",
        "load",
        "var",
        "add",
        "var",
        "var",
        "ret",
        "var",
        "var",
        "var",
        "val",
    ]


def test_node_full_texts(simple_ir_graph):
    full_texts = [
        n.features.feature["full_text"].bytes_list.value[0].decode("utf-8")
        for n in simple_ir_graph.node[1:]
    ]
    # The order of the last two variables may differ.
    print("\n".join(full_texts))
    assert full_texts == [
        "%3 = alloca i32, align 4",
        "%4 = alloca i32, align 4",
        "store i32 %0, i32* %3, align 4",
        "i32* %3",
        "i32*",
        "i32",
        "store i32 %1, i32* %4, align 4",
        "i32* %4",
        "%5 = load i32, i32* %3, align 4",
        "i32* %3",
        "%6 = load i32, i32* %4, align 4",
        "i32* %4",
        "%7 = add nsw i32 %5, %6",
        "i32 %5",
        "i32 %6",
        "ret i32 %7",
        "i32 %7",
        "i32 %0",
        "i32 %1",
        "i32 1",
    ] or full_texts == [
        "%3 = alloca i32, align 4",
        "%4 = alloca i32, align 4",
        "store i32 %0, i32* %3, align 4",
        "i32* %3",
        "i32*",
        "i32",
        "store i32 %1, i32* %4, align 4",
        "i32* %4",
        "%5 = load i32, i32* %3, align 4",
        "i32* %3",
        "%6 = load i32, i32* %4, align 4",
        "i32* %4",
        "%7 = add nsw i32 %5, %6",
        "i32 %5",
        "i32 %6",
        "ret i32 %7",
        "i32 %7",
        "i32 %1",
        "i32 %0",
        "i32 1",
    ]


@pytest.mark.parametrize("version", pg.create_ops.LLVM2GRAPH_BINARIES.keys())
def test_invalid_ir(version: str):
    """Test equivalence of nodes that pre-process to the same text."""
    with pytest.raises(pg.GraphCreationError, match="expected top-level entity"):
        pg.from_llvm_ir("foo bar", version=version)


@pytest.mark.parametrize("version", pg.create_ops.LLVM2GRAPH_BINARIES.keys())
def test_from_ir_string_smoke_test(llvm_ir: str, version: str):
    """Smoke test on real IRs."""
    graph = pg.from_llvm_ir(llvm_ir, version=version)
    assert isinstance(graph, pg.ProgramGraph)


def test_invalid_version():
    with pytest.raises(pg.UnsupportedCompiler):
        pg.from_llvm_ir("", version="invalid")


if __name__ == "__main__":
    main()
