// Copyright 2019-2020 the ProGraML authors.
//
// Contact Chris Cummins <chrisc.101@gmail.com>.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "programl/graph/format/cdfg.h"

#include "labm8/cpp/logging.h"
#include "labm8/cpp/test.h"
#include "programl/graph/program_graph_builder.h"

using labm8::Status;

namespace programl {
namespace graph {
namespace format {
namespace {

TEST(CDFGBuilderTest, GraphOne) {
  // A graph consisting of five instructions A-E, a single
  // variable v1, and a constant c1.
  //
  //     +---+                +---+   +----+
  //     | A |--------->-<----| D |<--| c1 |
  //     +---+          v     +---+   +----+
  //       |         +----+
  //   +-------+     | v1 |
  //   |       |     +----+
  //   v       v        |
  // +---+   +---+      v     +---+
  // | B |   | C |<-----+---->| E |
  // +---+   +---+            +---+
  ProgramGraphBuilder builder;
  auto mod = builder.AddModule("mod");
  auto fn = builder.AddFunction("fn", mod);
  auto v1 = builder.AddVariable("v1", fn);   // 0
  auto c1 = builder.AddConstant("c1");       // 1
  auto a = builder.AddInstruction("a", fn);  // 2
  auto b = builder.AddInstruction("b", fn);  // 3
  auto c = builder.AddInstruction("c", fn);  // 4
  auto d = builder.AddInstruction("d", fn);  // 5
  auto e = builder.AddInstruction("e", fn);  // 6
  CHECK(builder.AddControlEdge(0, builder.GetRootNode(), a).status().ok());
  CHECK(builder.AddControlEdge(0, a, b).status().ok());
  CHECK(builder.AddControlEdge(0, a, c).status().ok());
  CHECK(builder.AddControlEdge(0, a, d).status().ok());
  CHECK(builder.AddControlEdge(0, a, e).status().ok());
  CHECK(builder.AddDataEdge(0, c1, d).status().ok());
  CHECK(builder.AddDataEdge(0, a, v1).status().ok());
  CHECK(builder.AddDataEdge(0, d, v1).status().ok());
  CHECK(builder.AddDataEdge(0, v1, c).status().ok());
  CHECK(builder.AddDataEdge(0, v1, e).status().ok());

  const ProgramGraph graph = builder.Build().ValueOrDie();

  // The CDF for this looks like:
  //
  //     +---+                +---+
  //     | A |--------->+<----| D |
  //     +---+          |     +---+
  //       |            |
  //   +-------+        |
  //   |       |        |
  //   v       v        |
  // +---+   +---+      v     +---+
  // | B |   | C |<-----+---->| E |
  // +---+   +---+            +---+
  //
  CDFGBuilder cdfgBuilder;
  const ProgramGraph cdfg = cdfgBuilder.Build(graph);

  ASSERT_EQ(cdfg.node_size(), 6);
  ASSERT_EQ(cdfg.edge_size(), 9);

  // Check node equivalence.
  const auto map = NodeListToTranslationMap(cdfgBuilder.GetNodeList());
  ASSERT_EQ(map.size(), 6);
  for (const auto& it : map) {
    const auto& oldNode = graph.node(it.first);
    const auto& newNode = cdfg.node(it.second);
    EXPECT_EQ(oldNode.type(), newNode.type());
    EXPECT_EQ(oldNode.text(), newNode.text());
  }

  // Build the data-flow graph.
  vector<absl::flat_hash_set<int>> dfg;
  dfg.reserve(cdfg.node_size());
  for (int i = 0; i < cdfg.node_size(); ++i) {
    dfg.push_back({});
  }
  for (int i = 0; i < cdfg.edge_size(); ++i) {
    const auto& edge = cdfg.edge(i);
    if (edge.flow() == Edge::DATA) {
      dfg[edge.source()].insert(edge.target());
    }
  }

  // Check the replacement data flow edges.
  EXPECT_TRUE(dfg[1].contains(3));  // A -> C
  EXPECT_TRUE(dfg[1].contains(5));  // A -> E
  EXPECT_TRUE(dfg[4].contains(3));  // D -> C
  EXPECT_TRUE(dfg[4].contains(5));  // D -> E
}

}  // anonymous namespace
}  // namespace format
}  // namespace graph
}  // namespace programl

TEST_MAIN();
