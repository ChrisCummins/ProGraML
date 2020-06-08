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
#include "programl/graph/analysis/dominance.h"

#include "labm8/cpp/logging.h"
#include "labm8/cpp/status.h"
#include "labm8/cpp/test.h"
#include "programl/graph/program_graph_builder.h"
#include "programl/test/analysis_test_macros.h"
#include "programl/test/llvm_program_graphs.h"

using labm8::Status;

namespace programl {
namespace graph {
namespace analysis {
namespace {

class DominanceAnalysisTest : public labm8::Test {
 public:
  DominanceAnalysisTest() {
    {
      // A four statement graph with one data element.
      //
      //             +---+============++
      //    +--------+ a +--------+   ||  +---+
      //    |        +---+        |   ++=>| v1|
      //    V                     v       +---+
      //  +---+                 +-+-+       ||
      //  | b |                 | c +<======++
      //  +-+-+                 +-+-+
      //    |        +---+        |
      //    +------->+ d +<-------+
      //             +---+
      ProgramGraphBuilder builder;
      auto mod = builder.AddModule("mod");
      auto fn = builder.AddFunction("fn", mod);
      auto a = builder.AddInstruction("a", fn);
      auto b = builder.AddInstruction("b", fn);
      auto c = builder.AddInstruction("c", fn);
      auto d = builder.AddInstruction("d", fn);
      auto v1 = builder.AddVariable("v1", fn);
      CHECK(builder.AddControlEdge(0, builder.GetRootNode(), a).ok());
      CHECK(builder.AddControlEdge(0, a, b).ok());
      CHECK(builder.AddControlEdge(0, a, c).ok());
      CHECK(builder.AddControlEdge(0, b, d).ok());
      CHECK(builder.AddControlEdge(0, c, d).ok());
      CHECK(builder.AddDataEdge(0, a, v1).ok());
      CHECK(builder.AddDataEdge(0, v1, c).ok());
      g1_ = builder.Build().ValueOrDie();
    }
    {
      // A five statement graph with one data element.
      //
      // This is the same as g1, but the extra statement "e" means
      // that 'a' no longer dominates all of the other nodes.
      //
      //                        +---+============++
      //  +---+        +--------+ a +--------+   ||  +---+
      //  | e |        |        +---+        |   ++=>| v1|
      //  +-+-+        V                     v       +---+
      //    |        +---+                 +-+-+       ||
      //    +------->+ b |                 | c +<======++
      //             +-+-+                 +-+-+
      //               |        +---+        |
      //               +------->+ d +<-------+
      //                        +---+
      ProgramGraphBuilder builder;
      auto mod = builder.AddModule("mod");
      auto fn = builder.AddFunction("fn", mod);
      auto a = builder.AddInstruction("a", fn);
      auto b = builder.AddInstruction("b", fn);
      auto c = builder.AddInstruction("c", fn);
      auto d = builder.AddInstruction("d", fn);
      auto e = builder.AddInstruction("e", fn);
      auto v1 = builder.AddVariable("v1", fn);
      CHECK(builder.AddControlEdge(0, builder.GetRootNode(), a).ok());
      CHECK(builder.AddControlEdge(0, a, b).ok());
      CHECK(builder.AddControlEdge(0, a, c).ok());
      CHECK(builder.AddControlEdge(0, b, d).ok());
      CHECK(builder.AddControlEdge(0, c, d).ok());
      CHECK(builder.AddDataEdge(0, a, v1).ok());
      CHECK(builder.AddDataEdge(0, v1, c).ok());
      CHECK(builder.AddControlEdge(0, e, b).ok());
      g2_ = builder.Build().ValueOrDie();
    }
  }

 protected:
  ProgramGraph g1_;
  ProgramGraph g2_;
};

TEST_F(DominanceAnalysisTest, AnnotateG1FromRootA) {
  DominanceAnalysis analysis(g1_);
  analysis.Init();
  ProgramGraphFeatures f;

  ASSERT_OK(analysis.RunOne(1, &f));

  EXPECT_ACTIVE_NODE_COUNT(f, 4);
  EXPECT_STEP_COUNT(f, 4);

  // Features.
  EXPECT_NOT_ROOT(f, 0);
  EXPECT_ROOT(f, 1);
  EXPECT_NOT_ROOT(f, 2);
  EXPECT_NOT_ROOT(f, 3);
  EXPECT_NOT_ROOT(f, 4);
  EXPECT_NOT_ROOT(f, 5);

  // Labels.
  EXPECT_NODE_FALSE(f, 0);
  EXPECT_NODE_TRUE(f, 1);
  EXPECT_NODE_TRUE(f, 2);
  EXPECT_NODE_TRUE(f, 3);
  EXPECT_NODE_TRUE(f, 4);
  EXPECT_NODE_FALSE(f, 5);
}

TEST_F(DominanceAnalysisTest, AnnotateG2FromRootA) {
  DominanceAnalysis analysis(g2_);
  analysis.Init();
  ProgramGraphFeatures f;

  ASSERT_OK(analysis.RunOne(1, &f));

  EXPECT_ACTIVE_NODE_COUNT(f, 2);
  EXPECT_STEP_COUNT(f, 5);

  // Features.
  EXPECT_NOT_ROOT(f, 0);
  EXPECT_ROOT(f, 1);
  EXPECT_NOT_ROOT(f, 2);
  EXPECT_NOT_ROOT(f, 3);
  EXPECT_NOT_ROOT(f, 4);
  EXPECT_NOT_ROOT(f, 5);

  // Labels.
  EXPECT_NODE_FALSE(f, 0);
  EXPECT_NODE_TRUE(f, 1);
  EXPECT_NODE_FALSE(f, 2);
  EXPECT_NODE_TRUE(f, 3);
  EXPECT_NODE_FALSE(f, 4);
  EXPECT_NODE_FALSE(f, 5);
}

TEST_F(DominanceAnalysisTest, RealLlvmGraphs) {
  const auto llvmGraphs = test::ReadLlvmProgramGraphs();
  for (size_t i = 0; i < std::min(llvmGraphs.size(), size_t(50)); ++i) {
    const auto& graph = llvmGraphs[i];
    DominanceAnalysis analysis(graph);
    ProgramGraphFeaturesList features;
    auto status = analysis.Run(&features);
    if (status.ok()) {
      EXPECT_TRUE(features.graph_size());
    }
  }
}

}  // anonymous namespace
}  // namespace analysis
}  // namespace graph
}  // namespace programl

TEST_MAIN();