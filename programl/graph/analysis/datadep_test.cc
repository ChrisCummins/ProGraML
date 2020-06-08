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
#include "programl/graph/analysis/datadep.h"

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

class DatadepAnalysisTest : public labm8::Test {
 public:
  DatadepAnalysisTest() {
    ProgramGraphBuilder builder;
    //  A
    //  | \
    //  v  v1
    //  B /
    //  | \
    //  v  \
    //  C   v2
    //  |  /
    //  v /
    //  D
    auto mod = builder.AddModule("mod");
    auto fn = builder.AddFunction("fn", mod);
    auto a = builder.AddInstruction("a", fn);  // 1
    auto b = builder.AddInstruction("b", fn);  // 2
    auto c = builder.AddInstruction("c", fn);  // 3
    auto d = builder.AddInstruction("d", fn);  // 4

    auto v1 = builder.AddVariable("v1", fn);  // 5
    auto v2 = builder.AddVariable("v2", fn);  // 6

    CHECK(builder.AddControlEdge(0, builder.GetRootNode(), a).status().ok());
    CHECK(builder.AddControlEdge(0, a, b).status().ok());
    CHECK(builder.AddControlEdge(0, b, c).status().ok());
    CHECK(builder.AddControlEdge(0, c, d).status().ok());

    CHECK(builder.AddDataEdge(0, a, v1).status().ok());
    CHECK(builder.AddDataEdge(0, v1, b).status().ok());
    CHECK(builder.AddDataEdge(0, b, v2).status().ok());
    CHECK(builder.AddDataEdge(0, v2, d).status().ok());

    graph_ = builder.Build().ValueOrDie();
  }

 protected:
  ProgramGraph graph_;
};

TEST_F(DatadepAnalysisTest, FromRootV1) {
  DatadepAnalysis analysis(graph_);
  analysis.Init();
  ProgramGraphFeatures f;

  ASSERT_OK(analysis.RunOne(5, &f));
  EXPECT_ACTIVE_NODE_COUNT(f, 1);
  EXPECT_STEP_COUNT(f, 1);

  EXPECT_NOT_ROOT(f, 0);  // <root>
  EXPECT_NOT_ROOT(f, 1);  // a
  EXPECT_NOT_ROOT(f, 2);  // b
  EXPECT_NOT_ROOT(f, 3);  // c
  EXPECT_NOT_ROOT(f, 4);  // d
  EXPECT_ROOT(f, 5);      // v1
  EXPECT_NOT_ROOT(f, 6);  // v2

  EXPECT_NODE_FALSE(f, 0);  // <root>
  EXPECT_NODE_TRUE(f, 1);   // a
  EXPECT_NODE_FALSE(f, 2);  // b
  EXPECT_NODE_FALSE(f, 3);  // c
  EXPECT_NODE_FALSE(f, 4);  // d
  EXPECT_NODE_FALSE(f, 5);  // v1
  EXPECT_NODE_FALSE(f, 6);  // v2
}

TEST_F(DatadepAnalysisTest, FromRootv2) {
  DatadepAnalysis analysis(graph_);
  analysis.Init();
  ProgramGraphFeatures f;

  ASSERT_OK(analysis.RunOne(6, &f));
  EXPECT_ACTIVE_NODE_COUNT(f, 2);
  EXPECT_STEP_COUNT(f, 3);

  EXPECT_NOT_ROOT(f, 0);  // <root>
  EXPECT_NOT_ROOT(f, 1);  // a
  EXPECT_NOT_ROOT(f, 2);  // b
  EXPECT_NOT_ROOT(f, 3);  // c
  EXPECT_NOT_ROOT(f, 4);  // d
  EXPECT_NOT_ROOT(f, 5);  // v1
  EXPECT_ROOT(f, 6);      // v2

  EXPECT_NODE_FALSE(f, 0);  // <root>
  EXPECT_NODE_TRUE(f, 1);   // a
  EXPECT_NODE_TRUE(f, 2);   // b
  EXPECT_NODE_FALSE(f, 3);  // c
  EXPECT_NODE_FALSE(f, 4);  // d
  EXPECT_NODE_FALSE(f, 5);  // v1
  EXPECT_NODE_FALSE(f, 6);  // v2
}

TEST_F(DatadepAnalysisTest, FromRootC) {
  DatadepAnalysis analysis(graph_);
  analysis.Init();
  ProgramGraphFeatures f;

  ASSERT_OK(analysis.RunOne(3, &f));
  EXPECT_ACTIVE_NODE_COUNT(f, 1);
  EXPECT_STEP_COUNT(f, 0);

  EXPECT_NOT_ROOT(f, 0);  // <root>
  EXPECT_NOT_ROOT(f, 1);  // a
  EXPECT_NOT_ROOT(f, 2);  // b
  EXPECT_ROOT(f, 3);      // c
  EXPECT_NOT_ROOT(f, 4);  // d
  EXPECT_NOT_ROOT(f, 5);  // v1
  EXPECT_NOT_ROOT(f, 6);  // v2

  EXPECT_NODE_FALSE(f, 0);  // <root>
  EXPECT_NODE_FALSE(f, 1);  // a
  EXPECT_NODE_FALSE(f, 2);  // b
  EXPECT_NODE_TRUE(f, 3);   // c
  EXPECT_NODE_FALSE(f, 4);  // d
  EXPECT_NODE_FALSE(f, 5);  // v1
  EXPECT_NODE_FALSE(f, 6);  // v2
}

TEST_F(DatadepAnalysisTest, FromRootD) {
  DatadepAnalysis analysis(graph_);
  analysis.Init();
  ProgramGraphFeatures f;

  ASSERT_OK(analysis.RunOne(4, &f));
  EXPECT_ACTIVE_NODE_COUNT(f, 3);
  EXPECT_STEP_COUNT(f, 4);

  EXPECT_NOT_ROOT(f, 0);  // <root>
  EXPECT_NOT_ROOT(f, 1);  // a
  EXPECT_NOT_ROOT(f, 2);  // b
  EXPECT_NOT_ROOT(f, 3);  // c
  EXPECT_ROOT(f, 4);      // d
  EXPECT_NOT_ROOT(f, 5);  // v1
  EXPECT_NOT_ROOT(f, 6);  // v2

  EXPECT_NODE_FALSE(f, 0);  // <root>
  EXPECT_NODE_TRUE(f, 1);   // a
  EXPECT_NODE_TRUE(f, 2);   // b
  EXPECT_NODE_FALSE(f, 3);  // c
  EXPECT_NODE_TRUE(f, 4);   // d
  EXPECT_NODE_FALSE(f, 5);  // v1
  EXPECT_NODE_FALSE(f, 6);  // v2
}

TEST_F(DatadepAnalysisTest, RealLlvmGraphs) {
  for (const auto& graph : test::ReadLlvmProgramGraphs()) {
    DatadepAnalysis analysis(graph);
    ProgramGraphFeaturesList features;
    ASSERT_OK(analysis.Run(&features));
    EXPECT_TRUE(features.graph_size());
  }
}

}  // anonymous namespace
}  // namespace analysis
}  // namespace graph
}  // namespace programl

TEST_MAIN();