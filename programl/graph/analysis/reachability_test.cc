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
#include "programl/graph/analysis/reachability.h"

#include "labm8/cpp/logging.h"
#include "labm8/cpp/status.h"
#include "labm8/cpp/test.h"
#include "programl/graph/program_graph_builder.h"
#include "programl/test/analysis_testutil.h"
#include "programl/test/llvm_program_graphs.h"

using labm8::Status;

namespace programl {
namespace graph {
namespace analysis {
namespace {

class ReachabilityAnalysisTest : public labm8::Test {
 public:
  ReachabilityAnalysisTest() {
    ProgramGraphBuilder builder;
    Module* mod = builder.AddModule("mod");
    Function* fn = builder.AddFunction("fn", mod);
    Node* a = builder.AddInstruction("a", fn);
    Node* b = builder.AddInstruction("b", fn);
    Node* c = builder.AddInstruction("c", fn);
    Node* d = builder.AddInstruction("d", fn);
    CHECK(builder.AddControlEdge(0, builder.GetRootNode(), a).status().ok());
    CHECK(builder.AddControlEdge(0, a, b).status().ok());
    CHECK(builder.AddControlEdge(0, b, c).status().ok());
    CHECK(builder.AddControlEdge(0, c, d).status().ok());
    graph_ = builder.Build().ValueOrDie();
  }

 protected:
  ProgramGraph graph_;
};

TEST_F(ReachabilityAnalysisTest, ReachableNodeCountFromRootA) {
  ReachabilityAnalysis analysis(graph_);
  analysis.Init();
  ProgramGraphFeatures f;

  ASSERT_OK(analysis.RunOne(1, &f));
  EXPECT_ACTIVE_NODE_COUNT(f, 4);
}

TEST_F(ReachabilityAnalysisTest, ReachableNodeCountFromRootD) {
  ReachabilityAnalysis analysis(graph_);
  analysis.Init();
  ProgramGraphFeatures f;

  ASSERT_OK(analysis.RunOne(4, &f));
  EXPECT_ACTIVE_NODE_COUNT(f, 1);
}

TEST_F(ReachabilityAnalysisTest, StepCountFromRootA) {
  ReachabilityAnalysis analysis(graph_);
  analysis.Init();
  ProgramGraphFeatures f;

  ASSERT_OK(analysis.RunOne(1, &f));
  EXPECT_STEP_COUNT(f, 4);
}

TEST_F(ReachabilityAnalysisTest, StepCountFromRootD) {
  ReachabilityAnalysis analysis(graph_);
  analysis.Init();
  ProgramGraphFeatures f;

  ASSERT_OK(analysis.RunOne(4, &f));
  EXPECT_STEP_COUNT(f, 1);
}

TEST_F(ReachabilityAnalysisTest, NodeLabelsFromRootA) {
  ReachabilityAnalysis analysis(graph_);
  analysis.Init();
  ProgramGraphFeatures f;

  ASSERT_OK(analysis.RunOne(1, &f));
  EXPECT_NODE_FALSE(f, 0);
  EXPECT_NODE_TRUE(f, 1);
  EXPECT_NODE_TRUE(f, 2);
  EXPECT_NODE_TRUE(f, 3);
  EXPECT_NODE_TRUE(f, 4);
}

TEST_F(ReachabilityAnalysisTest, NodeLabelsFromRootD) {
  ReachabilityAnalysis analysis(graph_);
  analysis.Init();
  ProgramGraphFeatures f;

  ASSERT_OK(analysis.RunOne(4, &f));
  EXPECT_NODE_FALSE(f, 0);
  EXPECT_NODE_FALSE(f, 1);
  EXPECT_NODE_FALSE(f, 2);
  EXPECT_NODE_FALSE(f, 3);
  EXPECT_NODE_TRUE(f, 4);
}

TEST_F(ReachabilityAnalysisTest, RootNodeFromRootA) {
  ReachabilityAnalysis analysis(graph_);
  analysis.Init();
  ProgramGraphFeatures f;

  ASSERT_OK(analysis.RunOne(1, &f));
  EXPECT_NOT_ROOT(f, 0);
  EXPECT_ROOT(f, 1);
  EXPECT_NOT_ROOT(f, 2);
  EXPECT_NOT_ROOT(f, 3);
  EXPECT_NOT_ROOT(f, 4);
}

TEST_F(ReachabilityAnalysisTest, RootNodeFromRootD) {
  ReachabilityAnalysis analysis(graph_);
  analysis.Init();
  ProgramGraphFeatures f;

  ASSERT_OK(analysis.RunOne(4, &f));
  EXPECT_NOT_ROOT(f, 0);
  EXPECT_NOT_ROOT(f, 1);
  EXPECT_NOT_ROOT(f, 2);
  EXPECT_NOT_ROOT(f, 3);
  EXPECT_ROOT(f, 4);
}

TEST_F(ReachabilityAnalysisTest, RealLlvmGraphs) {
  for (const auto& graph : test::ReadLlvmProgramGraphs()) {
    ReachabilityAnalysis analysis(graph);
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