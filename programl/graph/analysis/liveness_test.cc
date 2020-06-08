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
#include "programl/graph/analysis/liveness.h"

#include "labm8/cpp/logging.h"
#include "labm8/cpp/status.h"
#include "labm8/cpp/test.h"
#include "programl/graph/program_graph_builder.h"
#include "programl/test/analysis_testutil.h"

using labm8::Status;
using ::testing::UnorderedElementsAre;

namespace programl {
namespace graph {
namespace analysis {
namespace {

class LivenessGraphBuilder {
 public:
  LivenessGraphBuilder() : builder_() {
    auto mod = builder_.AddModule("mod");
    fn_ = builder_.AddFunction("fn", mod);
  }

  Node* AddVariable() { return builder_.AddVariable("<var>", fn_); }

  Node* AddInstruction() { return builder_.AddInstruction("inst", fn_); }

  void AddControlEdge(const Node* src, const Node* dst) {
    CHECK(builder_.AddControlEdge(/*position=*/0, src, dst).ok());
  }

  void AddDataEdge(const Node* src, const Node* dst) {
    CHECK(builder_.AddDataEdge(/*position=*/0, src, dst).ok());
  }

  void AddCallEdge(const Node* src, const Node* dst) {
    CHECK(builder_.AddCallEdge(src, dst).ok());
  }

  const Node* GetRootNode() const { return builder_.GetRootNode(); }

  ProgramGraph Build() { return builder_.Build().ValueOrDie(); }

 private:
  ProgramGraphBuilder builder_;
  Function* fn_;
};

class LivenessAnalysisTest : public labm8::Test {
 public:
  LivenessAnalysisTest() {
    // Example graph taken from Wikipedia:
    // https://en.wikipedia.org/wiki/Live_variable_analysis
    //
    // // in: {}
    // b1: a = 3;
    //     b = 5;
    //     d = 4;
    //     x = 100; //x is never being used later thus not in the out set
    //     {a,b,d} if a > b then
    // // out: {a,b,d}    // union of all (in) successors of b1 => b2: {a,b},
    //                    // and b3:{b,d}
    //
    // // in: {a,b}
    // b2: c = a + b;
    //     d = 2;
    // // out: {b,d}
    //
    // // in: {b,d}
    // b3: endif
    //     c = 4;
    //     return b * d + c;
    // // out:{}
    LivenessGraphBuilder builder;

    // Variables.
    auto a = builder.AddVariable();  // 1
    auto b = builder.AddVariable();  // 2
    auto c = builder.AddVariable();  // 3
    auto d = builder.AddVariable();  // 4
    auto x = builder.AddVariable();  // 5

    // Blocks.
    auto b1 = builder.AddInstruction();   // 6
    auto b2 = builder.AddInstruction();   // 7
    auto b3a = builder.AddInstruction();  // 8
    auto b3b = builder.AddInstruction();  // 9

    // Control edges.
    builder.AddCallEdge(builder.GetRootNode(), b1);
    builder.AddControlEdge(b1, b2);
    builder.AddControlEdge(b1, b3a);
    builder.AddControlEdge(b2, b3a);
    builder.AddControlEdge(b3a, b3b);

    // Defs.
    builder.AddDataEdge(b1, a);
    builder.AddDataEdge(b1, b);
    builder.AddDataEdge(b1, d);
    builder.AddDataEdge(b1, x);
    builder.AddDataEdge(b2, c);
    builder.AddDataEdge(b2, d);
    builder.AddDataEdge(b3a, c);

    // Uses.
    builder.AddDataEdge(a, b2);
    builder.AddDataEdge(b, b2);
    builder.AddDataEdge(b, b3b);
    builder.AddDataEdge(c, b3b);
    builder.AddDataEdge(d, b3b);

    wiki_ = builder.Build();
  }

 protected:
  ProgramGraph wiki_;
};

TEST_F(LivenessAnalysisTest, WriteWithoutRead) {
  // a -> v1
  LivenessGraphBuilder builder;
  auto a = builder.AddInstruction();
  auto v1 = builder.AddVariable();
  builder.AddCallEdge(builder.GetRootNode(), a);
  builder.AddDataEdge(a, v1);
  const auto graph = builder.Build();

  LivenessAnalysis analysis(graph);
  analysis.Init();

  const auto liveOutSets = analysis.live_out_sets();
  ASSERT_EQ(liveOutSets.size(), 3);
  ASSERT_THAT(liveOutSets[0], UnorderedElementsAre());
  ASSERT_THAT(liveOutSets[1], UnorderedElementsAre());
  ASSERT_THAT(liveOutSets[2], UnorderedElementsAre());
}

TEST_F(LivenessAnalysisTest, ReadAfterWrite) {
  // a -> v1 -> b
  LivenessGraphBuilder builder;
  auto a = builder.AddInstruction();
  auto v1 = builder.AddVariable();
  auto b = builder.AddInstruction();
  builder.AddCallEdge(builder.GetRootNode(), a);
  builder.AddControlEdge(a, b);
  builder.AddDataEdge(a, v1);
  builder.AddDataEdge(v1, b);
  const auto graph = builder.Build();

  LivenessAnalysis analysis(graph);
  analysis.Init();

  const auto liveOutSets = analysis.live_out_sets();
  ASSERT_EQ(liveOutSets.size(), 4);
  ASSERT_THAT(liveOutSets[0], UnorderedElementsAre());
  ASSERT_THAT(liveOutSets[1], UnorderedElementsAre(2));  // Variable is read.
  ASSERT_THAT(liveOutSets[2], UnorderedElementsAre());
  ASSERT_THAT(liveOutSets[3], UnorderedElementsAre());
}

TEST_F(LivenessAnalysisTest, WriteAfterWrite) {
  // a -> v1
  // b -> v1
  // c <- v1
  LivenessGraphBuilder builder;
  auto a = builder.AddInstruction();
  auto b = builder.AddInstruction();
  auto c = builder.AddInstruction();
  auto v1 = builder.AddVariable();
  builder.AddCallEdge(builder.GetRootNode(), a);
  builder.AddControlEdge(a, b);
  builder.AddControlEdge(b, c);
  builder.AddDataEdge(a, v1);
  builder.AddDataEdge(b, v1);
  builder.AddDataEdge(v1, c);
  const auto graph = builder.Build();

  LivenessAnalysis analysis(graph);
  analysis.Init();

  const auto liveOutSets = analysis.live_out_sets();
  ASSERT_EQ(liveOutSets.size(), 5);
  ASSERT_THAT(liveOutSets[0], UnorderedElementsAre());
  ASSERT_THAT(liveOutSets[1], UnorderedElementsAre());
  ASSERT_THAT(liveOutSets[2], UnorderedElementsAre(4));
  ASSERT_THAT(liveOutSets[3], UnorderedElementsAre());
  ASSERT_THAT(liveOutSets[4], UnorderedElementsAre());
}

TEST_F(LivenessAnalysisTest, MultiVarWrite) {
  // a -> v1
  // b -> v2
  // c <- v1
  LivenessGraphBuilder builder;
  auto a = builder.AddInstruction();
  auto b = builder.AddInstruction();
  auto c = builder.AddInstruction();
  auto v1 = builder.AddVariable();
  auto v2 = builder.AddVariable();
  builder.AddCallEdge(builder.GetRootNode(), a);
  builder.AddControlEdge(a, b);
  builder.AddControlEdge(b, c);
  builder.AddDataEdge(a, v1);
  builder.AddDataEdge(b, v2);
  builder.AddDataEdge(v1, c);
  const auto graph = builder.Build();

  LivenessAnalysis analysis(graph);
  analysis.Init();

  const auto liveOutSets = analysis.live_out_sets();
  ASSERT_EQ(liveOutSets.size(), 6);
  ASSERT_THAT(liveOutSets[0], UnorderedElementsAre());
  ASSERT_THAT(liveOutSets[1], UnorderedElementsAre(4));
  ASSERT_THAT(liveOutSets[2], UnorderedElementsAre(4));
  ASSERT_THAT(liveOutSets[3], UnorderedElementsAre());
  ASSERT_THAT(liveOutSets[4], UnorderedElementsAre());
  ASSERT_THAT(liveOutSets[5], UnorderedElementsAre());
}

TEST_F(LivenessAnalysisTest, ReadAfterWriteBranchingControlFlow) {
  // if:
  //    x = 4
  // y = x
  LivenessGraphBuilder builder;
  auto s0 = builder.AddInstruction();  // 1
  auto s1 = builder.AddInstruction();  // 2
  auto s2 = builder.AddInstruction();  // 3
  auto x = builder.AddVariable();      // 4
  auto y = builder.AddVariable();      // 5
  builder.AddCallEdge(builder.GetRootNode(), s0);
  builder.AddControlEdge(s0, s1);
  builder.AddControlEdge(s0, s2);
  builder.AddControlEdge(s1, s2);
  builder.AddDataEdge(s1, x);
  builder.AddDataEdge(x, s2);
  builder.AddDataEdge(s2, y);
  const auto graph = builder.Build();

  LivenessAnalysis analysis(graph);
  analysis.Init();

  const auto liveOutSets = analysis.live_out_sets();
  ASSERT_EQ(liveOutSets.size(), 6);
  ASSERT_THAT(liveOutSets[0], UnorderedElementsAre());   // <root>
  ASSERT_THAT(liveOutSets[1], UnorderedElementsAre(4));  // s0
  ASSERT_THAT(liveOutSets[2], UnorderedElementsAre(4));  // s1
  ASSERT_THAT(liveOutSets[3], UnorderedElementsAre());   // s2
  ASSERT_THAT(liveOutSets[4], UnorderedElementsAre());   // x
  ASSERT_THAT(liveOutSets[5], UnorderedElementsAre());   // y
}

TEST_F(LivenessAnalysisTest, WikiLiveInSets) {
  LivenessAnalysis analysis(wiki_);
  analysis.Init();

  const auto liveInSets = analysis.live_in_sets();
  ASSERT_EQ(liveInSets.size(), 10);

  ASSERT_THAT(liveInSets[0], UnorderedElementsAre());         // <root>
  ASSERT_THAT(liveInSets[1], UnorderedElementsAre());         // a
  ASSERT_THAT(liveInSets[2], UnorderedElementsAre());         // b
  ASSERT_THAT(liveInSets[3], UnorderedElementsAre());         // c
  ASSERT_THAT(liveInSets[4], UnorderedElementsAre());         // d
  ASSERT_THAT(liveInSets[5], UnorderedElementsAre());         // x
  ASSERT_THAT(liveInSets[6], UnorderedElementsAre());         // b1
  ASSERT_THAT(liveInSets[7], UnorderedElementsAre(1, 2));     // b2
  ASSERT_THAT(liveInSets[8], UnorderedElementsAre(2, 4));     // b3a
  ASSERT_THAT(liveInSets[9], UnorderedElementsAre(2, 4, 3));  // b3b
}

TEST_F(LivenessAnalysisTest, WikiLiveOutSets) {
  LivenessAnalysis analysis(wiki_);
  analysis.Init();

  const auto liveOutSets = analysis.live_out_sets();
  ASSERT_EQ(liveOutSets.size(), 10);

  ASSERT_THAT(liveOutSets[0], UnorderedElementsAre());         // <root>
  ASSERT_THAT(liveOutSets[1], UnorderedElementsAre());         // a
  ASSERT_THAT(liveOutSets[2], UnorderedElementsAre());         // b
  ASSERT_THAT(liveOutSets[3], UnorderedElementsAre());         // c
  ASSERT_THAT(liveOutSets[4], UnorderedElementsAre());         // d
  ASSERT_THAT(liveOutSets[5], UnorderedElementsAre());         // x
  ASSERT_THAT(liveOutSets[6], UnorderedElementsAre(1, 2, 4));  // b1
  ASSERT_THAT(liveOutSets[7], UnorderedElementsAre(2, 4));     // b2
  ASSERT_THAT(liveOutSets[8], UnorderedElementsAre(2, 4, 3));  // b3a
  ASSERT_THAT(liveOutSets[9], UnorderedElementsAre());         // b3b
}

TEST_F(LivenessAnalysisTest, WikiWithRootB1) {
  LivenessAnalysis analysis(wiki_);
  analysis.Init();
  ProgramGraphFeatures g;

  ASSERT_OK(analysis.RunOne(6, &g));

  EXPECT_ACTIVE_NODE_COUNT(g, 3);
  EXPECT_STEP_COUNT(g, 3);

  // Features.
  EXPECT_NOT_ROOT(g, 0);  // <root>
  EXPECT_NOT_ROOT(g, 1);  // a
  EXPECT_NOT_ROOT(g, 2);  // b
  EXPECT_NOT_ROOT(g, 3);  // c
  EXPECT_NOT_ROOT(g, 4);  // d
  EXPECT_NOT_ROOT(g, 5);  // x
  EXPECT_ROOT(g, 6);      // b1
  EXPECT_NOT_ROOT(g, 7);  // b2
  EXPECT_NOT_ROOT(g, 8);  // b3a
  EXPECT_NOT_ROOT(g, 9);  // b3b

  // Labels.
  EXPECT_NODE_FALSE(g, 0);  // <root>
  EXPECT_NODE_TRUE(g, 1);   // a
  EXPECT_NODE_TRUE(g, 2);   // b
  EXPECT_NODE_FALSE(g, 3);  // c
  EXPECT_NODE_TRUE(g, 4);   // d
  EXPECT_NODE_FALSE(g, 5);  // x
  EXPECT_NODE_FALSE(g, 6);  // b1
  EXPECT_NODE_FALSE(g, 7);  // b2
  EXPECT_NODE_FALSE(g, 8);  // b3a
  EXPECT_NODE_FALSE(g, 9);  // b3b
}

TEST_F(LivenessAnalysisTest, WikiWithRootB2) {
  LivenessAnalysis analysis(wiki_);
  analysis.Init();
  ProgramGraphFeatures g;

  ASSERT_OK(analysis.RunOne(7, &g));
  EXPECT_ACTIVE_NODE_COUNT(g, 2);
  EXPECT_STEP_COUNT(g, 2);

  // Features.
  EXPECT_NOT_ROOT(g, 0);  // <root>
  EXPECT_NOT_ROOT(g, 1);  // a
  EXPECT_NOT_ROOT(g, 2);  // b
  EXPECT_NOT_ROOT(g, 3);  // c
  EXPECT_NOT_ROOT(g, 4);  // d
  EXPECT_NOT_ROOT(g, 5);  // x
  EXPECT_NOT_ROOT(g, 6);  // b1
  EXPECT_ROOT(g, 7);      // b2
  EXPECT_NOT_ROOT(g, 8);  // b3a
  EXPECT_NOT_ROOT(g, 9);  // b3b

  // Labels.
  EXPECT_NODE_FALSE(g, 0);  // <root>
  EXPECT_NODE_FALSE(g, 1);  // a
  EXPECT_NODE_TRUE(g, 2);   // b
  EXPECT_NODE_FALSE(g, 3);  // c
  EXPECT_NODE_TRUE(g, 4);   // d
  EXPECT_NODE_FALSE(g, 5);  // x
  EXPECT_NODE_FALSE(g, 6);  // b1
  EXPECT_NODE_FALSE(g, 7);  // b2
  EXPECT_NODE_FALSE(g, 8);  // b3a
  EXPECT_NODE_FALSE(g, 9);  // b3b
}

}  // anonymous namespace
}  // namespace analysis
}  // namespace graph
}  // namespace programl

TEST_MAIN();