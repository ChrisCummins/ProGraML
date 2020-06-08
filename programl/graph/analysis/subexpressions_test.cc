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
#include "programl/graph/analysis/subexpressions.h"

#include "labm8/cpp/logging.h"
#include "labm8/cpp/status.h"
#include "labm8/cpp/test.h"
#include "programl/graph/program_graph_builder.h"
#include "programl/test/analysis_testutil.h"
#include "programl/test/llvm_program_graphs.h"

using labm8::Status;
using ::testing::UnorderedElementsAre;

namespace programl {
namespace graph {
namespace analysis {
namespace {

class SubexpressionGraphBuilder {
 public:
  SubexpressionGraphBuilder() : builder_() {
    auto mod = builder_.AddModule("mod");
    fn_ = builder_.AddFunction("fn", mod);
  }

  Node* AddVariable() { return builder_.AddVariable("<var>", fn_); }

  Node* AddInstruction(const string& text) {
    return builder_.AddInstruction(text, fn_);
  }

  void AddControlEdge(const Node* src, const Node* dst) {
    CHECK(builder_.AddControlEdge(0, src, dst).ok());
  }

  void AddDataEdge(int position, const Node* src, const Node* dst) {
    CHECK(builder_.AddDataEdge(position, src, dst).ok());
  }

  const Node* GetRootNode() const { return builder_.GetRootNode(); }

  ProgramGraph Build() { return builder_.Build().ValueOrDie(); }

 private:
  ProgramGraphBuilder builder_;
  Function* fn_;
};

class SubexpressionsAnalysisTest : public labm8::Test {
 public:
  SubexpressionsAnalysisTest() {
    {
      // t0 = s0(b * c)
      // t1 = s1(b * c)
      // a = s2(t0 + g)
      // d = s3(t1 * e)
      SubexpressionGraphBuilder builder;
      auto t0 = builder.AddVariable();  // 1
      auto t1 = builder.AddVariable();  // 2
      auto a = builder.AddVariable();   // 3
      auto b = builder.AddVariable();   // 4
      auto c = builder.AddVariable();   // 5
      auto d = builder.AddVariable();   // 6
      auto e = builder.AddVariable();   // 7
      auto g = builder.AddVariable();   // 8

      auto s0 = builder.AddInstruction("mul");  // 9
      auto s1 = builder.AddInstruction("mul");  // 10
      auto s2 = builder.AddInstruction("add");  // 11
      auto s3 = builder.AddInstruction("mul");  // 12

      builder.AddControlEdge(builder.GetRootNode(), s0);
      builder.AddControlEdge(s0, s1);
      builder.AddControlEdge(s1, s2);
      builder.AddControlEdge(s2, s3);

      // Defs.
      builder.AddDataEdge(0, s0, t0);
      builder.AddDataEdge(0, s1, t1);
      builder.AddDataEdge(0, s2, a);
      builder.AddDataEdge(0, s2, d);

      // Uses.
      builder.AddDataEdge(0, b, s0);
      builder.AddDataEdge(1, c, s0);
      builder.AddDataEdge(0, b, s1);
      builder.AddDataEdge(1, c, s1);
      builder.AddDataEdge(0, t0, s2);
      builder.AddDataEdge(1, g, s2);
      builder.AddDataEdge(0, t1, s3);
      builder.AddDataEdge(1, e, s3);

      wiki_ = builder.Build();
    }

    {
      // The same the wiki graph, but with the order of the operands for t0 and
      // t1 flipped, and a non-commutative sdiv instruction used.
      //
      // t0 = s0(b / c)
      // t1 = s1(c / b)
      // a = s2(t0 + g)
      // d = s3(t1 * e)
      SubexpressionGraphBuilder builder;
      auto t0 = builder.AddVariable();  // 1
      auto t1 = builder.AddVariable();  // 2
      auto a = builder.AddVariable();   // 3
      auto b = builder.AddVariable();   // 4
      auto c = builder.AddVariable();   // 5
      auto d = builder.AddVariable();   // 6
      auto e = builder.AddVariable();   // 7
      auto g = builder.AddVariable();   // 8

      auto s0 = builder.AddInstruction("sdiv");  // 9
      auto s1 = builder.AddInstruction("sdiv");  // 10
      auto s2 = builder.AddInstruction("add");   // 11
      auto s3 = builder.AddInstruction("mul");   // 12

      builder.AddControlEdge(builder.GetRootNode(), s0);
      builder.AddControlEdge(s0, s1);
      builder.AddControlEdge(s1, s2);
      builder.AddControlEdge(s2, s3);

      // Defs.
      builder.AddDataEdge(0, s0, t0);
      builder.AddDataEdge(0, s1, t1);
      builder.AddDataEdge(0, s2, a);
      builder.AddDataEdge(0, s2, d);

      // Uses.
      builder.AddDataEdge(0, b, s0);
      builder.AddDataEdge(1, c, s0);
      builder.AddDataEdge(0, c, s1);  // operands are flipped
      builder.AddDataEdge(1, b, s1);
      builder.AddDataEdge(0, t0, s2);
      builder.AddDataEdge(1, g, s2);
      builder.AddDataEdge(0, t1, s3);
      builder.AddDataEdge(1, e, s3);

      wikiWithoutSubexpressions_ = builder.Build();
    }

    {
      // The same as the wiki graph, but with the order of the operands for t0
      // and t1 flipped, and a commutative mul instruction used.
      //
      // t0 = s0(b * c)
      // t1 = s1(c * b)
      // a = s2(t0 + g)
      // d = s3(t1 * e)
      SubexpressionGraphBuilder builder;
      auto t0 = builder.AddVariable();  // 1
      auto t1 = builder.AddVariable();  // 2
      auto a = builder.AddVariable();   // 3
      auto b = builder.AddVariable();   // 4
      auto c = builder.AddVariable();   // 5
      auto d = builder.AddVariable();   // 6
      auto e = builder.AddVariable();   // 7
      auto g = builder.AddVariable();   // 8

      auto s0 = builder.AddInstruction("mul");  // 9
      auto s1 = builder.AddInstruction("mul");  // 10
      auto s2 = builder.AddInstruction("add");  // 11
      auto s3 = builder.AddInstruction("mul");  // 12

      builder.AddControlEdge(builder.GetRootNode(), s0);
      builder.AddControlEdge(s0, s1);
      builder.AddControlEdge(s1, s2);
      builder.AddControlEdge(s2, s3);

      // Defs.
      builder.AddDataEdge(0, s0, t0);
      builder.AddDataEdge(0, s1, t1);
      builder.AddDataEdge(0, s2, a);
      builder.AddDataEdge(0, s2, d);

      // Uses.
      builder.AddDataEdge(0, b, s0);
      builder.AddDataEdge(1, c, s0);
      builder.AddDataEdge(0, c, s1);  // operands are flipped
      builder.AddDataEdge(1, b, s1);
      builder.AddDataEdge(0, t0, s2);
      builder.AddDataEdge(1, g, s2);
      builder.AddDataEdge(0, t1, s3);
      builder.AddDataEdge(1, e, s3);

      wikiWithSubexpressions_ = builder.Build();
    }
  }

 protected:
  ProgramGraph wiki_;
  ProgramGraph wikiWithoutSubexpressions_;
  ProgramGraph wikiWithSubexpressions_;
};

TEST_F(SubexpressionsAnalysisTest, WikiExpressionSets) {
  SubexpressionsAnalysis analysis(wiki_);
  analysis.Init();

  const auto expressionSets = analysis.subexpression_sets();
  ASSERT_EQ(expressionSets.size(), 1);
  ASSERT_THAT(expressionSets[0], UnorderedElementsAre(9, 10));
}

TEST_F(SubexpressionsAnalysisTest, WikiWithoutSubexpressionsExpressionSets) {
  SubexpressionsAnalysis analysis(wikiWithoutSubexpressions_);
  analysis.Init();

  const auto expressionSets = analysis.subexpression_sets();
  ASSERT_EQ(expressionSets.size(), 0);
}

TEST_F(SubexpressionsAnalysisTest, WikiWithSubexpSubexpressionSets) {
  SubexpressionsAnalysis analysis(wikiWithSubexpressions_);
  analysis.Init();

  const auto expressionSets = analysis.subexpression_sets();
  ASSERT_EQ(expressionSets.size(), 1);
  ASSERT_THAT(expressionSets[0], UnorderedElementsAre(9, 10));
}

TEST_F(SubexpressionsAnalysisTest, CommutativeInstruction) {
  SubexpressionsAnalysis analysis(wiki_);
  analysis.Init();
  ProgramGraphFeatures f;

  ASSERT_OK(analysis.RunOne(9, &f));
  EXPECT_ACTIVE_NODE_COUNT(f, 2);
  EXPECT_STEP_COUNT(f, 2);

  EXPECT_NOT_ROOT(f, 0);   // <root>
  EXPECT_NOT_ROOT(f, 1);   // t0
  EXPECT_NOT_ROOT(f, 2);   // t1
  EXPECT_NOT_ROOT(f, 3);   // a
  EXPECT_NOT_ROOT(f, 4);   // b
  EXPECT_NOT_ROOT(f, 5);   // c
  EXPECT_NOT_ROOT(f, 6);   // d
  EXPECT_NOT_ROOT(f, 7);   // e
  EXPECT_NOT_ROOT(f, 8);   // g
  EXPECT_ROOT(f, 9);       // s0
  EXPECT_NOT_ROOT(f, 10);  // s1
  EXPECT_NOT_ROOT(f, 11);  // s2
  EXPECT_NOT_ROOT(f, 12);  // s3

  EXPECT_NODE_FALSE(f, 0);   // <root>
  EXPECT_NODE_FALSE(f, 1);   // t0
  EXPECT_NODE_FALSE(f, 2);   // t1
  EXPECT_NODE_FALSE(f, 3);   // a
  EXPECT_NODE_FALSE(f, 4);   // b
  EXPECT_NODE_FALSE(f, 5);   // c
  EXPECT_NODE_FALSE(f, 6);   // d
  EXPECT_NODE_FALSE(f, 7);   // e
  EXPECT_NODE_FALSE(f, 8);   // g
  EXPECT_NODE_TRUE(f, 9);    // s0
  EXPECT_NODE_TRUE(f, 10);   // s1
  EXPECT_NODE_FALSE(f, 11);  // s2
  EXPECT_NODE_FALSE(f, 12);  // s3
}

TEST_F(SubexpressionsAnalysisTest, RealLlvmGraphs) {
  for (const auto& graph : test::ReadLlvmProgramGraphs()) {
    SubexpressionsAnalysis analysis(graph);
    ProgramGraphFeaturesList features;
    if (analysis.Run(&features).ok()) {
      EXPECT_TRUE(features.graph_size());
    }
  }
}

}  // anonymous namespace
}  // namespace analysis
}  // namespace graph
}  // namespace programl

TEST_MAIN();