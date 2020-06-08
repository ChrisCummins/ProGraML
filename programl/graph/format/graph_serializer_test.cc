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
#include "programl/graph/format/graph_serializer.h"

#include "programl/proto/program_graph.pb.h"

#include <vector>

#include "labm8/cpp/test.h"

namespace programl {
namespace graph {
namespace format {
namespace {

TEST(SerializeInstructionsInProgramGraph, EmptyGraph) {
  ProgramGraph graph;

  vector<int> serialized;
  SerializeInstructionsInProgramGraph(graph, &serialized, /*nodeMax=*/1000);
  ASSERT_EQ(0, serialized.size());
}

TEST(SerializeInstructionsInProgramGraph, RootNodeOnly) {
  ProgramGraph graph;
  Node* root = graph.add_node();
  root->set_type(Node::INSTRUCTION);

  vector<int> serialized;
  SerializeInstructionsInProgramGraph(graph, &serialized, /*nodeMax=*/1000);
  ASSERT_EQ(0, serialized.size());
}

TEST(SerializeInstructionsInProgramGraph, SingleFunction) {
  ProgramGraph graph;
  Node* root = graph.add_node();
  root->set_type(Node::INSTRUCTION);
  Node* a = graph.add_node();
  a->set_type(Node::INSTRUCTION);
  a->set_function(0);
  Edge* root_to_a = graph.add_edge();
  root_to_a->set_flow(Edge::CALL);
  root_to_a->set_source(0);
  root_to_a->set_target(1);

  vector<int> serialized;
  SerializeInstructionsInProgramGraph(graph, &serialized, /*nodeMax=*/1000);
  ASSERT_EQ(1, serialized.size());
  ASSERT_EQ(1, serialized[0]);
}

TEST(SerializeInstructionsInProgramGraph, SingleFunctionWithLoop) {
  ProgramGraph graph;
  Node* root = graph.add_node();
  root->set_type(Node::INSTRUCTION);
  Node* a = graph.add_node();
  a->set_type(Node::INSTRUCTION);
  a->set_function(0);
  Edge* root_to_a = graph.add_edge();
  root_to_a->set_flow(Edge::CALL);
  root_to_a->set_source(0);
  root_to_a->set_target(1);
  Node* b = graph.add_node();
  b->set_type(Node::INSTRUCTION);
  Edge* a_to_b = graph.add_edge();
  a_to_b->set_flow(Edge::CONTROL);
  a_to_b->set_source(1);
  a_to_b->set_target(2);
  Edge* b_to_a = graph.add_edge();
  b_to_a->set_flow(Edge::CONTROL);
  b_to_a->set_source(2);
  b_to_a->set_target(1);

  vector<int> serialized;
  SerializeInstructionsInProgramGraph(graph, &serialized, /*nodeMax=*/1000);
  ASSERT_EQ(2, serialized.size());
  ASSERT_EQ(1, serialized[0]);
  ASSERT_EQ(2, serialized[1]);
}

}  // namespace
}  // namespace format
}  // namespace graph
}  // namespace programl

TEST_MAIN();
