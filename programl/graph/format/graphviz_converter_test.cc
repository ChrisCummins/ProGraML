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
#include <iostream>

#include "programl/graph/format/graphviz_converter.h"

#include "labm8/cpp/status.h"
#include "labm8/cpp/test.h"
#include "programl/proto/program_graph.pb.h"

namespace programl {
namespace graph {
namespace format {
namespace {

TEST(SerializeGraphVizToString, EmptyGraph) {
  std::ostream ostream;
  ProgramGraph graph;
  EXPECT_EQ(SerializeGraphVizToString(graph, &ostream), labm8::Status::OK);
  EXPECT_NE(ostream.str(), "");
}

TEST(SerializeGraphVizToString, NonEmptyGraph) {
  std::ostream a;
  std::ostream b;
  ProgramGraph graph;
  EXPECT_EQ(SerializeGraphVizToString(graph, &a), labm8::Status::OK);
  graph.add_node();
  EXPECT_EQ(SerializeGraphVizToString(graph, &b), labm8::Status::OK);
  EXPECT_NE(a.str(), b.str())
}

}  // namespace
}  // namespace format
}  // namespace graph
}  // namespace programl

TEST_MAIN();
