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
#include <vector>

#include "programl/graph/format/graph_tuple.h"

#include "labm8/cpp/status.h"
#include "labm8/cpp/test.h"
#include "programl/proto/edge.pb.h"
#include "programl/proto/program_graph.pb.h"

using labm8::Status;

namespace programl {
namespace graph {
namespace format {
namespace {

TEST(GraphTuple, EmptyGraph) {
  GraphTuple tuple;
  EXPECT_EQ(tuple.adjacencies()[Edge::CONTROL].size(), 0);
  EXPECT_EQ(tuple.edge_positions()[Edge::CONTROL].size(), 0);
}

}  // namespace
}  // namespace format
}  // namespace graph
}  // namespace programl

TEST_MAIN();
