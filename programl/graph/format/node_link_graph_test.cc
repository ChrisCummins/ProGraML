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

#include "nlohmann/json.hpp"
#include "programl/graph/format/node_link_graph.h"

#include "labm8/cpp/status.h"
#include "labm8/cpp/string.h"
#include "labm8/cpp/test.h"
#include "programl/proto/program_graph.pb.h"

using json = nlohmann::json;
using labm8::Status;

namespace programl {
namespace graph {
namespace format {
namespace {

TEST(ProgramGraphToNodeLinkGraph, EmptyGraph) {
  ProgramGraph graph;
  auto nodeLinkGraph = json::object({});
  Status status = ProgramGraphToNodeLinkGraph(graph, &nodeLinkGraph);

  ASSERT_TRUE(status.ok());
  EXPECT_TRUE(nodeLinkGraph["directed"]);
  EXPECT_TRUE(nodeLinkGraph["multigraph"]);
}

TEST(CreateFeatureArray, EmptyFeature) {
  Feature feature;
  auto featureArray = json::array({});
  Status status = detail::CreateFeatureArray(feature, &featureArray);

  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.error_code(), labm8::error::Code::INTERNAL);
  ASSERT_EQ(status.error_message(), "Empty feature");
}

TEST(CreateFeatureArray, BytesList) {
  Feature feature;
  BytesList* values = feature.mutable_bytes_list();
  values->add_value("hello");
  values->add_value("world");
  auto featureArray = json::array({});
  std::vector<string> expected{"hello", "world"};
  Status status = detail::CreateFeatureArray(feature, &featureArray);

  ASSERT_TRUE(status.ok());
  ASSERT_EQ(featureArray, expected);
}

TEST(CreateFeatureArray, FloatList) {
  Feature feature;
  FloatList* values = feature.mutable_float_list();
  values->add_value(3.14);
  values->add_value(1.8);
  values->add_value(2);
  auto featureArray = json::array({});
  std::vector<float> expected{3.14, 1.8, 2};
  Status status = detail::CreateFeatureArray(feature, &featureArray);

  ASSERT_TRUE(status.ok());
  ASSERT_EQ(featureArray, expected);
}

TEST(CreateFeatureArray, Int64List) {
  Feature feature;
  Int64List* values = feature.mutable_int64_list();
  values->add_value(1);
  values->add_value(2);
  values->add_value(3);
  auto featureArray = json::array({});
  std::vector<int64_t> expected{1, 2, 3};
  Status status = detail::CreateFeatureArray(feature, &featureArray);

  ASSERT_TRUE(status.ok());
  ASSERT_EQ(featureArray, expected);
}

}  // namespace
}  // namespace format
}  // namespace graph
}  // namespace programl

TEST_MAIN();
