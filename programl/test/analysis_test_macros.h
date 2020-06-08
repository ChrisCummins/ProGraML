// This file defines macros for writing analaysis tests.
//
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
#pragma once

// Read a scalar int64 feature value.
#define SCALAR_INT64_GRAPH_FEATURE(programGraphFeatures, featureName)          \
  (*(programGraphFeatures).mutable_features()->mutable_feature())[featureName] \
      .int64_list()                                                            \
      .value(0)

// Read a scalar int64 node feature value.
#define SCALAR_INT64_NODE_FEATURE(programGraphFeatures, featureName, \
                                  nodeIndex)                         \
  (*(programGraphFeatures)                                           \
        .mutable_node_features()                                     \
        ->mutable_feature_list())[featureName]                       \
      .feature(nodeIndex)                                            \
      .int64_list()                                                  \
      .value(0)

#define EXPECT_STEP_COUNT(programGraphFeatures, val)            \
  EXPECT_EQ(SCALAR_INT64_GRAPH_FEATURE(programGraphFeatures,    \
                                       "data_flow_step_count"), \
            (val))

#define EXPECT_ACTIVE_NODE_COUNT(programGraphFeatures, val)            \
  EXPECT_EQ(SCALAR_INT64_GRAPH_FEATURE(programGraphFeatures,           \
                                       "data_flow_active_node_count"), \
            (val))

#define EXPECT_NODE_FALSE(programGraphFeatures, nodeIndex)                     \
  EXPECT_EQ(SCALAR_INT64_NODE_FEATURE(programGraphFeatures, "data_flow_value", \
                                      nodeIndex),                              \
            0)

#define EXPECT_NODE_TRUE(programGraphFeatures, nodeIndex)                      \
  EXPECT_EQ(SCALAR_INT64_NODE_FEATURE(programGraphFeatures, "data_flow_value", \
                                      nodeIndex),                              \
            1)

#define EXPECT_NOT_ROOT(programGraphFeatures, nodeIndex)                 \
  EXPECT_EQ(SCALAR_INT64_NODE_FEATURE(programGraphFeatures,              \
                                      "data_flow_root_node", nodeIndex), \
            0)

#define EXPECT_ROOT(programGraphFeatures, nodeIndex)                     \
  EXPECT_EQ(SCALAR_INT64_NODE_FEATURE(programGraphFeatures,              \
                                      "data_flow_root_node", nodeIndex), \
            1)
