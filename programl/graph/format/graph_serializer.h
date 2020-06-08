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

#include "programl/proto/node.pb.h"
#include "programl/proto/program_graph.pb.h"

#include "absl/container/flat_hash_map.h"

#include <vector>

namespace programl {
namespace graph {
namespace format {

using absl::flat_hash_map;
using std::vector;

// Produce a serialized list of node indices for the instructions in a graph.
// Order is determined by a depth first traversal of the instructions in each
// function.
void SerializeInstructionsInProgramGraph(const ProgramGraph& graph,
                                         vector<int>* serialized, int maxNodes);

// Produce a serialized list of node indices for the instructions in a graph.
// Order is determined by a depth first traversal of the instructions in each
// function.
void SerializeInstructionsInProgramGraph(const ProgramGraph& graph,
                                         NodeIndexList* serialized,
                                         int maxNodes);

// Produce a serialized list of node indices for the instructions in a graph.
// Order is determined by a depth first traversal of the instructions in each
// function.
bool SerializeInstructionsInFunction(
    const int& root,
    const flat_hash_map<int, vector<int>>& forward_control_edges,
    vector<int>* serialized, int maxNodes);

}  // namespace format
}  // namespace graph
}  // namespace programl
