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

#include <deque>

#include "absl/container/flat_hash_set.h"
#include "labm8/cpp/logging.h"

namespace programl {
namespace graph {
namespace format {

// We need to identify the root node in the graph so that we can mark all
// outgoing call edges as functions to visit.
static const int ROOT_NODE = 0;

void SerializeInstructionsInProgramGraph(const ProgramGraph& graph, std::vector<int>* serialized,
                                         int maxNodes) {
  // An array of function entries to visit, where each function entry is a node
  // that is the destination of an outgoing call edge from the root node.
  std::vector<int> function_entries_to_visit;
  // A map from source node to a list of destination control edges.
  absl::flat_hash_map<int, std::vector<int>> forward_control_edges;

  // First traverse the graph edges to build the list of function entries to
  // visit and forward control edge map.
  for (int i = 0; i < graph.edge_size(); ++i) {
    const auto& edge = graph.edge(i);

    switch (edge.flow()) {
      case Edge::CONTROL:
        forward_control_edges[edge.source()].push_back(edge.target());
        break;
      case Edge::CALL:
        if (edge.source() == ROOT_NODE) {
          function_entries_to_visit.push_back(edge.target());
        }
        break;
      case Edge::DATA:
        break;
      default:
        LOG(FATAL) << "unreachable";
    }
  }

  for (const auto& function_entry : function_entries_to_visit) {
    if (SerializeInstructionsInFunction(function_entry, forward_control_edges, serialized,
                                        maxNodes)) {
      return;
    }
  }
}

void SerializeInstructionsInProgramGraph(const ProgramGraph& graph, NodeIndexList* serialized,
                                         int maxNodes) {
  std::vector<int> vec;
  SerializeInstructionsInProgramGraph(graph, &vec, maxNodes);
  for (const auto& v : vec) {
    serialized->add_node(v);
  }
}

bool SerializeInstructionsInFunction(
    const int& root, const absl::flat_hash_map<int, std::vector<int>>& forward_control_edges,
    std::vector<int>* serialized, int maxNodes) {
  // A set of visited nodes.
  absl::flat_hash_set<int> visited;
  // A queue of nodes to visit.
  std::deque<int> q({root});

  while (q.size()) {
    const int node = q.front();
    q.pop_front();

    // Mark the node as visited.
    visited.insert(node);

    // Emit the node in the serialized node list.
    serialized->push_back(node);
    if (serialized->size() >= maxNodes) {
      return true;
    }

    // Add the unvisited outgoing control edges to the queue.
    const auto& outgoing_edges = forward_control_edges.find(node);
    if (outgoing_edges != forward_control_edges.end()) {
      for (const auto& successor : outgoing_edges->second) {
        if (visited.find(successor) == visited.end()) {
          q.push_back(successor);
        }
      }
    }
  }

  return false;
}

}  // namespace format
}  // namespace graph
}  // namespace programl
