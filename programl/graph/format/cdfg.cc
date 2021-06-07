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
#include "programl/graph/format/cdfg.h"

#include "absl/container/flat_hash_set.h"
#include "labm8/cpp/logging.h"
#include "programl/graph/features.h"
#include "programl/proto/program_graph.pb.h"

namespace programl {
namespace graph {
namespace format {

ProgramGraph CDFGBuilder::Build(const ProgramGraph& graph) {
  Clear();

  ProgramGraph cdfg;

  absl::flat_hash_map<int, int> nodeMap;

  // Insert the instruction nodes.
  for (int i = 0; i < graph.node_size(); ++i) {
    const Node& node = graph.node(i);

    if (node.type() == Node::INSTRUCTION) {
      nodeMap.insert({i, cdfg.node_size()});
      nodeList_.push_back(i);
      Node* newNode = cdfg.add_node();
      newNode->set_text(node.text());
      newNode->set_function(node.function());
      *newNode->mutable_features() = node.features();
      AddScalarFeature(newNode, "source_node_index", int64_t(i));
    }
  }

  // A map from variables to instructions which define it.
  absl::flat_hash_map<int, absl::flat_hash_set<int>> defs{};
  // A map from variables to instructions which use it.
  absl::flat_hash_map<int, absl::flat_hash_set<int>> uses{};

  for (int i = 0; i < graph.edge_size(); ++i) {
    const Edge& edge = graph.edge(i);
    Edge* newEdge;

    switch (edge.flow()) {
      case Edge::CONTROL:
      case Edge::CALL:
        // Call and control edges are copied verbatim.
        newEdge = cdfg.add_edge();
        newEdge->set_flow(edge.flow());
        newEdge->set_source(nodeMap[edge.source()]);
        newEdge->set_target(nodeMap[edge.target()]);
        *newEdge->mutable_features() = edge.features();
        break;
      case Edge::DATA:
        // Only variables are represented with data edges, constants are
        // excluded in CDFGs. Defer the creation of instruction-to-instruction
        // data edges until we have visited all edges in the graph.
        if (graph.node(edge.source()).type() == Node::VARIABLE &&
            graph.node(edge.target()).type() == Node::INSTRUCTION) {
          uses[edge.source()].insert(nodeMap[edge.target()]);
        } else if (graph.node(edge.target()).type() == Node::VARIABLE &&
                   graph.node(edge.source()).type() == Node::INSTRUCTION) {
          defs[edge.target()].insert(nodeMap[edge.source()]);
        }
        break;
      default:
        LOG(FATAL) << "unreachable";
    }
  }

  // Create the data flow edges. CDFG doesn't represent variables in the graphs,
  // so replace connections between instructions and variables with direct
  // instruction-to-instruction connections. For example, let's say we have an
  // instruction A which defines a variable v1, and instructions B and C use v1:
  //
  //     A -> v1 --> B
  //             \-> C
  //
  // The CDFG representation would be:
  //
  //     A --> B
  //       \-> C
  //
  for (const auto& it : defs) {
    int variable = it.first;

    for (const auto& def : it.second) {
      for (const auto& use : uses[variable]) {
        Edge* newEdge = cdfg.add_edge();
        newEdge->set_flow(Edge::DATA);
        newEdge->set_source(def);
        newEdge->set_target(use);
      }
    }
  }

  *cdfg.mutable_function() = graph.function();

  return cdfg;
}

const std::vector<int>& CDFGBuilder::GetNodeList() const { return nodeList_; }

void CDFGBuilder::Clear() { nodeList_.clear(); }

absl::flat_hash_map<int, int> NodeListToTranslationMap(const std::vector<int>& nodeList) {
  absl::flat_hash_map<int, int> map;
  map.reserve(nodeList.size());
  for (size_t i = 0; i < nodeList.size(); ++i) {
    map.insert({nodeList[i], i});
  }
  return map;
}

}  // namespace format
}  // namespace graph
}  // namespace programl
