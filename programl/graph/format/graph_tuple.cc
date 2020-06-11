// This file defines the pythons bindings for ProgramGraphBuilder.
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
#include "programl/graph/format/graph_tuple.h"

namespace programl {
namespace graph {
namespace format {

void GraphTuple::Clear() {
  graph_size_ = 0;
  node_size_ = 0;
  node_sizes_.clear();
  edge_sizes_.clear();
  adjacencies_[0].clear();
  adjacencies_[1].clear();
  adjacencies_[2].clear();
  edge_positions_[0].clear();
  edge_positions_[1].clear();
  edge_positions_[2].clear();
}

labm8::Status GraphTuple::AddProgramGraph(const ProgramGraph& graph) {
  for (int i = 0; i < graph.edge_size(); ++i) {
    const Edge& edge = graph.edge(i);

    (*mutable_adjacencies())[edge.flow()].push_back(
        {edge.source() + node_size(), edge.target() + node_size()});
    (*mutable_edge_positions())[edge.flow()].push_back(edge.position());
  }

  ++graph_size_;
  set_node_size(node_size() + graph.node_size());
  add_node_size(graph.node_size());
  add_edge_size(graph.edge_size());

  return labm8::Status::OK;
}

}  // namespace format
}  // namespace graph
}  // namespace programl
