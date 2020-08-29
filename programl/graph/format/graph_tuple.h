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
#pragma once

#include <array>
#include <utility>
#include <vector>

#include "labm8/cpp/status.h"
#include "programl/proto/program_graph.pb.h"

namespace programl {
namespace graph {
namespace format {

class GraphTuple {
 public:
  GraphTuple() = default;

  labm8::Status AddProgramGraph(const ProgramGraph& graph);

  // Remove all elements from the graph tuple.
  void Clear();

  const std::array<std::vector<std::pair<int, int>>, 3>& adjacencies() const {
    return adjacencies_;
  }
  const std::array<std::vector<int>, 3>& edge_positions() const { return edge_positions_; }

  inline size_t graph_size() const { return graph_size_; }
  inline size_t node_size() const { return node_size_; }
  inline size_t edge_size() {
    return adjacencies_[0].size() + adjacencies_[1].size() + adjacencies_[2].size();
  }
  inline size_t control_edge_size() const { return adjacencies_[Edge::CONTROL].size(); }
  inline size_t data_edge_size() const { return adjacencies_[Edge::DATA].size(); }
  inline size_t call_edge_size() const { return adjacencies_[Edge::CALL].size(); }
  inline const std::vector<size_t>& node_sizes() const { return node_sizes_; }
  inline const std::vector<size_t>& edge_sizes() const { return edge_sizes_; }

 protected:
  std::array<std::vector<std::pair<int, int>>, 3>* mutable_adjacencies() { return &adjacencies_; }

  std::array<std::vector<int>, 3>* mutable_edge_positions() { return &edge_positions_; }

  inline void set_node_size(size_t node_size) { node_size_ = node_size; }

  inline void add_node_size(size_t node_size) { node_sizes_.push_back(node_size); }

  inline void add_edge_size(size_t edge_size) { edge_sizes_.push_back(edge_size); }

 private:
  size_t graph_size_;
  size_t node_size_;
  std::vector<size_t> node_sizes_;
  std::vector<size_t> edge_sizes_;
  std::array<std::vector<std::pair<int, int>>, 3> adjacencies_;
  std::array<std::vector<int>, 3> edge_positions_;
};

}  // namespace format
}  // namespace graph
}  // namespace programl
