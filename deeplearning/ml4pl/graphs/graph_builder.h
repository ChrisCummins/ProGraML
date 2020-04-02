// This file defines a class for constructing program graphs.
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

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "deeplearning/ml4pl/graphs/programl.pb.h"
#include "labm8/cpp/string.h"

namespace ml4pl {

// An <entry, exits> pair which records the node numbers for a function's entry
// and exit statement nodes, respectively.
using FunctionEntryExits = std::pair<size_t, std::vector<size_t>>;

// A module for constructing a single program graph.
class GraphBuilder {
 public:
  GraphBuilder();

  // Construct a new function and return its number.
  std::pair<int, Function*> AddFunction(const string& name);

  // Node factories.
  std::pair<int, Node*> AddStatement(const string& text, int function);

  std::pair<int, Node*> AddIdentifier(const string& text, int function);

  std::pair<int, Node*> AddImmediate(const string& text);

  // Edge factories.

  // Add a control edge. Control edges are given a position automatically by
  // order in which they are added.
  void AddControlEdge(int sourceNode, int destinationNode);

  // Add a call edge. Call edges always have 0 position.
  void AddCallEdge(int sourceNode, int destinationNode);

  // Add a data edge with the given position.
  void AddDataEdge(int sourceNode, int destinationNode, int position);

  // Access the graph.
  const ProgramGraph& GetGraph();

 protected:
  std::pair<int, Node*> AddNode(const Node::Type& type);

  Edge* AddEdge(const Edge::Flow& flow, int sourceNode, int destinationNode,
                int position);

  size_t NextNodeNumber() const { return graph_.node_size(); }

  // Add outgoing and return call edges from a node to a function.
  void AddCallEdges(const size_t callingNode,
                    const FunctionEntryExits& calledFunction);

  // Return a mutable pointer to the graph protocol buffer.
  ProgramGraph* GetMutableProgramGraph() { return &graph_; }

  // Return the graph protocol buffer.
  const ProgramGraph& GetProgramGraph() const { return graph_; }

 private:
  ProgramGraph graph_;

  // Add a positional list of edges.
  void AddEdges(const std::vector<std::vector<size_t>>& adjacencies,
                const Edge::Flow& flow, std::vector<bool>* visitedNodes);

  // Add an unordered set of edges. All edges have position 0.
  void AddEdges(const std::vector<absl::flat_hash_set<size_t>>& adjacencies,
                const Edge::Flow& flow, std::vector<bool>* visitedNodes);

  void AddReverseEdges(
      const std::vector<std::vector<std::pair<size_t, int>>>& adjacencies,
      const Edge::Flow& flow, std::vector<bool>* visitedNodes);

  // Adjacency lists.
  std::vector<std::vector<size_t>> control_adjacencies_;
  std::vector<std::vector<std::pair<size_t, int>>> data_reverse_adjacencies_;
  std::vector<absl::flat_hash_set<size_t>> call_adjacencies_;

  bool finalized_;
};

}  // namespace ml4pl
