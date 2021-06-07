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

#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "labm8/cpp/statusor.h"
#include "labm8/cpp/string.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/proto/util.pb.h"

namespace programl {
namespace graph {

// A module for constructing a single program graph.
//
// Uses the builder pattern. Example usage:
//
//     ProgramGraphBuilder builder;
//
//     Module* mod = builder.AddModule("foo");
//     Function* fn = builder.AddFunction("A", mod);
//     Node* add = builder.AddInstruction("add", fn);
//     RETURN_IF_ERROR(builder.AddControlEdge(builder.GetRootNode(), add);
//     ...
//     ProgramGraph graph = builder.Build().ValueOrDie();
//
// After calling Build(), the object may be re-used.
//
class ProgramGraphBuilder {
 public:
  ProgramGraphBuilder() : ProgramGraphBuilder(ProgramGraphOptions{}) {}

  explicit ProgramGraphBuilder(const ProgramGraphOptions& options);

  // Construct a new function and return its number.
  Module* AddModule(const string& name);

  // Construct a new function and return its number.
  Function* AddFunction(const string& name, const Module* module);

  // Node factories.
  Node* AddInstruction(const string& text, const Function* function);

  Node* AddVariable(const string& text, const Function* function);

  Node* AddConstant(const string& text);

  // Edge factories.
  [[nodiscard]] labm8::StatusOr<Edge*> AddControlEdge(int32_t position, const Node* source,
                                                      const Node* target);

  [[nodiscard]] labm8::StatusOr<Edge*> AddDataEdge(int32_t position, const Node* source,
                                                   const Node* target);

  [[nodiscard]] labm8::StatusOr<Edge*> AddCallEdge(const Node* source, const Node* target);

  const Node* GetRootNode() const { return &graph_.node(0); }

  // Return the graph protocol buffer.
  const ProgramGraph& GetProgramGraph() const { return graph_; }

  // Return the program graph, and validate it first if strict option is set.
  [[nodiscard]] labm8::StatusOr<ProgramGraph> Build();

  // Validate the program graph.
  [[nodiscard]] labm8::Status ValidateGraph() const;

  // Reset builder state.
  void Clear();

 protected:
  inline const ProgramGraphOptions& options() const { return options_; }

  // Construct nodes.

  inline Node* AddNode(const Node::Type& type);

  inline Node* AddNode(const Node::Type& type, const string& text);

  inline Node* AddNode(const Node::Type& type, const string& text, const Function* function);

  // Construct edges.

  inline Edge* AddEdge(const Edge::Flow& flow, int32_t position, const Node* source,
                       const Node* target);

  // Return a mutable pointer to the graph protocol buffer.
  ProgramGraph* GetMutableProgramGraph() { return &graph_; }

 private:
  const ProgramGraphOptions options_;

  ProgramGraph graph_;

  // Get the index of an object in a repeated field lists.
  int32_t GetIndex(const Module* module);
  int32_t GetIndex(const Function* function);
  int32_t GetIndex(const Node* node);

  // Maps which covert store the index of objects in repeated field lists.
  absl::flat_hash_map<Module*, int32_t> moduleIndices_;
  absl::flat_hash_map<Function*, int32_t> functionIndices_;
  absl::flat_hash_map<Node*, int32_t> nodeIndices_;

  // Sets to track incomplete graph components. An object is added to one of
  // these sets when it is created, and removed from the set when it has met
  // the required criteria. Any elements remaining in these sets when Build()
  // is called will return an error status.
  absl::flat_hash_set<Module*> emptyModules_;
  absl::flat_hash_set<Function*> emptyFunctions_;
  absl::flat_hash_set<Node*> unconnectedNodes_;
};

}  // namespace graph
}  // namespace programl
