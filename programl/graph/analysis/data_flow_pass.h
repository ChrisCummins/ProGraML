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

#include <chrono>
#include <vector>

#include "labm8/cpp/status.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/proto/util.pb.h"

namespace programl {
namespace graph {
namespace analysis {

// Options which control the type of adjacency lists to construct.
struct AdjacencyListOptions {
  bool control;
  bool reverse_control;
  bool data;
  bool reverse_data;
  bool reverse_data_positions;
};

// A collection of adjacency lists.
struct AdjacencyLists {
  std::vector<std::vector<int>> control;
  std::vector<std::vector<int>> reverse_control;
  std::vector<std::vector<int>> data;
  std::vector<std::vector<int>> reverse_data;
  std::vector<std::vector<int>> reverse_data_positions;

  friend std::ostream& operator<<(std::ostream& os, const AdjacencyLists& dt);
};

// Base class for implementing data flow analysis passes.
class DataFlowPass {
 public:
  explicit DataFlowPass(const ProgramGraph& graph) : graph_(graph) {}

  [[nodiscard]] virtual labm8::Status Run(ProgramGraphFeaturesList* featuresList) = 0;

  const ProgramGraph& graph() const { return graph_; }

 protected:
  const AdjacencyLists& ComputeAdjacencies(const AdjacencyListOptions& options);
  const AdjacencyLists& adjacencies() const;

 private:
  const ProgramGraph& graph_;
  AdjacencyLists adjacencies_;
};

// A data flow analysis which begins at starts at a "root" node in the graph.
//
// For root-node data flow analyses, multiple sets of labels may be computed be
// selecting different root nodes from the set of elligible roots.
class RoodNodeDataFlowAnalysis : public DataFlowPass {
 public:
  RoodNodeDataFlowAnalysis(const ProgramGraph& graph) : RoodNodeDataFlowAnalysis(graph, 10) {}

  RoodNodeDataFlowAnalysis(const ProgramGraph& graph, int maxInstancesPerGraph)
      : DataFlowPass(graph),
        maxInstancesPerGraph_(maxInstancesPerGraph),
        seed_(std::chrono::system_clock::now().time_since_epoch().count()) {}

  // Return a list of nodes that are elligible for use as the root node of this
  // analysis.
  //
  // Consider using utility functions GetInstructionsInFunctionsNodeIndices() or
  // GetVariableNodeIndices() to implement this.
  virtual std::vector<int> GetEligibleRootNodes() = 0;

  [[nodiscard]] virtual labm8::Status Run(ProgramGraphFeaturesList* featuresList) override;

  [[nodiscard]] virtual labm8::Status Init();

  int max_instances_per_graph() const { return maxInstancesPerGraph_; }

  unsigned seed() const { return seed_; }
  void seed(unsigned seed) { seed_ = seed; }

 protected:
  virtual labm8::Status RunOne(int rootNode, ProgramGraphFeatures* features) = 0;

 private:
  const int maxInstancesPerGraph_;
  unsigned seed_;
};

// Utility function to add a new node feature a list of node features.
void AddNodeFeature(ProgramGraphFeatures* features, const string& name, const Feature& value);

std::vector<int> GetInstructionsInFunctionsNodeIndices(const ProgramGraph& graph);

std::vector<int> GetVariableNodeIndices(const ProgramGraph& graph);

}  // namespace analysis
}  // namespace graph
}  // namespace programl
