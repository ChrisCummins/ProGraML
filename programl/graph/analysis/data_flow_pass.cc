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

#include "programl/graph/analysis/data_flow_pass.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <random>

#include "labm8/cpp/status_macros.h"

using labm8::Status;
namespace error = labm8::error;
using std::vector;

namespace programl {
namespace graph {
namespace analysis {

namespace {
// Convenience function to format adjacency lists to an output stream.
//
// Example output:
//    0: [1, 2, 3]
//    2: [3, 4]
std::ostream& AdjacencyListToOstream(std::ostream& os, const vector<vector<int>>& adjacencies) {
  for (int i = 0; i < adjacencies.size(); ++i) {
    if (!adjacencies[i].size()) {
      continue;
    }
    os << "    " << i << ": [";
    for (int j = 0; j < adjacencies[i].size(); ++j) {
      if (j) {
        os << ", ";
      }
      os << adjacencies[i][j];
    }
    os << ']' << std::endl;
  }
  return os;
}

}  // namespace

std::ostream& operator<<(std::ostream& os, const AdjacencyLists& dt) {
  if (dt.control.size()) {
    os << "control:" << std::endl;
    AdjacencyListToOstream(os, dt.control);
  }
  if (dt.reverse_control.size()) {
    os << "reverse_control:" << std::endl;
    AdjacencyListToOstream(os, dt.reverse_control);
  }
  if (dt.data.size()) {
    os << "data:" << std::endl;
    AdjacencyListToOstream(os, dt.data);
  }
  if (dt.reverse_data.size()) {
    os << "reverse_data:" << std::endl;
    AdjacencyListToOstream(os, dt.reverse_data);
  }
  if (dt.reverse_data_positions.size()) {
    os << "reverse_data_positions:" << std::endl;
    AdjacencyListToOstream(os, dt.reverse_data_positions);
  }

  return os;
}

const AdjacencyLists& DataFlowPass::ComputeAdjacencies(const AdjacencyListOptions& options) {
  if (options.control) {
    adjacencies_.control.reserve(graph().node_size());
    for (int i = 0; i < graph().node_size(); ++i) {
      adjacencies_.control.emplace_back();
    }
  }
  if (options.reverse_control) {
    adjacencies_.reverse_control.reserve(graph().node_size());
    for (int i = 0; i < graph().node_size(); ++i) {
      adjacencies_.reverse_control.emplace_back();
    }
  }
  if (options.data) {
    adjacencies_.data.reserve(graph().node_size());
    for (int i = 0; i < graph().node_size(); ++i) {
      adjacencies_.data.emplace_back();
    }
  }
  if (options.reverse_data) {
    adjacencies_.reverse_data.reserve(graph().node_size());
    for (int i = 0; i < graph().node_size(); ++i) {
      adjacencies_.reverse_data.emplace_back();
    }
  }
  if (options.reverse_data_positions) {
    adjacencies_.reverse_data_positions.reserve(graph().node_size());
    for (int i = 0; i < graph().node_size(); ++i) {
      adjacencies_.reverse_data_positions.emplace_back();
    }
  }

  for (int i = 0; i < graph().edge_size(); ++i) {
    const Edge& edge = graph().edge(i);
    if (edge.flow() == Edge::CONTROL) {
      if (options.control) {
        adjacencies_.control[edge.source()].push_back(edge.target());
      }
      if (options.reverse_control) {
        adjacencies_.reverse_control[edge.target()].push_back(edge.source());
      }
    } else if (edge.flow() == Edge::DATA) {
      if (options.data) {
        adjacencies_.data[edge.source()].push_back(edge.target());
      }
      if (options.reverse_data) {
        adjacencies_.reverse_data[edge.target()].push_back(edge.source());
      }
      if (options.reverse_data_positions) {
        adjacencies_.reverse_data_positions[edge.target()].push_back(edge.position());
      }
    }
  }

  return adjacencies_;
}

const AdjacencyLists& DataFlowPass::adjacencies() const { return adjacencies_; }

Status RoodNodeDataFlowAnalysis::Run(ProgramGraphFeaturesList* featuresList) {
  RETURN_IF_ERROR(Init());

  vector<int> rootNodes = GetEligibleRootNodes();
  if (!rootNodes.size()) {
    return Status(error::Code::FAILED_PRECONDITION, "No eligible root nodes in graph with {} nodes",
                  graph().node_size());
  }
  std::shuffle(rootNodes.begin(), rootNodes.end(), std::default_random_engine(seed()));

  int numRoots =
      std::min(static_cast<int>(ceil(rootNodes.size() / 10.0)), max_instances_per_graph());
  for (int i = 0; i < numRoots; ++i) {
    ProgramGraphFeatures features;
    RETURN_IF_ERROR(RunOne(rootNodes[i], &features));
    *featuresList->add_graph() = features;
  }

  return Status::OK;
}

Status RoodNodeDataFlowAnalysis::Init() { return Status::OK; }

void AddNodeFeature(ProgramGraphFeatures* features, const string& name, const Feature& value) {
  (*(*features->mutable_node_features()->mutable_feature_list())[name].add_feature()) = value;
}

vector<int> GetInstructionsInFunctionsNodeIndices(const ProgramGraph& graph) {
  // Start at index 1 to skip the program root.
  vector<int> rootNodes;
  for (int i = 1; i < graph.node_size(); ++i) {
    if (graph.node(i).type() == Node::INSTRUCTION) {
      rootNodes.push_back(i);
    }
  }
  return rootNodes;
}

vector<int> GetVariableNodeIndices(const ProgramGraph& graph) {
  // Start at index 1 to skip the program root.
  vector<int> rootNodes;
  for (int i = 1; i < graph.node_size(); ++i) {
    if (graph.node(i).type() == Node::VARIABLE) {
      rootNodes.push_back(i);
    }
  }
  return rootNodes;
}

}  // namespace analysis
}  // namespace graph
}  // namespace programl
