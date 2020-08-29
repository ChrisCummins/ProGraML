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

#include "programl/graph/analysis/datadep.h"

#include <queue>
#include <utility>
#include <vector>

#include "labm8/cpp/logging.h"
#include "labm8/cpp/status.h"
#include "programl/graph/features.h"

using labm8::Status;
using std::vector;
namespace error = labm8::error;

namespace programl {
namespace graph {
namespace analysis {

Status DatadepAnalysis::Init() {
  ComputeAdjacencies(
      {.control = false, .reverse_control = false, .data = false, .reverse_data = true});
  return Status::OK;
}

vector<int> DatadepAnalysis::GetEligibleRootNodes() { return GetVariableNodeIndices(graph()); }

Status DatadepAnalysis::RunOne(int rootNode, ProgramGraphFeatures* features) {
  vector<bool> visited(graph().node_size(), false);

  const vector<vector<int>>& rdfg = adjacencies().reverse_data;
  DCHECK(rdfg.size() == graph().node_size()) << "RDFG size: " << rdfg.size() << " != "
                                             << " graph size: " << graph().node_size();

  int dataFlowStepCount = 1;
  std::queue<std::pair<int, int>> q;
  q.push({rootNode, 0});

  while (!q.empty()) {
    int current = q.front().first;
    dataFlowStepCount = q.front().second;
    q.pop();

    visited[current] = true;

    for (int next : rdfg[current]) {
      if (!visited[next]) {
        q.push({next, dataFlowStepCount + 1});
      }
    }
  }

  // Set the node features.
  Feature falseFeature = CreateFeature(0);
  Feature trueFeature = CreateFeature(1);

  int activeNodeCount = 0;
  for (int i = 0; i < graph().node_size(); ++i) {
    AddNodeFeature(features, "data_flow_root_node", i == rootNode ? trueFeature : falseFeature);
    if (visited[i] && graph().node(i).type() == Node::INSTRUCTION) {
      ++activeNodeCount;
      AddNodeFeature(features, "data_flow_value", trueFeature);
    } else {
      AddNodeFeature(features, "data_flow_value", falseFeature);
    }
  }

  SetFeature(features->mutable_features(), "data_flow_step_count",
             CreateFeature(dataFlowStepCount));
  SetFeature(features->mutable_features(), "data_flow_active_node_count",
             CreateFeature(activeNodeCount));

  return Status::OK;
}

}  // namespace analysis
}  // namespace graph
}  // namespace programl
