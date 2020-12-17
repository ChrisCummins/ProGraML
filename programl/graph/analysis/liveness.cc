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

#include "programl/graph/analysis/liveness.h"

#include <queue>

#include "labm8/cpp/logging.h"
#include "programl/graph/features.h"

using absl::flat_hash_set;
using labm8::Status;
using std::vector;
namespace error = labm8::error;

namespace programl {
namespace graph {
namespace analysis {

Status LivenessAnalysis::Init() {
  ComputeAdjacencies(
      {.control = true, .reverse_control = true, .data = true, .reverse_data = true});
  const auto& cfg = adjacencies().control;
  DCHECK(cfg.size() == graph().node_size()) << cfg.size() << " != " << graph().node_size();
  const auto& rcfg = adjacencies().reverse_control;
  DCHECK(rcfg.size() == graph().node_size()) << rcfg.size() << " != " << graph().node_size();
  const auto& dfg = adjacencies().data;
  DCHECK(dfg.size() == graph().node_size()) << dfg.size() << " != " << graph().node_size();
  const auto& rdfg = adjacencies().reverse_data;
  DCHECK(rdfg.size() == graph().node_size()) << rdfg.size() << " != " << graph().node_size();

  // Pre-compute all live-out sets.
  liveInSets_.reserve(graph().node_size());
  liveOutSets_.reserve(graph().node_size());
  for (int i = 0; i < graph().node_size(); ++i) {
    liveInSets_.emplace_back();
    liveOutSets_.emplace_back();
  }

  std::queue<int> workList;
  flat_hash_set<int> workSet;

  // Liveness is computed backwards starting from the program exit.
  // Build a work list of exit nodes to begin from.
  for (int i = 0; i < graph().node_size(); ++i) {
    if (graph().node(i).type() == Node::INSTRUCTION && !cfg[i].size()) {
      workList.push(i);
      workSet.insert(i);
    }
  }

  // A graph may not have any exit nodes.
  if (workList.empty()) {
    return Status::OK;
  }

  size_t stepCount = 0;
  while (!workList.empty()) {
    ++stepCount;

    if (stepCount > std::max(graph().node_size() * 2, 5000)) {
      return Status(error::FAILED_PRECONDITION,
                    "Failed to terminate liveness computation in 5000 steps");
    }

    int node = workList.front();
    workList.pop();
    workSet.erase(node);

    // Get immediate control and data neighbors.
    const auto& successors = cfg[node];
    const auto& predecessors = rcfg[node];
    const auto& defs = dfg[node];
    const auto& uses = rdfg[node];

    // LiveOut(n) = U {LiveIn(p) for p in succ(n)}
    flat_hash_set<int> newOutSet{};
    for (const auto& successor : successors) {
      newOutSet.merge(
          flat_hash_set<int>(liveInSets_[successor].begin(), liveInSets_[successor].end()));
    }

    // LiveIn(n) = Gen(n) U {LiveOut(n) - Kill(n)}
    flat_hash_set<int> newInSet{uses.begin(), uses.end()};
    flat_hash_set<int> newOutSetMinusDefs = newOutSet;
    for (const auto& d : defs) {
      newOutSetMinusDefs.erase(d);
    }
    newInSet.merge(newOutSetMinusDefs);

    // Visit predecessors if inSet is empty or has changed.
    if (newInSet.empty() || newInSet != liveInSets_[node]) {
      liveInSets_[node] = newInSet;
      for (const auto& predecessor : predecessors) {
        if (!workSet.contains(predecessor)) {
          workList.push(predecessor);
          workSet.insert(predecessor);
        }
      }
    }

    liveOutSets_[node] = newOutSet;
  }

  return Status::OK;
}

vector<int> LivenessAnalysis::GetEligibleRootNodes() {
  vector<int> nodes;
  for (int i = 0; i < liveOutSets_.size(); ++i) {
    if (liveOutSets_[i].size()) {
      nodes.push_back(i);
    }
  }
  return nodes;
}

Status LivenessAnalysis::RunOne(int rootNode, ProgramGraphFeatures* features) {
  if (rootNode < 0 || rootNode >= graph().node_size()) {
    return Status(error::INVALID_ARGUMENT, "Root node is out-of-range");
  }
  if (graph().node(rootNode).type() != Node::INSTRUCTION) {
    return Status(error::INVALID_ARGUMENT, "Root node must be an instruction");
  }

  Feature falseFeature = CreateFeature(0);
  Feature trueFeature = CreateFeature(1);

  Features notRootFeatures;
  SetFeature(&notRootFeatures, "data_flow_root_node", falseFeature);
  SetFeature(&notRootFeatures, "data_flow_value", falseFeature);

  Features rootNodeFeatures;
  SetFeature(&rootNodeFeatures, "data_flow_root_node", trueFeature);

  // We have already pre-computed the live-out sets, so just add the
  // annotations.
  const auto& outSet = liveOutSets_[rootNode];
  int dataFlowActiveNodeCount = 0;
  for (int i = 0; i < graph().node_size(); ++i) {
    AddNodeFeature(features, "data_flow_root_node", i == rootNode ? trueFeature : falseFeature);
    if (outSet.contains(i)) {
      ++dataFlowActiveNodeCount;
      AddNodeFeature(features, "data_flow_value", trueFeature);
    } else {
      AddNodeFeature(features, "data_flow_value", falseFeature);
    }
  }

  // BFS from root node to compute maximum distance from root node to a live-out
  // node.
  int maxDistance = 0;
  std::queue<std::pair<int, int>> q;
  q.push({rootNode, 1});
  vector<bool> visited(graph().node_size(), false);
  while (!q.empty()) {
    int node = q.front().first;
    int distance = q.front().second;
    q.pop();

    if (liveOutSets_[rootNode].contains(node)) {
      maxDistance = std::max(distance, maxDistance);
    }

    visited[node] = true;
    for (const auto next : adjacencies().control[node]) {
      if (!visited[next]) {
        q.push({next, distance + 1});
      }
    }
    for (const auto next : adjacencies().data[node]) {
      if (!visited[next]) {
        q.push({next, distance + 1});
      }
    }
  }

  AddScalarFeature(features, "data_flow_step_count", maxDistance);
  AddScalarFeature(features, "data_flow_active_node_count", dataFlowActiveNodeCount);

  return Status::OK;
}

}  // namespace analysis
}  // namespace graph
}  // namespace programl
