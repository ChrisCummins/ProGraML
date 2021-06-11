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
#include "labm8/cpp/status.h"
#include "programl/graph/analysis/data_flow_pass.h"
#include "programl/proto/program_graph.pb.h"

namespace programl {
namespace graph {
namespace analysis {

// Live-out analysis.
//
// Starting at instruction `n`, label all variable nodes which are live-out
// at node `n`.
class LivenessAnalysis : public RoodNodeDataFlowAnalysis {
 public:
  using RoodNodeDataFlowAnalysis::RoodNodeDataFlowAnalysis;

  labm8::Status RunOne(int rootNode, ProgramGraphFeatures* features) override;

  std::vector<int> GetEligibleRootNodes() override;

  labm8::Status Init() override;

  const std::vector<absl::flat_hash_set<int>>& live_in_sets() const { return liveInSets_; }

  const std::vector<absl::flat_hash_set<int>>& live_out_sets() const { return liveOutSets_; }

 private:
  // Live-in and live-out sets that are computed during Init().
  std::vector<absl::flat_hash_set<int>> liveInSets_;
  std::vector<absl::flat_hash_set<int>> liveOutSets_;
};

}  // namespace analysis
}  // namespace graph
}  // namespace programl
