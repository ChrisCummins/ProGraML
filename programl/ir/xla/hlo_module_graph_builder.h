// A class for constructing program graphs from LLVM modules.
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
#include "labm8/cpp/port.h"
#include "labm8/cpp/statusor.h"
#include "programl/graph/program_graph_builder.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/third_party/tensorflow/xla.pb.h"

namespace programl {
namespace ir {
namespace xla {

// An <entry, exits> pair which records the node numbers for a function's entry
// and exit statement nodes, respectively.
using FunctionEntryExits = std::pair<Node*, std::vector<Node*>>;

// A class for generating program graphs from HloProto messages.
class HloModuleGraphBuilder : graph::ProgramGraphBuilder {
 public:
  // Main entry point. Accepts a module as input and returns a graph as output,
  // or an error status if graph construction fails.
  labm8::StatusOr<ProgramGraph> Build(const ::xla::HloProto& proto);

 protected:
  [[nodiscard]] labm8::Status VisitModule(const ::xla::HloModuleProto& module);

  [[nodiscard]] labm8::StatusOr<FunctionEntryExits> VisitComputation(
      const ::xla::HloComputationProto& computation, const Module* module);

  [[nodiscard]] labm8::StatusOr<Node*> VisitInstruction(
      const ::xla::HloInstructionProto& instruction, Function* function, Node* entryInstruction);

 private:
  // A map from computations to their entry and exit nodes.
  absl::flat_hash_map<labm8::int64, FunctionEntryExits> computations_;
  // A map of instruction IDs to their node number.
  absl::flat_hash_map<labm8::int64, Node*> instructions_;
  // A map of instruction IDs to the data element produced by the instruction.
  absl::flat_hash_map<labm8::int64, Node*> producers_;
};

}  // namespace xla
}  // namespace ir
}  // namespace programl
