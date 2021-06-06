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
#include "programl/ir/xla/hlo_module_graph_builder.h"

#include <sstream>

#include "labm8/cpp/logging.h"
#include "labm8/cpp/status_macros.h"
#include "labm8/cpp/string.h"
#include "programl/ir/xla/xla_stringifier.h"
#include "programl/third_party/tensorflow/xla.pb.h"

namespace programl {
namespace ir {
namespace xla {

labm8::StatusOr<ProgramGraph> HloModuleGraphBuilder::Build(const ::xla::HloProto& proto) {
  RETURN_IF_ERROR(VisitModule(proto.hlo_module()));
  return GetProgramGraph();
}

labm8::Status HloModuleGraphBuilder::VisitModule(const ::xla::HloModuleProto& module) {
  Module* mod = AddModule(module.name());

  // Instantiate the "functions" from HloComputations. Functions are defined in
  // the order of dependencies.
  for (int i = 0; i < module.computations_size(); ++i) {
    FunctionEntryExits computation;
    ASSIGN_OR_RETURN(computation, VisitComputation(module.computations(i), mod));
    computations_.insert({module.computations(i).id(), computation});
  }

  // Add the call edges from the graph root to the entry computation.
  auto entryComputation = computations_.find(module.entry_computation_id());
  if (entryComputation == computations_.end()) {
    return labm8::Status(labm8::error::Code::INVALID_ARGUMENT,
                         "Failed to locate entry computation");
  }
  RETURN_IF_ERROR(AddCallEdge(GetRootNode(), entryComputation->second.first).status());
  for (const auto& exit : entryComputation->second.second) {
    RETURN_IF_ERROR(AddCallEdge(exit, GetRootNode()).status());
  }

  return labm8::Status::OK;
}

labm8::StatusOr<FunctionEntryExits> HloModuleGraphBuilder::VisitComputation(
    const ::xla::HloComputationProto& computation, const Module* module) {
  Function* fn = AddFunction(computation.name(), module);

  // Create an entry statement which acts as a common control predecessor of
  // the computation inputs. Since a HLO module is a dataflow graph, there may
  // be multiple inputs.
  Node* entryInstruction = AddInstruction("<entry>", fn);

  // Visit the instructions in-order. Instructions are ordered such that
  // producers appear before consumers.
  Node* lastInstruction = entryInstruction;
  for (int i = 0; i < computation.instructions_size(); ++i) {
    ASSIGN_OR_RETURN(lastInstruction,
                     VisitInstruction(computation.instructions(i), fn, entryInstruction));
  }

  // Since instructions are in a valid execution order, the last instruction
  // must be the final producer.
  return FunctionEntryExits{entryInstruction, {lastInstruction}};
}

labm8::StatusOr<Node*> HloModuleGraphBuilder::VisitInstruction(
    const ::xla::HloInstructionProto& instruction, Function* function, Node* entryInstruction) {
  bool isParam = instruction.opcode() == "parameter";
  const string name = isParam ? "<param>" : HloInstructionToText(instruction);

  // Generate the instruction node.
  Node* instructionNode = AddInstruction(name, function);
  instructions_.insert({instruction.id(), instructionNode});

  // Generate the identifier node for the data produced by the instruction.
  Node* data = AddVariable(ShapeProtoToString(instruction.shape()), function);
  producers_.insert({instruction.id(), data});
  RETURN_IF_ERROR(AddDataEdge(/*position=*/0, instructionNode, data).status());

  if (instruction.opcode() == "parameter") {
    // Add the implicit control edge from computation entry point to parameters.
    RETURN_IF_ERROR(AddControlEdge(/*position=*/0, entryInstruction, instructionNode).status());
  } else if (instruction.opcode() == "constant") {
    // Generate the immediate value nodes for constants.
    Node* literal = AddConstant(LiteralProtoToText(instruction.literal()));
    RETURN_IF_ERROR(AddDataEdge(instruction.operand_ids_size(), literal, instructionNode).status());
  }

  // Add data and control edges from consumer to producer..
  for (int i = 0; i < instruction.operand_ids_size(); ++i) {
    auto operandData = producers_.find(instruction.operand_ids(i));
    if (operandData == producers_.end()) {
      std::stringstream err;
      err << "Failed to find operand data " << instruction.id() << " <- "
          << instruction.operand_ids(i);
      return labm8::Status(labm8::error::Code::INVALID_ARGUMENT, err.str());
    }
    RETURN_IF_ERROR(AddDataEdge(/*position=*/i, operandData->second, instructionNode).status());

    auto pred = instructions_.find(instruction.operand_ids(i));
    if (pred == instructions_.end()) {
      return labm8::Status(labm8::error::Code::INVALID_ARGUMENT,
                           "Failed to find operand instruction");
    }
    RETURN_IF_ERROR(AddControlEdge(/*position=*/i, pred->second, instructionNode).status());
  }

  // Add explicit control dependencies.
  for (int i = 0; i < instruction.control_predecessor_ids_size(); ++i) {
    auto pred = instructions_.find(instruction.control_predecessor_ids(i));
    if (pred == instructions_.end()) {
      return labm8::Status(labm8::error::Code::INVALID_ARGUMENT,
                           "Failed to find control predecessor");
    }
    RETURN_IF_ERROR(AddControlEdge(/*position=*/0, pred->second, instructionNode).status());
  }

  // Add call edges from instructions to computations.
  for (int i = 0; i < instruction.called_computation_ids_size(); ++i) {
    labm8::int64 calledComputation = instruction.called_computation_ids(i);

    auto calledComputationEntryExits = computations_.find(calledComputation);
    if (calledComputationEntryExits == computations_.end()) {
      return labm8::Status(labm8::error::Code::INVALID_ARGUMENT,
                           "Failed to locate called computation");
    }
    RETURN_IF_ERROR(
        AddCallEdge(instructionNode, calledComputationEntryExits->second.first).status());
    for (const auto& exit : calledComputationEntryExits->second.second) {
      RETURN_IF_ERROR(AddCallEdge(exit, instructionNode).status());
    }
  }

  return instructionNode;
}

}  // namespace xla
}  // namespace ir
}  // namespace programl
