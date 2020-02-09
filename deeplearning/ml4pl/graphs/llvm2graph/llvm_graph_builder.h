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
#include "deeplearning/ml4pl/graphs/graph_builder.h"
#include "deeplearning/ml4pl/graphs/programl.pb.h"
#include "labm8/cpp/status.h"
#include "labm8/cpp/statusor.h"
#include "labm8/cpp/string.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

namespace ml4pl {

// An <entry, exit> pair which records the node numbers of a basic block's entry
// and exit statement nodes, respectively.
using BasicBlockEntryExit = std::pair<int, int>;

// A <node, position> pair which records a node number and a position argument.
using PositionalNode = std::pair<size_t, int>;

// A <source_instruction, destination_node> pair which records a data flow
// relation from a producer instruction to data element produced. Since
// LLVM does not have multi-assignment, the position of all DataEdges is assumed
// to be zero.
using DataEdge = std::pair<const llvm::Instruction*, size_t>;

// A map from instructions to their node number.
using InstructionNumberMap =
    absl::flat_hash_map<const llvm::Instruction*, size_t>;

using ArgumentConsumerMap =
    absl::flat_hash_map<const llvm::Argument*, std::vector<PositionalNode>>;

// Utility functions to generate the split "left"- and "right"-hand side
// components of an instruction.
//
//     E.g.      "%5 = add nsw i32 %3, %4"
//          LHS: "int64* %5"
//          RHS: "add nsw i32 %3, %4".
//
// Calling GetInstructionLhs() on an instruction with no LHS,
// e.g. "ret i32 %13", is an error.
//
// LLVM doesn't require "names" for instructions since it is in SSA form, so
// this method generates one by printing the instruction to a string (to
// generate identifiers), then splitting the LHS identifier name and
// concatenating it with the type.
//
// See: https://lists.llvm.org/pipermail/llvm-dev/2010-April/030726.html
string GetInstructionLhs(const llvm::Instruction& instruction);
string GetInstructionRhs(const llvm::Instruction& instruction);

// A module pass for constructing graphs.
class LlvmGraphBuilder : GraphBuilder {
 public:
  // Main entry point. Accepts a module as input and returns a graph as output,
  // or an error status if graph construction fails.
  labm8::StatusOr<ProgramGraph> Build(const llvm::Module& module);

 protected:
  labm8::StatusOr<FunctionEntryExits> VisitFunction(
      const llvm::Function& function, const int& functionNumber);

  labm8::StatusOr<BasicBlockEntryExit> VisitBasicBlock(
      const llvm::BasicBlock& block, const int& functionNumber,
      InstructionNumberMap* instructions,
      ArgumentConsumerMap* argumentConsumers,
      std::vector<DataEdge>* dataEdgesToAdd);

 private:
  // A map from node index to functions to mark call sites. This map is
  // populated by VisitBasicBlock() and consumed once all functions have been
  // visited.
  absl::flat_hash_map<size_t, const llvm::Function*> call_sites_;
  // A map from constant values to <node, position> uses. This map is
  // populated by VisitBasicBlock() and consumed once all functions have been
  // visited.
  absl::flat_hash_map<const llvm::Constant*, std::vector<PositionalNode>>
      constants_;
};

}  // namespace ml4pl
