// This file defines a class for constructing program graphs from LLVM modules.
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
#include "labm8/cpp/status.h"
#include "labm8/cpp/statusor.h"
#include "labm8/cpp/string.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "programl/graph/program_graph_builder.h"
#include "programl/ir/llvm/internal/text_encoder.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/proto/program_graph_options.pb.h"

namespace programl {
namespace ir {
namespace llvm {
namespace internal {

// An <entry, exits> pair which records the node numbers for a function's entry
// and exit statement nodes, respectively.
using FunctionEntryExits = std::pair<Node*, std::vector<Node*>>;

// An <entry, exit> pair which records the node numbers of a basic block's entry
// and exit statement nodes, respectively.
using BasicBlockEntryExit = std::pair<Node*, Node*>;

// A <node, position> pair which records a node number and a position argument.
using PositionalNode = std::pair<Node*, int32_t>;

// A <source_instruction, destination_node> pair which records a data flow
// relation from a producer instruction to data element produced. Since
// LLVM does not have multi-assignment, the position of all DataEdges is assumed
// to be zero.
using DataEdge = std::pair<const ::llvm::Instruction*, Node*>;

// A map from instructions to their node.
using InstructionMap = absl::flat_hash_map<const ::llvm::Instruction*, Node*>;

using ArgumentConsumerMap =
    absl::flat_hash_map<const ::llvm::Argument*, std::vector<PositionalNode>>;

// A specialized program graph builder for LLVM-IR.
class ProgramGraphBuilder : public programl::graph::ProgramGraphBuilder {
 public:
  explicit ProgramGraphBuilder(const ProgramGraphOptions& options);

  [[nodiscard]] labm8::StatusOr<ProgramGraph> Build(
      const ::llvm::Module& module);

  void Clear();

 protected:
  [[nodiscard]] labm8::StatusOr<FunctionEntryExits> VisitFunction(
      const ::llvm::Function& function, const Function* functionMessage);

  [[nodiscard]] labm8::StatusOr<BasicBlockEntryExit> VisitBasicBlock(
      const ::llvm::BasicBlock& block, const Function* functionMessage,
      InstructionMap* instructionMap, ArgumentConsumerMap* argumentConsumers,
      std::vector<DataEdge>* dataEdgesToAdd);

  [[nodiscard]] labm8::Status AddCallSite(const Node* source,
                                          const FunctionEntryExits& target);

  Node* AddLlvmInstruction(const ::llvm::Instruction* instruction,
                           const Function* function);
  Node* AddLlvmVariable(const ::llvm::Instruction* operand,
                        const Function* function);
  Node* AddLlvmVariable(const ::llvm::Argument* argument,
                        const Function* function);
  Node* AddLlvmConstant(const ::llvm::Constant* constant);

  // Add a string to the strings list and return its position.
  //
  // We use a graph-level "strings" feature to store a list of the original
  // LLVM-IR string corresponding to each graph nodes. This allows to us to
  // refer to the same string from multiple nodes without duplication.
  int32_t AddString(const string& text);

 private:
  const ProgramGraphOptions options_;

  TextEncoder textEncoder_;

  int32_t blockCount_;

  // A map from nodes to functions, used to track call sites. This map is
  // populated by VisitBasicBlock() and consumed once all functions have been
  // visited.
  absl::flat_hash_map<Node*, const ::llvm::Function*> callSites_;
  // A map from constant values to <node, position> uses. This map is
  // populated by VisitBasicBlock() and consumed once all functions have been
  // visited.
  absl::flat_hash_map<const ::llvm::Constant*, std::vector<PositionalNode>>
      constants_;

  // A mapping from string table value to its position in the "strings_table"
  // graph-level feature.
  absl::flat_hash_map<string, int32_t> stringsListPositions_;
  // The underlying storage for the strings table.
  BytesList* stringsList_;
};

}  // namespace internal
}  // namespace llvm
}  // namespace ir
}  // namespace programl
