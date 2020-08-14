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
#include "programl/ir/llvm/internal/program_graph_builder.h"

#include <deque>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/status_macros.h"
#include "labm8/cpp/string.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#if PROGRAML_LLVM_VERSION_MAJOR > 3
#include "llvm/IR/ProfileSummary.h"
#endif
#include "programl/graph/features.h"
#include "programl/ir/llvm/internal/text_encoder.h"
#include "programl/proto/program_graph.pb.h"

using labm8::Status;

namespace programl {
namespace ir {
namespace llvm {
namespace internal {

labm8::StatusOr<BasicBlockEntryExit> ProgramGraphBuilder::VisitBasicBlock(
    const ::llvm::BasicBlock& block, const Function* functionMessage, InstructionMap* instructions,
    ArgumentConsumerMap* argumentConsumers, std::vector<DataEdge>* dataEdgesToAdd) {
  if (!block.size()) {
    return Status(labm8::error::Code::FAILED_PRECONDITION, "Basic block contains no instructions");
  }

  Node* firstNode = nullptr;
  Node* lastNode = nullptr;

  // Iterate over the instructions of a basic block in-order.
  for (const ::llvm::Instruction& instruction : block) {
    const LlvmTextComponents text = textEncoder_.Encode(&instruction);

    // Create the graph node for the instruction.
    auto instructionMessage = AddLlvmInstruction(&instruction, functionMessage);

    // Record the instruction in the function-level instructions map.
    instructions->insert({&instruction, instructionMessage});

    // A basic block consists of a linear sequence of instructions, so we can
    // insert the control edge between instructions as we go.
    if (lastNode) {
      RETURN_IF_ERROR(AddControlEdge(/*position=*/0, lastNode, instructionMessage).status());
    }

    // If the instruction is a call, record the call site, which we will use
    // later to create all edges..
    if (auto* callInstruction = ::llvm::dyn_cast<::llvm::CallInst>(&instruction)) {
      auto calledFunction = callInstruction->getCalledFunction();
      // TODO(github.com/ChrisCummins/ProGraML/issues/46): Should we handle the
      // case when getCalledFunction() is nil?
      if (calledFunction) {
        callSites_.insert({instructionMessage, calledFunction});
      }
    }

    // Iterate over the usedef chains by reading operands of the currrent
    // instruction.
    int position = 0;
    for (const ::llvm::Use& use : instruction.operands()) {
      const ::llvm::Value* value = use.get();

      if (::llvm::dyn_cast<::llvm::Function>(value)) {
        // Functions are a subclass of ::llvm::Constant, but we do not want to
        // treat them the same way as other constants (i.e. generate data
        // elements for them). Instead we ignore them, as the call edges that
        // will be produced provide the information we want to capture.
      } else if (const auto* constant = ::llvm::dyn_cast<::llvm::Constant>(value)) {
        if (options().instructions_only()) {
          continue;
        }
        // If the operand is a constant value, insert a new entry into the map
        // of constants to node IDs and positions. We defer creating the
        // immediate nodes until we have traversed all
        constants_[constant].push_back({instructionMessage, position});
      } else if (const auto* operand = ::llvm::dyn_cast<::llvm::Instruction>(value)) {
        if (options().instructions_only()) {
          continue;
        }
        // We have an instruction operand which itself is another instruction.
        //
        // For example, take the following IR snippet:
        //
        //     %2 = load i32, i32* %1, align 4
        //     %3 = add nsw i32 %2, 1
        //
        // Here, instruction '%3' uses the result of '%2' as an operand, so we
        // want to construct a graph like:
        //
        //     STATEMENT:  load i32, i32* %1, align 4
        //                      |
        //                      V
        //     IDENTIFIER:      %2
        //                      |
        //                      V
        //     STATEMENT:  %3 = add nsw i32 %2, 1
        //
        // To this we create the intermediate data flow node '%2' immediately,
        // but defer adding the edge from the producer instruction, since we may
        // not have visited it yet.
        Node* variable = AddLlvmVariable(operand, functionMessage);

        // Connect the data -> consumer.
        RETURN_IF_ERROR(AddDataEdge(position, variable, instructionMessage).status());

        // Defer creation of the edge from producer -> data.
        dataEdgesToAdd->push_back({operand, variable});
      } else if (const auto* operand = ::llvm::dyn_cast<::llvm::Argument>(value)) {
        if (options().instructions_only()) {
          continue;
        }
        (*argumentConsumers)[operand].push_back({instructionMessage, position});
      } else if (!(
                     // Basic blocks are not considered data. There will instead
                     // be a control edge from this instruction to the entry
                     // node of the block.
                     ::llvm::dyn_cast<::llvm::BasicBlock>(value) ||
                     // Inline assembly is ignored.
                     ::llvm::dyn_cast<::llvm::InlineAsm>(value) ||
                     // Nothing to do here.
                     ::llvm::dyn_cast<::llvm::MetadataAsValue>(value))) {
        return Status(labm8::error::Code::INTERNAL,
                      "Unknown operand {} for instruction:\n\n  "
                      "Instruction:  {}\n",
                      "Operand:      {}\n", "Operand Type: {}\n", position,
                      textEncoder_.Encode(&instruction).text, textEncoder_.Encode(value).text,
                      textEncoder_.Encode(value->getType()).text);
      }

      // Advance to the next operand position.
      ++position;
    }

    // Update pointers.
    if (!firstNode) {
      firstNode = instructionMessage;
    }
    lastNode = instructionMessage;
  }

  ++blockCount_;

  return std::make_pair(firstNode, lastNode);
}

labm8::StatusOr<FunctionEntryExits> ProgramGraphBuilder::VisitFunction(
    const ::llvm::Function& function, const Function* functionMessage) {
  // A map from basic blocks to <entry,exit> nodes.
  absl::flat_hash_map<const ::llvm::BasicBlock*, BasicBlockEntryExit> blocks;
  // A map from function Arguments to the statements that consume them, and the
  // position of the argument in the statement operand list.
  ArgumentConsumerMap argumentConsumers;
  // A map of instruction numbers which will be used to resolve the node numbers
  // for inter-instruction data flow edges once all basic blocks have been
  // visited.
  InstructionMap instructions;

  // A mapping from producer instructions to consumer instructions.
  std::vector<DataEdge> dataEdgesToAdd;

  FunctionEntryExits functionEntryExits;

  if (function.isDeclaration()) {
    Node* node = AddInstruction("; undefined function", functionMessage);
    graph::AddScalarFeature(node, "full_text", "");
    functionEntryExits.first = node;
    functionEntryExits.second.push_back(node);
    return functionEntryExits;
  }

  // Visit all basic blocks in the function to construct the per-block graph
  // components.
  for (const ::llvm::BasicBlock& block : function) {
    BasicBlockEntryExit entryExit;
    ASSIGN_OR_RETURN(entryExit, VisitBasicBlock(block, functionMessage, &instructions,
                                                &argumentConsumers, &dataEdgesToAdd));
    blocks.insert({&block, entryExit});
  }
  if (!blocks.size()) {
    return Status(labm8::error::Code::FAILED_PRECONDITION, "Function contains no basic blocks: {}",
                  string(function.getName()));
  }

  // Construct the identifier data elements for arguments and connect data
  // edges.
  for (auto it : argumentConsumers) {
    Node* argument = AddLlvmVariable(it.first, functionMessage);
    for (auto argumentConsumer : it.second) {
      Node* argumentConsumerNode = argumentConsumer.first;
      int32_t position = argumentConsumer.second;
      RETURN_IF_ERROR(AddDataEdge(position, argument, argumentConsumerNode).status());
    }
  }

  // Construct the data edges from producer instructions to the data flow
  // elements that are produced.
  for (auto dataEdgeToAdd : dataEdgesToAdd) {
    auto producer = instructions.find(dataEdgeToAdd.first);
    if (producer == instructions.end()) {
      return Status(labm8::error::Code::FAILED_PRECONDITION,
                    "Operand references instruction that has not been visited: {}",
                    textEncoder_.Encode(dataEdgeToAdd.first).text);
    }
    RETURN_IF_ERROR(AddDataEdge(/*position=*/0, producer->second, dataEdgeToAdd.second).status());
  }

  const ::llvm::BasicBlock* entry = &function.getEntryBlock();
  if (!entry) {
    return Status(labm8::error::Code::FAILED_PRECONDITION, "No entry block for function: {}",
                  string(function.getName()));
  }

  // Construct the <entry, exits> pair.
  auto entryNode = blocks.find(entry);
  if (!entry) {
    return Status(labm8::error::Code::FAILED_PRECONDITION, "No entry block");
  }
  functionEntryExits.first = entryNode->second.first;

  // Traverse the basic blocks in the function, creating control edges between
  // them.
  absl::flat_hash_set<const ::llvm::BasicBlock*> visited{entry};
  std::deque<const ::llvm::BasicBlock*> q{entry};

  while (q.size()) {
    const ::llvm::BasicBlock* current = q.front();
    q.pop_front();

    auto it = blocks.find(current);
    if (it == blocks.end()) {
      return Status(labm8::error::Code::FAILED_PRECONDITION, "Block not found");
    }
    Node* currentExit = it->second.second;

    // For each current -> successor pair, construct a control edge from the
    // last instruction in the current basic block to the first instruction in
    // the successor block.
    int32_t successorNumber = 0;
    for (const ::llvm::BasicBlock* successor : ::llvm::successors(current)) {
      auto it = blocks.find(successor);
      if (it == blocks.end()) {
        return Status(labm8::error::Code::FAILED_PRECONDITION, "Block not found");
      }
      Node* successorEntry = it->second.first;

      RETURN_IF_ERROR(
          AddControlEdge(/*position=*/successorNumber, currentExit, successorEntry).status());

      if (visited.find(successor) == visited.end()) {
        q.push_back(successor);
        visited.insert(successor);
      }
      ++successorNumber;
    }

    // If the block has no successors, record the block exit instruction.
    if (successorNumber == 0) {
      functionEntryExits.second.push_back(currentExit);
    }
  }

  return functionEntryExits;
}

Status ProgramGraphBuilder::AddCallSite(const Node* source, const FunctionEntryExits& target) {
  RETURN_IF_ERROR(AddCallEdge(source, target.first).status());
  if (!options().ignore_call_returns()) {
    for (auto exitNode : target.second) {
      RETURN_IF_ERROR(AddCallEdge(exitNode, source).status());
    }
  }
  return Status::OK;
}

Node* ProgramGraphBuilder::AddLlvmInstruction(const ::llvm::Instruction* instruction,
                                              const Function* function) {
  const LlvmTextComponents text = textEncoder_.Encode(instruction);
  Node* node = AddInstruction(text.opcode_name, function);
  node->set_block(blockCount_);
  graph::AddScalarFeature(node, "full_text", text.text);

#if PROGRAML_LLVM_VERSION_MAJOR > 3
  // Add profiling information features, if available.
  uint64_t profTotalWeight;
  if (instruction->extractProfTotalWeight(profTotalWeight)) {
    graph::AddScalarFeature(node, "llvm_profile_total_weight", profTotalWeight);
  }
  uint64_t profTrueWeight;
  uint64_t profFalseWeight;
  if (instruction->extractProfMetadata(profTrueWeight, profFalseWeight)) {
    graph::AddScalarFeature(node, "llvm_profile_true_weight", profTrueWeight);
    graph::AddScalarFeature(node, "llvm_profile_false_weight", profFalseWeight);
  }
#endif

  return node;
}

Node* ProgramGraphBuilder::AddLlvmVariable(const ::llvm::Instruction* operand,
                                           const programl::Function* function) {
  const LlvmTextComponents text = textEncoder_.Encode(operand);
  Node* node = AddVariable("var", function);
  node->set_block(blockCount_);
  graph::AddScalarFeature(node, "full_text", text.lhs);

  compositeTypeParts_.clear();  // Reset after previous call.
  Node* type = GetOrCreateType(operand->getType());
  CHECK(AddTypeEdge(/*position=*/0, type, node).ok());

  return node;
}

Node* ProgramGraphBuilder::AddLlvmVariable(const ::llvm::Argument* argument,
                                           const programl::Function* function) {
  const LlvmTextComponents text = textEncoder_.Encode(argument);
  Node* node = AddVariable("var", function);
  node->set_block(blockCount_);
  graph::AddScalarFeature(node, "full_text", text.lhs);

  compositeTypeParts_.clear();  // Reset after previous call.
  Node* type = GetOrCreateType(argument->getType());
  CHECK(AddTypeEdge(/*position=*/0, type, node).ok());

  return node;
}

Node* ProgramGraphBuilder::AddLlvmConstant(const ::llvm::Constant* constant) {
  const LlvmTextComponents text = textEncoder_.Encode(constant);
  Node* node = AddConstant("val");
  node->set_block(blockCount_);
  graph::AddScalarFeature(node, "full_text", text.text);

  compositeTypeParts_.clear();  // Reset after previous call.
  Node* type = GetOrCreateType(constant->getType());
  CHECK(AddTypeEdge(/*position=*/0, type, node).ok());

  return node;
}

Node* ProgramGraphBuilder::AddLlvmType(const ::llvm::Type* type) {
  // Dispatch to the type-specific handlers.
  if (::llvm::dyn_cast<::llvm::StructType>(type)) {
    return AddLlvmType(::llvm::dyn_cast<::llvm::StructType>(type));
  } else if (::llvm::dyn_cast<::llvm::PointerType>(type)) {
    return AddLlvmType(::llvm::dyn_cast<::llvm::PointerType>(type));
  } else if (::llvm::dyn_cast<::llvm::FunctionType>(type)) {
    return AddLlvmType(::llvm::dyn_cast<::llvm::FunctionType>(type));
  } else if (::llvm::dyn_cast<::llvm::ArrayType>(type)) {
    return AddLlvmType(::llvm::dyn_cast<::llvm::ArrayType>(type));
  } else if (::llvm::dyn_cast<::llvm::VectorType>(type)) {
    return AddLlvmType(::llvm::dyn_cast<::llvm::VectorType>(type));
  } else {
    const LlvmTextComponents text = textEncoder_.Encode(type);
    Node* node = AddType(text.text);
    graph::AddScalarFeature(node, "full_text", text.text);
    return node;
  }
}

Node* ProgramGraphBuilder::AddLlvmType(const ::llvm::StructType* type) {
  Node* node = AddType("struct");
  compositeTypeParts_[type] = node;
  graph::AddScalarFeature(node, "full_text", textEncoder_.Encode(type).text);

  // Add types for the struct elements, and add type edges.
  for (int i = 0; i < type->getNumElements(); ++i) {
    const auto& member = type->elements()[i];
    // Don't re-use member types in structs, always create a new type. For
    // example, the code:
    //
    //     struct S {
    //         int a;
    //         int b;
    //     };
    //     int c;
    //     int d;
    //
    // would produce four type nodes: one for S.a, one for S.b, and one which
    // is shared by c and d.
    Node* memberNode = AddLlvmType(member);
    CHECK(AddTypeEdge(/*position=*/i, memberNode, node).ok());
  }

  return node;
}

Node* ProgramGraphBuilder::AddLlvmType(const ::llvm::PointerType* type) {
  Node* node = AddType("*");
  graph::AddScalarFeature(node, "full_text", textEncoder_.Encode(type).text);

  auto elementType = type->getElementType();
  auto parent = compositeTypeParts_.find(elementType);
  if (parent == compositeTypeParts_.end()) {
    // Re-use the type if it already exists to prevent duplication.
    auto elementNode = GetOrCreateType(type->getElementType());
    CHECK(AddTypeEdge(/*position=*/0, elementNode, node).ok());
  } else {
    // Bottom-out for self-referencing types.
    CHECK(AddTypeEdge(/*position=*/0, parent->second, node).ok());
  }

  return node;
}

Node* ProgramGraphBuilder::AddLlvmType(const ::llvm::FunctionType* type) {
  const std::string signature = textEncoder_.Encode(type).text;
  Node* node = AddType(signature);
  graph::AddScalarFeature(node, "full_text", signature);
  return node;
}

Node* ProgramGraphBuilder::AddLlvmType(const ::llvm::ArrayType* type) {
  Node* node = AddType("[]");
  graph::AddScalarFeature(node, "full_text", textEncoder_.Encode(type).text);
  // Re-use the type if it already exists to prevent duplication.
  auto elementType = GetOrCreateType(type->getElementType());
  CHECK(AddTypeEdge(/*position=*/0, elementType, node).ok());
  return node;
}

Node* ProgramGraphBuilder::AddLlvmType(const ::llvm::VectorType* type) {
  Node* node = AddType("vector");
  graph::AddScalarFeature(node, "full_text", textEncoder_.Encode(type).text);
  // Re-use the type if it already exists to prevent duplication.
  auto elementType = GetOrCreateType(type->getElementType());
  CHECK(AddTypeEdge(/*position=*/0, elementType, node).ok());
  return node;
}

namespace {

std::string getModuleName(const ::llvm::Module& module) {
#if PROGRAML_LLVM_VERSION_MAJOR == 3
  return module.getModuleIdentifier();
#else
  return module.getSourceFileName();
#endif
}

}  // anonymous namespace

labm8::StatusOr<ProgramGraph> ProgramGraphBuilder::Build(const ::llvm::Module& module) {
  // A map from functions to their entry and exit nodes.
  absl::flat_hash_map<const ::llvm::Function*, FunctionEntryExits> functions;

  Module* moduleMessage = AddModule(getModuleName(module));

  graph::AddScalarFeature(moduleMessage, "llvm_target_triple", module.getTargetTriple());
  graph::AddScalarFeature(moduleMessage, "llvm_data_layout", module.getDataLayoutStr());

  for (const ::llvm::Function& function : module) {
    // Create the function message.
    Function* functionMessage = AddFunction(function.getName(), moduleMessage);

#if PROGRAML_LLVM_VERSION_MAJOR > 6
    // Add profiling information, if available.
    if (function.hasProfileData()) {
      auto profileCount = function.getEntryCount();
      Feature feature;
      feature.mutable_int64_list()->add_value(profileCount.getCount());
      functionMessage->mutable_features()->mutable_feature()->insert(
          {"llvm_profile_entry_count", feature});
    }
#elif PROGRAML_LLVM_VERSION_MAJOR > 3
    // Add profiling information, if available.
    if (function.hasProfileData()) {
      auto profileCount = function.getEntryCount();
      Feature feature;
      if (profileCount.hasValue()) {
        feature.mutable_int64_list()->add_value(profileCount.getValue());
        functionMessage->mutable_features()->mutable_feature()->insert(
            {"llvm_profile_entry_count", feature});
      }
    }
#endif

    FunctionEntryExits functionEntryExits;
    ASSIGN_OR_RETURN(functionEntryExits, VisitFunction(function, functionMessage));

    functions.insert({&function, functionEntryExits});
  }

  // Add call edges to and from the root node.
  for (auto fn : functions) {
    RETURN_IF_ERROR(AddCallSite(GetRootNode(), fn.second));
  }

  // Add call edges to and from call sites.
  for (auto callSite : callSites_) {
    const auto& calledFunction = functions.find(callSite.second);
    if (calledFunction == functions.end()) {
      return Status(labm8::error::Code::FAILED_PRECONDITION, "Could not resolve call to function");
    }
    RETURN_IF_ERROR(AddCallSite(callSite.first, calledFunction->second));
  }

  // Create the constants.
  for (const auto& constant : constants_) {
    Node* constantMessage = AddLlvmConstant(constant.first);

    // Create data in-flow edges.
    for (auto destination : constant.second) {
      Node* destinationNode = destination.first;
      int32_t position = destination.second;
      RETURN_IF_ERROR(AddDataEdge(position, constantMessage, destinationNode).status());
    }
  }

#if PROGRAML_LLVM_VERSION_MAJOR > 3
  // Add profiling information, if available.
  ::llvm::Metadata* profileMetadata = module.getModuleFlag("ProfileSummary");
  if (profileMetadata) {
    ::llvm::ProfileSummary* profileSummary = ::llvm::ProfileSummary::getFromMD(profileMetadata);
    if (!profileSummary) {
      return Status(labm8::error::Code::FAILED_PRECONDITION, "llvm::Module ProfileSymmary is null");
    }
    graph::AddScalarFeature(moduleMessage, "llvm_profile_num_functions",
                            profileSummary->getNumFunctions());
    graph::AddScalarFeature(moduleMessage, "llvm_profile_max_function_count",
                            profileSummary->getMaxFunctionCount());
    graph::AddScalarFeature(moduleMessage, "llvm_profile_num_counts",
                            profileSummary->getNumCounts());
    graph::AddScalarFeature(moduleMessage, "llvm_profile_total_count",
                            profileSummary->getTotalCount());
    graph::AddScalarFeature(moduleMessage, "llvm_profile_max_count", profileSummary->getMaxCount());
    graph::AddScalarFeature(moduleMessage, "llvm_profile_max_internal_count",
                            profileSummary->getMaxInternalCount());
  }
#endif

  return programl::graph::ProgramGraphBuilder::Build();
}

void ProgramGraphBuilder::Clear() {
  textEncoder_.Clear();
  constants_.clear();
  blockCount_ = 0;
  callSites_.clear();
  programl::graph::ProgramGraphBuilder::Clear();
}

Node* ProgramGraphBuilder::GetOrCreateType(const ::llvm::Type* type) {
  auto it = types_.find(type);
  if (it == types_.end()) {
    Node* node = AddLlvmType(type);
    types_[type] = node;
    return node;
  }
  return it->second;
}

}  // namespace internal
}  // namespace llvm
}  // namespace ir
}  // namespace programl
