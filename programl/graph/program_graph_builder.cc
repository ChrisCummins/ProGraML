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

#include "programl/graph/program_graph_builder.h"

#include "labm8/cpp/logging.h"
#include "labm8/cpp/status.h"
#include "labm8/cpp/status_macros.h"

using labm8::Status;
namespace error = labm8::error;

namespace programl {
namespace graph {

ProgramGraphBuilder::ProgramGraphBuilder(const ProgramGraphOptions& options) : options_(options) {
  // Create the graph root node.
  AddNode(Node::INSTRUCTION, "[external]");
}

Module* ProgramGraphBuilder::AddModule(const string& name) {
  int32_t index = GetProgramGraph().module_size();
  Module* module = GetMutableProgramGraph()->add_module();
  module->set_name(name);
  moduleIndices_.insert({module, index});
  emptyModules_.insert(module);
  return module;
}

Function* ProgramGraphBuilder::AddFunction(const string& name, const Module* module) {
  DCHECK(module) << "nullptr argument";
  int32_t index = GetProgramGraph().function_size();
  Function* function = GetMutableProgramGraph()->add_function();
  function->set_name(name);
  function->set_module(GetIndex(module));
  functionIndices_.insert({function, index});
  emptyFunctions_.insert(function);
  emptyModules_.erase(module);
  return function;
}

Node* ProgramGraphBuilder::AddInstruction(const string& text, const Function* function) {
  DCHECK(function) << "nullptr argument";
  return AddNode(Node::INSTRUCTION, text, function);
}

Node* ProgramGraphBuilder::AddVariable(const string& text, const Function* function) {
  DCHECK(function) << "nullptr argument";
  return AddNode(Node::VARIABLE, text, function);
}

Node* ProgramGraphBuilder::AddConstant(const string& text) { return AddNode(Node::CONSTANT, text); }

Node* ProgramGraphBuilder::AddType(const string& text) { return AddNode(Node::TYPE, text); }

labm8::StatusOr<Edge*> ProgramGraphBuilder::AddControlEdge(int32_t position, const Node* source,
                                                           const Node* target) {
  DCHECK(source) << "nullptr argument";
  DCHECK(target) << "nullptr argument";

  if (source->type() != Node::INSTRUCTION) {
    return Status(labm8::error::Code::INVALID_ARGUMENT,
                  "Invalid control source type ({}). Expected instruction",
                  Node::Type_Name(source->type()));
  }
  if (target->type() != Node::INSTRUCTION) {
    return Status(labm8::error::Code::INVALID_ARGUMENT,
                  "Invalid control target type ({}). Expected instruction",
                  Node::Type_Name(target->type()));
  }
  int32_t sourceIndex = GetIndex(source);
  if (sourceIndex && source->function() != target->function()) {
    const string& sourceFunction = GetProgramGraph().function(source->function()).name();
    const string& targetFunction = GetProgramGraph().function(target->function()).name();
    return Status(labm8::error::Code::INVALID_ARGUMENT,
                  "Source and target instructions must belong to the same "
                  "function. Source instruction has function `{}`, "
                  "target function `{}`",
                  sourceFunction, targetFunction);
  }

  return AddEdge(Edge::CONTROL, position, source, target);
}

labm8::StatusOr<Edge*> ProgramGraphBuilder::AddDataEdge(int32_t position, const Node* source,
                                                        const Node* target) {
  DCHECK(source) << "nullptr argument";
  DCHECK(target) << "nullptr argument";

  bool sourceIsData = (source->type() == Node::VARIABLE || source->type() == Node::CONSTANT);
  bool targetIsData = (target->type() == Node::VARIABLE || target->type() == Node::CONSTANT);

  if (!((sourceIsData && targetIsData) || (sourceIsData && target->type() == Node::INSTRUCTION) ||
        (targetIsData && source->type() == Node::INSTRUCTION))) {
    return Status(labm8::error::Code::INVALID_ARGUMENT,
                  "Data edge must connect either an instruction with data "
                  "OR data with an instruction. "
                  "Source has type {} and target has type {}",
                  Node::Type_Name(source->type()), Node::Type_Name(target->type()));
  }

  return AddEdge(Edge::DATA, position, source, target);
}

labm8::StatusOr<Edge*> ProgramGraphBuilder::AddCallEdge(const Node* source, const Node* target) {
  DCHECK(source) << "nullptr argument";
  DCHECK(target) << "nullptr argument";

  if (source->type() != Node::INSTRUCTION) {
    return Status(labm8::error::Code::INVALID_ARGUMENT,
                  "Invalid call source type ({}). Expected instruction",
                  Node::Type_Name(source->type()));
  }
  if (target->type() != Node::INSTRUCTION) {
    return Status(labm8::error::Code::INVALID_ARGUMENT,
                  "Invalid call target type ({}). Expected instruction",
                  Node::Type_Name(target->type()));
  }

  return AddEdge(Edge::CALL, /*position=*/0, source, target);
}

labm8::StatusOr<Edge*> ProgramGraphBuilder::AddTypeEdge(int32_t position, const Node* source,
                                                        const Node* target) {
  DCHECK(source) << "nullptr argument";
  DCHECK(target) << "nullptr argument";

  if (source->type() != Node::TYPE) {
    return Status(labm8::error::Code::INVALID_ARGUMENT,
                  "Invalid source type ({}) for type edge. Expected type",
                  Node::Type_Name(source->type()));
  }
  if (target->type() == Node::INSTRUCTION) {
    return Status(labm8::error::Code::INVALID_ARGUMENT,
                  "Invalid destination type (instruction) for type edge. "
                  "Expected {variable,constant,type}");
  }

  return AddEdge(Edge::TYPE, position, source, target);
}

labm8::StatusOr<ProgramGraph> ProgramGraphBuilder::Build() {
  if (options().strict()) {
    RETURN_IF_ERROR(ValidateGraph());
  }
  return GetProgramGraph();
}

labm8::Status ProgramGraphBuilder::ValidateGraph() const {
  // Check that all nodes except the root are connected. The root is allowed to
  // have no connections in the case where it is an empty graph.
  if (!emptyModules_.empty()) {
    return Status(labm8::error::Code::FAILED_PRECONDITION, "Module `{}` is empty",
                  (*emptyModules_.begin())->name());
  }

  if (!emptyFunctions_.empty()) {
    return Status(labm8::error::Code::FAILED_PRECONDITION, "Function `{}` is empty",
                  (*emptyFunctions_.begin())->name());
  }

  if (!unconnectedNodes_.empty()) {
    return Status(labm8::error::Code::FAILED_PRECONDITION, "{} has no connections: `{}`",
                  Node::Type_Name((*unconnectedNodes_.begin())->type()),
                  (*unconnectedNodes_.begin())->text());
  }

  return Status::OK;
}

void ProgramGraphBuilder::Clear() {
  graph_ = ProgramGraph::default_instance();

  moduleIndices_.clear();
  functionIndices_.clear();
  nodeIndices_.clear();

  emptyModules_.clear();
  emptyFunctions_.clear();
  unconnectedNodes_.clear();
}

Node* ProgramGraphBuilder::AddNode(const Node::Type& type) {
  int32_t index = GetProgramGraph().node_size();
  Node* node = GetMutableProgramGraph()->add_node();
  node->set_type(type);

  nodeIndices_.insert({node, index});
  unconnectedNodes_.insert(node);

  return node;
}

Node* ProgramGraphBuilder::AddNode(const Node::Type& type, const string& text) {
  Node* node = AddNode(type);
  node->set_text(text);
  return node;
}

Node* ProgramGraphBuilder::AddNode(const Node::Type& type, const string& text,
                                   const Function* function) {
  DCHECK(function) << "nullptr argument";

  Node* node = AddNode(type, text);
  node->set_function(GetIndex(function));
  emptyFunctions_.erase(function);
  return node;
}

Edge* ProgramGraphBuilder::AddEdge(const Edge::Flow& flow, int32_t position, const Node* source,
                                   const Node* target) {
  DCHECK(source) << "nullptr argument";
  DCHECK(target) << "nullptr argument";

  int32_t sourceIndex = GetIndex(source);
  int32_t targetIndex = GetIndex(target);

  Edge* edge = GetMutableProgramGraph()->add_edge();
  edge->set_source(sourceIndex);
  edge->set_target(targetIndex);
  edge->set_flow(flow);
  edge->set_position(position);

  unconnectedNodes_.erase(source);
  unconnectedNodes_.erase(target);

  return edge;
}

namespace {

template <typename T>
int32_t GetIndexOrDie(const absl::flat_hash_map<T*, int32_t>& map, const T* element) {
  DCHECK(element) << "nullptr argument";

  auto it = map.find(element);
  CHECK(it != map.end());
  return it->second;
}

}  // anonymous namespace

int32_t ProgramGraphBuilder::GetIndex(const Module* module) {
  DCHECK(module) << "nullptr argument";
  return GetIndexOrDie(moduleIndices_, module);
}

int32_t ProgramGraphBuilder::GetIndex(const Function* function) {
  DCHECK(function) << "nullptr argument";
  return GetIndexOrDie(functionIndices_, function);
}

int32_t ProgramGraphBuilder::GetIndex(const Node* node) {
  DCHECK(node) << "nullptr argument";
  return GetIndexOrDie(nodeIndices_, node);
}

}  // namespace graph
}  // namespace programl
