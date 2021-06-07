// An LLVM Module pass for constructing a program graph from a module.
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

#include "labm8/cpp/statusor.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "programl/ir/llvm/internal/program_graph_builder.h"
#include "programl/proto/program_graph.pb.h"

namespace programl {
namespace ir {
namespace llvm {
namespace internal {

// A module pass for constructing a graph.
//
// The way to use this class is to construct a pass pipeline with this in,
// feed it a single LLVM module, then call GetGraph() to access the graph that
// it constructed. This pass can only be run on a single module.
class ProgramGraphBuilderPass : public ::llvm::ModulePass {
 public:
  static char ID;

  explicit ProgramGraphBuilderPass(const ProgramGraphOptions& options)
      : ModulePass(ID),
        graph_(labm8::Status(labm8::error::Code::FAILED_PRECONDITION, "runOnModule() not called")),
        graphBuilder_(options) {}

  bool runOnModule(::llvm::Module& module) override;

  labm8::StatusOr<ProgramGraph> GetProgramGraph() const { return graph_; }

 private:
  labm8::StatusOr<ProgramGraph> graph_;
  ProgramGraphBuilder graphBuilder_;
};

}  // namespace internal
}  // namespace llvm
}  // namespace ir
}  // namespace programl
