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
#include "programl/ir/llvm/internal/program_graph_builder_pass.h"

namespace programl {
namespace ir {
namespace llvm {
namespace internal {

char ProgramGraphBuilderPass::ID = 0;

bool ProgramGraphBuilderPass::runOnModule(::llvm::Module& module) {
  graph_ = graphBuilder_.Build(module);
  return /*modified=*/false;
}

}  // namespace internal
}  // namespace llvm
}  // namespace ir
}  // namespace programl
