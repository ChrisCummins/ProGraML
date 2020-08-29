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
#include "programl/ir/llvm/llvm.h"

#include "labm8/cpp/status.h"
#include "labm8/cpp/status_macros.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "programl/ir/llvm/internal/program_graph_builder_pass.h"
#include "programl/proto/program_graph.pb.h"

using labm8::Status;

namespace programl {
namespace ir {
namespace llvm {

Status BuildProgramGraph(::llvm::Module& module, ProgramGraph* graph,
                         const ProgramGraphOptions& options) {
  ::llvm::legacy::PassManager passManager;
  ::llvm::PassManagerBuilder passManagerBuilder;
  passManagerBuilder.OptLevel = options.opt_level();
  passManagerBuilder.populateModulePassManager(passManager);

  // Create a graph builder pass. Ownership of this pointer is transferred to
  // legacy::PassManager on add().
  internal::ProgramGraphBuilderPass* pass = new internal::ProgramGraphBuilderPass(options);
  passManager.add(pass);

  passManager.run(module);
  ASSIGN_OR_RETURN(*graph, pass->GetProgramGraph());
  return Status::OK;
}

Status BuildProgramGraph(const ::llvm::MemoryBuffer& irBuffer, ProgramGraph* graph,
                         const ProgramGraphOptions& options) {
  ::llvm::SMDiagnostic error;
  ::llvm::LLVMContext ctx;
  auto module = ::llvm::parseIR(irBuffer.getMemBufferRef(), error, ctx);
  if (!module) {
    // Format an error message in the style of clang, complete with line number,
    // column number, then the offending line and a caret pointing at the
    // column. For example:
    //
    // 1:0: error: expected top-level entity
    //     node {
    //     ^
    return Status(labm8::error::Code::INVALID_ARGUMENT, "{}:{}: error: {}\n    {}\n    {}^",
                  error.getLineNo(), error.getColumnNo(), error.getMessage().str(),
                  error.getLineContents().str(), string(std::max(error.getColumnNo() - 1, 0), ' '));
  }

  return BuildProgramGraph(*module, graph, options);
}

Status BuildProgramGraph(const string& irString, ProgramGraph* graph,
                         const ProgramGraphOptions& options) {
  const auto irBuffer = ::llvm::MemoryBuffer::getMemBuffer(irString);
  return BuildProgramGraph(*irBuffer, graph, options);
}

}  // namespace llvm
}  // namespace ir
}  // namespace programl
