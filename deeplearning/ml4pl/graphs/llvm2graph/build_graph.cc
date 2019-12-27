#include "deeplearning/ml4pl/graphs/llvm2graph/build_graph.h"

#include "deeplearning/ml4pl/graphs/llvm2graph/graph_builder_pass.h"
#include "deeplearning/ml4pl/graphs/programl.pb.h"
#include "labm8/cpp/statusor.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

namespace ml4pl {

labm8::StatusOr<ProgramGraph> BuildGraph(llvm::Module& module) {
  // Construct an empty pass manager.
  llvm::legacy::PassManager passManager;
  llvm::PassManagerBuilder passManagerBuilder;
  passManagerBuilder.populateModulePassManager(passManager);

  // Create a graph builder pass. Ownership of this pointer is transferred to
  // legacy::PassManager on add().
  auto graph_builder_pass = new ml4pl::GraphBuilderPass();

  passManager.add(graph_builder_pass);

  // Run the pass on the module.
  passManager.run(module);

  return graph_builder_pass->GetGraph();
}

labm8::StatusOr<ProgramGraph> BuildGraph(const llvm::MemoryBuffer& irBuffer) {
  // Parse the IR module.
  llvm::SMDiagnostic error;
  llvm::LLVMContext ctx;
  auto module = llvm::parseIR(irBuffer.getMemBufferRef(), error, ctx);
  if (!module) {
    return labm8::Status(labm8::error::Code::INVALID_ARGUMENT,
                         error.getMessage().str());
  }

  // Build the graph.
  return BuildGraph(*module);
}

labm8::StatusOr<ProgramGraph> BuildGraph(const string& irString) {
  const auto irBuffer = llvm::MemoryBuffer::getMemBuffer(irString);
  return BuildGraph(*irBuffer);
}

}  // namespace ml4pl
