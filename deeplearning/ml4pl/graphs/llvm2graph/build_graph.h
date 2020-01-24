#include "deeplearning/ml4pl/graphs/programl.pb.h"
#include "labm8/cpp/statusor.h"
#include "labm8/cpp/string.h"
#include "llvm/IR/Module.h"

namespace ml4pl {

// Construct a program graph from the given module.
labm8::StatusOr<ProgramGraph> BuildGraph(llvm::Module& module);

// Construct a program graph from a buffer for a module.
labm8::StatusOr<ProgramGraph> BuildGraph(const llvm::MemoryBuffer& irBuffer);

// Construct a program graph from a string of IR.
labm8::StatusOr<ProgramGraph> BuildGraph(const string& irString);

}  // namespace ml4pl
