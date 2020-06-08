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
#include <iomanip>
#include <iostream>

#include "labm8/cpp/status.h"
#include "programl/ir/llvm/llvm.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/proto/program_graph_options.pb.h"

#include "llvm/Support/SourceMgr.h"

using labm8::Status;

int main(int argc, char** argv) {
  programl::ProgramGraphOptions options;
  if (!options.ParseFromIstream(&std::cin)) {
    std::cerr << "Failed to read input" << std::endl;
    return 1;
  }

  auto buffer = llvm::MemoryBuffer::getFileOrSTDIN(options.ir_path());
  if (!buffer) {
    std::cerr << "Failed to read input IR" << std::endl;
    return 1;
  }

  programl::ProgramGraph graph;
  Status status =
      programl::ir::llvm::BuildProgramGraph(*buffer.get(), &graph, options);
  if (!status.ok()) {
    std::cerr << status.error_message() << std::endl;
    return 2;
  }

  graph.SerializeToOstream(&std::cout);

  return 0;
}
