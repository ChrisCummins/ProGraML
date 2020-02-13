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
#include <iostream>
#include <memory>

#include "deeplearning/ml4pl/graphs/graphviz_converter.h"
#include "deeplearning/ml4pl/graphs/llvm2graph/build_graph.h"
#include "labm8/cpp/app.h"
#include "labm8/cpp/logging.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"

DEFINE_string(stdout_fmt, "pbtxt",
              "The type of output format to use. Valid options are: "
              "\"pb\" which prints binary protocol buffer, \"pbtxt\" which "
              "prints a text format protocol buffer, or \"dot\" which prints a "
              "graphviz dot string.");

// Assert that the stdout format is legal.
static bool ValidateStdoutFormat(const char* flagname, const string& value) {
  if (value == "pb" || value == "pbtxt" || value == "dot") {
    return true;
  }

  LOG(FATAL) << "Unknown --" << flagname << ": `" << value << "`. Supported "
             << "formats: pb,pbtxt,dot";
  return false;
}
DEFINE_validator(stdout_fmt, &ValidateStdoutFormat);

static const char* usage =
    "Generate program graph from an IR.\n"
    "\n"
    "Read an LLVM IR module from file and print the program graph to stdout:\n"
    "\n"
    "  $ llvm2graph /path/to/llvm.ir\n"
    "\n"
    "Use the \"-\" argument to read the input from stdin:\n"
    "\n"
    "  $ clang foo.c -emit-llvm -o - | llvm2graph -\n"
    "\n"
    "The output format is the textual representation of a ProgramGraph \n"
    "protocol buffer. A binary protocol buffer can be printed using:\n"
    "\n"
    "  $ llvm2graph /path/to/llvm.ir --stdout_fmt=pb > /tmp/llvm.pb\n"
    "\n"
    "Or a dot string using:\n"
    "\n"
    "  $ llvm2graph /path/to/llvm.ir --stdout_fmt=dot > /tmp/llvm.dot\n"
    "\n"
    "The output can then be processed by Graphviz.";

int main(int argc, char** argv) {
  labm8::InitApp(&argc, &argv, usage);

  if (argc != 2) {
    std::cerr << "Usage: llvm2graph <filepath>" << std::endl;
    return 4;
  }

  // Read the input from stdin or from file.
  auto buffer = llvm::MemoryBuffer::getFileOrSTDIN(argv[1]);
  if (!buffer) {
    std::cerr << "File not found: " << argv[1] << std::endl;
    return 1;
  }

  // Build the graph.
  const auto build = ml4pl::BuildGraph(*buffer.get());
  if (!build.ok()) {
    std::cerr << build.status().error_message() << std::endl;
    return 1;
  }

  // Print the generated graph.
  const auto graph = build.ValueOrDie();
  if (FLAGS_stdout_fmt == "pb") {
    graph.SerializeToOstream(&std::cout);
  } else if (FLAGS_stdout_fmt == "pbtxt") {
    std::cout << graph.DebugString();
  } else if (FLAGS_stdout_fmt == "dot") {
    ml4pl::SerializeGraphVizToString(graph, &std::cout);
  } else {
    LOG(FATAL) << "unreachable";
  }

  return 0;
}
