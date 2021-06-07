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
#include <fstream>
#include <iomanip>
#include <iostream>

#include "labm8/cpp/app.h"
#include "labm8/cpp/status.h"
#include "labm8/cpp/statusor.h"
#include "labm8/cpp/strutil.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/SourceMgr.h"
#include "programl/ir/llvm/llvm.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/proto/util.pb.h"
#include "programl/util/stdout_fmt.h"
#include "programl/version.h"

using labm8::Status;
using labm8::StatusOr;

const char* usage = R"(Generate a program graph from LLVM-IR.

Read an LLVM-IR module from stdin and print the program graph to stdout. For example:

  $ llvm2graph < /path/to/llvm.ll

Or pipe the output directly from clang:

  $ clang foo.c -emit-llvm -o - | llvm2graph -

If the filename has suffix '.Ir.pb', the file is parsed as an Ir protocol buffer:

  $ llvm2graph /path/to/llvm.Ir.pb

If the filename has suffix '.IrList.pb', the file is parsed as an IrList protocol buffer
and the IR at position --ir_list_index (zero-based) is used:

  $ llvm2graph /path/to/list.IrList.pb --ir_list_index=2)";

DEFINE_bool(instructions_only, false, "Include only instructions in the generated program graph.");
DEFINE_bool(ignore_call_returns, false,
            "Include only instructions in the generated program graph.");
DEFINE_int32(ir_list_index, 0,
             "If reading an IrList protocol buffer, use this value to index "
             "into the list.");
DEFINE_bool(strict, false,
            "Validate that the generated graph conforms to expectations of "
            "what a graph should look like - i.e. the module is not empty, "
            "every function contains instructions, and it does not contain "
            "unreachable code.");

StatusOr<programl::ProgramGraphOptions> GetProgramGraphOptionsFromFlags() {
  programl::ProgramGraphOptions options;
  if (FLAGS_instructions_only) {
    options.set_instructions_only(true);
  }
  if (FLAGS_ignore_call_returns) {
    options.set_ignore_call_returns(true);
  }
  if (FLAGS_strict) {
    options.set_strict(true);
  }
  return options;
}

// Read the IR input as an LLVM buffer based on the filename:
//   * if '-', read stdin;
//   * if '.Ir.pb' suffix, read an Ir protocol buffer;
//   * if '.IrList.pb' suffix, read IrList protocol buffer and return index
//   --ir_list_index;
//   * else read text file.
llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> GetInputAsBuffer(const string& filename) {
  if (labm8::HasSuffixString(filename, ".IrList.pb")) {
    std::ifstream file(filename);
    programl::IrList irList;
    if (!irList.ParseFromIstream(&file)) {
      LOG(ERROR) << "Failed to parse IrList protocol buffer";
      return std::unique_ptr<llvm::MemoryBuffer>(nullptr);
    }
    const auto& ir = irList.ir(FLAGS_ir_list_index);
    return llvm::MemoryBuffer::getMemBuffer(ir.text());
  } else if (labm8::HasSuffixString(filename, ".Ir.pb")) {
    std::ifstream file(filename);
    programl::Ir ir;
    if (!ir.ParseFromIstream(&file)) {
      LOG(ERROR) << "Failed to parse Ir protocol buffer";
      return std::unique_ptr<llvm::MemoryBuffer>(nullptr);
    }
    return llvm::MemoryBuffer::getMemBuffer(ir.text());
  } else {
    auto buf = llvm::MemoryBuffer::getFileOrSTDIN(filename);
    if (!buf) {
      LOG(ERROR) << "File not found: " << filename;
      return std::unique_ptr<llvm::MemoryBuffer>(nullptr);
    }
    return buf;
  }
}

int main(int argc, char** argv) {
  gflags::SetVersionString(PROGRAML_VERSION);
  labm8::InitApp(&argc, &argv, usage);

  string filename("-");

  if (argc > 2) {
    std::cerr << usage;
    return 4;
  } else if (argc == 2) {
    filename = string(argv[1]);
  }

  auto buffer = GetInputAsBuffer(filename);
  if (!buffer) {
    return 1;
  }

  auto options = GetProgramGraphOptionsFromFlags();
  if (!options.ok()) {
    LOG(ERROR) << options.status().error_message();
    return 4;
  }

  programl::ProgramGraph graph;
  Status status =
      programl::ir::llvm::BuildProgramGraph(*buffer.get(), &graph, options.ValueOrDie());
  if (!status.ok()) {
    LOG(ERROR) << status.error_message();
    return 2;
  }

  programl::util::WriteStdout(graph);

  return 0;
}
