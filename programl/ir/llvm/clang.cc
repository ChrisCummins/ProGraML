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
#include "programl/ir/llvm/clang.h"

#include "absl/strings/str_format.h"
#include "labm8/cpp/bazelutil.h"
#include "labm8/cpp/status.h"
#include "subprocess/subprocess.hpp"

using labm8::Status;
namespace error = labm8::error;

namespace programl {
namespace ir {
namespace llvm {

#ifdef __APPLE__
const char* kClangPath = "clang-llvm-10.0.0-x86_64-apple-darwin/bin/clang++";
#else
const char* kClangPath = "clang-llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/bin/clang++";
#endif

Status Clang::Compile(const string& src, IrList* irs) const {
  for (size_t i = 0; i < compileCommands_.size(); ++i) {
    auto process = subprocess::Popen(compileCommands_[i], subprocess::input{subprocess::PIPE},
                                     subprocess::output{subprocess::PIPE},
                                     subprocess::error{subprocess::PIPE});
    auto stdout = process.communicate(src.c_str(), src.size()).first;
    if (process.retcode()) {
      return Status(error::Code::INVALID_ARGUMENT, "Compilation failed");
    }

    Ir* ir = irs->add_ir();
    ir->set_type(Ir::LLVM);
    ir->set_compiler_version(600);
    ir->set_cmd(compileCommandsWithoutAbspath_[i]);
    ir->set_text(string(stdout.buf.begin(), stdout.buf.end()));
  }

  return Status::OK;
}

std::vector<string> Clang::BuildCompileCommands(const string& baseFlags, int timeout,
                                                bool abspath) {
  const string clangPath = (abspath ? labm8::BazelDataPathOrDie(kClangPath).string() : "clang++");

  const std::vector<string> opts{
      "-O0",
      "-O1",
      "-O2",
      "-O3",
      "-O0 -ffast-math",
      "-O1 -ffast-math",
      "-O2 -ffast-math",
      "-O3 -ffast-math",
  };

  std::vector<string> commands;
  commands.reserve(opts.size());
  for (const string& opt : opts) {
    commands.push_back(clangPath + " -emit-llvm -c -S " + baseFlags + " " + opt +
                       " - -o - -Wno-everything");
  }
  return commands;
}

}  // namespace llvm
}  // namespace ir
}  // namespace programl
