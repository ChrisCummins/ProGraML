// A programmatic interface to clang.
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

#include <vector>

#include "labm8/cpp/status.h"
#include "labm8/cpp/string.h"
#include "programl/proto/util.pb.h"

namespace programl {
namespace ir {
namespace llvm {

// A programmatic interface to clang.
//
// Instances of this object can be used to convert source files into LLVM-IR
// modules.
//
// This uses the data augmentation described in [1] to produce eight LLVM-IRs
// from each input source, compiling with different optimization levels -O0 to
// -O3 and with and without -ffast-math.
//
//     Ben-Nun, T., Jakobovits, A. S., & Hoefler, T. (2018). Neural Code
//     Comprehension: A Learnable Representation of Code Semantics. NeurIPS.
//
class Clang {
 public:
  // TODO(cummins): timeout is unused!
  Clang(const string& baseFlags, int timeout = 60)
      : compileCommands_(BuildCompileCommands(baseFlags, timeout, /*abspath=*/true)),
        compileCommandsWithoutAbspath_(
            BuildCompileCommands(baseFlags, timeout, /*abspath=*/false)) {}

  const std::vector<string> GetCompileCommands() { return compileCommands_; };

  // Compile the given source file to produce a list of LLVM-IRs. If any of the
  // compilation commands fails, returns error status.
  labm8::Status Compile(const string& src, IrList* irs) const;

 private:
  static std::vector<string> BuildCompileCommands(const string& baseFlags, int timeout,
                                                  bool abspath);

  const std::vector<string> compileCommands_;
  const std::vector<string> compileCommandsWithoutAbspath_;
};

}  // namespace llvm
}  // namespace ir
}  // namespace programl
