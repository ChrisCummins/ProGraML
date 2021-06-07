// This file defines the APIs for constructing program graphs from LLVM-IR.
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

#include <utility>
#include <vector>

#include "labm8/cpp/status.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/proto/util.pb.h"

namespace programl {
namespace ir {
namespace llvm {

const static ProgramGraphOptions defaultOptions;

// Construct a program graph from the given module.
labm8::Status BuildProgramGraph(::llvm::Module& module, ProgramGraph* graph,
                                const ProgramGraphOptions& options = defaultOptions);

// Construct a program graph from a buffer for a module.
labm8::Status BuildProgramGraph(const ::llvm::MemoryBuffer& irBuffer, ProgramGraph* graph,
                                const ProgramGraphOptions& options = defaultOptions);

// Construct a program graph from a string of IR.
labm8::Status BuildProgramGraph(const string& irString, ProgramGraph* graph,
                                const ProgramGraphOptions& options = defaultOptions);

}  // namespace llvm
}  // namespace ir
}  // namespace programl
