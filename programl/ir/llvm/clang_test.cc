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
#include "programl/ir/llvm/clang.h"

#include <vector>

#include "labm8/cpp/status.h"
#include "labm8/cpp/test.h"
#include "programl/proto/ir.pb.h"

using labm8::Status;
namespace error = labm8::error;

namespace programl {
namespace ir {
namespace llvm {
namespace {

TEST(Clang, SimpleCProgram) {
  IrList irs;
  Clang compiler("-xc");
  Status status = compiler.Compile("int main() { return 0; }", &irs);

  ASSERT_TRUE(status.ok());
  EXPECT_EQ(irs.ir_size(), 8);
}

TEST(Clang, InvalidCProgram) {
  IrList irs;
  Clang compiler("-xc");
  Status status = compiler.Compile("~~~invalid~~~", &irs);

  EXPECT_EQ(status.code(), error::Code::INVALID_ARGUMENT);
  EXPECT_EQ(irs.ir_size(), 0);
}

}  // anonymous namespace
}  // namespace llvm
}  // namespace ir
}  // namespace programl

TEST_MAIN();
