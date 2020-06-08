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
#include "programl/ir/xla/hlo_proto_reader.h"

#include "labm8/cpp/test.h"

#include <fstream>

namespace programl {
namespace ir {
namespace xla {

namespace {

class HloProtoReaderTest : public ::labm8::Test {};

TEST_F(HloProtoReaderTest, ReadStdinStringEmptyString) {
  std::istringstream input("");
  auto statusOr = ReadFileOrStdin("-", input);

  ASSERT_TRUE(statusOr.ok());
  EXPECT_EQ(statusOr.ValueOrDie(), "");
}

TEST_F(HloProtoReaderTest, ReadStdinStringHelloWorld) {
  std::istringstream input("Hello, world");
  auto statusOr = ReadFileOrStdin("-", input);

  ASSERT_TRUE(statusOr.ok());
  EXPECT_EQ(statusOr.ValueOrDie(), "Hello, world");
}

TEST_F(HloProtoReaderTest, NonExistentFile) {
  auto statusOr = ReadFileOrStdin(GetTempFile().string());

  ASSERT_FALSE(statusOr.ok());
}

TEST_F(HloProtoReaderTest, FileContents) {
  auto path = GetTempFile().string();

  std::ofstream file;
  file.open(path);
  file << "Hello world";
  file.close();

  auto statusOr = ReadFileOrStdin(path);

  ASSERT_TRUE(statusOr.ok());
  EXPECT_EQ(statusOr.ValueOrDie(), "Hello world");
}

TEST_F(HloProtoReaderTest, ParseEmptyFileAsProto) {
  auto path = GetTempFile().string();

  std::ofstream file;
  file.open(path);
  file << "";
  file.close();

  auto statusOr = GetHloProtoFromFileOrStdin(path);

  ASSERT_TRUE(statusOr.ok());
}

TEST_F(HloProtoReaderTest, NonExistentProto) {
  ASSERT_FALSE(GetHloProtoFromFileOrStdin(GetTempFile().string()).ok());
}

}  // namespace

}  // namespace xla
}  // namespace ir
}  // namespace programl

TEST_MAIN();
