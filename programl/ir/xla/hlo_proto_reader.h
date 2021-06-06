// This file defines utility functions for reading input protos.
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

#include <istream>

#include "labm8/cpp/statusor.h"
#include "labm8/cpp/string.h"
#include "programl/third_party/tensorflow/xla.pb.h"

namespace programl {
namespace ir {
namespace xla {

// Read the contents of a file to string.
labm8::StatusOr<string> ReadFileOrStdin(const string& path, std::istream& stdin = std::cin);

// Read a HloProto message from a file.
//
// Args:
//     wireFormat: If true, read wire format proto. Else, read text
//         format proto.
labm8::StatusOr<::xla::HloProto> GetHloProtoFromFileOrStdin(const string& path,
                                                            bool wireFormat = true);

}  // namespace xla
}  // namespace ir
}  // namespace programl
