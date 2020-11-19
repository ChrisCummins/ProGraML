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

#include <fstream>
#include <sstream>
#include <streambuf>

#include "labm8/cpp/status_macros.h"

namespace programl {
namespace ir {
namespace xla {

labm8::StatusOr<string> ReadFileOrStdin(const string& path, std::istream& ins) {
  string str;

  if (path == "-") {
    str.assign((std::istreambuf_iterator<char>(ins)), std::istreambuf_iterator<char>());
  } else {
    std::ifstream file(path, std::ios::ate);
    if (!file) {
      std::stringstream err;
      err << "File not found: " << path;
      return labm8::Status(labm8::error::Code::INVALID_ARGUMENT, err.str());
    }

    str.reserve(file.tellg());
    file.seekg(0, std::ios::beg);

    str.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  }

  return str;
}

labm8::StatusOr<::xla::HloProto> GetHloProtoFromFileOrStdin(const string& path, bool wireFormat) {
  string serializedProto;
  ASSIGN_OR_RETURN(serializedProto, ReadFileOrStdin(path));

  ::xla::HloProto proto;
  if (wireFormat) {
    if (!proto.ParseFromString(serializedProto)) {
      return labm8::Status(labm8::error::Code::INVALID_ARGUMENT, "Failed to parse HloProto\n");
    }
  } else {
    return labm8::Status(labm8::error::Code::UNIMPLEMENTED, "Only wire format is supported");
  }

  return proto;
}

}  // namespace xla
}  // namespace ir
}  // namespace programl
