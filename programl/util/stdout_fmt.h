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

#include <iostream>

#include "google/protobuf/util/json_util.h"
#include "labm8/cpp/app.h"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/status.h"

DECLARE_string(stdout_fmt);

namespace programl {
namespace util {

// Write the given protocol buffer to stdout.
// The format is determined by the --stdout_fmt flag.
template <typename ProtocolBuffer>
void WriteStdout(const ProtocolBuffer& message) {
  if (FLAGS_stdout_fmt == "pb") {
    message.SerializeToOstream(&std::cout);
  } else if (FLAGS_stdout_fmt == "pbtxt") {
    std::cout << message.DebugString();
  } else if (FLAGS_stdout_fmt == "json") {
    google::protobuf::util::JsonPrintOptions opts;
    opts.add_whitespace = false;
    opts.always_print_primitive_fields = false;
    opts.always_print_enums_as_ints = false;
    opts.preserve_proto_field_names = true;
    std::string json;
    CHECK(google::protobuf::util::MessageToJsonString(message, &json, opts).ok());
    std::cout << json << std::endl;
  } else {
    LOG(FATAL) << "unreachable! Unrecognized --stdout_fmt";
  }
}

}  // namespace util
}  // namespace programl
