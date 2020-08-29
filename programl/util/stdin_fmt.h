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
#include <sstream>

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/json_util.h"
#include "labm8/cpp/app.h"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/status.h"

DECLARE_string(stdin_fmt);

namespace programl {
namespace util {

// Read the entire stdin and parse as a protocol buffer of the given type.
// The format is determined by the --stdin_fmt flag.
template <typename ProtocolBuffer, int exitCode = 4>
void ParseStdinOrDie(ProtocolBuffer* message) {
  if (FLAGS_stdin_fmt == "pb") {
    if (!message->ParseFromIstream(&std::cin)) {
      LOG(ERROR) << "Failed to parse binary protocol buffer from stdin";
      exit(exitCode);
    }
  } else if (FLAGS_stdin_fmt == "pbtxt") {
    google::protobuf::io::IstreamInputStream istream(&std::cin);
    if (!google::protobuf::TextFormat::Parse(&istream, message)) {
      LOG(ERROR) << "Failed to parse text-format protocol buffer from stdin";
      exit(exitCode);
    }
  } else if (FLAGS_stdin_fmt == "json") {
    std::string input(std::istreambuf_iterator<char>(std::cin), {});
    if (!google::protobuf::util::JsonStringToMessage(input, message).ok()) {
      LOG(ERROR) << "Failed to parse JSON-format protocol buffer from stdin";
      exit(exitCode);
    }
  } else {
    LOG(FATAL) << "unreachable! Unrecognized --stdin_fmt";
  }
}

}  // namespace util
}  // namespace programl
