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
#include "programl/util/stdout_fmt.h"

DEFINE_string(stdout_fmt, "pbtxt",
              "The format of output. Valid options are: "
              "\"pbtxt\" for a text-format protocol buffer, "
              "\"pb\" for a binary format protocol buffer, "
              "or \"json\" for JSON. "
              "Text format protocol buffers are recommended for human-readable output, "
              "binary-format for efficient and fast file storage, and JSON for "
              "processing "
              "with third-party tools such as `jq`.");

// Assert that the stdout format is legal.
static bool ValidateStdoutFormat(const char* flagname, const string& value) {
  if (value == "pb" || value == "pbtxt" || value == "json") {
    return true;
  }

  LOG(FATAL) << "Unknown --" << flagname << ": `" << value << "`. Supported "
             << "formats: pb,pbtxt,json";
  return false;
}
DEFINE_validator(stdout_fmt, &ValidateStdoutFormat);
