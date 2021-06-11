// Generate a program graph from a HLO module.
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
#include <fstream>
#include <sstream>
#include <streambuf>

#include "labm8/cpp/app.h"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/status.h"
#include "labm8/cpp/statusor.h"
#include "labm8/cpp/string.h"
#include "programl/ir/xla/hlo_module_graph_builder.h"
#include "programl/third_party/tensorflow/xla.pb.h"
#include "programl/util/stdin_fmt.h"
#include "programl/util/stdout_fmt.h"
#include "programl/version.h"

static const char* usage = R"(Generate program graph from a HLO module.

Read a HloProto message from file and print the program graph to stdout.

Tensorflow, JAX, Julia, and PyTorch can all be used as XLA frontends. To
run TensorFlow using XLA and dump HloProto files, run:

  $ TF_XLA_FLAGS=--tf_xla_auto_jit=2 \
    XLA_FLAGS="--xla_dump_hlo_as_proto --xla_dump_to=/tmp/hlo" \
    path/to/your/tf/program

Then read and convert the HloProto to a ProgramGraph using:

  $ xla2graph /tmp/hlo/module_0000.before_optimizations.hlo.pb)";

using labm8::Status;
namespace error = labm8::error;

template <typename ProtocolBuffer, int exitCode = 4>
void ParseInputOrDie(const string& filename, ProtocolBuffer* message) {
  // Read from stdin if filename is '-'.
  if (filename == "-") {
    programl::util::ParseStdinOrDie(message);
    return;
  }

  std::ifstream file(filename, std::ios::ate);
  if (!file) {
    LOG(ERROR) << "File not found: " << filename;
    exit(exitCode);
  }

  string str;
  str.reserve(file.tellg());
  file.seekg(0, std::ios::beg);

  str.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  if (!message->ParseFromString(str)) {
    LOG(ERROR) << "Failed to parse proto file: " << filename;
    exit(exitCode);
  }
}

int main(int argc, char** argv) {
  gflags::SetVersionString(PROGRAML_VERSION);
  labm8::InitApp(&argc, &argv, usage);
  if (argc > 2) {
    std::cerr << usage;
    return 4;
  }

  string filename("-");
  if (argc == 2) {
    filename = string(argv[1]);
  }

  xla::HloProto hlo;
  ParseInputOrDie(filename, &hlo);

  programl::ir::xla::HloModuleGraphBuilder builder;
  auto graphOr = builder.Build(hlo);
  if (!graphOr.ok()) {
    std::cerr << graphOr.status().error_message() << std::endl;
    return 1;
  }

  programl::util::WriteStdout(graphOr.ValueOrDie());

  return 0;
}
