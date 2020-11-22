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
#include "labm8/cpp/app.h"
#include "programl/graph/format/cdfg.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/util/stdin_fmt.h"
#include "programl/util/stdout_fmt.h"
#include "programl/version.h"

const char* usage =
    R"(Convert a ProgramGraph message to a Control and Data Flow Graph (CDFG).

Usage:

    graph2cdfg [--stdin_fmt={pb,pbtxt}]

The CDFG format is a subset of a ProgramGraph which excludes data elements
from the graph representation. Instead, data edges are connected directly
between defining instructions and users. The format is described in:

  Brauckmann, A., Ertel, S., Goens, A., & Castrillon, J. (2020). Compiler-Based Graph
  Representations for Deep Learning Models of Code. CC.)";

int main(int argc, char** argv) {
  gflags::SetVersionString(PROGRAML_VERSION);
  labm8::InitApp(&argc, &argv, usage);
  if (argc != 1) {
    std::cerr << usage;
    return 4;
  }

  programl::ProgramGraph graph;
  programl::util::ParseStdinOrDie(&graph);

  programl::graph::format::CDFGBuilder builder;
  programl::ProgramGraph cdfg = builder.Build(graph);
  programl::util::WriteStdout(cdfg);

  return 0;
}
