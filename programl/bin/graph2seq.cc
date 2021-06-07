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
#include "programl/graph/format/graph_serializer.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/util/stdin_fmt.h"
#include "programl/util/stdout_fmt.h"
#include "programl/version.h"

const char* usage = R"(Serialize the instruction nodes in a program graph.

Usage:

    graph2seq [--stdin_fmt={pb,pbtxt}] < program_graph.pbtxt

Read a ProgramGraph message from stdin and print a serialized list
of instruction node indices to stdout.)";

int main(int argc, char** argv) {
  gflags::SetVersionString(PROGRAML_VERSION);
  labm8::InitApp(&argc, &argv, usage);

  programl::ProgramGraph graph;
  programl::util::ParseStdinOrDie(&graph);

  programl::NodeIndexList serialized;
  programl::graph::format::SerializeInstructionsInProgramGraph(graph, &serialized,
                                                               /*maxNodes=*/1000000);
  programl::util::WriteStdout(serialized);

  return 0;
}
