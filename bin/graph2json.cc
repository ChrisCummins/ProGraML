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
#include <iomanip>

#include "labm8/cpp/app.h"
#include "programl/graph/format/node_link_graph.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/util/stdin_fmt.h"
#include "programl/version.h"

const char* usage = R"(Convert a ProgramGraph message to JSON node link graph.

Usage:

    graph2json < program_graph.pbtxt)";

DEFINE_bool(pretty_print, false, "Pretty-print the generated JSON.");

using json = nlohmann::json;
using labm8::Status;

int main(int argc, char** argv) {
  gflags::SetVersionString(PROGRAML_VERSION);
  labm8::InitApp(&argc, &argv, usage);
  if (argc != 1) {
    std::cerr << usage;
    return 4;
  }

  programl::ProgramGraph graph;
  programl::util::ParseStdinOrDie(&graph);

  auto nodeLinkGraph = json({});
  Status status = programl::graph::format::ProgramGraphToNodeLinkGraph(graph, &nodeLinkGraph);
  if (!status.ok()) {
    std::cerr << "fatal: failed to convert ProgramGraph to node link graph ("
              << status.error_message() << ')' << std::endl;
    return 2;
  }

  if (FLAGS_pretty_print) {
    std::cout << std::setw(2) << nodeLinkGraph << std::endl;
  } else {
    std::cout << nodeLinkGraph << std::endl;
  }

  return 0;
}
