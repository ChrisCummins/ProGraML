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
#include "programl/graph/format/graphviz_converter.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/util/stdin_fmt.h"
#include "programl/version.h"

const char* usage = R"(Convert a ProgramGraph message to GraphViz dot.

Usage:

    graph2dot [--node_label={,text,<feature>] [--stdin_fmt={pb,pbtxt}] < program_graph.pbtxt)

Where --node_label is the attribute to use for producing node labels. Possible values:
  --node_label=           No node labels
  --node_label=text       Use the Node.text field as node labels.
  --node_label=<feature>  Use the given Node.features.feature. If the feature is not found,
                          an error is raised.)";

DEFINE_string(node_label, "text", "The node attribute to use for node labels.");

using namespace programl;
using labm8::Status;
using namespace programl::graph::format;

void NodeLabelFromFlagsOrDie(NodeLabel* label, string* nodeFeatureName) {
  if (FLAGS_node_label == "text") {
    *label = kText;
  } else if (FLAGS_node_label == "") {
    *label = kNone;
  } else {
    *label = kFeature;
    *nodeFeatureName = FLAGS_node_label;
  }
}

int main(int argc, char** argv) {
  gflags::SetVersionString(PROGRAML_VERSION);
  labm8::InitApp(&argc, &argv, usage);
  if (argc != 1) {
    std::cerr << usage;
    return 4;
  }

  // NodeLabel label = kText;
  NodeLabel label;
  string nodeFeatureName;
  NodeLabelFromFlagsOrDie(&label, &nodeFeatureName);

  ProgramGraph graph;
  util::ParseStdinOrDie(&graph);

  Status status = SerializeGraphVizToString(graph, &std::cout, label, nodeFeatureName);
  if (!status.ok()) {
    std::cerr << "fatal: " << status.error_message() << std::endl;
    return 2;
  }

  return 0;
}
