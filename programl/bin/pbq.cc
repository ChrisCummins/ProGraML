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
#include <iostream>

#include "labm8/cpp/app.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/proto/util.pb.h"
#include "programl/third_party/tensorflow/xla.pb.h"
#include "programl/util/stdin_fmt.h"
#include "programl/util/stdout_fmt.h"
#include "programl/version.h"

const char* usage = R"(Query properties of a protocol buffer.

Usage:

    pbq <query> [--stdin_fmt=<fmt>] < <proto>

Decode a protocol buffer from stdin and query properties of it.
Accepts either text or binary-format protocol buffers, controlled
using --stdin_fmt={pb,pbtxt}.

Supported Queries
-----------------

    pbq ProgramGraph
      Prints an entire program graph.

    pbq ProgramGraph.Node.text
      Prints the text of each node in a graph, one node per line.

    pbq ProgramGraphList
      Prints an entire list of program graphs.

    pbq ProgramGraphFeaturesList
      Prints an entire program graph features list.

    pbq Ir
      Prints an entire IR message.

    pbq Ir.text
      Prints the text of an IR.

    pbq IrList
      Prints an entire IR message list.

    pbq SourceFile
      Prints an entire SourceFile message.

    pbq 'SELECT COUNT(graph) FROM ProgramGraphFeaturesList'
      Count the number of program graphs in a list of program graph features.

    pbq 'SELECT COUNT(node_features.data_flow_value) FROM ProgramGraphFeaturesList.graph'
      Count the total number of data flow values in a list of program graph features.

Example usages
--------------

Read a binary program graph to a human-readable format:

  $ pbq ProgramGraph < graph.pb

Print the IR text from a binary-format IR protocol buffer:

  $ cat program.Ir.pb | pbq Ir.text --stdin_fmt=pb

Print the text representation of the nodes in a graph derived from LLVM-IR:

  $ llvm2graph foo.ll | pbq ProgramGraph.Node.text)";

template <typename ProtocolBuffer>
ProtocolBuffer ReadStdin() {
  ProtocolBuffer proto;
  programl::util::ParseStdinOrDie(&proto);
  return proto;
}

template <typename ProtocolBuffer>
void DecodeAndPrint() {
  ProtocolBuffer proto = ReadStdin<ProtocolBuffer>();
  programl::util::WriteStdout(proto);
}

int main(int argc, char** argv) {
  gflags::SetVersionString(PROGRAML_VERSION);
  labm8::InitApp(&argc, &argv, usage);

  if (argc != 2) {
    std::cerr << usage;
    return 4;
  }

  const std::string name(argv[1]);

  if (name == "--help") {
    std::cerr << usage;
    return 4;
  }

  // TODO(cummins): Rewrite this to use a more general query language that
  // allows for composable queries, for example:
  //
  //      SELECT COUNT(node) FROM ProgramGraph where node.type = INSTRUCTION
  if (name == "ProgramGraph") {
    DecodeAndPrint<programl::ProgramGraph>();
  } else if (name == "ProgramGraph.Node.text") {
    const auto graph = ReadStdin<programl::ProgramGraph>();
    for (int i = 0; i < graph.node_size(); ++i) {
      std::cout << graph.node(i).text() << std::endl;
    }
  } else if (name == "ProgramGraphList") {
    DecodeAndPrint<programl::ProgramGraphList>();
  } else if (name == "ProgramGraphFeatures") {
    DecodeAndPrint<programl::ProgramGraphFeatures>();
  } else if (name == "ProgramGraphFeaturesList") {
    DecodeAndPrint<programl::ProgramGraphFeaturesList>();
  } else if (name == "Ir") {
    DecodeAndPrint<programl::Ir>();
  } else if (name == "Ir.text") {
    std::cout << ReadStdin<programl::Ir>().text() << std::endl;
  } else if (name == "IrList") {
    DecodeAndPrint<programl::IrList>();
  } else if (name == "SourceFile") {
    DecodeAndPrint<programl::SourceFile>();
  } else if (name == "SELECT COUNT(graph) FROM ProgramGraphFeaturesList") {
    const auto featuresList = ReadStdin<programl::ProgramGraphFeaturesList>();
    std::cout << featuresList.graph_size() << std::endl;
  } else if (name ==
             "SELECT COUNT(node_features.data_flow_value) FROM "
             "ProgramGraphFeaturesList.graph") {
    const auto featuresList = ReadStdin<programl::ProgramGraphFeaturesList>();
    size_t nodeFeaturesCount = 0;
    for (int i = 0; i < featuresList.graph_size(); ++i) {
      const auto& it = featuresList.graph(i).node_features().feature_list().find("data_flow_value");
      if (it != featuresList.graph(i).node_features().feature_list().end()) {
        nodeFeaturesCount += it->second.feature_size();
      }
    }
    std::cout << nodeFeaturesCount << std::endl;
  } else if (name == "HloProto") {
    DecodeAndPrint<xla::HloProto>();
  } else {
    std::cerr << "Unrecognized attribute: " << name << std::endl;
    return 4;
  }

  return 0;
}
