// This file defines the pythons bindings for ProgramGraphBuilder.
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
#include "programl/graph/format/graph_serializer.h"

#include "labm8/cpp/string.h"
#include "programl/proto/program_graph.pb.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace programl {
namespace graph {
namespace format {

PYBIND11_MODULE(graph_serializer_pybind, m) {
  m.doc() = "Python bindings for serializing program graphs";

  m.def("_SerializeInstructionsInProgramGraph",
        [&](const string& serializedProto, int maxNodes) {
          // De-serialize the input graph.
          ProgramGraph graph;
          if (!graph.ParseFromString(serializedProto)) {
            throw std::runtime_error("Failed to parse input proto");
          }

          vector<int> nodeList;
          SerializeInstructionsInProgramGraph(graph, &nodeList, maxNodes);

          return nodeList;
        },
        "Convert a program graph to a serialized list of instruction nodes.");
}

}  // namespace format
}  // namespace graph
}  // namespace programl
