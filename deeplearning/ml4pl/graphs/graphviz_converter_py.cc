// A python binding for the C++ graphviz conversion routine.
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

#include "deeplearning/ml4pl/graphs/graphviz_converter.h"
#include "deeplearning/ml4pl/graphs/programl.pb.h"

#include "labm8/cpp/string.h"

#include <pybind11/pybind11.h>

// Convert the given serialized program graph proto to a graphviz string.
string ProgramGraphToGraphviz(const string& serializedProto) {
  // Parse the input protocol buffer.
  ml4pl::ProgramGraph graph;
  graph.ParseFromString(serializedProto);

  // Serialize to graphviz.
  std::stringstream buffer;
  ml4pl::SerializeGraphVizToString(graph, &buffer);
  return buffer.str();
}

PYBIND11_MODULE(graphviz_converter_py, m) {
  m.doc() = "Convert ProgramGraph protocol buffers to graphviz dot strings.";

  m.def("ProgramGraphToGraphviz", &ProgramGraphToGraphviz,
        "Convert the given serialized program graph proto to graphviz.");
}
