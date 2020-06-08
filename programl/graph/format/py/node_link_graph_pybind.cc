// This file defines python bindings for node_link_graph.
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
#include <pybind11/pybind11.h>
#include <sstream>
#include "labm8/cpp/status.h"
#include "labm8/cpp/string.h"
#include "nlohmann/json.hpp"
#include "programl/graph/format/node_link_graph.h"
#include "programl/proto/program_graph.pb.h"
#include "pybind11_json/pybind11_json.hpp"

using json = nlohmann::json;
using labm8::Status;
namespace py = pybind11;

namespace programl {
namespace graph {
namespace format {

PYBIND11_MODULE(node_link_graph_pybind, m) {
  m.doc() = "This module converts program graphs to JSON node link graph";

  m.def("ProgramGraphToNodeLinkGraph",
        [&](const string& serializedProto) {
          // De-serialize the input graph.
          ProgramGraph graph;
          if (!graph.ParseFromString(serializedProto)) {
            throw std::runtime_error("Failed to parse input proto");
          }

          // Convert to a node link graph.
          json dict;
          Status status = ProgramGraphToNodeLinkGraph(graph, &dict);
          if (!status.ok()) {
            throw std::runtime_error(status.ToString());
          }

          // Serialize JSON string.
          py::object obj = dict;
          return std::move(obj);
        },
        "Convert a program graph to JSON node link graph.");
}

}  // namespace format
}  // namespace graph
}  // namespace programl
