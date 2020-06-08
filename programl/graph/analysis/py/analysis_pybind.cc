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
#include <sstream>
#include "labm8/cpp/status.h"
#include "labm8/cpp/string.h"
#include "programl/graph/analysis/analysis.h"
#include "programl/proto/program_graph.pb.h"
#include "programl/proto/program_graph_features.pb.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;
using labm8::Status;

namespace programl {
namespace graph {
namespace analysis {

PYBIND11_MODULE(analysis_pybind, m) {
  m.doc() = "Python bindings for analysis passes";

  m.def("RunAnalysis",
        [](const string& analysis, const string& serializedProgramGraph) {
          ProgramGraph graph;
          graph.ParseFromString(serializedProgramGraph);

          ProgramGraphFeaturesList featuresList;
          RunAnalysis(analysis, graph, &featuresList).RaiseException();

          std::stringstream str;
          featuresList.SerializeToOstream(&str);
          return py::bytes(str.str());
        });
}

}  // namespace analysis
}  // namespace graph
}  // namespace programl