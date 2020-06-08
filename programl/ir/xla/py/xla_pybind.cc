// This file defines the pybind11 bindings for xla2graph.
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
#include "labm8/cpp/statusor.h"
#include "labm8/cpp/string.h"
#include "programl/ir/xla/hlo_module_graph_builder.h"

namespace py = pybind11;

PYBIND11_MODULE(xla_pybind, m) {
  m.doc() = "Generate program graphs from XLA HLO modules.";

  m.def("BuildProgramGraphProto",
        [&](const string& serializedProto) {
          programl::ir::xla::HloModuleGraphBuilder builder;
          ::xla::HloProto proto;
          if (!proto.ParseFromString(serializedProto)) {
            throw std::runtime_error("Failed to parse input proto");
          }

          auto graph = builder.Build(proto).ValueOrException();

          std::stringstream out;
          graph.SerializeToOstream(&out);
          return py::bytes(out.str());
        },
        "Build a serialized ProgramGraph from a serialized HloProto.");
}
