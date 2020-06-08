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
#include "programl/graph/format/graph_tuple.h"

#include "labm8/cpp/string.h"
#include "programl/proto/program_graph.pb.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace programl {
namespace graph {
namespace format {

PYBIND11_MODULE(graph_tuple_pybind, m) {
  m.doc() = "Python bindings for GraphTupless";

  py::class_<GraphTuple>(m, "GraphTuple")
      .def(py::init<>())
      .def("_AddProgramGraph",
           [&](GraphTuple& tuple, const string& serializedProto) {
             ProgramGraph graph;
             graph.ParseFromString(serializedProto);
             return tuple.AddProgramGraph(graph).RaiseException();
           })
      .def("Clear", &GraphTuple::Clear)
      .def_property_readonly("_adjacencies", &GraphTuple::adjacencies)
      .def_property_readonly("_edge_positions", &GraphTuple::edge_positions)
      .def_property_readonly("node_size", &GraphTuple::node_size)
      .def_property_readonly("edge_size", &GraphTuple::edge_size)
      .def_property_readonly("node_sizes", &GraphTuple::node_size)
      .def_property_readonly("edge_sizes", &GraphTuple::edge_size)
      .def_property_readonly("graph_size", &GraphTuple::graph_size)
      .def_property_readonly("control_edge_size",
                             &GraphTuple::control_edge_size)
      .def_property_readonly("data_edge_size", &GraphTuple::data_edge_size)
      .def_property_readonly("call_edge_size", &GraphTuple::call_edge_size);
}

}  // namespace format
}  // namespace graph
}  // namespace programl
