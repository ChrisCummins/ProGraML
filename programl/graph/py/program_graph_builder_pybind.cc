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
#include "programl/graph/program_graph_builder.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace programl {
namespace graph {
PYBIND11_MODULE(program_graph_builder_pybind, m) {
  m.doc() = "A class for building program graphs";

  py::class_<ProgramGraphBuilder>(m, "ProgramGraphBuilder")
      .def(py::init<>())
      .def("_Build",
           [&](ProgramGraphBuilder& builder) {
             ProgramGraph graph = builder.Build().ValueOrException();
             std::stringstream str;
             graph.SerializeToOstream(&str);
             return py::bytes(str.str());
           })
      .def("Clear", &ProgramGraphBuilder::Clear)
      .def("AddModule",
           [&](ProgramGraphBuilder& builder, const string& name) {
             int index = builder.GetProgramGraph().module_size();
             builder.AddModule(name);
             return index;
           },
           py::arg("name"))
      .def("AddFunction",
           [&](ProgramGraphBuilder& builder, const string& name, int module) {
             const Module* mod = &builder.GetProgramGraph().module(module);
             int index = builder.GetProgramGraph().function_size();
             builder.AddFunction(name, mod);
             return index;
           },
           py::arg("name"), py::arg("module"))
      .def("AddInstruction",
           [&](ProgramGraphBuilder& builder, const string& text, int function) {
             const Function* fn = &builder.GetProgramGraph().function(function);
             int index = builder.GetProgramGraph().node_size();
             builder.AddInstruction(text, fn);
             return index;
           },
           py::arg("text"), py::arg("function"))
      .def("AddVariable",
           [&](ProgramGraphBuilder& builder, const string& text, int function) {
             const Function* fn = &builder.GetProgramGraph().function(function);
             int index = builder.GetProgramGraph().node_size();
             builder.AddVariable(text, fn);
             return index;
           },
           py::arg("text"), py::arg("function"))
      .def("AddConstant",
           [&](ProgramGraphBuilder& builder, const string& text) {
             int index = builder.GetProgramGraph().node_size();
             builder.AddConstant(text);
             return index;
           },
           py::arg("text"))

      .def("AddControlEdge",
           [&](ProgramGraphBuilder& builder, int source, int target,
               int position) {
             const Node* sourceNode = &builder.GetProgramGraph().node(source);
             const Node* targetNode = &builder.GetProgramGraph().node(target);
             builder.AddControlEdge(position, sourceNode, targetNode)
                 .status()
                 .RaiseException();
           },
           py::arg("source"), py::arg("target"), py::arg("position"))

      .def("AddDataEdge",
           [&](ProgramGraphBuilder& builder, int source, int target,
               int position) {
             const Node* sourceNode = &builder.GetProgramGraph().node(source);
             const Node* targetNode = &builder.GetProgramGraph().node(target);
             builder.AddDataEdge(position, sourceNode, targetNode)
                 .status()
                 .RaiseException();
           },
           py::arg("source"), py::arg("target"), py::arg("position"))

      .def("AddCallEdge",
           [&](ProgramGraphBuilder& builder, int source, int target) {
             const Node* sourceNode = &builder.GetProgramGraph().node(source);
             const Node* targetNode = &builder.GetProgramGraph().node(target);
             builder.AddCallEdge(sourceNode, targetNode)
                 .status()
                 .RaiseException();
           },
           py::arg("source"), py::arg("target"))

      .def_property_readonly("root",
                             [&](ProgramGraphBuilder& builder) { return 0; });
}

}  // namespace graph
}  // namespace programl