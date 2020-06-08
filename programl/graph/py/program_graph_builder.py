# Copyright 2019-2020 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This file defines a class for constructing program graphs."""
from programl.graph.py import program_graph_builder_pybind
from programl.proto import program_graph_pb2


class ProgramGraphBuilder(program_graph_builder_pybind.ProgramGraphBuilder):
  """A module for constructing a single program graph.

  Uses the builder pattern. Example usage:

      builder = ProgramGraphBuilder();
      mod = builder.AddModule("foo")
      fn = builder.AddFunction("A", mod)
      add = builder.AddInstruction("add", fn)
      builder.AddControlEdge(builder.root, a, position=0)

      graph = builder.Build().

    After calling Build(), the object may be re-used.
    """

  def Build(self) -> program_graph_pb2.ProgramGraph:
    proto = program_graph_pb2.ProgramGraph()
    proto.ParseFromString(self._Build())
    return proto
