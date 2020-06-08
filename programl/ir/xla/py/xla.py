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
"""Generate program graphs from XLA HLO modules."""
from typing import Optional

from tensorflow.compiler.xla.service import hlo_pb2

from labm8.py import app
from programl.ir.xla.py import xla_pybind
from programl.proto import program_graph_pb2

FLAGS = app.FLAGS


def BuildProgramGraphProto(
  hlo_proto: hlo_pb2.HloProto,
) -> program_graph_pb2.ProgramGraph:
  """Construct a program graph for the given LLVM IR.

  Args:
    hlo_proto: The LLVM IR string for a module.

  Returns:
    A ProgramGraph message instance.

  Raises:
    ValueError: If graph construction fails.
  """
  # This requires a round trip serialized to / from strings, since I can't
  # figure out a way to get pybind11 to auto-generate bindings for protocol
  # buffers.
  graph = program_graph_pb2.ProgramGraph()
  serialized_graph = xla_pybind.BuildProgramGraphProto(
    hlo_proto.SerializeToString()
  )
  graph.ParseFromString(serialized_graph)
  return graph
