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
"""This module defines convertors from ProgramGraph to CDFG."""
import subprocess
from typing import Optional

import google.protobuf.message

from labm8.py import bazelutil
from programl.proto import program_graph_pb2

GRAPH2CDFG = bazelutil.DataPath("phd/programl/cmd/graph2cdfg")


def FromProgramGraphFile(path) -> Optional[program_graph_pb2.ProgramGraph]:
  """Convert a binary ProgramGraph message file to a CDFG.

  Args:
    path: The path of a ProgramGraph protocol buffer.

  Returns:
    A ProgramGraph instance, or None if graph conversion failed.

  Raises:
    ValueError: If the graph cannot be converted.
  """
  graph = program_graph_pb2.ProgramGraph()

  with open(path, "rb") as f:
    p = subprocess.Popen(
      [str(GRAPH2CDFG), "--stdin_fmt", "pb", "--stdout_fmt", "pb"],
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
    )
    stdout, _ = p.communicate(f.read())

  if p.returncode:
    return None

  try:
    graph.ParseFromString(stdout)
  except google.protobuf.message.DecodeError:
    return None

  return graph
