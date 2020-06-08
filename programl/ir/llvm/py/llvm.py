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
"""Python interface for building program graphs from LLVM-IR."""
import subprocess
import tempfile

from labm8.py import bazelutil
from programl.proto import program_graph_options_pb2
from programl.proto import program_graph_pb2

GRAPH_BUILDER_BIN = bazelutil.DataPath(
  "phd/programl/ir/llvm/py/graph_builder_bin"
)

DefaultOptions = program_graph_options_pb2.ProgramGraphOptions()


def BuildProgramGraph(
  ir: str,
  options: program_graph_options_pb2.ProgramGraphOptions = DefaultOptions,
  timeout: int = 60,
) -> program_graph_pb2.ProgramGraph:
  """Construct a program graph from an LLVM-IR.

  Args:
    ir: The text of an LLVM-IR Module.
    options: The graph construction options.
    timeout: The number of seconds to permit before timing out.

  Returns:
    A ProgramGraph instance.

  Raises:
    ValueError: In case graph construction fails.
    TimeoutError: If timeout is reached.
    OsError: In case of other error.
  """
  # Write the ProgramGraphOptions to a temporary file and pass it to a
  # worker subprocess which generates the graph and produces a ProgramGraph
  # message on stdout.
  with tempfile.NamedTemporaryFile("w") as f:
    f.write(ir)
    f.flush()
    options.ir_path = f.name
    process = subprocess.Popen(
      ["timeout", "-s9", str(timeout), str(GRAPH_BUILDER_BIN)],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      stdin=subprocess.PIPE,
    )
    stdout, stderr = process.communicate(options.SerializeToString())

  proto = program_graph_pb2.ProgramGraph()
  if process.returncode == 2:
    raise ValueError(stderr.decode("utf-8").rstrip())
  elif process.returncode == 9 or process.returncode == -9:
    raise TimeoutError(f"Program graph construction exceeded {timeout} seconds")
  elif process.returncode:
    raise OSError(stderr.decode("utf-8").rstrip())
  proto.ParseFromString(stdout)
  return proto
