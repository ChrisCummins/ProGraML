# Copyright 2019 the ProGraML authors.
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
import subprocess
from typing import Optional

import networkx as nx

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import humanize

FLAGS = app.FLAGS

# The path of the //deeplearning/ml4pl/graphs/llvm2graph executable.
LLVM2GRAPH = bazelutil.DataPath(
  "phd/deeplearning/ml4pl/graphs/llvm2graph/llvm2graph"
)


def BuildProgramGraphProto(
  module: str,
  timeout: int = 120,
  graph: Optional[programl_pb2.ProgramGraph] = None,
) -> programl_pb2.ProgramGraph:
  """Construct a program graph for the given LLVM IR.

  Args:
    module: The LLVM IR string for a module.
    timeout: The maximum number of seconds to allow graph construction to run
      for.
    graph: An existing graph message to write the result to. If not provided,
      a new graph message is constructed.

  Returns:
    A ProgramGraph message instance.

  Raises:
    TimeoutError: If graph construction fails to complete within timeout
      seconds.
    ValueError: If graph construction fails.
  """
  # NOTE: Ideally we would wrap the ml4pl::BuildProto() C++ function using
  # pybind11 and call directly into it (see
  # //deeplearning/ml4pl/graphs:graphviz_conter_py for an example of this).
  # However, I have so far been unable to make a pybind11 module work due to
  # double free / heap corruption errors when linking both pybind11 and LLVM
  # libraries in a single library. Because of this, we instead call the C++
  # binary as a subprocess and feed deserialize the protocol buffer output
  # from stdout. This has a higher overhead (requiring an extra proto serialize
  # and deserialize per call).

  graph = graph or programl_pb2.ProgramGraph()

  # Build and execute a llvm2graph command.
  cmd = [
    "timeout",
    "-s9",
    str(timeout),
    str(LLVM2GRAPH),
    "-",
    "--stdout_fmt",
    "pb",
  ]
  process = subprocess.Popen(
    cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
  )

  # Pass the module input to the binary.
  stdout, stderr = process.communicate(module.encode("utf-8"))

  # Handle an error in the llvm2graph command.
  if process.returncode == 9 or process.returncode == -9:
    raise TimeoutError(
      f"llvm2graph took longer than {timeout} seconds on "
      f"{humanize.BinaryPrefix(len(module), 'B')} input"
    )
  if process.returncode:
    message = "unknown error"
    try:
      message = stderr.decode("utf-8")
    finally:
      raise ValueError(message)

  # Parse the binary graph.
  graph.ParseFromString(stdout)
  return graph


def BuildProgramGraphNetworkX(
  module: str,
  timeout: int = 120,
  graph: Optional[programl_pb2.ProgramGraph] = None,
) -> nx.MultiDiGraph:
  """Construct a NetworkX program graph for the given LLVM IR.

  Args:
    module: The LLVM IR string for a module.
    timeout: The maximum number of seconds to allow graph construction to run
      for.
    graph: An existing graph message to write the result to. If not provided,
      a new graph message is constructed.

  Returns:
    A NetworkX MultiDiGraph instance.

  Raises:
    TimeoutError: If graph construction fails to complete within timeout
      seconds.
    ValueError: If graph construction fails.
  """
  # NOTE: A more direct approach to generating a networkx graph instance would
  # be to add a --stdout_fmt=json option to
  # //deeplearning/ml4pl/graphs/llvm2graph which would produce output in the
  # format expected by networkx. See:
  # https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.convert.to_dict_of_dicts.html#networkx.convert.to_dict_of_dicts
  return programl.ProgramGraphToNetworkX(
    BuildProgramGraphProto(module=module, timeout=timeout, graph=graph)
  )
