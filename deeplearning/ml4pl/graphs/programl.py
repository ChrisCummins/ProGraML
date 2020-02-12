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
"""Utility functions for working with program graphs.

When executed as a binary, this program reads a single program graph from
stdin, and writes a the same graph to stdout. Use --stdin_fmt and --stdout_fmt
to convert between different graph types.

Example usage:

  Convert a binary protocol buffer to a text version:

    $ bazel run //deeplearning/ml4pl/graphs:programl -- \
        --stdin_fmt=pb \
        --stdout_fmt=pbtxt \
        < /tmp/proto.pb > /tmp/proto.pbtxt
"""
import enum
import pickle
import sys
from typing import List
from typing import Optional

import networkx as nx
import numpy as np

from deeplearning.ml4pl.graphs import graphviz_converter_py
from deeplearning.ml4pl.graphs import programl_pb2
from labm8.py import app
from labm8.py import pbutil

FLAGS = app.FLAGS


class StdinGraphFormat(enum.Enum):
  """The format of a graph read from stdin."""

  # A binary protocol buffer.
  PB = 1
  # A text protocol buffer.
  PBTXT = 2
  # A pickled networkx graph.
  NX = 3


class StdoutGraphFormat(enum.Enum):
  """The format of a graph written to stdout."""

  # A binary protocol buffer.
  PB = 1
  # A text protocol buffer.
  PBTXT = 2
  # A pickled networkx graph.
  NX = 3
  # A graphviz dot string. WARNING: Conversion to DOT format is lossy. A DOT
  # format graph cannot be converted back to a protocol buffer.
  DOT = 4


app.DEFINE_enum(
  "stdin_fmt",
  StdinGraphFormat,
  StdinGraphFormat.PBTXT,
  "The format for input program graph.",
)
app.DEFINE_enum(
  "stdout_fmt",
  StdoutGraphFormat,
  StdoutGraphFormat.PBTXT,
  "The format for output program graphs.",
)
app.DEFINE_string(
  "node_labels",
  "text",
  "The Node message field to use for graphviz node labels.",
)


class GraphBuilder(object):
  """A helper object for constructing a well-formed program graph."""

  def __init__(
    self, x: Optional[List[int]] = None, y: Optional[List[int]] = None
  ):
    """Create a new graph.

    Args:
      x: A list of graph features.
      y: A list of graph labels.
    """
    self.g = nx.MultiDiGraph()
    self.g.graph["x"] = x or []
    self.g.graph["y"] = y or []

    self.last_node_index = -1
    self.last_function_counter = 0
    self.functions: List[str] = []

  @property
  def proto(self) -> programl_pb2.ProgramGraph:
    """Access the program graph as a protocol buffer."""
    return NetworkXToProgramGraph(self.g)

  def AddFunction(self, name: Optional[str] = None) -> str:
    """Create a new function and return its name.

    Args:
      name: The function name. If not given, one is generated.

    Returns:
      The function name.
    """
    if name is None:
      self.last_function_counter += 1
      name = f"fn_{self.last_function_counter:06d}"
    self.functions.append(name)
    return name

  def AddNode(
    self,
    type: programl_pb2.Node.Type = programl_pb2.Node.STATEMENT,
    text: str = "",
    preprocessed_text: str = "",
    function: Optional[str] = None,
    x: Optional[List[int]] = None,
    y: Optional[List[int]] = None,
  ) -> int:
    """Construct a new node.

    Args:
      type: The node type.
      text: The node text.
      preprocessed_text: The preprocessed node text.
      function: The name of a function created using AddFunction().
      x: A list of node features.
      y: A list of node labels.

    Returns:
      The integer index of the node.
    """
    self.last_node_index += 1
    self.g.add_node(
      self.last_node_index,
      type=type,
      x=x or [],
      y=y or [],
      text=text,
      function=function,
      preprocessed_text=preprocessed_text,
    )
    return self.last_node_index

  def AddEdge(
    self,
    source_node_index: int,
    destination_node_index: int,
    flow: programl_pb2.Edge.Flow = programl_pb2.Edge.CONTROL,
    position: int = 0,
  ):
    self.g.add_edge(
      source_node_index,
      destination_node_index,
      flow=flow,
      position=position,
      key=flow,
    )


def ProgramGraphToNetworkX(proto: programl_pb2) -> nx.MultiDiGraph:
  """Convert a ProgramGraph proto to a networkx graph.

  The networkx representation of a program graph uses node and edge-level
  attributes to encode the information of protobufs.

  The mapping from protocol buffer fields to networkx graph attributes is:

  Graph:
      * x (List[int]): ProgramGraph.x
      * y (List[int]): ProgramGraph.y
      * data_flow_root_node: ProgramGraph.data_flow_root_node
      * data_flow_steps: ProgramGraph.data_flow_steps
      * data_flow_positive_node_count: ProgramGraph.data_flow_positive_node_count
      
      Profile info if available:
      * llvm_profile_num_functions (int):
          ProgramGraph.LlvmProfile.num_functions (uint32)
      * llvm_profile_max_function_count (int):
          ProgramGraph.LlvmProfile.max_function_count (uint64)
      * llvm_profile_num_counts (int):
          ProgramGraph.LlvmProfile.num_counts (uint64)
      * llvm_profile_total_count (int):
          ProgramGraph.LlvmProfile.total_count (uint32)
      * llvm_profile_max_count (int):
          ProgramGraph.LlvmProfile.max_count (uint32)
      * llvm_profile_max_internal_count (int):
          ProgramGraph.LlvmProfile.max_internal_count (uint32)
      * llvm_function_entry_count (Dict[str, int]): A dictionary mapping
          Function.name to Function.llvm_entry_count (uint64).

  Nodes:
      * type (Node.Type enum): Node.type
      * text (str): Node.text
      * preprocessed_text (str): Node.preprocessed_text
      * function (Union[str, None]): Function.name
      * x (List[int]): Node.x
      * y (List[int]): Node.y
      
      Profile info if available:
        * llvm_profile_true_weight (int): Node.llvm_profile_true_weight
        * llvm_profile_false_weight (int): Node.llvm_profile_false_weight
        * llvm_profile_total_weight (int): Node.llvm_profile_total_weight

  Edges:
      * flow (Edge.Flow enum): Edge.flow
      * position (int): Edge.position
  """
  g = nx.MultiDiGraph()

  # Add graph-level features and labels.
  g.graph["x"] = list(proto.x)
  g.graph["y"] = list(proto.y)
  if proto.HasField("data_flow_root_node"):
    g.graph["data_flow_root_node"] = proto.data_flow_root_node
  if proto.HasField("data_flow_steps"):
    g.graph["data_flow_steps"] = proto.data_flow_steps
  if proto.HasField("data_flow_positive_node_count"):
    g.graph[
      "data_flow_positive_node_count"
    ] = proto.data_flow_positive_node_count

  # Graph-level LLVM profiling info.
  if proto.llvm_profile.HasField("num_functions"):
    g.graph["llvm_profile_num_functions"] = proto.llvm_profile.num_functions
  if proto.llvm_profile.HasField("max_function_count"):
    g.graph[
      "llvm_profile_max_function_count"
    ] = proto.llvm_profile.max_function_count
  if proto.llvm_profile.HasField("num_counts"):
    g.graph["llvm_profile_num_counts"] = proto.llvm_profile.num_counts
  if proto.llvm_profile.HasField("total_count"):
    g.graph["llvm_profile_total_count"] = proto.llvm_profile.total_count
  if proto.llvm_profile.HasField("max_count"):
    g.graph["llvm_profile_max_count"] = proto.llvm_profile.max_count
  if proto.llvm_profile.HasField("max_internal_count"):
    g.graph[
      "llvm_profile_max_internal_count"
    ] = proto.llvm_profile.max_internal_count

  # Function-level LLVM profiling info.
  function_llvm_entry_counts = {}
  for function in proto.function:
    if function.HasField("llvm_entry_count"):
      function_llvm_entry_counts[function.name] = function.llvm_entry_count
  if function_llvm_entry_counts:
    g.graph["llvm_function_entry_count"] = function_llvm_entry_counts

  # Build the nodes.
  for i, node in enumerate(proto.node):
    node_data = {
      "type": node.type,
      "text": node.text,
      "preprocessed_text": node.preprocessed_text,
      "function": (
        proto.function[node.function].name
        if node.HasField("function")
        else None
      ),
      "x": list(node.x),
      "y": list(node.y),
    }
    # Node-level LLVM profiling info.
    if node.HasField("llvm_profile_true_weight"):
      node_data["llvm_profile_true_weight"] = node.llvm_profile_true_weight
    if node.HasField("llvm_profile_false_weight"):
      node_data["llvm_profile_false_weight"] = node.llvm_profile_false_weight
    if node.HasField("llvm_profile_total_weight"):
      node_data["llvm_profile_total_weight"] = node.llvm_profile_total_weight

    g.add_node(i, **node_data)

  # Build the edges.
  for edge in proto.edge:
    g.add_edge(
      edge.source_node,
      edge.destination_node,
      flow=edge.flow,
      position=edge.position,
    )

  return g


def NetworkXToProgramGraph(
  g: nx.MultiDiGraph,
  proto: Optional[programl_pb2.ProgramGraph] = None,
  **proto_fields,
) -> programl_pb2.ProgramGraph:
  """Perform the inverse transformation from networkx graph -> protobuf.

  See ProgramGraphToNetworkX() for details.

  Arguments:
    g: A networkx graph.
    proto: An optional protocol buffer instance to use. Else a new one is
      created. Calling code is reponsible for clearning the protocol buffer.
    **proto_fields: Optional keyword arguments to use when constructing a proto.
      Has no effect if proto argument is set.

  Returns:
    A ProgramGraph proto instance.
  """
  proto = proto or programl_pb2.ProgramGraph(**proto_fields)

  # Create a map from function name to function ID.
  function_names = list(
    sorted(set([fn for _, fn in g.nodes(data="function") if fn]))
  )
  function_to_idx_map = {fn: i for i, fn in enumerate(function_names)}

  # Create the function protos.
  for function_name in function_names:
    function_proto = proto.function.add()
    function_proto.name = function_name
    # Function-level LLVM profiling info.
    if (
      "llvm_function_entry_count" in g.graph
      and function_name in g.graph["llvm_function_entry_count"]
    ):
      function_proto.llvm_entry_count = g.graph["llvm_function_entry_count"][
        function_name
      ]

  # Set the graph-level features and labels.
  proto.x[:] = np.array(g.graph["x"], dtype=np.int64).tolist()
  proto.y[:] = np.array(g.graph["y"], dtype=np.int64).tolist()
  if "data_flow_root_node" in g.graph:
    proto.data_flow_root_node = g.graph["data_flow_root_node"]
  if "data_flow_steps" in g.graph:
    proto.data_flow_steps = g.graph["data_flow_steps"]
  if "data_flow_positive_node_count" in g.graph:
    proto.data_flow_positive_node_count = g.graph[
      "data_flow_positive_node_count"
    ]
  # Graph-level LLVM profiling info.
  if "llvm_profile_num_functions" in g.graph:
    proto.llvm_profile.num_functions = g.graph["llvm_profile_num_functions"]
  if "llvm_profile_max_function_count" in g.graph:
    proto.llvm_profile.max_function_count = g.graph[
      "llvm_profile_max_function_count"
    ]
  if "llvm_profile_num_counts" in g.graph:
    proto.llvm_profile.num_counts = g.graph["llvm_profile_num_counts"]
  if "llvm_profile_total_count" in g.graph:
    proto.llvm_profile.total_count = g.graph["llvm_profile_total_count"]
  if "llvm_profile_max_count" in g.graph:
    proto.llvm_profile.max_count = g.graph["llvm_profile_max_count"]
  if "llvm_profile_max_internal_count" in g.graph:
    proto.llvm_profile.max_internal_count = g.graph[
      "llvm_profile_max_internal_count"
    ]

  # Create the node list.
  for node, data in g.nodes(data=True):
    node_proto = proto.node.add()
    node_proto.type = data["type"]
    node_proto.text = data["text"]
    node_proto.preprocessed_text = data["preprocessed_text"]
    if data["function"] is not None:
      node_proto.function = function_to_idx_map[data["function"]]
    node_proto.x[:] = np.array(data["x"], dtype=np.int64).tolist()
    node_proto.y[:] = np.array(data["y"], dtype=np.int64).tolist()
    # Node-level LLVM profiling info.
    if data.get("llvm_profile_true_weight") is not None:
      node_proto.llvm_profile_true_weight = data["llvm_profile_true_weight"]
    if data.get("llvm_profile_false_weight") is not None:
      node_proto.llvm_profile_false_weight = data["llvm_profile_false_weight"]
    if data.get("llvm_profile_total_weight") is not None:
      node_proto.llvm_profile_total_weight = data["llvm_profile_total_weight"]

  # Create the edge list.
  for src, dst, data in g.edges(data=True):
    edge_proto = proto.edge.add()
    edge_proto.source_node = src
    edge_proto.destination_node = dst
    edge_proto.flow = data["flow"]
    edge_proto.position = data["position"]

  return proto


def ProgramGraphToGraphviz(
  proto: programl_pb2, node_labels: Optional[str] = None
) -> str:
  """Convert a program graph protocol buffer to a graphviz dot string.

  Wraps the C++ method defined the graphviz_convert_py.cc pybind module.

  Args:
    proto: A program graph protocol buffer.

  Returns:
    A string suitable for feeding into `dot`.
  """
  proto_str = proto.SerializeToString()
  node_labels = node_labels or FLAGS.node_labels
  return graphviz_converter_py.ProgramGraphToGraphviz(proto_str, node_labels)


def FromBytes(
  data: bytes,
  fmt: StdinGraphFormat,
  proto: Optional[programl_pb2.ProgramGraph] = None,
  empty_okay: bool = False,
) -> programl_pb2.ProgramGraph:
  """Decode a byte array to a program graph proto.

  Args:
    data: The binary data to decode.
    fmt: The format of the binary data.
    proto: A ProgramGraph instance to reuse.
    empty_okay: If False, raise an error if the protocol buffer is not
      initialized, or contains no nodes.

  Returns:
    A program graph protocol buffer.
  """
  proto = proto or programl_pb2.ProgramGraph()
  if fmt == StdinGraphFormat.PB:
    proto.ParseFromString(data)
  elif fmt == StdinGraphFormat.PBTXT:
    pbutil.FromString(data.decode("utf-8"), proto)
  elif fmt == StdinGraphFormat.NX:
    NetworkXToProgramGraph(pickle.loads(data), proto=proto)
  else:
    raise ValueError(f"Unknown program graph format: {fmt}")

  if not empty_okay:
    if not proto.IsInitialized():
      raise ValueError("Program graph is uninitialized")
    if not proto.node:
      raise ValueError("Program graph contains no nodes")

  return proto


def StdoutGraphFormatToFileExtension(fmt: StdoutGraphFormat):
  if fmt == StdoutGraphFormat.PB:
    return ".pb"
  elif fmt == StdoutGraphFormat.PBTXT:
    return ".pbtxt"
  elif fmt == StdoutGraphFormat.NX:
    return ".nx.pickle"
  elif fmt == StdoutGraphFormat.DOT:
    return ".dot"
  else:
    raise TypeError(f"Unknown fmt: {fmt}")


def StdoutGraphFormatToStdinGraphFormat(fmt: StdoutGraphFormat):
  if fmt == StdoutGraphFormat.PB:
    return StdinGraphFormat.PB
  elif fmt == StdoutGraphFormat.PBTXT:
    return StdinGraphFormat.PBTXT
  elif fmt == StdoutGraphFormat.NX:
    return StdinGraphFormat.NX
  elif fmt == StdoutGraphFormat.DOT:
    raise TypeError("Cannot construct graphs from dot format")
  else:
    raise TypeError(f"Unknown fmt: {fmt}")


def ToBytes(
  program_graph: programl_pb2.ProgramGraph, fmt: StdoutGraphFormat
) -> bytes:
  """Convert a program graph to a byte array.

  Args:
    program_graph: A program graph.
    fmt: The desired binary format.

  Returns:
    A byte array.
  """
  if fmt == StdoutGraphFormat.PB:
    return program_graph.SerializeToString()
  elif fmt == StdoutGraphFormat.PBTXT:
    return str(program_graph).encode("utf-8")
  elif fmt == StdoutGraphFormat.NX:
    return pickle.dumps(ProgramGraphToNetworkX(program_graph))
  elif fmt == StdoutGraphFormat.DOT:
    return ProgramGraphToGraphviz(program_graph).encode("utf-8")
  else:
    raise ValueError(f"Unknown program graph format: {fmt}")


def SerializedProgramGraphToBytes(
  serialized_proto: bytes, fmt: StdoutGraphFormat
) -> bytes:
  """Convert a serialized ProgramGraphProto to a byte array.

  Args:
    serialized_proto: The serialized program graph proto.
    fmt: The output format of the byte array.

  Returns:
    An array of bytes.
  """
  if fmt == StdoutGraphFormat.PB:
    return serialized_proto
  proto = programl_pb2.ProgramGraph()
  proto.ParseFromString(serialized_proto)
  return ToBytes(proto, fmt)


def ReadStdin() -> programl_pb2.ProgramGraph:
  """Read a program graph from stdin using --stdin_fmt."""
  return FromBytes(sys.stdin.buffer.read(), FLAGS.stdin_fmt())


def WriteStdout(proto: pbutil.ProtocolBuffer) -> None:
  """Write a graph to stdout using --stdout_fmt."""
  sys.stdout.buffer.write(ToBytes(proto, FLAGS.stdout_fmt()))


def Main():
  """Main entry point."""
  WriteStdout(ReadStdin())


if __name__ == "__main__":
  app.Run(Main)
