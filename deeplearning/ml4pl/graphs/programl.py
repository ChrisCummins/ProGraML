"""Utility functions for working with program graph protos."""
from typing import List
from typing import Optional

import networkx as nx
import numpy as np

from deeplearning.ml4pl.graphs import programl_pb2
from labm8.py import app

FLAGS = app.FLAGS


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

  Nodes:
      * type (Node.Type enum): Node.type
      * text (str): Node.text
      * preprocessed_text (str): Node.preprocessed_text
      * function (Union[str, None]): Function.name
      * x (List[int]): Node.x
      * y (List[int]): Node.y

  Edges:
      * flow (Edge.Flow enum): Edge.flow
      * position (int): Edge.position
  """
  g = nx.MultiDiGraph()

  # Add graph-level features and labels.
  g.graph["x"] = list(proto.x)
  g.graph["y"] = list(proto.y)

  # Build the nodes.
  for i, node in enumerate(proto.node):
    g.add_node(
      i,
      type=node.type,
      text=node.text,
      preprocessed_text=node.preprocessed_text,
      function=(
        proto.function[node.function].name
        if node.HasField("function")
        else None
      ),
      x=list(node.x),
      y=list(node.y),
    )

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

  # Set the graph-level features and labels.
  proto.x[:] = np.array(g.graph["x"], dtype=np.int32).tolist()
  proto.y[:] = np.array(g.graph["y"], dtype=np.int32).tolist()

  # Create the node list.
  for node, data in g.nodes(data=True):
    node_proto = proto.node.add()
    node_proto.type = data["type"]
    node_proto.text = data["text"]
    node_proto.preprocessed_text = data["preprocessed_text"]
    if data["function"] is not None:
      node_proto.function = function_to_idx_map[data["function"]]
    node_proto.x[:] = np.array(data["x"], dtype=np.int32).tolist()
    node_proto.y[:] = np.array(data["y"], dtype=np.int32).tolist()

  # Create the edge list.
  for src, dst, data in g.edges(data=True):
    edge_proto = proto.edge.add()
    edge_proto.source_node = src
    edge_proto.destination_node = dst
    edge_proto.flow = data["flow"]
    edge_proto.position = data["position"]

  return proto
