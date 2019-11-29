"""Utility functions for working with program graph protos."""
import networkx as nx
import numpy as np

from deeplearning.ml4pl.graphs import programl_pb2
from labm8.py import app

FLAGS = app.FLAGS


def ProgramGraphToNetworkX(proto: programl_pb2) -> nx.MultiDiGraph:
  """Convert a ProgramGraph proto to a networkx graph.

  The networkx representation of a program graph uses node and edge-level
  attributes to encode the information of protobufs.

  The mapping from protocol buffer fields to networkx graph attributes is:

  Graph:
      * discrete_x (Optional, np.array of np.int32): ProgramGraph.discrete_x
      * discrete_y (Optional, np.array of np.int32): ProgramGraph.discrete_y
      * real_x (Optional, np.array of np.float32): ProgramGraph.real_x
      * real_y (Optional, np.array of np.float32): ProgramGraph.real_y

  Nodes:
      * type (Node.Type enum): Node.type
      * text (str): Node.text
      * preprocessed_text (str): Node.preprocessed_text
      * function (Optional, str): Function.name
      * discrete_x (Optional, np.array of np.int32): Node.discrete_x
      * discrete_y (Optional, np.array of np.int32): Node.discrete_y
      * real_x (Optional, np.array of np.float32): Node.real_x
      * real_y (Optional, np.array of np.float32): Node.real_y

  Edges:
      * flow (Edge.Flow enum): Edge.flow
      * position (int): Edge.position
  """
  g = nx.MultiDiGraph()

  # Add graph-level features and labels.
  if proto.discrete_x:
    g.graph["discrete_x"] = np.array(proto.discrete_x, dtype=np.int32)
  if proto.discrete_y:
    g.graph["discrete_y"] = np.array(proto.discrete_y, dtype=np.int32)
  if proto.real_x:
    g.graph["real_x"] = np.array(proto.real_x, dtype=np.float32)
  if proto.real_y:
    g.graph["real_y"] = np.array(proto.real_y, dtype=np.float32)

  # Build the nodes.
  for i, node in enumerate(proto.node):
    data = {
      "type": node.type,
      "text": node.text,
      "preprocessed_text": node.preprocessed_text,
    }
    if node.HasField("function"):
      data["function"]: str = proto.function[node.function].name
    if node.discrete_x:
      data["discrete_x"] = np.array(node.discrete_x, dtype=np.int32)
    if node.discrete_y:
      data["discrete_y"] = np.array(node.discrete_y, dtype=np.int32)
    if node.real_x:
      data["real_x"] = np.array(node.real_x, dtype=np.float32)
    if node.real_y:
      data["real_y"] = np.array(node.real_y, dtype=np.float32)
    g.add_node(i, **data)

  # Build the edges.
  for edge in proto.edge:
    g.add_edge(
      edge.source_node,
      edge.destination_node,
      flow=edge.flow,
      position=edge.position,
    )

  app.Log(
    1, "ProgramGraphToNetworkX PROTO FUNCTION COUNT %s", len(proto.function)
  )
  app.Log(
    1,
    "ProgramGraphToNetworkX GRAPH FUNCTION COUNT %s",
    len(set(fn for _, fn in g.nodes(data="function") if fn)),
  )

  return g


def NetworkXToProgramGraph(g: nx.MultiDiGraph) -> programl_pb2.ProgramGraph:
  """Perform the inverse transformation from networkx graph -> protobuf.

  See ProgramGraphToNetworkX() for details.
  """
  proto = programl_pb2.ProgramGraph()

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
  if "discrete_x" in g.graph:
    proto.discrete_x[:] = g.graph["discrete_x"].tolist()
  if "discrete_y" in g.graph:
    proto.discrete_y[:] = g.graph["discrete_y"].tolist()
  if "real_x" in g.graph:
    proto.real_x[:] = g.graph["real_x"].tolist()
  if "real_y" in g.graph:
    proto.real_y[:] = g.graph["real_y"].tolist()

  # Create the node list.
  for node, data in g.nodes(data=True):
    node_proto = proto.node.add()
    node_proto.type = data["type"]
    node_proto.text = data["text"]
    node_proto.preprocessed_text = data["preprocessed_text"]
    if "function" in data:
      node_proto.function = function_to_idx_map[data["function"]]
    if "discrete_x" in data:
      node_proto.discrete_x[:] = data["discrete_x"].tolist()
    if "discrete_y" in data:
      node_proto.discrete_y[:] = data["discrete_y"].tolist()
    if "real_x" in data:
      node_proto.real_x[:] = data["real_x"].tolist()
    if "real_y" in data:
      node_proto.real_y[:] = data["real_y"].tolist()

  # Create the edge list.
  for src, dst, data in g.edges(data=True):
    edge_proto = proto.edge.add()
    edge_proto.source_node = src
    edge_proto.destination_node = dst
    edge_proto.flow = data["flow"]
    edge_proto.position = data["position"]

  return proto
