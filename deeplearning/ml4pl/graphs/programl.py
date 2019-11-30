"""Utility functions for working with program graph protos."""
from typing import Optional

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
      * discrete_x (List[int]): ProgramGraph.discrete_x
      * discrete_y (List[int]): ProgramGraph.discrete_y
      * real_x (List[float]): ProgramGraph.real_x
      * real_y (List[float]): ProgramGraph.real_y

  Nodes:
      * type (Node.Type enum): Node.type
      * text (str): Node.text
      * preprocessed_text (str): Node.preprocessed_text
      * function (Union[str, None]): Function.name
      * discrete_x (List[int]): Node.discrete_x
      * discrete_y (List[int]): Node.discrete_y
      * real_x (List[float]): Node.real_x
      * real_y (List[float]): Node.real_y

  Edges:
      * flow (Edge.Flow enum): Edge.flow
      * position (int): Edge.position
  """
  g = nx.MultiDiGraph()

  # Add graph-level features and labels.
  g.graph["discrete_x"] = list(proto.discrete_x)
  g.graph["discrete_y"] = list(proto.discrete_y)
  g.graph["real_x"] = list(proto.real_x)
  g.graph["real_y"] = list(proto.real_y)

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
      discrete_x=list(node.discrete_x),
      discrete_y=list(node.discrete_y),
      real_x=list(node.real_x),
      real_y=list(node.real_y),
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
  g: nx.MultiDiGraph, proto: Optional[programl_pb2.ProgramGraph] = None
) -> programl_pb2.ProgramGraph:
  """Perform the inverse transformation from networkx graph -> protobuf.

  See ProgramGraphToNetworkX() for details.

  Arguments:
    g: A networkx graph.

  Returns:
    A ProgramGraph proto instance.
  """
  proto = proto or programl_pb2.ProgramGraph()

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
  proto.discrete_x[:] = np.array(g.graph["discrete_x"], dtype=np.int32).tolist()
  proto.discrete_y[:] = np.array(g.graph["discrete_y"], dtype=np.int32).tolist()
  proto.real_x[:] = np.array(g.graph["real_x"], dtype=np.float32).tolist()
  proto.real_y[:] = np.array(g.graph["real_y"], dtype=np.float32).tolist()

  # Create the node list.
  for node, data in g.nodes(data=True):
    node_proto = proto.node.add()
    node_proto.type = data["type"]
    node_proto.text = data["text"]
    node_proto.preprocessed_text = data["preprocessed_text"]
    if data["function"] is not None:
      node_proto.function = function_to_idx_map[data["function"]]
    node_proto.discrete_x[:] = np.array(
      data["discrete_x"], dtype=np.int32
    ).tolist()
    node_proto.discrete_y[:] = np.array(
      data["discrete_y"], dtype=np.int32
    ).tolist()
    node_proto.real_x[:] = np.array(data["real_x"], dtype=np.float32).tolist()
    node_proto.real_y[:] = np.array(data["real_y"], dtype=np.float32).tolist()

  # Create the edge list.
  for src, dst, data in g.edges(data=True):
    edge_proto = proto.edge.add()
    edge_proto.source_node = src
    edge_proto.destination_node = dst
    edge_proto.flow = data["flow"]
    edge_proto.position = data["position"]

  return proto
