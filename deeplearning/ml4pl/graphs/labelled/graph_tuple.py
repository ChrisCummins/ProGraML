"""The module implements conversion of graphs to tuples of arrays."""
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import networkx as nx
import numpy as np

from deeplearning.ml4pl.graphs import programl_pb2
from labm8.py import app

FLAGS = app.FLAGS


class GraphTuple(NamedTuple):
  """The graph tuple: a compact representation of a labelled graph for machine
  learning.

  The transformation of ProgramGraph protocol buffer to GraphTuple is lossy
  (omitting attributes such as node text), and is partly specialized to the
  machine learning tasks that we have considered so far.

  See <github.com/ChrisCummins/ProGraML/issues/22>.
  """

  # A list of adjacency lists, one for each flow type, where an entry in an
  # adjacency list is a <src,dst> tuple of node indices.
  # Shape (edge_flow_count, edge_count, 2), dtype int32:
  adjacency_lists: np.array

  # A list of edge positions, one for each edge type. An edge position is an
  # integer in the range 0 <= x < edge_position_max.
  # Shape (edge_type_count, edge_count), dtype int32:
  edge_positions: np.array

  # A list of node feature arrays. Each row is a node, and each column is an
  # feature for that node.
  # Shape (node_count, node_x_dimensionality), dtype int32
  node_x: np.array

  # (optional) A list of node labels arrays.
  # Shape (node_count, node_y_dimensionality), dtype float32
  node_y: Optional[np.array] = None

  # (optional) A list of graph features arrays.
  # Shape (graph_x_dimensionality), dtype int32:
  graph_x: Optional[np.array] = None

  # (optional) A vector of graph labels arrays.
  # Shape (graph_y_dimensionality), dtype float32:
  graph_y: Optional[np.array] = None

  @property
  def node_count(self) -> int:
    """Return the number of nodes in the graph."""
    return self.node_x.shape[0]

  @property
  def edge_count(self) -> int:
    """Return the number of edges."""
    return self.adjacency_lists.shape[0]

  @property
  def edge_position_max(self) -> int:
    """Return the maximum edge position."""
    return max(self.edge_positions)

  @property
  def has_node_y(self) -> bool:
    """Return whether graph tuple has node labels."""
    return self.node_y is not None

  @property
  def has_graph_x(self) -> bool:
    """Return whether graph tuple has graph features."""
    return self.graph_x is not None

  @property
  def has_graph_y(self) -> bool:
    """Return whether graph tuple has graph labels."""
    return self.graph_y is not None

  @property
  def node_x_dimensionality(self) -> int:
    """Return the dimensionality of node features."""
    return self.node_x.shape[1]

  @property
  def node_y_dimensionality(self) -> int:
    """Return the dimensionality of node labels."""
    return self.node_y.shape[1] if self.has_node_y else 0

  @property
  def graph_x_dimensionality(self) -> int:
    """Return the dimensionality of graph features."""
    return self.graph_x.shape[1] if self.has_graph_x else 0

  @property
  def graph_y_dimensionality(self) -> int:
    """Return the dimensionality of graph labels."""
    return self.graph_y.shape[1] if self.has_graph_y else 0

  @classmethod
  def CreateFromNetworkX(cls, g: nx.MultiDiGraph) -> "GraphTuple":
    """Construct a graph tuple from a networkx graph.

    Args:
      g: The graph to convert to a graph_tuple. See
        deeplearning.ml4pl.graphs.programl.ProgramGraphToNetworkX() for a
        description of the networkx format.

    Returns:
      A GraphTuple instance.
    """
    # Create an adjacency list for each edge type.
    adjacency_lists: List[List[Tuple[int, int]]] = [
      [],
      [],
      [],  # {control, data, call} types.
    ]
    # Create an edge position list for each edge type.
    edge_positions: List[List[int]] = [
      [],
      [],
      [],  # {control, data, call} types.
    ]

    # Build the adjacency and positions lists.
    for src, dst, data in g.edges(data=True):
      adjacency_lists[data["flow"]].append((src, dst))
      edge_positions[data["flow"]].append(data["position"])

    # Convert the edge lists to numpy arrays.
    # Shape (edge_count, 2):
    adjacency_lists = np.array(
      [
        np.array(adjacency_list, dtype=np.int32)
        for adjacency_list in adjacency_lists
      ]
    )
    # Shape (edge_count, 1):
    edge_positions = np.array(
      [
        np.array(edge_position, dtype=np.int32)
        for edge_position in edge_positions
      ]
    )

    # Note(github.com/ChrisCummins/ProGraML/issues/22): Hardcoded to support
    # only discrete node features, real-valued node labels, discrete graph
    # features, and discrete graph labels.

    # Set the node embedding indices.
    node_x = [None] * g.number_of_nodes()
    for node, discrete_x in g.nodes(data="discrete_x"):
      node_x[node] = np.array(discrete_x, dtype=np.int32)
    # Shape (node_count, node_x_count):
    node_x = np.vstack(node_x)

    # Set the node labels.
    node_targets = [None] * g.number_of_nodes()
    for node, real_y in g.nodes(data="real_y"):
      node_targets[node] = real_y
    # Shape (node_count, node_real_y_count):
    node_y = np.vstack(node_targets)

    # Get the optional graph-level features and labels.
    graph_x = (
      np.array(g.graph["discrete_x"], dtype=np.int32)
      if g.graph["discrete_x"]
      else None
    )
    graph_y = (
      np.array(g.graph["discrete_y"], dtype=np.int32)
      if g.graph["discrete_y"]
      else None
    )

    # End of specialised tuple representation.

    return GraphTuple(
      adjacency_lists=adjacency_lists,
      edge_positions=edge_positions,
      node_x=node_x,
      node_y=node_y,
      graph_x=graph_x,
      graph_y=graph_y,
    )

  def ToNetworkx(self) -> nx.MultiDiGraph:
    """Construct a networkx graph from a graph tuple.

    Use this function for producing interpretable representation of graph
    tuples, but note that this is not an inverse of the CreateFromNetworkX()
    function, since critical information is lost, e.g. the text attribute of
    nodes, etc.
    """
    g = nx.MultiDiGraph()

    # Reconstruct the graph edges.
    for flow, (adjacency_list, position_list) in enumerate(
      zip(self.adjacency_lists, self.edge_positions)
    ):
      for (src, dst), position in zip(adjacency_list, position_list):
        g.add_edge(src, dst, flow=programl_pb2.Edge(flow), position=position)

    # Note(github.com/ChrisCummins/ProGraML/issues/22): Hardcoded to support
    # only discrete node features, real-valued node labels, discrete graph
    # features, and discrete graph labels.

    for i, x in enumerate(self.node_x):
      g.nodes[i]["discrete_x"] = x.tolist()

    if self.has_node_y:
      for i, y in enumerate(self.node_y):
        g.nodes[i]["real_y"] = y.tolist()

    if self.has_graph_x:
      g.graph["discrete_x"] = self.graph_x

    if self.has_graph_y:
      g.graph["discrete_y"] = self.graph_y

    # End of specialised tuple representation.

    return g
