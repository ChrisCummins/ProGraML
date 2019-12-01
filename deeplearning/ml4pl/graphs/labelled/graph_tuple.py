"""The module implements conversion of graphs to tuples of arrays."""
from typing import Iterable
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
  """The graph tuple: a compact representation of one or more graphs.

  The transformation of ProgramGraph protocol buffer to GraphTuples is lossy
  (omitting attributes such as node types and texts).

  See <github.com/ChrisCummins/ProGraML/issues/22>.
  """

  # A list of adjacency lists, one for each flow type, where an entry in an
  # adjacency list is a <src,dst> tuple of node indices.
  # Shape (edge_flow_count, edge_count, 2), dtype int32:
  adjacencies: np.array

  # A list of edge positions, one for each edge type. An edge position is an
  # integer in the range 0 <= x < edge_position_max.
  # Shape (edge_flow_count, edge_count), dtype int32:
  edge_positions: np.array

  # A list of node feature arrays. Each row is a node, and each column is an
  # feature for that node.
  # Shape (node_count, node_x_dimensionality), dtype int32
  node_x: np.array

  # (optional) A list of node labels arrays.
  # Shape (node_count, node_y_dimensionality), dtype float32
  node_y: Optional[np.array] = None

  # (optional) A list of graph features arrays.
  # Shape (graph_x_dimensionality) OR (graph_count, graph_x_dimensionality) if
  # graph_count > 1, dtype int32:
  graph_x: Optional[np.array] = None

  # (optional) A vector of graph labels arrays.
  # Shape (graph_y_dimensionality) OR (graph_count, graph_y_dimensionality) if
  # graph_count > 1, dtype float32:
  graph_y: Optional[np.array] = None

  # Disjoint graph properties:

  # The number of disconnected graphs in the tuple.
  graph_count: int = 1

  # A list of integers which segment the nodes by graph. E.g. with a GraphTuple
  # of two distinct graphs, both with three nodes, nodes_list will be
  # [0, 0, 0, 1, 1, 1].
  # Shape (node_count), dtype int32:
  nodes_list: np.array = None

  def ToGraphTuples(self) -> Iterable["GraphTuple"]:
    """Perform the inverse transformation from disjoint graph to a list of
    individual graph tuples.

    Returns:
      An iterable of graph instances.

    Raises:
      ValueError: If the graph tuple is not disjoint, or if the graph tuple
        is invalid.
    """
    if self.graph_count <= 1:
      raise ValueError(
        "ToGraphTuples() called on graph tuple of a single graph"
      )

    # Split the list of node indices into individual lists for each graph.
    graph_split_indices = (
      np.where(self.nodes_list[:-1] != self.nodes_list[1:])[0] + 1
    )
    if len(graph_split_indices) != self.graph_count:
      raise ValueError(
        f"Graph tuple contains {self.graph_count} disjoint "
        f"graphs but only found {len(graph_split_indices)} "
        "splits"
      )
    # Shape (graph_count, ?), dtype=np.int32.
    nodes_per_graph = np.split(self.nodes_list, graph_split_indices)

    # Iterate over the per-graph list of nodes.
    node_count = 0
    for current_graph, nodes in enumerate(nodes_per_graph):
      graph_node_count = len(nodes)

      # Per-flow edge attributes.
      current_adjacencies = [None, None, None]
      current_edge_positions = [None, None, None]

      for edge_flow, (adjacency_list, position_list) in enumerate(
        zip(self.adjacencies, self.edge_positions)
      ):
        # No edges of this type in the entire graph batch.
        if not adjacency_list.size:
          continue

        # The adjacency list contains the adjacencies for all graphs. Determine
        # those that are in this graph by selecting only those with a source
        # node in the list of this graph's nodes.
        srcs = adjacency_list[:, 0]
        edge_indicies = np.where(
          np.logical_and(
            srcs >= node_count, srcs < node_count + graph_node_count
          )
        )
        current_adjacencies[edge_flow] = adjacency_list[edge_indicies]
        current_edge_positions[edge_flow] = position_list[edge_indicies]

        # Negate the positive offset into the adjacency lists.
        offset = np.array((node_count, node_count), dtype=np.int32)
        current_adjacencies[edge_flow] -= offset

      # Read node features.
      current_node_x = self.node_x[node_count : node_count + graph_node_count]
      if len(current_node_x) != graph_node_count:
        raise ValueError(
          f"Graph has {len(current_node_x)} nodes but expected "
          f"{graph_node_count}"
        )

      # Read optional node labels.
      if self.has_node_y:
        current_node_y = self.node_y[node_count : node_count + graph_node_count]
        if len(current_node_y) != graph_node_count:
          raise ValueError(
            f"Graph has {len(current_node_y)} nodes but expected "
            f"{graph_node_count}"
          )
      else:
        current_node_y = None

      yield GraphTuple(
        adjacencies=current_adjacencies,
        edge_positions=current_edge_positions,
        node_x=current_node_x,
        node_y=current_node_y,
        graph_x=self.graph_x[current_graph] if self.has_graph_x else None,
        graph_y=self.graph_y[current_graph] if self.has_graph_y else None,
      )

      node_count += graph_node_count

  @property
  def is_disjoint_graph(self) -> bool:
    """Return whether the graph tuple is disjoint."""
    return self.graph_count > 1

  # End disjoint graph properties.

  @property
  def node_count(self) -> int:
    """Return the number of nodes in the graph."""
    return self.node_x.shape[0]

  @property
  def edge_count(self) -> int:
    """Return the total number of edges of all flow types."""
    return sum(len(adjacency_list) for adjacency_list in self.adjacencies)

  @property
  def control_edge_count(self) -> int:
    return self.adjacencies[programl_pb2.Edge.CONTROL].shape[0]

  @property
  def data_edge_count(self) -> int:
    return self.adjacencies[programl_pb2.Edge.DATA].shape[0]

  @property
  def call_edge_count(self) -> int:
    return self.adjacencies[programl_pb2.Edge.CALL].shape[0]

  @property
  def edge_position_max(self) -> int:
    """Return the maximum edge position."""
    return max(
      [
        position_list.max() if position_list.size else 0
        for position_list in self.edge_positions
      ]
    )

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
    if self.has_graph_x and self.is_disjoint_graph:
      return self.graph_x.shape[1]
    elif self.has_graph_x:
      return self.graph_x.shape[0]
    else:
      return 0

  @property
  def graph_y_dimensionality(self) -> int:
    """Return the dimensionality of graph labels."""
    if self.has_graph_y and self.is_disjoint_graph:
      return self.graph_y.shape[1]
    elif self.has_graph_y:
      return self.graph_y.shape[0]
    else:
      return 0

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
    adjacencies: List[List[Tuple[int, int]]] = [
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
      adjacencies[data["flow"]].append((src, dst))
      edge_positions[data["flow"]].append(data["position"])

    # Convert the edge lists to numpy arrays.
    # Shape (edge_count, 2):
    adjacencies = np.array(
      [
        np.array(adjacency_list, dtype=np.int32)
        for adjacency_list in adjacencies
      ]
    )
    # Shape (edge_count, 1):
    edge_positions = np.array(
      [
        np.array(edge_position, dtype=np.int32)
        for edge_position in edge_positions
      ]
    )

    # Set the node features.
    node_x = [None] * g.number_of_nodes()
    for node, x in g.nodes(data="x"):
      node_x[node] = np.array(x, dtype=np.int32)
    # Shape (node_count, node_x_dimensionality):
    node_x = np.vstack(node_x)

    # Set the node labels.
    node_targets = [None] * g.number_of_nodes()
    node_y = None
    for node, y in g.nodes(data="y"):
      # Node labels are optional. If there are no labels, break.
      if not y:
        break
      node_targets[node] = y
    else:
      # Shape (node_count, node_y_dimensionality):
      node_y = np.vstack(node_targets).astype(np.int32)

    # Get the optional graph-level features and labels.
    graph_x = np.array(g.graph["x"], dtype=np.int32) if g.graph["x"] else None
    graph_y = np.array(g.graph["y"], dtype=np.int32) if g.graph["y"] else None

    # End of specialised tuple representation.

    return GraphTuple(
      adjacencies=adjacencies,
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
      zip(self.adjacencies, self.edge_positions)
    ):
      for (src, dst), position in zip(adjacency_list, position_list):
        g.add_edge(src, dst, key=flow, flow=flow, position=position)

    for i, x in enumerate(self.node_x):
      g.nodes[i]["x"] = x.tolist()

    if self.has_node_y:
      for i, y in enumerate(self.node_y):
        g.nodes[i]["y"] = y.tolist()
    else:
      for node, data in g.nodes(data=True):
        data["y"] = []

    g.graph["x"] = self.graph_x.tolist() if self.has_graph_x else []
    g.graph["y"] = self.graph_y.tolist() if self.has_graph_y else []

    return g
