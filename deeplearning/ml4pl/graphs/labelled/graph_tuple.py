"""The module implements conversion of graphs to tuples of arrays."""
import pathlib
import pickle
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

  ##############################################################################
  # Disjoint graph properties
  ##############################################################################

  # The number of disconnected graphs in the tuple.
  disjoint_graph_count: int = 1

  # A list of integers which segment the nodes by graph. E.g. with a GraphTuple
  # of two distinct graphs, both with three nodes, nodes_list will be
  # [0, 0, 0, 1, 1, 1].
  # Shape (node_count), dtype int32:
  disjoint_nodes_list: np.array = None

  @property
  def is_disjoint_graph(self) -> bool:
    """Return whether the graph tuple is disjoint."""
    return self.disjoint_nodes_list is not None

  ##############################################################################
  # Properties
  ##############################################################################

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
      position_list.max() if position_list.size else 0
      for position_list in self.edge_positions
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
    # Disjoint graphs have a list of feature vectors, one for each graph.
    if self.has_graph_x and self.is_disjoint_graph:
      return self.graph_x.shape[1]
    elif self.has_graph_x:
      return self.graph_x.shape[0]
    else:
      return 0

  @property
  def graph_y_dimensionality(self) -> int:
    """Return the dimensionality of graph labels."""
    # Disjoint graphs have a list of label vectors, one for each graph.
    if self.has_graph_y and self.is_disjoint_graph:
      return self.graph_y.shape[1]
    elif self.has_graph_y:
      return self.graph_y.shape[0]
    else:
      return 0

  def SetLabels(
    self, node_y=None, graph_y=None, copy: bool = True
  ) -> "GraphTuple":
    """Create a graph tuple with new labels.

    Args:
      node_y: New node labels.
      graph_y: New graph labels.
      copy: If true, copy the underlying numpy arrays from the existing graph
        tuple. Else, the new graph tuple references the old one, meaning that
        modifications to either will be reflected on both.

    Returns:
      A graph tuple instance.
    """
    # Check that the user provided new labels.
    if node_y is None and graph_y is None:
      raise ValueError("Must set node_y or graph_y")

    # Check that the new labels have the correct sizes.
    if node_y is not None and node_y.shape != self.node_y.shape:
      raise TypeError(
        f"New node_y shape {node_y.shape} does not match "
        f"existing shape {self.node_y.shape}"
      )
    if graph_y is not None and graph_y.shape != self.graph_y.shape:
      raise TypeError(
        f"New graph_y shape {graph_y.shape} does not match "
        f"existing shape {self.graph_y.shape}"
      )

    # Determine whether to copy the underlying numpy arrays or creat new
    # references to them.
    new = np.copy if copy else lambda x: x

    return GraphTuple(
      adjacencies=new(self.adjacencies),
      edge_positions=new(self.edge_positions),
      node_x=new(self.node_x),
      node_y=node_y,
      graph_x=new(self.graph_x),
      graph_y=graph_y,
    )

  ##############################################################################
  # Factory methods
  ##############################################################################

  @staticmethod
  def FromFile(path: pathlib.Path):
    """Construct a graph tuple from a file generated by ToFile().

    Args:
      path: The path of the file to read.

    Returns:
      A GraphTuple instance.
    """
    with open(path, "rb") as f:
      return pickle.load(f)

  @classmethod
  def CreateFromNetworkX(cls, g: nx.MultiDiGraph) -> "GraphTuple":
    """Construct a graph tuple from a networkx graph.

    Args:
      g: The graph to convert to a graph. See
        deeplearning.ml4pl.graphs.programl.ProgramGraphToNetworkX() for a
        description of the networkx format.

    Returns:
      A GraphTuple instance.
    """
    # Create an adjacency list for each edge type.
    # {control, data, call} types.
    adjacencies: List[List[Tuple[int, int]]] = [
      [],
      [],
      [],
    ]
    # Create an edge position list for each edge type.
    # {control, data, call} types.
    edge_positions: List[List[int]] = [
      [],
      [],
      [],
    ]

    # Build the adjacency and positions lists.
    for src, dst, data in g.edges(data=True):
      adjacencies[data["flow"]].append((src, dst))
      edge_positions[data["flow"]].append(data["position"])

    # Convert the edge lists to numpy arrays.
    # Shape (edge_flow_count, edge_count, 2):
    for i in range(len(adjacencies)):
      if len(adjacencies[i]):
        adjacencies[i] = np.array(adjacencies[i], dtype=np.int32)
      else:
        adjacencies[i] = np.zeros((0, 2), dtype=np.int32)

    # Shape (edge_flow_count, edge_count):
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

    return GraphTuple(
      adjacencies=np.array(adjacencies),
      edge_positions=edge_positions,
      node_x=node_x,
      node_y=node_y,
      graph_x=graph_x,
      graph_y=graph_y,
    )

  @classmethod
  def FromGraphTuples(
    cls, graph_tuples: Iterable["GraphTuple"]
  ) -> "GraphTuple":
    """Construct a graph tuple by merging multiple tuples into a single
    disjoint graph.

    Args:
      graph_tuples: The tuples to combine.

    Returns:
      A GraphTuple instance.
    """
    adjacencies: List[List[Tuple[int, int]]] = [[], [], []]
    edge_positions: List[List[int]] = [[], [], []]
    disjoint_nodes_list: List[List[int]] = []

    node_x: List[List[int]] = []
    node_y: List[List[int]] = []
    graph_x: List[List[int]] = []
    graph_y: List[List[int]] = []

    disjoint_graph_count = 0
    node_count = 0

    # Iterate over each graph, merging them.
    for graph in graph_tuples:
      disjoint_nodes_list.append(
        np.full(
          shape=[graph.node_count],
          fill_value=disjoint_graph_count,
          dtype=np.int32,
        )
      )

      for edge_flow, (adjacency_list, position_list) in enumerate(
        zip(graph.adjacencies, graph.edge_positions)
      ):
        if adjacency_list.size:
          # Offset the adjacency list node indices.
          offset = np.array((node_count, node_count), dtype=np.int32)
          adjacencies[edge_flow].append(adjacency_list + offset)
          edge_positions[edge_flow].append(position_list)

      # Add features and labels.

      # Shape (graph.node_count, node_x_dimensionality):
      node_x.extend(graph.node_x)

      if graph.has_node_y:
        # Shape (graph.node_count, node_y_dimensionality):
        node_y.extend(graph.node_y)

      if graph.has_graph_x:
        graph_x.append(graph.graph_x)

      if graph.has_graph_y:
        graph_y.append(graph.graph_y)

      # Update the counters.
      disjoint_graph_count += 1
      node_count += graph.node_count

    # Concatenate and convert lists to numpy arrays.
    for i in range(len(adjacencies)):
      if len(adjacencies[i]):
        adjacencies[i] = np.concatenate(adjacencies[i])
      else:
        adjacencies[i] = np.zeros((0, 2), dtype=np.int32)

      if len(edge_positions[i]):
        edge_positions[i] = np.concatenate(edge_positions[i])
      else:
        edge_positions[i] = np.array([], dtype=np.int32)

    return cls(
      adjacencies=np.array(adjacencies),
      edge_positions=np.array(edge_positions),
      node_x=np.array(node_x, dtype=np.int32),
      node_y=np.array(node_y, dtype=np.int32) if node_y else None,
      graph_x=np.array(graph_x, dtype=np.int32) if graph_x else None,
      graph_y=np.array(graph_y, dtype=np.int32) if graph_y else None,
      disjoint_graph_count=disjoint_graph_count,
      disjoint_nodes_list=np.concatenate(disjoint_nodes_list),
    )

  ##############################################################################
  # Convertor methods
  ##############################################################################

  def ToFile(self, path: pathlib.Path) -> None:
    """Dump the pickled graph tuple to file.

    This is lossy, as the ir_id column is not dumped.

    Args:
      path: The path of the graph tuple to write.
    """
    with open(path, "wb") as f:
      pickle.dump(self, f)

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

  def ToGraphTuples(self) -> Iterable["GraphTuple"]:
    """Perform the inverse transformation from disjoint graph to a list of
    individual graph tuples.

    Returns:
      An iterable of graph instances.

    Raises:
      ValueError: If the graph tuple is not disjoint, or if the graph tuple
        is invalid.
    """
    if not self.is_disjoint_graph:
      raise ValueError("ToGraphTuples() called on non-disjoint graph tuple")

    # Split the list of node indices into individual lists for each graph.
    graph_split_indices = (
      np.where(self.disjoint_nodes_list[:-1] != self.disjoint_nodes_list[1:])[0]
      + 1
    )
    if len(graph_split_indices) + 1 != self.disjoint_graph_count:
      raise ValueError(
        f"Graph tuple contains {self.disjoint_graph_count} disjoint "
        f"graphs but only found {len(graph_split_indices) + 1} "
        "splits"
      )
    # Shape (disjoint_graph_count, ?), dtype=np.int32.
    nodes_per_graph = np.split(self.disjoint_nodes_list, graph_split_indices)

    # The starting index of the node.
    node_offset = 0

    # Iterate over the per-graph list of nodes.
    for current_graph, graph_node_count in enumerate(
      [len(n) for n in nodes_per_graph]
    ):

      # Per-flow edge attributes.
      adjacencies = [None, None, None]
      edge_positions = [None, None, None]

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
            srcs >= node_offset, srcs < node_offset + graph_node_count
          )
        )

        adjacencies[edge_flow] = adjacency_list[edge_indicies]
        edge_positions[edge_flow] = position_list[edge_indicies]

        # Negate the positive offset into the adjacency lists.
        offset = np.array((node_offset, node_offset), dtype=np.int32)
        adjacencies[edge_flow] -= offset

      # Read node features.
      current_node_x = self.node_x[node_offset : node_offset + graph_node_count]
      if len(current_node_x) != graph_node_count:
        raise ValueError(
          f"Graph has {len(current_node_x)} nodes but expected "
          f"{graph_node_count}"
        )

      # Read optional node labels.
      if self.has_node_y:
        current_node_y = self.node_y[
          node_offset : node_offset + graph_node_count
        ]
        if len(current_node_y) != graph_node_count:
          raise ValueError(
            f"Graph has {len(current_node_y)} nodes but expected "
            f"{graph_node_count}"
          )
      else:
        current_node_y = None

      yield GraphTuple(
        adjacencies=np.array(adjacencies),
        edge_positions=np.array(edge_positions),
        node_x=current_node_x,
        node_y=current_node_y,
        graph_x=self.graph_x[current_graph] if self.has_graph_x else None,
        graph_y=self.graph_y[current_graph] if self.has_graph_y else None,
      )

      node_offset += graph_node_count
