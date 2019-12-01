"""A module for batching graph tuples."""
import time
import typing

import networkx as nx
import numpy as np
import sqlalchemy as sql

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs import graph_database_stats as graph_stats
from deeplearning.ml4pl.graphs.labelled import (
  graph_database_reader as graph_readers,
)
from deeplearning.ml4pl.models import log_database
from labm8.py import app
from labm8.py import humanize

FLAGS = app.FLAGS


class GraphBatchOptions(typing.NamedTuple):
  """A tuple of options for constructing graph batches."""

  # The maximum number of graphs to include in a batch. If zero, the number of
  # graphs is not limited.
  max_graphs: int = 0

  # The maximum number of graphs to include in batch, across all graphs. If
  # zero, the number of nodes is not limited.
  max_nodes: int = 0

  # Filter graphs to these "GraphMeta.group" columns. None value means no
  # filter.
  groups: typing.List[str] = None

  # Only include graphs which can be computed in less than or equal to this
  # number of data flow steps. A value of zero means no limit.
  data_flow_max_steps_required: int = 0

  def ShouldAddToBatch(
    self, graph: graph_database.GraphMeta, log: log_database.BatchLogMeta
  ) -> bool:
    """Return whether the given graph should be added to the batch.

    Args:
      graph: A graph.
      log: The current state of the batch. The node_count and graph_count
        attributes are accessed.
    """
    # Check if we already have enough graphs.
    if self.max_nodes and log.node_count + graph.node_count >= self.max_nodes:
      return False

    # Check if we already have enough graphs.
    if self.max_graphs and log.graph_count >= self.max_graphs:
      return False

    # De-serialize pickled data in database and process.
    if graph.graph is None:
      app.Error("Failed to read data on graph %s", graph.id)
      return False

    return True

  def GetDatabaseQueryFilters(self) -> typing.List[typing.Callable[[], bool]]:
    """Convert the given batcher options to a set of SQL query filters.

    Returns:
      A list of lambdas, where each lambda when called provides a sqlalchemy
      filter.
    """
    filters = []
    if self.max_nodes:
      filters.append(
        lambda: graph_database.GraphMeta.node_count <= self.max_nodes
      )
    if self.groups:
      filters.append(lambda: graph_database.GraphMeta.group.in_(self.groups))
    if self.data_flow_max_steps_required:
      filters.append(
        lambda: (
          graph_database.GraphMeta.data_flow_max_steps_required
          <= self.data_flow_max_steps_required
        )
      )
    return filters


class GraphBatch(typing.NamedTuple):
  """An extension to GraphTuple to support multiple disconnected graphs."""

  # A list of adjacency lists, one for each edge_type, where an entry in an
  # adjacency list is a <src,dst> tuple of node indices.
  adjacency_lists: np.array  # Shape [edge_type_count, ?, 2], dtype int32

  # A list of edge positions, one for each edge type. An edge position is an
  # integer in the range 0 <= x < max_edge_position.
  edge_positions: np.array  # Shape [edge_type_count, ?], dtype int32

  # A list of incoming edge count dicts, one for each edge_type. Use
  # IncomingEdgeCountsToDense() to convert this to a dense representation.
  incoming_edge_counts: np.array  # Shape [edge_type_count, ?])

  # A matrix of indices into the node features table. Each row is a node,
  # and each column is an embedding index for that node. Multiple embedding
  # indices read from multiple embedding tables.
  # Shape [node_count,node_embeddings_count], dtype int32
  node_x_indices: np.array

  # A list of shape [node_count] which segments the nodes by graph.
  graph_nodes_list: np.array

  # The number of disconnected graphs in the batch.
  graph_count: int

  # A batch log.
  log: log_database.BatchLogMeta

  # (optional) A list of node labels arrays.
  # Shape [node_count, node_label_dimensionality]
  node_y: typing.Optional[np.array] = None

  # (optional) A list of graph features arrays.
  # Shape [graph_count,graph_feature_dimensionality]
  graph_x: typing.Optional[np.array] = None

  # (optional) A vector of graph labels arrays.
  # Shape [graph_count,graph_label_dimensionality]
  graph_y: typing.Optional[np.array] = None

  @property
  def has_node_y(self) -> bool:
    """Return whether graph tuple has node labels."""
    return self.node_y is not None

  @property
  def has_graph_x(self) -> bool:
    """Return whether graph tuple has graph features."""
    return self.graph_x is not None

  @property
  def has_edge_positions(self) -> bool:
    """Return whether graph tuple has graph features."""
    return self.edge_positions is not None

  @property
  def has_graph_y(self) -> bool:
    """Return whether graph tuple has graph labels."""
    return self.graph_y is not None

  @property
  def node_count(self) -> int:
    """Return the number of nodes in the graph."""
    return len(self.node_x_indices)

  @property
  def edge_type_count(self) -> int:
    """Return the number of edge types."""
    return len(self.adjacency_lists)

  @property
  def dense_incoming_edge_counts(self) -> np.array:
    """Return counters for incoming edges as a dense array."""
    dense = np.zeros((self.node_count, self.edge_type_count))
    for edge_type, incoming_edge_dict in enumerate(self.incoming_edge_counts):
      for node_id, edge_count in incoming_edge_dict.items():
        dense[node_id, edge_type] = edge_count
    return dense

  @staticmethod
  def NextGraph(
    graphs: typing.Iterable[graph_database.GraphMeta],
    options: GraphBatchOptions,
  ) -> typing.Optional[graph_database.GraphMeta]:
    """Read the next graph from graph iterable, or None if no more graphs.

    Args:
      graphs: An iterator over graphs.

    Returns:
      A graph, or None.

    Raises:
      ValueError: If the graph is larger than permitted by the batch size.
    """
    try:
      graph = next(graphs)
      if options.max_nodes and graph.node_count > options.max_nodes:
        raise ValueError(
          f"Graph `{graph.id}` with {graph.node_count} is larger "
          f"than batch size {options.max_nodes}"
        )
      return graph
    except StopIteration:  # We have run out of graphs.
      return None

  @classmethod
  def CreateFromGraphMetas(
    cls,
    graphs: typing.Iterable[graph_database.GraphMeta],
    stats: graph_stats.GraphTupleDatabaseStats,
    options: GraphBatchOptions,
  ) -> typing.Optional["GraphBatch"]:
    """Construct a graph batch.

    Args:
      graphs: An iterator of graphs to construct the batch from.
      stats: A database stats instance.
      options: The options for the graph batch.

    Returns:
      The graph batch. If there are no graphs to batch then None is returned.
    """
    graph = cls.NextGraph(graphs, options)
    if not graph:  # We have run out of graphs.
      return None

    data_flow_max_steps_required = graph.data_flow_max_steps_required
    max_edge_count = graph.edge_count

    edge_type_count = stats.edge_type_count

    # The batch log contains properties describing the batch (such as the list
    # of graphs used).
    log = log_database.BatchLogMeta(
      graph_count=0, node_count=0, group=graph.group
    )

    graph_ids: typing.List[int] = []
    bytecode_ids: typing.List[int] = []
    adjacency_lists = [[] for _ in range(edge_type_count)]
    position_lists = [[] for _ in range(edge_type_count)]
    incoming_edge_counts = []
    graph_nodes_list = []
    node_x_indices = []

    has_node_labels = stats.node_labels_dimensionality > 0
    has_graph_features = stats.graph_features_dimensionality > 0
    has_graph_labels = stats.graph_labels_dimensionality > 0

    if has_node_labels:
      node_y = []
    else:
      node_y = None

    if has_graph_features:
      graph_x = []
    else:
      graph_x = None

    if has_graph_labels:
      graph_y = []
    else:
      graph_y = None

    # Pack until we cannot fit more graphs in the batch.
    while graph and options.ShouldAddToBatch(graph, log):
      graph_tuple = graph.data

      graph_ids.append(graph.id)
      bytecode_ids.append(graph.bytecode_id)

      graph_nodes_list.append(
        np.full(
          shape=[graph.node_count], fill_value=log.graph_count, dtype=np.int32,
        )
      )

      # Offset the adjacency list node indices.
      for edge_type, (adjacency_list, position_list) in enumerate(
        zip(graph_tuple.adjacency_lists, graph_tuple.edge_positions)
      ):
        if adjacency_list.size:
          offset = np.array((log.node_count, log.node_count), dtype=np.int32)
          adjacency_lists[edge_type].append(adjacency_list + offset)
          position_lists[edge_type].append(position_list)

      incoming_edge_counts.append(graph_tuple.dense_incoming_edge_counts)

      # Add features and labels.

      # Shape: [graph.node_count, node_embbeddings_count]
      node_x_indices.extend(graph_tuple.node_x_indices)

      if has_node_labels:
        # Shape: [graph_tuple.node_count, node_labels_dimensionality]
        node_y.extend(graph_tuple.node_y)

      if has_graph_features:
        graph_x.append(graph_tuple.graph_x)

      if has_graph_labels:
        graph_y.append(graph_tuple.graph_y)

      # Update batch counters.
      log.graph_count += 1
      log.node_count += graph.node_count

      graph = cls.NextGraph(graphs, options)
      if not graph:  # We have run out of graphs.
        break
      data_flow_max_steps_required = max(
        data_flow_max_steps_required, graph.data_flow_max_steps_required
      )
      max_edge_count = max(max_edge_count, graph.edge_count)

    # Empty batch
    if not len(incoming_edge_counts):
      return None

    # Return None as well if we are in the fixed number of graphs mode and
    # the batch has less than that number of graphs.
    if options.max_graphs and log.graph_count < options.max_graphs:
      return None

    # Concatenate and convert lists to numpy arrays.

    for i in range(stats.edge_type_count):
      if len(adjacency_lists[i]):
        adjacency_lists[i] = np.concatenate(adjacency_lists[i])
      else:
        adjacency_lists[i] = np.zeros((0, 2), dtype=np.int32)

      if len(position_lists[i]):
        position_lists[i] = np.concatenate(position_lists[i])
      else:
        position_lists[i] = np.array([], dtype=np.int32)

    incoming_edge_counts = np.concatenate(incoming_edge_counts, axis=0)
    graph_nodes_list = np.concatenate(graph_nodes_list)
    node_x_indices = np.array(node_x_indices, dtype=np.int32)
    if has_node_labels:
      node_y = np.array(node_y)
    if has_graph_features:
      graph_x = np.array(graph_x)
    if has_graph_labels:
      graph_y = np.array(graph_y)

    # Record the graphs that we used in this batch to an unmapped property.
    # TODO(github.com/ChrisCummins/ProGraML/issues/17): Setting an attribute on
    # a mapped object at run time like this is shitty. Rethink.
    log._transient_data = {
      "graph_indices": graph_ids,
      "bytecode_ids": bytecode_ids,
      "data_flow_max_steps_required": data_flow_max_steps_required,
      "max_edge_count": max_edge_count,
    }

    return cls(
      adjacency_lists=adjacency_lists,
      edge_positions=position_lists,
      incoming_edge_counts=incoming_edge_counts,
      node_x_indices=node_x_indices,
      node_y=node_y,
      graph_x=graph_x,
      graph_y=graph_y,
      graph_nodes_list=graph_nodes_list,
      graph_count=log.graph_count,
      log=log,
    )

  def ToNetworkXGraphs(self) -> typing.Iterable[nx.MultiDiGraph]:
    """Perform the inverse transformation from batch_dict to list of graphs.

    Args:
      batch_dict: The batch dictionary to construct graphs from.

    Returns:
      A generator of graph instances.
    """
    node_count = 0
    for graph_count in range(self.graph_count):
      g = nx.MultiDiGraph()
      # Mask the nodes from the node list to determine how many nodes are in
      # the graph.
      nodes = self.graph_nodes_list[self.graph_nodes_list == graph_count]
      graph_node_count = len(nodes)

      for edge_type in range(len(self.adjacency_lists)):
        adjacency_list = self.adjacency_lists[edge_type]
        position_list = self.edge_positions[edge_type]

        # No edges of this type.
        if not adjacency_list.size:
          continue

        # The adjacency list contains the adjacencies for all graphs. Determine
        # those that are in this graph by selecting only those with a source
        # node in the list of this graph's nodes.
        srcs = adjacency_list[:, 0]
        adjacency_list_indices = np.where(
          np.logical_and(
            srcs >= node_count, srcs < node_count + graph_node_count
          )
        )
        adjacency_list = adjacency_list[adjacency_list_indices]
        position_list = position_list[adjacency_list_indices]

        # Negate the positive offset into adjacency lists.
        offset = np.array((node_count, node_count), dtype=np.int32)
        adjacency_list -= offset

        # Add the edges to the graph.
        for (src, dst), position in zip(adjacency_list, position_list):
          g.add_edge(src, dst, flow=edge_type, position=position)

      node_x_embedding_indices = self.node_x_indices[
        node_count : node_count + graph_node_count
      ]
      if len(node_x_embedding_indices) != g.number_of_nodes():
        raise ValueError(
          f"Graph has {g.number_of_nodes()} nodes but "
          f"expected {len(node_x_embedding_indices)}"
        )
      for i, x in enumerate(node_x_embedding_indices):
        g.nodes[i]["x"] = x

      if self.has_node_y:
        node_y = self.node_y[node_count : node_count + graph_node_count]
        if len(node_y) != g.number_of_nodes():
          raise ValueError(
            f"Graph has {g.number_of_nodes()} nodes but "
            f"expected {len(node_y)}"
          )
        for i, values in enumerate(node_y):
          g.nodes[i]["y"] = values

      if self.has_graph_x:
        g.x = self.graph_x[graph_count]

      if self.has_graph_y:
        g.y = self.graph_y[graph_count]

      yield g

      node_count += graph_node_count


class GraphBatcher(object):
  """A generalised graph batcher which flattens adjacency matrices into a single
  adjacency matrix with multiple disconnected components. Supports all feature
  and label types of graph tuples.
  """

  def __init__(self, db: graph_database.Database):
    """Constructor.

    Args:
      db: The database to read and batch graphs from.
      message_passing_step_count: The number of message passing steps in the
        model that this batcher is feeding. This value is used when the
        --{limit,match}_data_flow_max_steps_required_to_message_passing_steps
        flags are set to limit the graphs which are used to construct batches.
    """
    self.db = db
    self.stats = graph_stats.GraphTupleDatabaseStats(self.db)
    app.Log(1, "%s", self.stats)

  def GetGraphsInGroupCount(self, groups: typing.List[str]) -> int:
    """Get the number of graphs in the given group(s)."""
    with self.db.Session() as s:
      q = s.query(sql.func.count(graph_database.GraphMeta.id))
      q = q.filter(graph_database.GraphMeta.group.in_(groups))
      q = q.filter(graph_database.GraphMeta.node_count > 1)
      num_rows = q.one()[0]
    return num_rows

  def MakeGraphBatchIterator(
    self,
    options: GraphBatchOptions,
    # TODO(cec): This duplicates the logic of the GraphTuplesOptions field.
    # Consolidate these.
    max_instance_count: int = 0,
    print_context: typing.Any = None,
  ) -> typing.Iterable[GraphBatch]:
    """Make a batch iterator over the given group.

    Args:
      max_instance_count: Limit the total number of graphs returned across all
        graph batches. A value of zero means no limit.

    Returns:
      An iterator over graph batch tuples.
    """
    filters = options.GetDatabaseQueryFilters()

    if FLAGS.graph_reader_order == "in_order":
      order = graph_readers.BufferedGraphReaderOrder.IN_ORDER
    elif FLAGS.graph_reader_order == "global_random":
      order = graph_readers.BufferedGraphReaderOrder.GLOBAL_RANDOM
    elif FLAGS.graph_reader_order == "batch_random":
      order = graph_readers.BufferedGraphReaderOrder.BATCH_RANDOM
    else:
      raise app.UsageError(
        f"Unknown --graph_reader_order=`{FLAGS.graph_reader_order}`."
      )

    graph_reader = graph_readers.BufferedGraphReader(
      self.db,
      filters=filters,
      order=order,
      eager_graph_loading=True,
      # Magic constant to try and get a reasonable balance between memory
      # requirements and database round trips.
      buffer_size=FLAGS.graph_reader_buffer_size,
      limit=max_instance_count,
      print_context=print_context,
    )

    # Batch creation outer-loop.
    while True:
      start_time = time.time()
      batch = GraphBatch.CreateFromGraphMetas(graph_reader, self.stats, options)
      if batch:
        elapsed_time = time.time() - start_time
        app.Log(
          5,
          "Created batch of %s graphs (%s nodes) in %s " "(%s graphs/sec)",
          humanize.Commas(batch.log.graph_count),
          humanize.Commas(batch.log.node_count),
          humanize.Duration(elapsed_time),
          humanize.Commas(batch.log.graph_count / elapsed_time),
          print_context=print_context,
        )
        assert batch.log.graph_count > 0
        yield batch
      else:
        return
