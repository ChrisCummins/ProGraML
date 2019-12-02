"""A module for batching graph tuples."""
from typing import Iterable
from typing import List
from typing import Optional

from deeplearning.ml4pl.graphs.labelled import graph_tuple
from labm8.py import app
from labm8.py import progress

FLAGS = app.FLAGS


class GraphBatcher(object):
  """A graph batcher collects and combines input graphs into disjoint graph
  tuples.
  """

  def __init__(
    self,
    graphs: Iterable[graph_tuple.GraphTuple],
    max_graph_count: int = 0,
    exact_graph_count: int = 0,
    max_node_count: int = 0,
    ctx: progress.ProgressContext = progress.NullContext,
  ):
    """Constructor.

    Args:
      graphs: An iterator of graphs to batch.
      max_graph_count: The maximum number of graphs to include in a batch. If
        zero, the number of graphs is not limited.
      exact_graph_count: The exact number of graphs to include in a batch.
        Unlike max_graph_count, a batch will only be returned if it has this
        exact size. If zero, no exact graph count is used.
      max_node_count: The maximum number of graphs to include in a batch, across
        all graphs. If zero, the number of nodes is not limited.
      ctx: A progress context.
    """
    self.graphs = graphs
    self.max_graph_count = max_graph_count
    self.exact_graph_count = exact_graph_count
    self.max_node_count = max_node_count
    self.ctx = ctx

    # Hold onto the last read graph so that if we don't decide to include it in
    # a batch we may still include it in subsequent batches.
    self.last_graph: Optional[graph_tuple.GraphTuple] = None

  def __iter__(self):
    return self

  def __next__(self) -> Optional[graph_tuple.GraphTuple]:
    """Construct a graph batch.

    Returns:
      A batch of graphs as a disjointed graph tuple. If there are no graphs to
      batch then None is returned.
    """
    with self.ctx.Profile(
      2, lambda t: f"Constructed a batch of {len(graphs)} graphs"
    ):
      graphs: List[graph_tuple.GraphTuple] = []
      node_count = 0

      while True:
        if self.last_graph:
          # Pop the last visited graph.
          graph = self.last_graph
        else:
          # Read a new graph.
          graph = self._ReadNextGraph()
          if graph is None:
            break

        self.last_graph = graph

        # Check if the node count fits in the batch.
        if (
          self.max_node_count
          and node_count + graph.node_count > self.max_node_count
        ):
          break

        # Check if we have enough graphs.
        if self.max_graph_count and len(graphs) >= self.max_graph_count:
          break

        # Check if we have the exact number of graphs.
        if self.exact_graph_count and len(graphs) == self.exact_graph_count:
          break

        graphs.append(graph)
        node_count += graph.node_count
        # Pop the last visited graph.
        self.last_graph = None

      if self.exact_graph_count and len(graphs) != self.exact_graph_count:
        # We require batches of an exact size, but we don't have that many
        # graphs to return.
        raise StopIteration
      if graphs:
        # We have graphs to batch.
        return graph_tuple.GraphTuple.FromGraphTuples(graphs)
      else:
        raise StopIteration

  def _ReadNextGraph(self,) -> Optional[graph_tuple.GraphTuple]:
    """Read the next graph from graph iterable, or None if no more graphs.

    Returns:
      A graph, or None.

    Raises:
      ValueError: If the graph is larger than permitted by the batch size.
    """
    try:
      graph = next(self.graphs)
      if self.max_node_count and graph.node_count > self.max_node_count:
        raise ValueError(
          f"Graph with node_count={graph.node_count} is larger "
          f"than max_node_count={self.max_node_count}"
        )
      return graph
    except StopIteration:  # We have run out of graphs.
      return None
