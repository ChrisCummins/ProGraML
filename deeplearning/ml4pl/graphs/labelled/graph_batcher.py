"""A module for batching graph tuples.

This can be executed as a script to dump graph tuples to file, example:

    $ bazel run //deeplearning/ml4pl/graphs/labelled:graph_batcher -- \
          --graph_db='sqlite:////tmp/graphs.db' \
          --graph_batch_outdir=/tmp/graphs \
          --graph_batch_size=10 \
          --graph_reader_limit=1000 \
          --vmodule='*'=3
"""
import pathlib
from typing import Iterable
from typing import List
from typing import Optional

from deeplearning.ml4pl.graphs.labelled import graph_database_reader
from deeplearning.ml4pl.graphs.labelled import graph_tuple
from labm8.py import app
from labm8.py import progress

FLAGS = app.FLAGS


app.DEFINE_integer(
  "graph_batch_size",
  0,
  "The maximum number of graphs to include in a batch of graphs.",
)
app.DEFINE_integer(
  "graph_batch_exact_size",
  0,
  "The number of graphs to include in a batch of graphs.",
)
app.DEFINE_integer(
  "graph_batch_node_count",
  0,
  "The maximum number of nodes to include in a batch of graphs.",
)
app.DEFINE_string(
  "max_node_count_limit_handler",
  "error",
  "This determines what happens when we try to batch a graph which exceeds "
  "--graph_batch_node_count. Possible values are: skip (skip the graph but "
  "print a warning), error (raise an error), or include (include the graph in "
  "the batch anyway). This flag has no effect when --graph_batch_node_count "
  "is not set.",
)
app.DEFINE_output_path(
  "graph_batch_outdir",
  None,
  "When //deeplearning/ml4pl/graphs/labelled:graph_batcher is executed as a "
  "script, this determines the directory to write pickled graph tuples to.",
)


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
    max_node_count_limit_handler: str = "error",
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
      max_node_count_limit_handler: Determines what happens when we try to
        batch a graph which exceeds max_node_count. Possible values are: skip
        (skip the graph but print a warning), error (raise an error), or
        include (include the graph in the batch anyway). Has no effect when
        max_node_count is not set.
      ctx: A progress context.
    """
    self.graphs = graphs
    self.max_graph_count = max_graph_count
    self.exact_graph_count = exact_graph_count
    self.max_node_count = max_node_count
    self.max_node_count_limit_handler = max_node_count_limit_handler
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
        # Determine the behaviour when we find a graph that is larger than
        # the graph node limit.
        msg = (
          f"Graph with node_count={graph.node_count} is larger "
          f"than max_node_count={self.max_node_count}"
        )
        if self.max_node_count_limit_handler == "skip":
          self.ctx.Warning("%s, skipping it", msg)
          return None
        if self.max_node_count_limit_handler == "error":
          raise ValueError(msg)
        elif self.max_node_count_limit_handler == "include":
          return graph
        else:
          raise app.UsageError(
            "Unknown max_node_count_limit_handler: "
            f"{self.max_node_count_limit_handler}"
          )
      return graph
    except StopIteration:  # We have run out of graphs.
      return None

  @classmethod
  def CreateFromFlags(
    cls,
    graphs: Iterable[graph_tuple.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ):
    return cls(
      graphs,
      max_graph_count=FLAGS.graph_batch_size,
      exact_graph_count=FLAGS.graph_batch_exact_size,
      max_node_count=FLAGS.graph_batch_node_count,
      max_node_count_limit_handler=FLAGS.max_node_count_limit_handler,
      ctx=ctx,
    )


class WriteGraphsToFile(progress.Progress):
  """Write graphs in a graph database to pickled files.

  This is for debugging.
  """

  def __init__(self, outdir: pathlib.Path):
    reader = graph_database_reader.BufferedGraphReader.CreateFromFlags()

    def GraphTupleIterator(graph_db_reader):
      """Make an iterator over graph tuples from an iterator over graph tuples
      in the database."""
      for graph in graph_db_reader:
        yield graph.tuple

    self.batcher = GraphBatcher.CreateFromFlags(GraphTupleIterator(reader))
    self.outdir = outdir
    self.outdir.mkdir(parents=True, exist_ok=True)
    super(WriteGraphsToFile, self).__init__("read_db", 0, reader.n)
    reader.ctx = self.ctx
    self.batcher.ctx = self.ctx

  def Run(self):
    """Read and write the graphs."""
    for i, graph_tuple in enumerate(self.batcher):
      self.ctx.i += graph_tuple.disjoint_graph_count
      path = self.outdir / f"batched_graph_tuple_{i:08}.pickle"
      graph_tuple.ToFile(path)


def Main():
  """Main entry point."""
  if not FLAGS.graph_batch_outdir:
    raise app.UsageError("--graph_batch_outdir must be set")
  progress.Run(WriteGraphsToFile(FLAGS.graph_batch_outdir))


if __name__ == "__main__":
  app.Run(Main)
