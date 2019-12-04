"""A module for reaching graphs from graph databases."""
import enum
import pathlib
import random
from typing import Callable
from typing import List
from typing import Optional

import sqlalchemy as sql

from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from labm8.py import app
from labm8.py import humanize
from labm8.py import progress

FLAGS = app.FLAGS

app.DEFINE_string(
  "graph_reader_order",
  "batch_random",
  "The order to read graphs from. See BufferedGraphReaderOrder. One of"
  "{in_order,global_random,batch_random}",
)
app.DEFINE_integer(
  "graph_reader_buffer_size_mb",
  64,
  "Tuning parameter. The size of the graph database reader buffer, in "
  "megabytes. A larger buffer means fewer costly SQL queries, but requires "
  "more memory to store the results.",
)
app.DEFINE_integer(
  "graph_reader_limit",
  None,
  "The maximum number of rows to read from the graph reader.",
)
app.DEFINE_output_path(
  "graph_reader_outdir",
  None,
  "When //deeplearning/ml4pl/graphs/labelled:graph_database_reader is executed "
  "as a script, this determines the directory to write pickled graph tuples to.",
)


class BufferedGraphReaderOrder(enum.Enum):
  """Determine the order to read graphs from a database."""

  # In-order reading always starts at the smallest graph ID and proceeds
  # incrementally through the graph table.
  IN_ORDER = 1
  # Global random order means that the graphs are selected form the entire graph
  # table using a random order.
  GLOBAL_RANDOM = 2
  # Batch random order means that the graph table is read in order, but once
  # each batch is read, the graphs are then shuffled locally. This aims to
  # strike a balance between the randomness of the graph order and the speed
  # at which graphs can be read, as reading from the table in order generally
  # requires fewer disk seeks than the global random option. Note that this ties
  # the randomness of the graphs to the size of the graph buffer. A larger graph
  # buffer will increase the randomness of the graphs. When buffer_size >= the
  # size of the graph table, this is equivalent to GLOBAL_RANDOM. When
  # buffer_size == 1, this is the same as IN_ORDER.
  BATCH_RANDOM = 3
  # Ordering by data flow max steps required is an optimization for when testing
  # on large graphs by enabling graph batches to be constructed with largely
  # similar sized graphs, minimizing the amount of redundant
  DATA_FLOW_STEPS = 4


class BufferedGraphReader(object):
  """An iterator over a database of graph tuples."""

  def __init__(
    self,
    db: graph_tuple_database.Database,
    buffer_size_mb: int,
    filters: Optional[List[Callable[[], bool]]] = None,
    order: BufferedGraphReaderOrder = BufferedGraphReaderOrder.IN_ORDER,
    eager_graph_loading: bool = True,
    limit: Optional[int] = None,
    ctx: progress.ProgressContext = progress.NullContext,
  ):
    """Constructor.

    Args:
      db: The database to iterate over.
      filters: An optional list of callbacks, where each callback returns a
        filter condition on the GraphTuple table.
      order: Determine the order to read graphs. See BufferedGraphReaderOrder.
      eager_graph_loading: If true, load the contents of the Graph table eagerly,
        preventing the need for subsequent SQL queries to access the graph data.
      buffer_size_mb: The number of graphs to query from the database at a time. A
        larger number reduces the number of queries, but increases the memory
        requirement.
      limit: Limit the total number of rows returned to this value.

    Raises:
      ValueError: If the query with the given filters returns no results.
    """
    self.db = db
    self.order = order
    self.max_buffer_size = buffer_size_mb * 1024 * 1024
    self.eager_graph_loading = eager_graph_loading
    self.filters = filters or []
    self.ctx = ctx

    # Graphs that fail during dataset generation are inserted as zero-node
    # entries. Ignore those.
    self.filters.append(lambda: graph_tuple_database.GraphTuple.node_count > 1)

    with ctx.Profile(
      2,
      lambda _: (
        f"Selected {humanize.Commas(self.n)} of "
        f"{humanize.Commas(self.total_graph_count)} graphs from database"
      ),
    ):
      with db.Session() as session:
        self.total_graph_count = session.query(
          sql.func.count(graph_tuple_database.GraphTuple.id)
        ).scalar()

        # Random ordering means that we can't use
        # labm8.py.sqlutil.OffsetLimitBatchedQuery() to read results as each
        # query will produce a different random order. Instead, first run a
        # query to read all of the IDs and the corresponding tuple sizes that
        # match the query, then iterate through the list of IDs.
        query = session.query(
          graph_tuple_database.GraphTuple.id,
          graph_tuple_database.GraphTuple.pickled_graph_tuple_size.label(
            "size"
          ),
        )

        # Apply the requested filters.
        for filter_cb in self.filters:
          query = query.filter(filter_cb())

        # If we are ordering with global random then we can scan through the
        # graph table using index range checks, so we need the IDs sorted.
        if order == BufferedGraphReaderOrder.DATA_FLOW_STEPS:
          self.ordered_ids = False
          query = query.order_by(
            graph_tuple_database.GraphTuple.data_flow_steps
          )
        elif order == BufferedGraphReaderOrder.GLOBAL_RANDOM:
          self.ordered_ids = False
          query = query.order_by(db.Random())
        else:
          self.ordered_ids = True
          query = query.order_by(graph_tuple_database.GraphTuple.id)

        # Read the full set of graph IDs and sizes.
        self.ids_and_sizes = [(row.id, row.size) for row in query.all()]

      if not self.ids_and_sizes:
        raise ValueError(
          f"Query on database `{db.url}` returned no results: `{query}`"
        )

      # When we are limiting the number of rows and not reading the table in
      # order, pick a random starting point in the list of IDs.
      if limit and order != BufferedGraphReaderOrder.IN_ORDER:
        batch_start = random.randint(
          0, max(len(self.ids_and_sizes) - limit - 1, 0)
        )
        self.ids_and_sizes = self.ids_and_sizes[
          batch_start : batch_start + limit
        ]
      elif limit:
        # If we are reading the table in order, we must still respect the limit
        # argument.
        self.ids_and_sizes = self.ids_and_sizes[:limit]

      self.i = 0
      self.n = len(self.ids_and_sizes)

      # The local buffer of graphs, and an index into that buffer.
      self.buffer = []
      self.buffer_i = 0

  def __iter__(self):
    return self

  def __next__(self):
    """Get the next graph."""
    if self.buffer_i < len(self.buffer):
      graph = self.buffer[self.buffer_i]
      self.buffer_i += 1
      return graph
    else:
      self.buffer = self.GetNextBuffer()
      self.buffer_i = 1
      return self.buffer[0]

  def GetNextBuffer(self) -> List[graph_tuple_database.GraphTuple]:
    """Fetch the next buffer of graphs from the database."""
    if self.i >= self.n:
      # We have run out of graphs to read.
      raise StopIteration

    # Select a batch of IDs to fetch from the database.
    end_i = self.i
    current_buffer_size = 0
    # Build up our ID list until we have the requested buffer size.
    while current_buffer_size < self.max_buffer_size:
      current_buffer_size += self.ids_and_sizes[end_i][1]
      end_i += 1
      if end_i >= len(self.ids_and_sizes):
        # We have reached the end of the graph list, fetch whatever remains
        # into the local buffer.
        break

    with self.db.Session() as session:
      # Build the query to fetch the graph data from the database.
      query = session.query(graph_tuple_database.GraphTuple)

      # Perform the joined eager load.
      if self.eager_graph_loading:
        query = query.options(
          sql.orm.joinedload(graph_tuple_database.GraphTuple.data)
        )

      # If we are reading the IDs in-order then we can use ID range checks to
      # return graphs. Else, we must perform ID value lookups.
      if self.ordered_ids:
        start_id = self.ids_and_sizes[self.i][0]
        end_id = self.ids_and_sizes[end_i - 1][0]
        query = query.filter(
          graph_tuple_database.GraphTuple.id >= start_id,
          graph_tuple_database.GraphTuple.id <= end_id,
        )
        # For index range comparisons we must repeat the same filters as when
        # initially reading the graph IDs.
        for filter in self.filters:
          query = query.filter(filter())
      else:
        batch_ids = [
          id_and_size[0] for id_and_size in self.ids_and_sizes[self.i : end_i]
        ]
        query = query.filter(graph_tuple_database.GraphTuple.id.in_(batch_ids))

      # Randomize the order of results for random orders.
      if (
        self.order == BufferedGraphReaderOrder.BATCH_RANDOM
        or self.order == BufferedGraphReaderOrder.GLOBAL_RANDOM
      ):
        query = query.order_by(self.db.Random())

      # Fetch the buffer data.
      with self.ctx.Profile(
        3,
        f"Read {humanize.BinaryPrefix(current_buffer_size, 'B')} "
        f"buffer of {end_i - self.i} graph tuples",
      ):
        buffer = query.all()
      if len(buffer) != end_i - self.i:
        raise OSError(
          f"Requested buffer of {end_i - self.i} graphs but received "
          f"{len(buffer)} graphs"
        )

      # Update the index into the list of graph IDs.
      self.i = end_i

      return buffer

  @classmethod
  def CreateFromFlags(
    cls,
    filters: Optional[List[Callable[[], bool]]] = None,
    eager_graph_loading: bool = True,
    limit: int = None,
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> "BufferedGraphReader":
    """Construct a database reader using global flag values for options.

    The relevant flags are:
        --graph_db: The database.
        --graph_reader_order: The order of graphs read.
        --graph_reader_buffer_size_mb: The size of the buffer.

    Ars:
      filters: A list of filter callbacks.
      eager_graph_loading: Whether to load eagerly load the graph data.
      limit: The maximum number of graphs read.

    Returns:
      A BufferedGraphReader instance.
    """
    graph_db = FLAGS.graph_db()

    if FLAGS.graph_reader_order == "in_order":
      order = BufferedGraphReaderOrder.IN_ORDER
    elif FLAGS.graph_reader_order == "global_random":
      order = BufferedGraphReaderOrder.GLOBAL_RANDOM
    elif FLAGS.graph_reader_order == "batch_random":
      order = BufferedGraphReaderOrder.BATCH_RANDOM
    elif FLAGS.graph_reader_order == "data_flow_steps":
      order = BufferedGraphReaderOrder.DATA_FLOW_STEPS
    else:
      raise app.UsageError(
        f"Unknown --graph_reader_order=`{FLAGS.graph_reader_order}`."
      )

    return cls(
      graph_db,
      filters=filters,
      order=order,
      eager_graph_loading=eager_graph_loading,
      buffer_size_mb=FLAGS.graph_reader_buffer_size_mb,
      limit=limit or FLAGS.graph_reader_limit,
      ctx=ctx,
    )


class WriteGraphsToFile(progress.Progress):
  """Write graphs in a graph database to pickled files.

  This is for debugging.
  """

  def __init__(self, outdir: pathlib.Path):
    self.reader = BufferedGraphReader.CreateFromFlags()
    self.outdir = outdir
    self.outdir.mkdir(parents=True, exist_ok=True)
    super(WriteGraphsToFile, self).__init__("read_db", 0, self.reader.n)
    self.reader.ctx = self.ctx

  def Run(self):
    """Read and write the graphs."""
    for self.ctx.i, graph_tuple in enumerate(self.reader):
      path = self.outdir / f"graph_tuple_{graph_tuple.id:08}.pickle"
      graph_tuple.ToFile(path)


def Main():
  """Main entry point."""
  if not FLAGS.graph_reader_outdir:
    raise app.UsageError("--graph_reader_outdir must be set")
  progress.Run(WriteGraphsToFile(FLAGS.graph_reader_outdir))


if __name__ == "__main__":
  app.Run(Main)
