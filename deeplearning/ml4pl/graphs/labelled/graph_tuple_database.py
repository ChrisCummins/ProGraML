"""This module defines a database for storing graph tuples."""
import datetime
import pathlib
import pickle
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import sqlalchemy as sql

from deeplearning.ml4pl import run_id
from deeplearning.ml4pl.graphs.labelled import graph_tuple as graph_tuple_lib
from deeplearning.ml4pl.graphs.labelled.dataflow import data_flow_graphs
from labm8.py import app
from labm8.py import crypto
from labm8.py import decorators
from labm8.py import humanize
from labm8.py import jsonutil
from labm8.py import progress
from labm8.py import sqlutil


FLAGS = app.FLAGS
# Note we declare a graph_db flag at the bottom of this file, after declaring
# the Database class.

Base = sql.ext.declarative.declarative_base()


class Meta(Base, sqlutil.TablenameFromClassNameMixin):
  """A key-value database metadata store, with additional run ID."""

  # Unused integer ID for this row.
  id: int = sql.Column(sql.Integer, primary_key=True)

  # The run ID that generated this <key,value> pair.
  run_id: str = run_id.RunId.SqlStringColumn()

  timestamp: datetime.datetime = sqlutil.ColumnFactory.MillisecondDatetime()

  # The <key,value> pair.
  key: str = sql.Column(sql.String(128), index=True)
  pickled_value: bytes = sql.Column(
    sqlutil.ColumnTypes.LargeBinary(), nullable=False
  )

  @property
  def value(self) -> Any:
    """De-pickle the column value."""
    return pickle.loads(self.pickled_value)

  @classmethod
  def Create(cls, key: str, value: Any):
    """Construct a table entry."""
    return Meta(key=key, pickled_value=pickle.dumps(value))


class GraphTuple(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """A table of graph tuples.

  For every GraphTuple, there should be a corresponding GraphTupleData row
  containing the pickled graph tuple as a binary blob. The reason for dividing
  the data horizontally across two tables is to enable fast scanning
  of graph metadata, without needing to churn through a table of pickled binary
  blobs.
  """

  id: int = sql.Column(sql.Integer, primary_key=True)

  # A reference to the 'id' column of a
  # deeplearning.ml4pl.ir.ir_database.IntermediateRepresentationFile database
  # row. There is no foreign key relationship here because they are separate
  # databases.
  ir_id: int = sql.Column(sql.Integer, nullable=False, index=True)

  # An integer used to split databases of graphs into separate graphs, e.g.
  # train/val/test split.
  split: Optional[int] = sql.Column(sql.Integer, nullable=True, index=True)

  # The size of the program graph.
  node_count: int = sql.Column(sql.Integer, nullable=False)
  control_edge_count: int = sql.Column(sql.Integer, nullable=False)
  data_edge_count: int = sql.Column(sql.Integer, nullable=False)
  call_edge_count: int = sql.Column(sql.Integer, nullable=False)

  # The maximum value of the 'position' attribute of edges.
  edge_position_max: int = sql.Column(sql.Integer, nullable=False)

  # The dimensionality of node-level features and labels.
  node_x_dimensionality: int = sql.Column(
    sql.Integer, default=0, nullable=False
  )
  node_y_dimensionality: int = sql.Column(
    sql.Integer, default=0, nullable=False
  )

  # The dimensionality of graph-level features and labels.
  graph_x_dimensionality: int = sql.Column(
    sql.Integer, default=0, nullable=False
  )
  graph_y_dimensionality: int = sql.Column(
    sql.Integer, default=0, nullable=False
  )

  # The size of the pickled graph tuple in bytes.
  pickled_graph_tuple_size: int = sql.Column(sql.Integer, nullable=False)

  # A copy of attributes from the
  # deeplearning.ml4pl.graphs.labelled.data_flow_graphs.DataFlowAnnotatedGraph
  # tuple for storing metadata of data flow analysis graphs. If not relevant ,
  # these columns may be null.
  data_flow_steps: int = sql.Column(sql.Integer, nullable=True)
  data_flow_root_node: int = sql.Column(sql.Integer, nullable=True)
  data_flow_positive_node_count: int = sql.Column(sql.Integer, nullable=True)

  timestamp: datetime.datetime = sqlutil.ColumnFactory.MillisecondDatetime()

  # Create the one-to-one relationship from GraphTuple to GraphTupleData.
  data: "GraphTupleData" = sql.orm.relationship(
    "GraphTupleData", uselist=False, cascade="all, delete-orphan"
  )

  @property
  def edge_count(self) -> int:
    return self.control_edge_count + self.data_edge_count + self.call_edge_count

  # Joined table accessors:

  @property
  def sha1(self) -> str:
    """Return the sha1 of the graph tuple."""
    return self.data.sha1

  @decorators.memoized_property
  def tuple(self) -> graph_tuple_lib.GraphTuple:
    """Un-pickle the graph tuple and cache the binary results."""
    return pickle.loads(self.data.pickled_graph_tuple)

  def ToFile(self, path: pathlib.Path) -> None:
    """Dump the pickled graph tuple to file.

    This is lossy, as the ir_id column is not dumped.

    Args:
      path: The path of the graph tuple to write.
    """
    with open(path, "wb") as f:
      pickle.dump(self.tuple, f)

  # Factory methods:

  @classmethod
  def FromFile(cls, path: pathlib.Path, ir_id: int):
    """Construct a mapped database instance from a file generated by ToFile().

    Args:
      path: The path of the file to read.
      ir_id: The IR id of the graph tuple.

    Returns:
      A GraphTuple instance.
    """
    with open(path, "rb") as f:
      graph_tuple = pickle.load(f)

    return cls.CreateFromGraphTuple(graph_tuple, ir_id)

  @classmethod
  def CreateFromGraphTuple(
    cls,
    graph_tuple: graph_tuple_lib.GraphTuple,
    ir_id: int,
    split: Optional[int] = None,
  ) -> "GraphTuple":
    """Create a mapped database instance from the given graph tuple.

    This is the preferred method of populating databases of graph tuples, as
    it contains the boilerplate to extract and set the metadata columns, and
    handles the join between the two data/metadata tables invisibly.

    Args:
      graph_tuple: The graph tuple to map.
      ir_id: The intermediate representation ID.
      split: The split value of this graph.

    Returns:
      A GraphTuple instance.
    """
    pickled_graph_tuple = pickle.dumps(graph_tuple)
    return GraphTuple(
      ir_id=ir_id,
      split=split,
      node_count=graph_tuple.node_count,
      control_edge_count=graph_tuple.control_edge_count,
      data_edge_count=graph_tuple.data_edge_count,
      call_edge_count=graph_tuple.call_edge_count,
      edge_position_max=graph_tuple.edge_position_max,
      node_x_dimensionality=graph_tuple.node_x_dimensionality,
      node_y_dimensionality=graph_tuple.node_y_dimensionality,
      graph_x_dimensionality=graph_tuple.graph_x_dimensionality,
      graph_y_dimensionality=graph_tuple.graph_y_dimensionality,
      pickled_graph_tuple_size=len(pickled_graph_tuple),
      data=GraphTupleData(
        sha1=crypto.sha1(pickled_graph_tuple),
        pickled_graph_tuple=pickled_graph_tuple,
      ),
    )

  @classmethod
  def CreateEmpty(cls, ir_id: int) -> "GraphTuple":
    """Create an "empty" graph tuple.

    An empty graph tuple can be used to signal that the conversion to GraphTuple
    failed, and is signalled by a node_count of 0. An empty graph tuple has
    no corresponding GraphTupleData row.
    """
    return GraphTuple(
      ir_id=ir_id,
      node_count=0,
      control_edge_count=0,
      data_edge_count=0,
      call_edge_count=0,
      edge_position_max=0,
      pickled_graph_tuple_size=0,
    )

  @classmethod
  def CreateFromDataFlowAnnotatedGraph(
    cls,
    annotated_graph: data_flow_graphs.DataFlowAnnotatedGraph,
    ir_id: int,
    split: Optional[int] = None,
  ) -> "GraphTuple":
    """Create a mapped database instance from the given annotated graph.

    This is the preferred method of populating databases of graph tuples, as
    it contains the boilerplate to extract and set the metadata columns, and
    handles the join between the two data/metadata tables invisibly.

    Args:
      annotated_graph: A DataFlowAnnotatedGraph instance.
      ir_id: The intermediate representation ID.
      split: The split value of this graph.

    Returns:
      A GraphTuple instance.
    """
    graph_tuple = graph_tuple_lib.GraphTuple.CreateFromNetworkX(
      annotated_graph.g
    )
    mapped = cls.CreateFromGraphTuple(graph_tuple, ir_id, split)
    mapped.data_flow_steps = annotated_graph.data_flow_steps
    mapped.data_flow_root_node = annotated_graph.root_node
    mapped.data_flow_positive_node_count = annotated_graph.positive_node_count
    return mapped


class GraphTupleData(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """The pickled graph tuple data. See GraphTuple for the parent table."""

  id: int = sql.Column(
    sql.Integer,
    sql.ForeignKey("graph_tuples.id", onupdate="CASCADE", ondelete="CASCADE"),
    primary_key=True,
  )

  # The sha1sum of the 'pickled_graph_tuple' column. There is no requirement
  # that graph tuples be unique, but, should you wish to enforce this,
  # you can group by this sha1 column and prune the duplicates.
  sha1: str = sql.Column(sql.String(40), nullable=False, index=True)

  # The pickled GraphTuple data.
  pickled_graph_tuple: bytes = sql.Column(
    sqlutil.ColumnTypes.LargeBinary(), nullable=False
  )


# A registry of database statics, where each entry is a <name, property> tuple.
database_statistics_registry: List[Tuple[str, Callable[["Database"], Any]]] = []


def database_statistic(func):
  """A decorator to mark a method on a Database as a database static.

  Database statistics can be accessed using Database.stats_json property to
  retrieve a <name, vale> dictionary.
  """
  global database_statistics_registry
  database_statistics_registry.append((func.__name__, func))
  return property(func)


class Database(sqlutil.Database):
  """A database of GraphTuples."""

  def __init__(
    self,
    url: str,
    must_exist: bool = False,
    ctx: progress.ProgressContext = progress.NullContext,
  ):
    super(Database, self).__init__(url, Base, must_exist=must_exist)
    self.ctx = ctx

    # Lazily evaluated attributes.
    self._graph_tuple_stats = None
    self._splits = None
    self._split_counts = None

  ##############################################################################
  # Database stats. These are evaluated lazily and the results cached. There is
  # no cache invalidation strategy - after modifying the database, you must
  # manually call RefreshStats() to ensure that stale stats are re-computed.
  ##############################################################################

  @database_statistic
  def graph_count(self) -> int:
    """The number of non-empty graphs in the database."""
    return int(self.graph_tuple_stats.graph_count)

  @database_statistic
  def ir_count(self) -> int:
    """The number of distinct intermediate representations that the non-empty
    graphs are constructed from.
    """
    return int(self.graph_tuple_stats.ir_count or 0)

  @database_statistic
  def split_count(self) -> int:
    """The number of distinct splits in the database."""
    return int(self.graph_tuple_stats.split_count or 0)

  @database_statistic
  def node_count(self) -> int:
    """The total node count in non-empty graphs."""
    return int(self.graph_tuple_stats.node_count or 0)

  @database_statistic
  def edge_count(self) -> int:
    """The total edge count in non-empty graphs."""
    return int(self.graph_tuple_stats.edge_count or 0)

  @database_statistic
  def control_edge_count(self) -> int:
    """The total control edge count in non-empty graphs."""
    return int(self.graph_tuple_stats.control_edge_count or 0)

  @database_statistic
  def data_edge_count(self) -> int:
    """The total data edge count in non-empty graphs."""
    return int(self.graph_tuple_stats.data_edge_count or 0)

  @database_statistic
  def call_edge_count(self) -> int:
    """The total call edge count in non-empty graphs."""
    return int(self.graph_tuple_stats.call_edge_count or 0)

  @database_statistic
  def node_count_max(self) -> int:
    """The maximum node count in non-empty graphs."""
    return int(self.graph_tuple_stats.node_count_max or 0)

  @database_statistic
  def edge_count_max(self) -> int:
    """The maximum edge count in non-empty graphs."""
    return int(self.graph_tuple_stats.edge_count_max or 0)

  @database_statistic
  def control_edge_count_max(self) -> int:
    """The maximum control edge count in non-empty graphs."""
    return int(self.graph_tuple_stats.control_edge_count_max or 0)

  @database_statistic
  def data_edge_count_max(self) -> int:
    """The maximum data edge count in non-empty graphs."""
    return int(self.graph_tuple_stats.data_edge_count_max or 0)

  @database_statistic
  def call_edge_count_max(self) -> int:
    """The maximum call edge count in non-empty graphs."""
    return int(self.graph_tuple_stats.call_edge_count_max or 0)

  @database_statistic
  def edge_position_max(self) -> int:
    """The maximum edge position in non-empty graphs."""
    return int(self.graph_tuple_stats.edge_position_max or 0)

  @database_statistic
  def node_x_dimensionality(self) -> int:
    """The node x dimensionality of all non-empty graphs."""
    return int(self.graph_tuple_stats.node_x_dimensionality or 0)

  @database_statistic
  def node_y_dimensionality(self) -> int:
    """The node y dimensionality of all non-empty graphs."""
    return int(self.graph_tuple_stats.node_y_dimensionality or 0)

  @database_statistic
  def graph_x_dimensionality(self) -> int:
    """The graph x dimensionality of all non-empty graphs."""
    return int(self.graph_tuple_stats.graph_x_dimensionality or 0)

  @database_statistic
  def graph_y_dimensionality(self) -> int:
    """The graph y dimensionality of all non-empty graphs."""
    return int(self.graph_tuple_stats.graph_y_dimensionality or 0)

  @database_statistic
  def graph_data_size(self) -> int:
    """The total size of the non-empty graph data, in bytes."""
    return int(self.graph_tuple_stats.graph_data_size or 0)

  @database_statistic
  def graph_data_size_min(self) -> int:
    """The minimum size of the non-empty graph tuple data, in bytes."""
    return int(self.graph_tuple_stats.graph_data_size_min or 0)

  @database_statistic
  def graph_data_size_avg(self) -> float:
    """The average size of the non-empty graph tuple data, in bytes."""
    return float(self.graph_tuple_stats.graph_data_size_avg or 0)

  @database_statistic
  def graph_data_size_max(self) -> int:
    """The maximum size of the non-empty graph tuple data, in bytes."""
    return int(self.graph_tuple_stats.graph_data_size_max or 0)

  @database_statistic
  def has_data_flow(self) -> bool:
    """Return whether the graph database has data flow annotations."""
    return (
      self.graph_count
      and self.graph_tuple_stats.data_flow_steps_null_count == 0
    )

  @database_statistic
  def data_flow_steps_min(self) -> Optional[int]:
    """The minimum data flow steps for non-empty graphs."""
    if self.has_data_flow:
      return int(self.graph_tuple_stats.data_flow_steps_min or 0)

  @database_statistic
  def data_flow_steps_avg(self) -> Optional[float]:
    """The average data flow steps for non-empty graphs."""
    if self.has_data_flow:
      return float(self.graph_tuple_stats.data_flow_steps_avg)

  @database_statistic
  def data_flow_steps_max(self) -> Optional[int]:
    """The maximum data flow steps for non-empty graphs."""
    if self.has_data_flow:
      return int(self.graph_tuple_stats.data_flow_steps_max or 0)

  @database_statistic
  def data_flow_positive_node_count_min(self) -> Optional[int]:
    """The minimum data flow positive node count for non-empty graphs."""
    if self.has_data_flow:
      return int(self.graph_tuple_stats.data_flow_positive_node_count_min or 0)

  @database_statistic
  def data_flow_positive_node_count_avg(self) -> Optional[int]:
    """The minimum data flow average node count for non-empty graphs."""
    if self.has_data_flow:
      return int(self.graph_tuple_stats.data_flow_positive_node_count_avg or 0)

  @database_statistic
  def data_flow_positive_node_count_max(self) -> Optional[int]:
    """The minimum data flow max node count for non-empty graphs."""
    if self.has_data_flow:
      return int(self.graph_tuple_stats.data_flow_positive_node_count_max or 0)

  @database_statistic
  def splits(self) -> List[int]:
    """Return a list of unique split values."""
    if self._splits is None:
      self.RefreshStats()
    return self._splits

  @database_statistic
  def split_counts(self) -> Dict[int, int]:
    """Return a dictionary mapping split to the number of graphs."""
    if self._split_counts is None:
      self.RefreshStats()
    return self._split_counts

  def RefreshStats(self):
    """Compute the database stats for access via the instance properties.

    Raises:
      ValueError: If the database contains invalid entries, e.g. inconsistent
        vector dimensionalities.
    """
    with self.ctx.Profile(
      2,
      lambda t: (
        "Computed stats over "
        f"{humanize.BinaryPrefix(stats.graph_data_size, 'B')} database "
        f"({humanize.Plural(stats.graph_count, 'graph')})"
      ),
    ), self.Session() as session:
      query = session.query(
        # Graph and IR counts.
        sql.func.count(GraphTuple.id).label("graph_count"),
        sql.func.count(sql.func.distinct(GraphTuple.ir_id)).label("ir_count"),
        sql.func.count(sql.func.distinct(GraphTuple.split)).label(
          "split_count"
        ),
        # Node and edge attribute sums.
        sql.func.sum(GraphTuple.node_count).label("node_count"),
        sql.func.sum(GraphTuple.control_edge_count).label("control_edge_count"),
        sql.func.sum(GraphTuple.data_edge_count).label("data_edge_count"),
        sql.func.sum(GraphTuple.call_edge_count).label("call_edge_count"),
        sql.func.sum(
          GraphTuple.control_edge_count
          + GraphTuple.data_edge_count
          + GraphTuple.call_edge_count
        ).label("edge_count"),
        # Node and edge attribute maximums.
        sql.func.max(GraphTuple.node_count).label("node_count_max"),
        sql.func.max(GraphTuple.control_edge_count).label(
          "control_edge_count_max"
        ),
        sql.func.max(GraphTuple.data_edge_count).label("data_edge_count_max"),
        sql.func.max(GraphTuple.call_edge_count).label("call_edge_count_max"),
        sql.func.max(GraphTuple.call_edge_count).label("call_edge_count_max"),
        sql.func.max(
          GraphTuple.control_edge_count
          + GraphTuple.data_edge_count
          + GraphTuple.call_edge_count
        ).label("edge_count_max"),
        # Edge position max.
        # NOTE: For some strange reason, sqlalchemy likes to interpret the
        # value sql.func.max(GraphTuple.edge_position_max) as a bytes array.
        # Forcing a cast to integer fixes this.
        sql.cast(sql.func.max(GraphTuple.edge_position_max), sql.Integer).label(
          "edge_position_max"
        ),
        # Feature and label dimensionality counts. Each of these columns
        # should be one, showing that there is a single value for all graph
        # tuples.
        sql.func.count(
          sql.func.distinct(GraphTuple.node_x_dimensionality)
        ).label("node_x_dimensionality_count"),
        sql.func.count(
          sql.func.distinct(GraphTuple.node_y_dimensionality)
        ).label("node_y_dimensionality_count"),
        sql.func.count(
          sql.func.distinct(GraphTuple.graph_x_dimensionality)
        ).label("graph_x_dimensionality_count"),
        sql.func.count(
          sql.func.distinct(GraphTuple.graph_y_dimensionality)
        ).label("graph_y_dimensionality_count"),
        # Feature and label dimensionalities.
        sql.func.max(GraphTuple.node_x_dimensionality).label(
          "node_x_dimensionality"
        ),
        sql.func.max(GraphTuple.node_y_dimensionality).label(
          "node_y_dimensionality"
        ),
        sql.func.max(GraphTuple.graph_x_dimensionality).label(
          "graph_x_dimensionality"
        ),
        sql.func.max(GraphTuple.graph_y_dimensionality).label(
          "graph_y_dimensionality"
        ),
        # Graph tuple sizes.
        sql.func.sum(GraphTuple.pickled_graph_tuple_size).label(
          "graph_data_size"
        ),
        sql.func.min(GraphTuple.pickled_graph_tuple_size).label(
          "graph_data_size_min"
        ),
        sql.func.avg(GraphTuple.pickled_graph_tuple_size).label(
          "graph_data_size_avg"
        ),
        sql.func.max(GraphTuple.pickled_graph_tuple_size).label(
          "graph_data_size_max"
        ),
        # Data flow column null counts.
        sql.func.count(GraphTuple.data_flow_steps == None).label(
          "data_flow_steps_null_count"
        ),
        sql.func.count(GraphTuple.data_flow_steps == None).label(
          "data_flow_positive_node_count_null_count"
        ),
        # Data flow step counts.
        sql.func.min(GraphTuple.data_flow_steps).label("data_flow_steps_min"),
        sql.func.avg(GraphTuple.data_flow_steps).label("data_flow_steps_avg"),
        sql.func.max(GraphTuple.data_flow_steps).label("data_flow_steps_max"),
        # Data flow positive node count.
        sql.func.min(GraphTuple.data_flow_positive_node_count).label(
          "data_flow_positive_node_count_min"
        ),
        sql.func.avg(GraphTuple.data_flow_positive_node_count).label(
          "data_flow_positive_node_count_avg"
        ),
        sql.func.max(GraphTuple.data_flow_positive_node_count).label(
          "data_flow_positive_node_count_max"
        ),
      )

      # Ignore "empty" graph nodes.
      query = query.filter(GraphTuple.node_count > 1)

      # Compute the stats.
      stats = query.one()

      # Check that databases have a consistent value for dimensionalities.
      if stats.node_x_dimensionality_count > 1:
        raise ValueError(
          f"Database contains {stats.node_x_dimensionality_count} "
          "distinct node x dimensionalities"
        )
      if stats.node_y_dimensionality_count > 1:
        raise ValueError(
          f"Database contains {stats.node_y_dimensionality_count} "
          "distinct node y dimensionalities"
        )
      if stats.graph_x_dimensionality_count > 1:
        raise ValueError(
          f"Database contains {stats.graph_x_dimensionality_count} "
          "distinct graph x dimensionalities"
        )
      if stats.graph_y_dimensionality_count > 1:
        raise ValueError(
          f"Database contains {stats.graph_y_dimensionality_count} "
          "distinct graph y dimensionalities"
        )

      # Check that every graph has data flow attributes, or none of them do.
      if (
        stats.data_flow_steps_null_count != 0
        and stats.data_flow_steps_null_count != stats.graph_count
      ):
        raise ValueError(
          f"{stats.data_flow_steps_null_count} of "
          f"{stats.graph_count} graphs have no data_flow_steps "
          "value"
        )

      if (
        stats.data_flow_positive_node_count_null_count != 0
        and stats.data_flow_positive_node_count_null_count != stats.graph_count
      ):
        raise ValueError(
          f"{stats.data_flow_positive_node_count_null_count} of "
          f"{stats.graph_count} graphs have no "
          " data_flow_positive_node_count value"
        )

      self._graph_tuple_stats = stats

      with self.Session() as session:
        self._splits = sorted(
          set(
            [
              row.split
              for row in session.query(GraphTuple.split).group_by(
                GraphTuple.split
              )
            ]
          )
        )

        self._split_counts = {
          split: session.query(sql.func.count(GraphTuple.id))
          .filter(GraphTuple.split == split)
          .scalar()
          for split in self._splits
        }

  @property
  def graph_tuple_stats(self):
    """Fetch aggregate graph tuple stats, or compute them if not set."""
    if self._graph_tuple_stats is None:
      self.RefreshStats()
    return self._graph_tuple_stats

  @property
  def stats_json(self) -> Dict[str, Any]:
    """Fetch the database statics as a JSON dictionary."""
    return {
      name: function(self) for name, function in database_statistics_registry
    }

  def __repr__(self) -> str:
    return (
      f"Database of {humanize.DecimalPrefix(self.graph_count, 'graph')} with "
      f"dimensionalities: node_x={self.node_x_dimensionality}, "
      f"node_y={self.node_y_dimensionality}, "
      f"graph_x={self.graph_x_dimensionality}, "
      f"graph_y={self.graph_y_dimensionality}."
    )


# Deferred declaration of flags because we need to reference Database class.
app.DEFINE_database(
  "graph_db", Database, None, "The database to read graph tuples from.",
)


def Main():
  """Main entry point."""
  graph_db = FLAGS.graph_db()
  print(jsonutil.format_json(graph_db.stats_json))


if __name__ == "__main__":
  app.Run(Main)
