"""This module defines a database for storing graph tuples."""
import datetime
import pickle
import typing

import sqlalchemy as sql

from deeplearning.ml4pl import run_id
from deeplearning.ml4pl.graphs.labelled import data_flow_graphs
from deeplearning.ml4pl.graphs.labelled import graph_tuple as graph_tuple_lib
from labm8.py import app
from labm8.py import crypto
from labm8.py import sqlutil


FLAGS = app.FLAGS


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
  def value(self) -> typing.Any:
    """De-pickle the column value."""
    return pickle.loads(self.pickled_value)

  @classmethod
  def Create(cls, key: str, value: typing.Any):
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

  @property
  def tuple(self) -> graph_tuple_lib.GraphTuple:
    """Un-pickle the graph tuple."""
    return pickle.loads(self.data.pickled_graph_tuple)

  @classmethod
  def CreateFromGraphTuple(
    cls, graph_tuple: graph_tuple_lib.GraphTuple, ir_id: int
  ) -> "GraphTuple":
    """Create a mapped database instance from the given graph tuple.

    This is the preferred method of populating databases of graph tuples, as
    it contains the boilerplate to extract and set the metadata columns, and
    handles the join between the two data/metadata tables invisibly.

    Args:
      graph_tuple: The graph tuple to map.
      ir_id: The intermediate representation ID.

    Returns:
      A GraphTuple instance.
    """
    pickled_graph_tuple = pickle.dumps(graph_tuple)
    return GraphTuple(
      ir_id=ir_id,
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
    cls, annotated_graph: data_flow_graphs.DataFlowAnnotatedGraph, ir_id: int
  ) -> "GraphTuple":
    """Create a mapped database instance from the given annotated graph.

    This is the preferred method of populating databases of graph tuples, as
    it contains the boilerplate to extract and set the metadata columns, and
    handles the join between the two data/metadata tables invisibly.

    Args:
      annotated_graph: A DataFlowAnnotatedGraph instance.
      ir_id: The intermediate representation ID.

    Returns:
      A GraphTuple instance.
    """
    graph_tuple = graph_tuple_lib.GraphTuple.CreateFromNetworkX(
      annotated_graph.g
    )
    mapped = cls.CreateFromGraphTuple(graph_tuple, ir_id)
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


class Database(sqlutil.Database):
  """A database of GraphTuples."""

  def __init__(self, url: str, must_exist: bool = False):
    super(Database, self).__init__(url, Base, must_exist=must_exist)
