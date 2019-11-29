"""A database of labelled graph tuples."""
import datetime
import pickle
import typing

import sqlalchemy as sql
from sqlalchemy.dialects import mysql
from sqlalchemy.ext import declarative

from deeplearning.ml4pl.graphs.labelled.graph_tuple import graph_tuple
from deeplearning.ml4pl.graphs.unlabelled import programl_pb2
from labm8.py import app
from labm8.py import crypto
from labm8.py import labdate
from labm8.py import sqlutil

FLAGS = app.FLAGS

Base = declarative.declarative_base()


class Meta(Base, sqlutil.TablenameFromClassNameMixin):
  """Key-value database metadata store."""

  key: str = sql.Column(sql.String(64), primary_key=True)
  pickled_value: str = sql.Column(
      sqlutil.ColumnTypes.LargeBinary(), nullable=False
  )
  date_added: datetime.datetime = sql.Column(
      sql.DateTime().with_variant(mysql.DATETIME(fsp=3), "mysql"),
      nullable=False,
      default=labdate.GetUtcMillisecondsNow,
  )

  @property
  def value(self) -> typing.Any:
    return pickle.loads(self.pickled_value)

  @classmethod
  def Create(cls, key: str, value: typing.Any):
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

  # A string name to split the graphs in a database into discrete buckets, e.g.
  # "train", "val", "test"; or "1", "2", ... k for k-fold classification.
  split: str = sql.Column(sql.String(8), nullable=False, index=True)

  # The size of the program graph.
  node_count: int = sql.Column(sql.Integer, nullable=False)
  edge_count: int = sql.Column(sql.Integer, nullable=False)

  # The number of distinct node and edge types.
  node_type_count: int = sql.Column(sql.Integer, nullable=False)
  edge_type_count: int = sql.Column(sql.Integer, nullable=False)

  # The dimensionality of node-level features and labels.
  node_features_dimensionality: int = sql.Column(
      sql.Integer, default=0, nullable=False
  )
  node_labels_dimensionality: int = sql.Column(
      sql.Integer, default=0, nullable=False
  )

  # The dimensionality of graph-level features and labels.
  graph_features_dimensionality: int = sql.Column(
      sql.Integer, default=0, nullable=False
  )
  graph_labels_dimensionality: int = sql.Column(
      sql.Integer, default=0, nullable=False
  )

  # The maximum value of the 'position' attribute of edges.
  edge_position_max: int = sql.Column(sql.Integer, nullable=False)

  # The size of the pickled graph tuple in bytes.
  graph_tuple_size: int = sql.Column(sql.Integer, nullable=False)

  date_added: datetime.datetime = sql.Column(
      sql.DateTime().with_variant(mysql.DATETIME(fsp=3), "mysql"),
      nullable=False,
      default=labdate.GetUtcMillisecondsNow,
  )

  # Create the one-to-one relationship from GraphTuple to GraphTupleData.
  data: "GraphTupleData" = sql.orm.relationship(
      "GraphTupleData", uselist=False, cascade="all, delete-orphan"
  )

  # Joined table accessors:

  @property
  def sha1(self) -> str:
    """Return the sha1 of the graph tuple."""
    return self.data.sha1

  @property
  def tuple(self) -> graph_tuple.GraphTuple:
    """Deserialize and load the protocol buffer."""
    return pickle.loads(self.data.pickled_graph_tuple)

  @classmethod
  def Create(cls, unlabelled_graph: unlabelled_graph_database.ProgramGraph):
    """Create a GraphTuple from the given protocol buffer.

    This is the preferred method of populating databases of program graphs, as
    it contains the boilerplate to extract and set the metadata columns, and
    handles the join between the two proto/metadata invisibly.

    Args:
      proto: The protocol buffer to instantiate a program graph from.
      split: The name of the split that this graph belongs to.

    Returns:
      A GraphTuple instance.
    """
    # Gather the edge attributes in a single pass of the proto.
    edge_attributes = [(edge.type, edge.position) for edge in proto.edge]
    edge_types = set([x[0] for x in edge_attributes])
    edge_position_max = max([x[1] for x in edge_attributes])
    del edge_attributes

    # Gather the node attributes in a single pass.
    node_types = set()
    node_texts = []
    node_preprocessed_texts = []
    node_encodeds = []
    for node in proto.node:
      node_types.add(node.type)
      if node.HasField("text"):
        node_texts.append(node.text)
      if node.HasField("preprocessed_text"):
        node_preprocessed_texts.append(node.preprocessed_text)
      if node.HasField("encoded"):
        node_encodeds.append(node.encoded)

    serialized_proto = proto.SerializeToString()

    return GraphTuple(
        split=split,
        ir_id=ir_id,
        node_count=len(proto.node),
        edge_count=len(proto.edge),
        node_type_count=len(node_types),
        edge_type_count=len(edge_types),
        node_text_count=len(node_texts),
        node_unique_text_count=len(set(node_texts)),
        node_preprocessed_text_count=len(node_preprocessed_texts),
        node_unique_preprocessed_text_count=len(set(node_preprocessed_texts)),
        node_encoded_count=len(node_encodeds),
        node_unique_encoded_count=len(set(node_encodeds)),
        edge_position_max=edge_position_max,
        serialized_proto_size=len(serialized_proto),
        data=GraphTupleData(
            sha1=crypto.sha1(serialized_proto), serialized_proto=serialized_proto,
        ),
    )


class GraphTupleData(
    Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin
):
  """The protocol buffer of a program graph.

  See GraphTuple for the parent table.
  """

  id: int = sql.Column(
      sql.Integer, sql.ForeignKey("program_graphs.id"), primary_key=True
  )

  # The sha1sum of the 'serialized_proto' column. There is no requirement
  # that unlabelled graphs be unique, but, should you wish to enforce this,
  # you can group by this sha1 column and prune the duplicates.
  sha1: str = sql.Column(sql.String(40), nullable=False, index=True)

  # A binary-serialized GraphTuple protocol buffer.
  serialized_proto: bytes = sql.Column(
      sqlutil.ColumnTypes.LargeBinary(), nullable=False
  )


class Database(sqlutil.Database):
  """A database of GraphTuple protocol buffers."""

  def __init__(self, url: str, must_exist: bool = False):
    super(Database, self).__init__(url, Base, must_exist=must_exist)
