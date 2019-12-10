"""This module defines a base class for implementing data flow analyses over
networkx graphs.
"""
import copy
import random
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Optional

import networkx as nx

from deeplearning.ml4pl.graphs import programl_pb2
from labm8.py import app

FLAGS = app.FLAGS


class DataFlowAnnotatedGraph(NamedTuple):
  """A networkx graph with data flow analysis annotations."""

  # A graph with {x, y} {node, graph}-level annotations set.
  g: nx.MultiDiGraph
  # For iterative data flow analyses which have a defined "starting point", this
  # is the node index of that starting point.
  root_node: int = 0
  # For iterative data flow analyses, this is the number of steps that the data
  # flow analysis required to compute the annotations.
  data_flow_steps: int = 0
  # The number of nodes with "positive" analysis results.
  positive_node_count: int = 0


class DataFlowGraphAnnotator(object):
  """Abstract base class for implement networkx graph data flow annotators.

  A data flow annotator takes as input a networkx graph and a root node, and
  generates networkx graphs annotated
  """

  def RootNodeType(self) -> programl_pb2.Node.Type:
    """Return the Node.Type enum for root nodes."""
    raise NotImplementedError("abstract class")

  def Annotate(
    self, g: nx.MultiDiGraph, root_node: int
  ) -> DataFlowAnnotatedGraph:
    """Annotate a networkx graph in-place."""
    raise NotImplementedError("abstract class")

  def MakeAnnotated(
    self, g: nx.MultiDiGraph, n: Optional[int] = None
  ) -> Iterable[DataFlowAnnotatedGraph]:
    """Produce up to "n" annotated graphs.

    Args:
      g: The graph used to produce annotated graphs. The graph is copied and
        left unmodified.
      n: The maximum number of annotated graphs to produce. Multiple graphs are
        produced by selecting different root nodes for creating annotations.
        If `n` is provided, the number of annotated graphs generated will be in
        the range 1 <= x <= min(root_node_count, n). Else, the number of graphs
        will be equal to root_node_count (i.e. one graph for each root node in
        the input graph).

    Returns:
      An iterator of DataFlowAnnotatedGraph tuples.
    """
    root_nodes: List[int] = [
      n for n, type_ in g.nodes(data="type") if type_ == self.RootNodeType()
    ]

    # Impose the limit on the maximum number of graphs to generate.
    if n and n < len(root_nodes):
      random.shuffle(root_nodes)
      root_nodes = root_nodes[:n]

    for root_node in root_nodes:
      # Note that a deep copy is required to ensure that x/y lists are
      # duplicated.
      yield self.Annotate(copy.deepcopy(g), root_node)


# The x value for specifying the root node.
ROOT_NODE_NO = 0
ROOT_NODE_YES = 1
