"""This module defines a base class for implementing data flow analyses over
networkx graphs.
"""
import copy
import random
from typing import List
from typing import Optional

import networkx as nx

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from labm8.py import app

FLAGS = app.FLAGS


class DataFlowGraphs(object):
  """A set of data-flow annotated graphs that abstract the the difference
  between proto and networkx representations.
  """

  @property
  def graphs(self) -> List[nx.MultiDiGraph]:
    """Access the data flow graphs as networkx."""
    raise NotImplementedError("abstract class")

  @property
  def protos(self) -> List[programl_pb2.ProgramGraph]:
    """Access the data flow graphs as protos."""
    raise NotImplementedError("abstract class")


class NetworkxDataFlowGraphs(DataFlowGraphs):
  """A set of data-flow annotated graphs."""

  def __init__(self, graphs: List[nx.MultiDiGraph]):
    self._graphs = graphs

  @property
  def graphs(self) -> List[nx.MultiDiGraph]:
    """Access the underlying networkx graphs."""
    return self._graphs

  @property
  def protos(self) -> List[programl_pb2.ProgramGraph]:
    """Convert the networkx graphs to program graph protos."""
    return [programl.NetworkXToProgramGraph(g) for g in self.graphs]


class DataFlowGraphAnnotator(object):
  """Abstract base class for implement data flow analysis graph annotators."""

  def MakeAnnotated(
    self, unlabelled_graph: programl_pb2.ProgramGraph, n: Optional[int] = None
  ) -> DataFlowGraphs:
    """Produce up to "n" annotated graphs.

    Args:
      unlabelled_graph: The unlabelled program graph used to produce annotated
        graphs.
      n: The maximum number of annotated graphs to produce. For analyses that
        produce only a single graph, this parameter has no effect.

    Returns:
      An iterator of annotated program graphs.
    """
    raise NotImplementedError("abstract class")


class NetworkXDataFlowGraphAnnotator(DataFlowGraphAnnotator):
  """A data flow annotator takes as input a networkx graph."""

  def RootNodeType(self) -> programl_pb2.Node.Type:
    """Return the Node.Type enum for root nodes."""
    raise NotImplementedError("abstract class")

  def Annotate(self, g: nx.MultiDiGraph, root_node: int) -> nx.MultiDiGraph:
    """Annotate a networkx graph in-place."""
    raise NotImplementedError("abstract class")

  def MakeAnnotated(
    self, unlabelled_graph: programl_pb2.ProgramGraph, n: Optional[int] = None
  ) -> DataFlowGraphs:
    """Produce up to "n" annotated graphs.

    Args:
      unlabelled_graph: The unlabelled program graph used to produce annotated
        graphs.
      n: The maximum number of annotated graphs to produce. Multiple graphs are
        produced by selecting different root nodes for creating annotations.
        If `n` is provided, the number of annotated graphs generated will be in
        the range 1 <= x <= min(root_node_count, n). Else, the number of graphs
        will be equal to root_node_count (i.e. one graph for each root node in
        the input graph).

    Returns:
      An AnnotatedGraph instance..
    """
    g = programl.ProgramGraphToNetworkX(unlabelled_graph)

    # Create a pool of candidate root nodes. A root node must be of the expected
    # type (e.g. statement/identifier/immediate) and belong to a function, i.e.
    # not be a "magic" node like the entry node.
    root_nodes: List[int] = [
      n
      for n, data in g.nodes(data=True)
      if data["type"] == self.RootNodeType() and data["function"] is not None
    ]

    # Impose the limit on the maximum number of graphs to generate.
    if n and n < len(root_nodes):
      random.shuffle(root_nodes)
      root_nodes = root_nodes[:n]

    annotated_graphs = []
    for root_node in root_nodes:
      # Note that a deep copy is required to ensure that lists in x/y attributes
      # are duplicated.
      annotated_graph = self.Annotate(copy.deepcopy(g), root_node)
      # Ignore graphs that require no data flow steps.
      if annotated_graph.graph["data_flow_steps"]:
        annotated_graphs.append(annotated_graph)

    return NetworkxDataFlowGraphs(annotated_graphs)


# The x value for specifying the root node for iterative data flow analyses
# that have a defined "starting point".
ROOT_NODE_NO = 0
ROOT_NODE_YES = 1
