"""Library for labelling program graphs with reachability information."""
import collections

import networkx as nx

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled.dataflow import data_flow_graphs
from labm8.py import app

FLAGS = app.FLAGS


# The real_y arrays for node reachability:
REACHABLE_NO = [1, 0]
REACHABLE_YES = [0, 1]


class ReachabilityAnnotator(data_flow_graphs.NetworkXDataFlowGraphAnnotator):
  """Annotate graphs with reachability anlaysis.

  Statement node A is reachable from statement node B iff there exists some
  control flow path from B >> A. Non-statement nodes are never reachable.
  """

  def RootNodeType(self) -> programl_pb2.Node.Type:
    """Reachability is a statement-based analysis."""
    return programl_pb2.Node.STATEMENT

  def Annotate(
    self, g: nx.MultiDiGraph, root_node: int
  ) -> programl_pb2.ProgramGraph:
    """Annotate nodes in the graph with their reachability.

    The 'root node' annotation is a [0,1] value appended to node x vectors.
    The reachability label is a 1-hot binary vector set to y node vectors.

    Args:
      g: The graph to annotate.
      root_node: The source node for determining reachability.

    Returns:
      The program graph with reachability annotations.
    """
    # Initialize all nodes as unreachable and not root node, except the root
    # node.
    for node, data in g.nodes(data=True):
      data["x"].append(data_flow_graphs.ROOT_NODE_NO)
      data["y"] = REACHABLE_NO
    g.nodes[root_node]["x"][-1] = data_flow_graphs.ROOT_NODE_YES

    # Perform a breadth-first traversal to mark reachable nodes.
    data_flow_steps = 0
    reachable_node_count = 0
    visited = set()
    q = collections.deque()

    # Only begin reachable BFS if the root is a statement.
    if g.nodes[root_node]["type"] == programl_pb2.Node.STATEMENT:
      q.append((root_node, 1))

    while q:
      node, data_flow_steps = q.popleft()
      reachable_node_count += 1
      g.nodes[node]["y"] = REACHABLE_YES
      visited.add(node)

      for _, next, flow in g.out_edges(node, data="flow"):
        if flow == programl_pb2.Edge.CONTROL and next not in visited:
          q.append((next, data_flow_steps + 1))

    return programl.NetworkXToProgramGraph(
      g,
      data_flow_root_node=root_node,
      data_flow_steps=data_flow_steps,
      data_flow_positive_node_count=reachable_node_count,
    )
