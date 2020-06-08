# Copyright 2019-2020 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module defines convertors between networkx and ProgramGraph objects.

Conversion between ProgramGraph protocol buffers and NetworkX graphs is
lossless.
"""
import networkx as nx

from programl.graph.format.py import node_link_graph_pybind
from programl.proto import program_graph_pb2


def ProgramGraphToNetworkX(
  proto: program_graph_pb2.ProgramGraph,
) -> nx.MultiDiGraph:
  """Convert a ProgramGraph message to a NetworkX graph.

  Args:
    proto: The program graph to convert.

  Returns:
    A NetworkX MultiDiGraph.

  Raises:
    ValueError: If the graph cannot be converted.
  """
  node_link_graph = node_link_graph_pybind.ProgramGraphToNodeLinkGraph(
    proto.SerializeToString()
  )
  return nx.readwrite.json_graph.node_link_graph(node_link_graph)
