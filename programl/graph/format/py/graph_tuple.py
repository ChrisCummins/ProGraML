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
"""The module implements conversion of graphs to tuples of arrays."""
from typing import NamedTuple

import numpy as np


class GraphTuple(NamedTuple):
  """The graph tuple: a compact representation of graph features and labels.

  The transformation of ProgramGraph protocol message to GraphTuples is lossy
  (omitting attributes such as node types and features).
  """

  # A list of adjacency lists, one for each flow type, where an entry in an
  # adjacency list is a <src,dst> tuple of node indices.
  # Shape (3, ?, 2), dtype int32:
  adjacencies: np.array

  # A list of edge positions, one for each edge type. An edge position is an
  # integer in the range 0 <= x < edge_position_max.
  # Shape (3, ?), dtype int32:
  edge_positions: np.array

  # A list of integers which segment the nodes by graph. E.g. with a GraphTuple
  # of two distinct graphs, both with three nodes, nodes_list will be
  # [0, 0, 0, 1, 1, 1].
  # Shape (?), dtype int32
  node_sizes: np.array
  edge_sizes: np.array

  # The number of graphs.
  graph_size: int
  # The total number of nodes across the disjoint graphs.
  node_size: int
  # The total number of edges of all types.
  edge_size: int

  @property
  def control_edge_size(self) -> int:
    return self.adjacencies[0].shape[0]

  @property
  def data_edge_size(self) -> int:
    return self.adjacencies[1].shape[0]

  @property
  def call_edge_size(self) -> int:
    return self.adjacencies[2].shape[0]

  @property
  def edge_position_max(self) -> int:
    """Return the maximum edge position."""
    return max(
      [
        self.adjacencies[0].max(),
        self.adjacencies[1].max(),
        self.adjacencies[2].max(),
      ]
    )
