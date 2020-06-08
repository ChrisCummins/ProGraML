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
"""A batch of GGNN data."""
from typing import List
from typing import NamedTuple
from typing import Optional

import numpy as np

from programl.graph.format.py.graph_tuple import GraphTuple
from programl.proto import program_graph_pb2


class GgnnBatchData(NamedTuple):
  """The model-specific data generated for a batch."""

  # A combination of one or more graphs into a single disconnected graph.
  graph_tuple: GraphTuple
  vocab_ids: np.array
  selector_ids: np.array

  # A list of graphs that were used to construct the disjoint graph.
  # This can be useful for debugging, but is not required by the model.
  graphs: Optional[List[program_graph_pb2.ProgramGraph]] = None

  # Shape: (node_size, num_classes), dtype np.int32
  node_labels: Optional[np.array] = None
  # Shape: (num_classes), dtype np.int32
  graph_labels: Optional[np.array] = None

  # Graph-level feature vectors.
  graph_features: Optional[np.array] = None
