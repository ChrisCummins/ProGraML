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

import numpy as np


class LstmBatchData(NamedTuple):
  """The model-specific data generated for a batch."""

  # An array of shape (batch_size) which indicates the the number of each
  # nodes for each graph in the batch, before truncation or padding. For
  # example, if padded_sequence_length=64 and given an array of graph_node_sizes
  # [10, 104], this means that the first graph in the batch has 54 padding
  # nodes, and the second graph had the final 40 nodes truncated.
  graph_node_sizes: np.array

  # Shape (batch_size, padded_sequence_length, 1), dtype np.int32
  encoded_sequences: np.array
  # Shape (batch_size, padded_sequence_length, 2), dtype np.int32
  selector_vectors: np.array
  # Shape (batch_size, padded_sequence_length, node_y_dimensionality),
  # dtype np.float32
  node_labels: np.array
  # Shape (batch_size, ?, 1), dtype np.int32.
  targets: List[np.array]
