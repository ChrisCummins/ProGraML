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
"""Batch build for GGNN graphs."""
from typing import Dict
from typing import Optional

import numpy as np

from labm8.py import app
from programl.graph.format.py.graph_tuple_builder import GraphTupleBuilder
from programl.models.base_batch_builder import BaseBatchBuilder
from programl.models.base_graph_loader import BaseGraphLoader
from programl.models.batch_data import BatchData
from programl.models.ggnn.ggnn_batch import GgnnBatchData


class DataflowGgnnBatchBuilder(BaseBatchBuilder):
  """The GGNN batch builder for data flow graphs.

  Constructs a graph tuple per-batch.
  """

  def __init__(
    self,
    graph_loader: BaseGraphLoader,
    vocabulary: Dict[str, int],
    max_node_size: int = 10000,
    use_cdfg: bool = False,
    max_batch_count: int = None,
    max_queue_size: int = 100,
  ):
    self.vocabulary = vocabulary
    self.max_node_size = max_node_size
    self.use_cdfg = use_cdfg

    # Mutable state.
    self.builder = GraphTupleBuilder()
    self.vocab_ids = []
    self.selector_ids = []
    self.node_labels = []
    self.node_size = 0

    # Call super-constructor last since it starts the worker thread.
    super(DataflowGgnnBatchBuilder, self).__init__(
      graph_loader=graph_loader,
      max_batch_count=max_batch_count,
      max_queue_size=max_queue_size,
    )

  def OnItem(self, item) -> Optional[BatchData]:
    graph, features = item

    # Determine the node list.
    if self.use_cdfg:
      node_list = [
        node.features.feature["source_node_index"].int64_list.value[0]
        for node in graph.node
      ]
    else:
      node_list = list(range(len(graph.node)))

    if self.node_size + len(graph.node) > self.max_node_size:
      if self.node_size:
        return self._Build()
      else:
        return

    # Add the graph to the batch.
    try:
      # Find the vocabulary indices for the nodes in the graph.
      vocab_ids = [
        self.vocabulary.get(node.text, len(self.vocabulary))
        for node in graph.node
      ]
      # Read the graph node features using the given node list.
      selector_ids = [
        features.node_features.feature_list["data_flow_root_node"]
        .feature[n]
        .int64_list.value[0]
        for n in node_list
      ]
      node_labels = [
        features.node_features.feature_list["data_flow_value"]
        .feature[n]
        .int64_list.value[0]
        for n in node_list
      ]
    except IndexError:
      app.Log(2, "Encoding error")
      return

    self.builder.AddProgramGraph(graph)
    self.vocab_ids += vocab_ids
    self.selector_ids += selector_ids
    self.node_labels += node_labels
    self.node_size += len(node_list)

  def EndOfItems(self) -> Optional[BatchData]:
    # We've ran out of graphs, but may have an in-progress batch.
    if self.node_size:
      return self._Build()

  def _Build(self) -> BatchData:
    """Construct and return a batch, resetting mutable state."""
    gt = self.builder.Build()

    # Expand node labels to 1-hot.
    indices = np.arange(len(self.node_labels))
    node_labels_1hot = np.zeros((len(self.node_labels), 2), dtype=np.int32)
    node_labels_1hot[indices, self.node_labels] = 1

    batch = BatchData(
      graph_count=gt.graph_size,
      model_data=GgnnBatchData(
        graph_tuple=gt,
        vocab_ids=np.array(self.vocab_ids, dtype=np.int32),
        selector_ids=np.array(self.selector_ids, dtype=np.int32),
        node_labels=node_labels_1hot,
      ),
    )

    # Reset mutable state.
    self.vocab_ids = []
    self.selector_ids = []
    self.node_labels = []
    self.node_size = 0

    return batch
