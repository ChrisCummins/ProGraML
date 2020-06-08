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
from programl.graph.format.py import graph_serializer
from programl.models.base_batch_builder import BaseBatchBuilder
from programl.models.base_graph_loader import BaseGraphLoader
from programl.models.batch_data import BatchData
from programl.models.lstm.lstm_batch import LstmBatchData
from third_party.py.tensorflow import tf

FLAGS = app.FLAGS


class DataflowLstmBatchBuilder(BaseBatchBuilder):
  """The LSTM batch builder."""

  def __init__(
    self,
    graph_loader: BaseGraphLoader,
    vocabulary: Dict[str, int],
    node_y_dimensionality: int,
    batch_size: int = 256,
    padded_sequence_length: int = 256,
    max_batch_count: int = None,
    max_queue_size: int = 100,
  ):
    self.vocabulary = vocabulary
    self.node_y_dimensionality = node_y_dimensionality
    self.batch_size = batch_size
    self.padded_sequence_length = padded_sequence_length

    # Mutable state.
    self.graph_node_sizes = []
    self.vocab_ids = []
    self.selector_vectors = []
    self.targets = []

    # Padding values.
    self._vocab_id_pad = len(self.vocabulary) + 1
    self._selector_vector_pad = np.zeros((0, 2), dtype=np.int32)
    self._node_label_pad = np.zeros(
      (0, self.node_y_dimensionality), dtype=np.int32
    )

    # Call super-constructor last since it starts the worker thread.
    super(DataflowLstmBatchBuilder, self).__init__(
      graph_loader=graph_loader,
      max_batch_count=max_batch_count,
      max_queue_size=max_queue_size,
    )

  def _Build(self) -> BatchData:
    # A batch may contain fewer graphs than the required batch_size.
    # If so, pad with empty "graphs". These padding graphs will be discarded
    # once processed.
    if len(self.graph_node_sizes) < self.batch_size:
      pad_count = self.batch_size - len(self.graph_node_sizes)
      self.vocab_ids += [
        np.array([self._vocab_id_pad], dtype=np.int32)
      ] * pad_count
      self.selector_vectors += [self._selector_vector_pad] * pad_count
      self.targets += [self._node_label_pad] * pad_count

    batch = BatchData(
      graph_count=len(self.graph_node_sizes),
      model_data=LstmBatchData(
        graph_node_sizes=np.array(self.graph_node_sizes, dtype=np.int32),
        encoded_sequences=tf.compat.v1.keras.preprocessing.sequence.pad_sequences(
          self.vocab_ids,
          maxlen=self.padded_sequence_length,
          dtype="int32",
          padding="pre",
          truncating="post",
          value=self._vocab_id_pad,
        ),
        selector_vectors=tf.compat.v1.keras.preprocessing.sequence.pad_sequences(
          self.selector_vectors,
          maxlen=self.padded_sequence_length,
          dtype="float32",
          padding="pre",
          truncating="post",
          value=np.zeros(2, dtype=np.float32),
        ),
        node_labels=tf.compat.v1.keras.preprocessing.sequence.pad_sequences(
          self.targets,
          maxlen=self.padded_sequence_length,
          dtype="float32",
          padding="pre",
          truncating="post",
          value=np.zeros(self.node_y_dimensionality, dtype=np.float32),
        ),
        # We don't pad or truncate targets.
        targets=self.targets,
      ),
    )

    # Reset mutable state.
    self.graph_node_sizes = []
    self.vocab_ids = []
    self.selector_vectors = []
    self.targets = []

    return batch

  def OnItem(self, item) -> Optional[BatchData]:
    graph, features = item

    # Get the list of graph node indices that produced the serialized encoded
    # graph representation. We use this to construct predictions for the
    # "full" graph through padding.
    node_list = graph_serializer.SerializeInstructionsInProgramGraph(
      graph, self.padded_sequence_length
    )

    try:
      vocab_ids = [
        self.vocabulary.get(
          graph.node[n]
          .features.feature["inst2vec_preprocessed"]
          .bytes_list.value[0]
          .decode("utf-8"),
          self.vocabulary["!UNK"],
        )
        for n in node_list
      ]
      selector_values = np.array(
        [
          features.node_features.feature_list["data_flow_root_node"]
          .feature[n]
          .int64_list.value[0]
          for n in node_list
        ],
        dtype=np.int32,
      )
      selector_vectors = np.zeros((selector_values.size, 2), dtype=np.float32)
      selector_vectors[
        np.arange(selector_values.size), selector_values
      ] = FLAGS.selector_embedding_value
      targets = np.array(
        [
          features.node_features.feature_list["data_flow_value"]
          .feature[n]
          .int64_list.value[0]
          for n in node_list
        ],
        dtype=np.int32,
      )
      targets_1hot = np.zeros(
        (targets.size, self.node_y_dimensionality), dtype=np.float32
      )
      targets_1hot[np.arange(targets.size), targets] = 1
    except IndexError:
      app.Log(2, "Encoding error")
      return

    self.graph_node_sizes.append(len(node_list))
    self.vocab_ids.append(vocab_ids)
    self.selector_vectors.append(selector_vectors)
    self.targets.append(targets_1hot)

    if len(self.graph_node_sizes) >= self.batch_size:
      return self._Build()

  def EndOfItems(self) -> Optional[BatchData]:
    if len(self.graph_node_sizes):
      return self._Build()
