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
"""This module defines an iterator over batches, divided into epochs."""
from typing import Iterable

from programl.models.base_batch_builder import BaseBatchBuilder
from programl.models.batch_data import BatchData


def _LimitBatchesToTargetGraphCount(
  batches: Iterable[BatchData], target_graph_count: int
):
  """Return an iterator over the input batches which terminates once the
  target number of graphs has been reached.
  """
  graph_count = 0
  for batch in batches:
    yield batch
    graph_count += batch.graph_count
    if graph_count >= target_graph_count:
      break


def EpochBatchIterator(
  batch_builder: BaseBatchBuilder,
  target_graph_counts: Iterable[int],
  start_graph_count: int = 0,
):
  """Divide a sequence of batches into chunks of the given graph counts.

  Args:
    batch_builder: A batch builder.
    target_graph_counts: A list of target graph counts.

  Returns:
    A iterable sequence of <target_graph_count, total_graph_count, batches>
    tuples of length len(target_graph_counts).
  """
  total_graph_count = start_graph_count
  for target_graph_count in target_graph_counts:
    total_graph_count += target_graph_count
    batches = _LimitBatchesToTargetGraphCount(batch_builder, target_graph_count)
    yield target_graph_count, total_graph_count, batches
  batch_builder.Stop()
