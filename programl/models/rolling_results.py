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
"""This module defines a rolling batch results data structure."""
import time
from typing import Optional

from programl.models.batch_data import BatchData
from programl.models.batch_results import BatchResults
from programl.proto import epoch_pb2


class RollingResults(object):
  """Maintain weighted rolling averages across batches."""

  def __init__(self):
    self.weight_sum = 0
    self.batch_count = 0
    self.graph_count = 0
    self.target_count = 0
    self.weighted_iteration_count_sum = 0
    self.weighted_model_converged_sum = 0
    self.has_learning_rate = False
    self.weighted_learning_rate_sum = 0
    self.has_loss = False
    self.weighted_loss_sum = 0
    self.weighted_accuracy_sum = 0
    self.weighted_precision_sum = 0
    self.weighted_recall_sum = 0
    self.weighted_f1_sum = 0
    self.confusion_matrix = None
    self.start_time = time.time()

  def Update(
    self, data: BatchData, results: BatchResults, weight: Optional[float] = None
  ) -> None:
    """Update the rolling results with a new batch.

    Args:
      data: The batch data used to produce the results.
      results: The batch results to update the current state with.
      weight: A weight to assign to weighted sums. E.g. to weight results
        across all targets, use weight=results.target_count. To weight across
        targets, use weight=batch.target_count. To weight across
        graphs, use weight=batch.graph_count. By default, weight by target
        count.
    """
    if weight is None:
      weight = results.target_count

    self.weight_sum += weight
    self.batch_count += 1
    self.graph_count += data.graph_count
    self.target_count += results.target_count
    self.weighted_iteration_count_sum += results.iteration_count * weight
    self.weighted_model_converged_sum += (
      weight if results.model_converged else 0
    )
    if results.has_learning_rate:
      self.has_learning_rate = True
      self.weighted_learning_rate_sum += results.learning_rate * weight
    if results.has_loss:
      self.has_loss = True
      self.weighted_loss_sum += results.loss * weight
    self.weighted_accuracy_sum += results.accuracy * weight
    self.weighted_precision_sum += results.precision * weight
    self.weighted_recall_sum += results.recall * weight
    self.weighted_f1_sum += results.f1 * weight
    if self.confusion_matrix is None:
      self.confusion_matrix = results.confusion_matrix
    else:
      self.confusion_matrix += results.confusion_matrix

  @property
  def iteration_count(self) -> float:
    return self.weighted_iteration_count_sum / max(self.weight_sum, 1)

  @property
  def model_converged(self) -> float:
    return self.weighted_model_converged_sum / max(self.weight_sum, 1)

  @property
  def learning_rate(self) -> Optional[float]:
    if self.has_learning_rate:
      return self.weighted_learning_rate_sum / max(self.weight_sum, 1)

  @property
  def loss(self) -> Optional[float]:
    if self.has_loss:
      return self.weighted_loss_sum / max(self.weight_sum, 1)

  @property
  def accuracy(self) -> float:
    return self.weighted_accuracy_sum / max(self.weight_sum, 1)

  @property
  def precision(self) -> float:
    return self.weighted_precision_sum / max(self.weight_sum, 1)

  @property
  def recall(self) -> float:
    return self.weighted_recall_sum / max(self.weight_sum, 1)

  @property
  def f1(self) -> float:
    return self.weighted_f1_sum / max(self.weight_sum, 1)

  def ToEpochResults(self) -> epoch_pb2.EpochResults:
    return epoch_pb2.EpochResults(
      graph_count=self.graph_count,
      batch_count=self.batch_count,
      target_count=self.target_count,
      mean_iteration_count=self.iteration_count,
      mean_model_converged=self.model_converged,
      mean_learning_rate=self.learning_rate,
      mean_loss=self.loss,
      mean_accuracy=self.accuracy,
      mean_precision=self.precision,
      mean_recall=self.recall,
      mean_f1=self.f1,
      walltime_seconds=time.time() - self.start_time,
      confusion_matrix=epoch_pb2.ConfusionMatrix(
        column=[
          epoch_pb2.ConfusionMatrixRow(row=column)
          for column in self.confusion_matrix.T.tolist()
        ],
      ),
    )

  def __repr__(self) -> str:
    return (
      f"batch={self.batch_count}, "
      f"graphs={self.graph_count}, "
      f"loss={self.loss:.6f}, "
      f"acc={self.accuracy:.3%}%, "
      f"prec={self.precision:.4f}, "
      f"rec={self.recall:.4f}, "
      f"f1={self.f1:.4f}"
    )
