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
"""This module defines data structures for model results for a mini-batch."""
from typing import NamedTuple
from typing import Optional

import numpy as np
import sklearn.metrics

from labm8.py import app

app.DEFINE_string(
  "batch_results_averaging_method",
  "weighted",
  "Selects the averaging method to use when computing recall/precision/F1 "
  "scores. See <https://scikit-learn.org/stable/modules/generated/sklearn"
  ".metrics.f1_score.html>",
)

FLAGS = app.FLAGS


class BatchResults(NamedTuple):
  """The results of running a batch through a model.

  Don't instantiate this tuple directly, use Results.Create().
  """

  targets: np.array
  predictions: np.array
  # The number of model iterations to compute the final results. This is used
  # by iterative models such as message passing networks.
  iteration_count: int
  # For iterative models, this indicates whether the state of the model at
  # iteration_count had converged on a solution.
  model_converged: bool
  # The learning rate and loss of models, if applicable.
  learning_rate: Optional[float]
  loss: Optional[float]
  # Batch-level average performance metrics.
  accuracy: float
  precision: float
  recall: float
  f1: float
  # The confusion matrix.
  confusion_matrix: np.array

  @property
  def has_learning_rate(self) -> bool:
    return self.learning_rate is not None

  @property
  def has_loss(self) -> bool:
    return self.loss is not None

  @property
  def target_count(self) -> int:
    """Get the number of targets in the batch.

    E.g. the number of graphs in the batch for graph-level models, the number
    of nodes in the batch for node-level models, etc.
    """
    return self.targets.shape[0]

  def __repr__(self) -> str:
    return (
      f"loss={self.loss:.6f}, "
      f"acc={self.accuracy:.3%}%, "
      f"prec={self.precision:.4f}, "
      f"rec={self.recall:.4f}, "
      f"f1={self.f1:.4f}"
    )

  def __eq__(self, rhs: "BatchResults"):
    """Batch results are compared by model accuracy."""
    return self.accuracy == rhs.accuracy

  def __gt__(self, rhs: "BatchResults"):
    """Batch results are compared by model accuracy."""
    return self.accuracy > rhs.accuracy

  @classmethod
  def Create(
    cls,
    targets: np.array,
    predictions: np.array,
    iteration_count: int = 0,
    model_converged: bool = False,
    learning_rate: Optional[float] = None,
    loss: Optional[float] = None,
  ):
    """Construct a results instance from 1-hot targets and predictions.

    This is the preferred means of construct a BatchResults instance as it takes
    care of evaluating all of the metrics for you. The behavior of metrics
    calculation is dependent on the --batch_results_averaging_method flag.

    Args:
      targets: An array of 1-hot target vectors with
        shape (y_count, y_dimensionality), dtype int32.
      predictions: An array of 1-hot prediction vectors with
        shape (y_count, y_dimensionality), dtype int32.
      iteration_count: For iterative models, the number of model iterations to
        compute the final result.
      model_converged: For iterative models, whether model converged.
      learning_rate: The model learning rate, if applicable.
      loss: The model loss, if applicable.

    Returns:
      A Results instance.
    """
    if targets.shape != predictions.shape:
      raise TypeError(
        f"Expected model to produce targets with shape {targets.shape} but "
        f"instead received predictions with shape {predictions.shape}"
      )

    y_dimensionality = targets.shape[1]
    if y_dimensionality < 2:
      raise TypeError(
        f"Expected label dimensionality > 1, received {y_dimensionality}"
      )

    # Create dense arrays of shape (target_count).
    true_y = np.argmax(targets, axis=1)
    pred_y = np.argmax(predictions, axis=1)

    # NOTE(github.com/ChrisCummins/ProGraML/issues/22): This assumes that
    # labels use the values [0,...n).
    labels = np.arange(y_dimensionality, dtype=np.int64)

    return cls(
      targets=targets,
      predictions=predictions,
      iteration_count=iteration_count,
      model_converged=model_converged,
      learning_rate=learning_rate,
      loss=loss,
      accuracy=sklearn.metrics.accuracy_score(true_y, pred_y),
      precision=sklearn.metrics.precision_score(
        true_y,
        pred_y,
        labels=labels,
        average=FLAGS.batch_results_averaging_method,
      ),
      recall=sklearn.metrics.recall_score(
        true_y,
        pred_y,
        labels=labels,
        average=FLAGS.batch_results_averaging_method,
      ),
      f1=sklearn.metrics.f1_score(
        true_y,
        pred_y,
        labels=labels,
        average=FLAGS.batch_results_averaging_method,
      ),
      confusion_matrix=BuildConfusionMatrix(
        targets=true_y, predictions=pred_y, num_classes=y_dimensionality
      ),
    )


def BuildConfusionMatrix(
  targets: np.array, predictions: np.array, num_classes: int
) -> np.array:
  """Build a confusion matrix.

  Args:
    targets: A list of target classes, dtype int32.
    predictions: A list of predicted classes of the same length as targets,
      dtype float32.
    num_classes: The number of classes. All values in targets and predictions
      lists must be in the range 0 <= x <= num_classes.

  Returns:
    A matrix of shape [num_classes, num_classes] where the rows indicate true
    target class, the columns indicate predicted target class, and the element
    values are the number of instances of this type in the batch.
  """
  if targets.shape != predictions.shape:
    raise TypeError(
      f"Predictions shape {predictions.shape} must match targets "
      f"shape {targets.shape}"
    )

  confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
  for target, prediction in zip(targets, predictions):
    confusion_matrix[target][prediction] += 1

  return confusion_matrix
