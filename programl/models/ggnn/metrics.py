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
"""GGNN performance metrics."""
import torch
from torch import nn


class Metrics(nn.Module):
  """Common metrics and info for inspection of results.
  Args:
  logits, labels
  Returns:
  (accuracy, pred_targets, correct_preds, targets)"""

  def __init__(self):
    super().__init__()

  def forward(self, logits, labels):
    # be flexible with 1hot labels vs indices
    if len(labels.size()) == 2:
      targets = labels.argmax(dim=1)
    elif len(labels.size()) == 1:
      targets = labels
    else:
      raise ValueError(
        f"labels={labels.size()} tensor is is neither 1 nor 2-dimensional. :/"
      )

    pred_targets = logits.argmax(dim=1)
    correct_preds = targets.eq(pred_targets).float()
    accuracy = torch.mean(correct_preds)
    return accuracy, logits, correct_preds, targets
