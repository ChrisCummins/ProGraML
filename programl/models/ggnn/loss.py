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
"""A torch loss module."""
import torch
from torch import nn


class Loss(nn.Module):
  """[Binary] Cross Entropy loss with weighted intermediate loss."""

  def __init__(
    self,
    num_classes: int,
    has_aux_input: bool,
    intermediate_loss_weight: float,
    class_prevalence_weighting: bool,
  ):
    super().__init__()
    self.has_aux_input = has_aux_input
    self.intermediate_loss_weight = intermediate_loss_weight
    self.class_prevalence_weighting = class_prevalence_weighting

    if num_classes == 1:
      self.loss = nn.BCELoss()  # in: (N, *), target: (N, *)
    else:
      # TODO(github.com/ChrisCummins/ProGraML/issues/27): Class labels '-1'
      # don't contribute to the gradient. I was under the impression that we
      # wanted to exploit this fact somewhere. I.e. not predicting labels on
      # nodes that don't constitute branching statements. Let's discuss.

      # obs: no need to normalize if reduction='mean'
      weight = (
        torch.tensor(
          [
            1.0 - self.class_prevalence_weighting,
            self.class_prevalence_weighting,
          ]
        )
        if self.class_prevalence_weighting != 0.5
        else None
      )
      self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=-1)

  def forward(self, inputs, targets):
    """inputs: (predictions) or (predictions, intermediate_predictions)"""
    loss = self.loss(inputs[0], targets)
    if self.has_aux_input:
      loss += self.intermediate_loss_weight * self.loss(inputs[1], targets)
    return loss
