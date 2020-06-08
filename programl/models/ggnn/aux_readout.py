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
"""The auxiliary readout torch module."""
import torch
from torch import nn


class AuxiliaryReadout(nn.Module):
  """Produces per-graph predictions by combining
    the per-graph predictions with auxiliary features"""

  # TODO(github.com/ChrisCummins/ProGraML/issues/27): I don't like that we only
  # introduce the global features AFTER the per node predictions have been made
  # and not while we do those! This is limiting the expressivity of the model.
  def __init__(
    self,
    num_classes: int,
    log1p_graph_x: bool,
    output_dropout: float,
    graph_x_layer_size: int,
    graph_x_dimensionality: int,
  ):
    super().__init__()
    self.log1p_graph_x = log1p_graph_x
    self.batch_norm = nn.BatchNorm1d(num_classes + graph_x_dimensionality)
    self.feed_forward = nn.Sequential(
      nn.Linear(num_classes + graph_x_dimensionality, graph_x_layer_size,),
      nn.ReLU(),
      nn.Dropout(1 - output_dropout),
      nn.Linear(graph_x_layer_size, num_classes),
    )

  def forward(self, graph_features, auxiliary_features):
    assert (
      graph_features.size()[0] == auxiliary_features.size()[0]
    ), "every graph needs aux_features. Dimension mismatch."
    if self.log1p_graph_x:
      auxiliary_features.log1p_()

    aggregate_features = torch.cat((graph_features, auxiliary_features), dim=1)

    normed_features = self.batch_norm(aggregate_features)
    out = self.feed_forward(normed_features)
    return out, graph_features
