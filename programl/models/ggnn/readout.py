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
"""The readout torch module."""
import torch
from torch import nn

from programl.models.ggnn.linear_net import LinearNet


class Readout(nn.Module):
  """aka GatedRegression. See Eq. 4 in Gilmer et al. 2017 MPNN."""

  def __init__(
    self,
    num_classes: int,
    has_graph_labels: bool,
    hidden_size: int,
    output_dropout: float,
  ):
    super().__init__()
    self.has_graph_labels = has_graph_labels
    self.num_classes = num_classes

    self.regression_gate = LinearNet(
      2 * hidden_size, self.num_classes, dropout=output_dropout,
    )
    self.regression_transform = LinearNet(
      hidden_size, self.num_classes, dropout=output_dropout,
    )

  def forward(
    self, raw_node_in, raw_node_out, graph_nodes_list=None, num_graphs=None
  ):
    gate_input = torch.cat((raw_node_in, raw_node_out), dim=-1)
    gating = torch.sigmoid(self.regression_gate(gate_input))
    nodewise_readout = gating * self.regression_transform(raw_node_out)

    graph_readout = None
    if self.has_graph_labels:
      assert (
        graph_nodes_list is not None and num_graphs is not None
      ), "has_graph_labels requires graph_nodes_list and num_graphs tensors."
      # aggregate via sums over graphs
      device = raw_node_out.device
      graph_readout = torch.zeros(num_graphs, self.num_classes, device=device)
      graph_readout.index_add_(
        dim=0, index=graph_nodes_list, source=nodewise_readout
      )
    return nodewise_readout, graph_readout
