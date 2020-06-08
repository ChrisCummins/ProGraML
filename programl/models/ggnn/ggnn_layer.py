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
"""The GGNN layer."""
import torch
import torch.nn.functional as F
from torch import nn


class GGNNLayer(nn.Module):
  def __init__(self, hidden_size: int, dropout: float, use_typed_nodes: bool):
    super().__init__()
    self.dropout = dropout

    self.gru = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)

    # Only admits node types 0 and 1 for statements and identifiers.
    self.use_typed_nodes = use_typed_nodes
    if self.use_typed_nodes:
      self.id_gru = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)

  def forward(self, messages, node_states, node_types=None):
    if self.use_typed_nodes:
      assert (
        node_types is not None
      ), "Need to provide node_types <N> if `use_typed_nodes=True`"
      output = torch.zeros_like(node_states, device=node_states.device)
      stmt_mask = node_types == 0
      output[stmt_mask] = self.gru(messages[stmt_mask], node_states[stmt_mask])
      id_mask = node_types == 1
      output[id_mask] = self.id_gru(messages[id_mask], node_states[id_mask])
    else:
      output = self.gru(messages, node_states)

    if self.dropout > 0.0:
      F.dropout(output, p=self.dropout, training=self.training, inplace=True)

    return output
