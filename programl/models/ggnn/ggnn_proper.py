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
"""The GGNN model."""
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from programl.models.ggnn.ggnn_layer import GGNNLayer
from programl.models.ggnn.messaging_layer import MessagingLayer
from programl.models.ggnn.readout import Readout


class GGNNProper(nn.Module):
  def __init__(
    self,
    readout: Readout,
    text_embedding_dimensionality: int,
    selector_embedding_dimensionality: int,
    forward_edge_type_count: int,
    unroll_strategy: str,
    use_backward_edges: bool,
    layer_timesteps: List[int],
    use_position_embeddings: bool,
    use_edge_bias: bool,
    msg_mean_aggregation: bool,
    max_timesteps: int,
    unroll_convergence_threshold: float,
    unroll_convergence_steps: int,
    graph_state_dropout: float,
    edge_weight_dropout: float,
    edge_position_max: int = 1024,
  ):
    super().__init__()
    self.readout = readout
    self.use_backward_edges = use_backward_edges
    self.layer_timesteps = layer_timesteps
    self.position_embeddings = use_position_embeddings
    self.msg_mean_aggregation = msg_mean_aggregation
    self.edge_position_max = edge_position_max

    # optional eval time unrolling parameter
    self.unroll_strategy = unroll_strategy
    self.max_timesteps = max_timesteps
    self.unroll_convergence_threshold = unroll_convergence_threshold
    self.unroll_convergence_steps = unroll_convergence_steps

    # Make readout available to label_convergence tests in GGNN Proper (at
    # runtime).
    assert (
      self.unroll_strategy != "label_convergence"
      or len(self.layer_timesteps) == 1
    ), (
      "Label convergence only supports one-layer GGNNs, but "
      f"{len(self.layer_timesteps)} are configured in "
      f"layer_timesteps: {self.layer_timesteps}"
    )

    # Message and update layers.
    self.message = nn.ModuleList()
    for i in range(len(self.layer_timesteps)):
      self.message.append(
        MessagingLayer(
          text_embedding_dimensionality=text_embedding_dimensionality,
          selector_embedding_dimensionality=selector_embedding_dimensionality,
          forward_edge_type_count=forward_edge_type_count,
          use_backward_edges=use_backward_edges,
          use_position_embeddings=use_position_embeddings,
          use_edge_bias=use_edge_bias,
          edge_weight_dropout=edge_weight_dropout,
          edge_position_max=edge_position_max,
        )
      )

    self.update = nn.ModuleList()

    # NB: Maybe decouple hidden GRU size: make hidden GRU size larger and edge
    # size non-square instead? Or implement stacking GRU layers between message
    # passing steps.
    hidden_size = (
      text_embedding_dimensionality + selector_embedding_dimensionality
    )
    for i in range(len(self.layer_timesteps)):
      # NB: Node types are currently unused as the only way of differentiating
      # the type of a node is by looking at the text embedding value, but may
      # be added in the future.
      self.update.append(
        GGNNLayer(
          hidden_size=hidden_size,
          dropout=graph_state_dropout,
          use_typed_nodes=False,
        )
      )

  def forward(
    self, edge_lists, node_states, pos_lists=None, node_types=None,
  ):
    old_node_states = node_states.clone()

    if self.use_backward_edges:
      back_edge_lists = [x.flip([1]) for x in edge_lists]
      edge_lists.extend(back_edge_lists)
      # For backward edges we keep the positions of the forward edge.
      if self.position_embeddings:
        pos_lists.extend(pos_lists)

    if self.unroll_strategy == "label_convergence":
      node_states, unroll_steps, converged = self.label_convergence_forward(
        edge_lists,
        node_states,
        pos_lists,
        node_types,
        initial_node_states=old_node_states,
      )
      return node_states, old_node_states, unroll_steps, converged

    # Compute incoming message divisor.
    if self.msg_mean_aggregation:
      bincount = torch.zeros(
        node_states.size()[0], dtype=torch.long, device=node_states.device
      )
      for edge_list in edge_lists:
        edge_targets = edge_list[:, 1]
        edge_bincount = edge_targets.bincount(minlength=node_states.size()[0])
        bincount += edge_bincount
      # Avoid division by zero for lonely nodes.
      bincount[bincount == 0] = 1
      msg_mean_divisor = bincount.float().unsqueeze_(1)
    else:
      msg_mean_divisor = None

    # Clamp the position lists in the range [0,edge_position_max) to match
    # the pre-computed position embeddings table.
    if pos_lists:
      for pos_list in pos_lists:
        pos_list.clamp_(0, self.edge_position_max - 1)

    for (layer_idx, num_timesteps) in enumerate(self.layer_timesteps):
      for t in range(num_timesteps):
        messages = self.message[layer_idx](
          edge_lists,
          node_states,
          msg_mean_divisor=msg_mean_divisor,
          pos_lists=pos_lists,
        )
        node_states = self.update[layer_idx](messages, node_states, node_types)

    return node_states, old_node_states

  def label_convergence_forward(
    self, edge_lists, node_states, pos_lists, node_types, initial_node_states
  ):
    stable_steps = 0
    old_tentative_labels = self.ComputeTentativeLabels(
      initial_node_states, node_states
    )

    for i in range(self.max_timesteps):
      messages = self.message[0](edge_lists, node_states, pos_lists)
      node_states = self.update[0](messages, node_states, node_types)
      new_tentative_labels = self.ComputeTentativeLabels(
        initial_node_states, node_states
      )
      i += 1

      # Return the new node states if their predictions match the old node
      # states' predictions. It doesn't matter during testing since the
      # predictions are the same anyway.
      stability = (
        (new_tentative_labels == old_tentative_labels)
        .to(dtype=torch.get_default_dtype())
        .mean()
      )
      if stability >= self.unroll_convergence_threshold:
        stable_steps += 1
        if stable_steps >= self.unroll_convergence_steps:
          break

      old_tentative_labels = new_tentative_labels

    return node_states, i, False

  def ComputeTentativeLabels(self, initial_node_states, node_states):
    logits, _ = self.readout(initial_node_states, node_states)
    preds = F.softmax(logits, dim=1)
    predicted_labels = torch.argmax(preds, dim=1)
    return predicted_labels
