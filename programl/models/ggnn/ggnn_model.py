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
from typing import Optional

import torch
from torch import nn
from torch import optim

from labm8.py import app
from labm8.py import gpu_scheduler
from programl.models.ggnn.aux_readout import AuxiliaryReadout
from programl.models.ggnn.ggnn_proper import GGNNProper
from programl.models.ggnn.loss import Loss
from programl.models.ggnn.metrics import Metrics
from programl.models.ggnn.node_embeddings import NodeEmbeddings

FLAGS = app.FLAGS

app.DEFINE_boolean(
  "block_gpu", True, "Prevent model from hitchhiking on an occupied gpu."
)


class GGNNModel(nn.Module):
  def __init__(
    self,
    node_embeddings: NodeEmbeddings,
    ggnn: GGNNProper,
    aux_readout: Optional[AuxiliaryReadout],
    loss: Loss,
    has_graph_labels: bool,
    test_only: bool,
    learning_rate: float,
    lr_decay_rate: float,
  ):
    super().__init__()

    self.node_embeddings = node_embeddings
    self.ggnn = ggnn
    self.aux_readout = aux_readout
    self.has_graph_labels = has_graph_labels

    self.loss = loss
    self.metrics = Metrics()

    # Move the model to device before making the optimizer.
    if FLAGS.block_gpu:
      self.dev = (
        torch.device("cuda")
        if gpu_scheduler.LockExclusiveProcessGpuAccess()
        else torch.device("cpu")
      )
    else:
      self.dev = torch.device("cuda")

    self.to(self.dev)

    # Not instantiating the optimizer should save 2 x #model_params of GPU
    # memory, bc. Adam carries two momentum params per trainable model
    # parameter.
    if test_only:
      self.opt = None
      self.eval()
      self.scheduler = None
    else:
      self.opt = self.GetOptimizer(learning_rate)
      self.scheduler = self.GetLRScheduler(self.opt, lr_decay_rate)

  @property
  def learning_rate(self) -> float:
    if self.scheduler is None:
      return 0.0
    else:
      return self.scheduler.get_lr()[0]

  def GetOptimizer(self, learning_rate: float):
    return optim.AdamW(self.parameters(), lr=learning_rate)

  def GetLRScheduler(self, optimizer, gamma):
    """Exponential decay LR schedule. at each schedule.step(), the LR is
    multiplied by gamma."""
    return optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)

  def forward(
    self,
    vocab_ids,
    labels,
    edge_lists,
    selector_ids=None,
    pos_lists=None,
    num_graphs=None,
    graph_nodes_list=None,
    node_types=None,
    aux_in=None,
  ):
    raw_in = self.node_embeddings(vocab_ids, selector_ids)
    # self.ggnn might change raw_in inplace, so use the two outputs instead!
    raw_out, raw_in, *unroll_stats = self.ggnn(
      edge_lists, raw_in, pos_lists, node_types
    )

    if self.has_graph_labels:
      assert (
        graph_nodes_list is not None and num_graphs is not None
      ), "has_graph_labels requires graph_nodes_list and num_graphs tensors."

    nodewise_readout, graphwise_readout = self.ggnn.readout(
      raw_in, raw_out, graph_nodes_list=graph_nodes_list, num_graphs=num_graphs
    )

    logits = graphwise_readout if self.has_graph_labels else nodewise_readout

    if self.aux_readout:
      logits, graphwise_readout = self.aux_readout(logits, aux_in)

    # accuracy, pred_targets, correct, targets
    # metrics_tuple = self.metrics(logits, labels)
    targets = labels.argmax(dim=1)

    outputs = (targets, logits, graphwise_readout,) + tuple(unroll_stats)

    return outputs

  @property
  def trainable_parameter_count(self) -> int:
    """Compute the trainable parameter count in this module and its children."""
    return sum(
      param.numel()
      for param in self.parameters(recurse=True)
      if param.requires_grad
    )
