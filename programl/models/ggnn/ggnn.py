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
"""A gated graph neural network classifier."""
import typing
from typing import Dict
from typing import Tuple

import numpy as np
import torch
from torch import nn

from labm8.py import app
from labm8.py.progress import NullContext
from labm8.py.progress import ProgressContext
from programl.graph.format.py.graph_tuple import GraphTuple
from programl.models.batch_data import BatchData
from programl.models.batch_results import BatchResults
from programl.models.ggnn.aux_readout import AuxiliaryReadout
from programl.models.ggnn.ggnn_batch import GgnnBatchData
from programl.models.ggnn.ggnn_model import GGNNModel
from programl.models.ggnn.ggnn_proper import GGNNProper
from programl.models.ggnn.loss import Loss
from programl.models.ggnn.node_embeddings import NodeEmbeddings
from programl.models.ggnn.readout import Readout
from programl.models.model import Model
from programl.proto import epoch_pb2

FLAGS = app.FLAGS

# Graph unrolling flags.
app.DEFINE_string(
  "unroll_strategy",
  "constant",
  "The unroll strategy to use. One of: "
  "{none, constant, edge_count, data_flow_max_steps, label_convergence} "
  "constant: Unroll by a constant number of steps.",
)
app.DEFINE_float(
  "unroll_convergence_threshold",
  0.995,
  "convergence interval: fraction of labels that need to be stable",
)
app.DEFINE_integer(
  "unroll_convergence_steps",
  1,
  "required number of consecutive steps within the convergence interval",
)
app.DEFINE_integer(
  "unroll_max_steps",
  1000,
  "The maximum number of iterations to attempt to reach label convergence. "
  "No effect when --unroll_strategy is not label_convergence.",
)

app.DEFINE_list(
  "layer_timesteps",
  ["30"],
  "A list of layers, and the number of steps for each layer.",
)
app.DEFINE_float("learning_rate", 0.00025, "The initial learning rate.")
app.DEFINE_float(
  "lr_decay_rate",
  0.95,
  "Learning rate decay; multiplicative factor for lr after every epoch.",
)
app.DEFINE_integer(
  "lr_decay_steps", 1000, "Steps until next LR decay.",
)

app.DEFINE_float("clip_gradient_norm", 0.0, "Clip gradients to L-2 norm.")

# Edge and message flags.
app.DEFINE_boolean("use_backward_edges", True, "Add backward edges.")
app.DEFINE_boolean("use_edge_bias", True, "")
app.DEFINE_boolean(
  "msg_mean_aggregation",
  True,
  "If true, normalize incoming messages by the number of incoming messages.",
)
# Embeddings options.
app.DEFINE_string(
  "text_embedding_type",
  "random",
  "The type of node embeddings to use. One of "
  "{constant_zero, constant_random, random}.",
)
app.DEFINE_integer(
  "text_embedding_dimensionality",
  32,
  "The dimensionality of node text embeddings.",
)
app.DEFINE_boolean(
  "use_position_embeddings",
  True,
  "Whether to use position embeddings as signals for edge order. "
  "False may be a good default for small datasets.",
)
app.DEFINE_float(
  "selector_embedding_value",
  50,
  "The value used for the positive class in the 1-hot selector embedding "
  "vectors. Has no effect when selector embeddings are not used.",
)
# Loss.
app.DEFINE_float(
  "intermediate_loss_weight",
  0.2,
  "The true loss is computed as loss + factor * intermediate_loss. Only "
  "applicable when graph_x_dimensionality > 0.",
)
# Graph features flags.
app.DEFINE_integer(
  "graph_x_layer_size",
  32,
  "Size for MLP that combines graph_features and aux_in features",
)
app.DEFINE_boolean(
  "log1p_graph_x",
  True,
  "If set, apply a log(x + 1) transformation to incoming auxiliary graph-level features.",
)
# Dropout flags.
app.DEFINE_float(
  "graph_state_dropout", 0.2, "Graph state dropout rate.",
)
app.DEFINE_float(
  "edge_weight_dropout", 0.0, "Edge weight dropout rate.",
)
app.DEFINE_float(
  "output_layer_dropout", 0.0, "Dropout rate on the output layer.",
)

# Loss flags
app.DEFINE_float(
  "loss_weighting",
  0.5,
  "Weight loss contribution in batch by inverse class prevalence"
  "to mitigate class imbalance in the dataset."
  "currently implemented as a float w --> [1 - w, w] weighting for 2 class problems"
  "this flag will crash the program if set to trum and num_classes != 2.",
)

# not implemented yet
# app.DEFINE_boolean("loss_masking",
#                   False,
#                   "Mask loss computation on nodes chosen at random from each class"
#                   "such that balanced class distributions (per batch) remain")

# Debug flags.
app.DEFINE_boolean(
  "debug_nan_hooks",
  False,
  "If set, add hooks to model execution to trap on NaNs.",
)


def NanHook(self, _, output):
  """Checks return values of any forward() function for NaN"""
  if not isinstance(output, tuple):
    outputs = [output]
  else:
    outputs = output

  for i, out in enumerate(outputs):
    nan_mask = torch.isnan(out)
    if nan_mask.any():
      print("In", self.__class__.__name__)
      raise RuntimeError(
        f"Found NAN in output {i} at indices: ",
        nan_mask.nonzero(),
        "where:",
        out[nan_mask.nonzero()[:, 0].unique(sorted=True)],
      )


class Ggnn(Model):
  """A gated graph neural network."""

  def __init__(
    self,
    vocabulary: Dict[str, int],
    node_y_dimensionality: int,
    graph_y_dimensionality: int,
    graph_x_dimensionality: int,
    use_selector_embeddings: bool,
    test_only: bool = False,
    name: str = "ggnn",
  ):
    """Constructor."""
    super(Ggnn, self).__init__(
      name=name, vocabulary=vocabulary, test_only=test_only
    )

    # Graph attribute shapes.
    self.node_y_dimensionality = node_y_dimensionality
    self.graph_x_dimensionality = graph_x_dimensionality
    self.graph_y_dimensionality = graph_y_dimensionality
    self.node_selector_dimensionality = 2 if use_selector_embeddings else 0

    if graph_y_dimensionality and node_y_dimensionality:
      raise ValueError(
        "Cannot use both node and graph-level classification at"
        "the same time."
      )

    node_embeddings = NodeEmbeddings(
      node_embeddings_type=FLAGS.text_embedding_type,
      use_selector_embeddings=self.node_selector_dimensionality > 0,
      selector_embedding_value=FLAGS.selector_embedding_value,
      embedding_shape=(
        # Add one to the vocabulary size to account for the out-of-vocab token.
        len(vocabulary) + 1,
        FLAGS.text_embedding_dimensionality,
      ),
    )

    self.clip_gradient_norm = FLAGS.clip_gradient_norm

    if self.has_aux_input:
      aux_readout = AuxiliaryReadout(
        num_classes=self.num_classes,
        log1p_graph_x=FLAGS.log1p_graph_x,
        output_dropout=FLAGS.output_layer_dropout,
        graph_x_layer_size=FLAGS.graph_x_layer_size,
        graph_x_dimensionality=self.graph_x_dimensionality,
      )
    else:
      aux_readout = None

    self.model = GGNNModel(
      node_embeddings=node_embeddings,
      ggnn=GGNNProper(
        readout=Readout(
          num_classes=self.num_classes,
          has_graph_labels=self.has_graph_labels,
          hidden_size=node_embeddings.embedding_dimensionality,
          output_dropout=FLAGS.output_layer_dropout,
        ),
        text_embedding_dimensionality=node_embeddings.text_embedding_dimensionality,
        selector_embedding_dimensionality=node_embeddings.selector_embedding_dimensionality,
        forward_edge_type_count=3,
        unroll_strategy=FLAGS.unroll_strategy,
        use_backward_edges=FLAGS.use_backward_edges,
        layer_timesteps=self.layer_timesteps,
        use_position_embeddings=FLAGS.use_position_embeddings,
        use_edge_bias=FLAGS.use_edge_bias,
        msg_mean_aggregation=FLAGS.msg_mean_aggregation,
        max_timesteps=FLAGS.unroll_max_steps,
        unroll_convergence_threshold=FLAGS.unroll_convergence_threshold,
        unroll_convergence_steps=FLAGS.unroll_convergence_steps,
        graph_state_dropout=FLAGS.graph_state_dropout,
        edge_weight_dropout=FLAGS.edge_weight_dropout,
      ),
      aux_readout=aux_readout,
      loss=Loss(
        num_classes=self.num_classes,
        has_aux_input=self.has_aux_input,
        intermediate_loss_weight=FLAGS.intermediate_loss_weight,
        class_prevalence_weighting=FLAGS.loss_weighting,
      ),
      has_graph_labels=self.has_graph_labels,
      test_only=self.test_only,
      learning_rate=FLAGS.learning_rate,
      lr_decay_rate=FLAGS.lr_decay_rate,
    )

    if FLAGS.debug_nan_hooks:
      for submodule in self.model.modules():
        submodule.register_forward_hook(NanHook)

  @property
  def opt_step_count(self) -> int:
    step = self.model.opt.state[self.model.opt.param_groups[0]["params"][0]][
      "step"
    ]
    return step

  @property
  def num_classes(self) -> int:
    return self.node_y_dimensionality or self.graph_y_dimensionality

  @property
  def has_graph_labels(self) -> bool:
    return self.graph_y_dimensionality > 0

  @property
  def has_aux_input(self) -> bool:
    return self.graph_x_dimensionality > 0

  @property
  def message_passing_step_count(self) -> int:
    return self.layer_timesteps.sum()

  @property
  def layer_timesteps(self) -> np.array:
    return np.array([int(x) for x in FLAGS.layer_timesteps])

  @property
  def trainable_parameter_count(self) -> int:
    """Compute the trainable parameter count in this module and its children."""
    return self.model.trainable_parameter_count

  def PrepareModelInputs(
    self, epoch_type: epoch_pb2.EpochType, batch: BatchData
  ) -> Dict[str, torch.Tensor]:
    """RunBatch() helper method to prepare inputs to model.

    Args:
      epoch_type: The type of epoch the model is performing.
      batch: A batch of data to prepare inputs from:

    Returns:
      A dictionary of model inputs.
    """
    del epoch_type
    batch_data: GgnnBatchData = batch.model_data
    graph_tuple: GraphTuple = batch_data.graph_tuple

    # Batch to model-inputs. torch.from_numpy() shares memory with numpy.
    # TODO(github.com/ChrisCummins/ProGraML/issues/27): maybe we can save
    # memory copies in the training loop if we can turn the data into the
    # required types (np.int64 and np.float32) once they come off the network
    # from the database, where smaller i/o size (int32) is more important.
    vocab_ids = torch.from_numpy(batch_data.vocab_ids).to(
      self.model.dev, torch.long
    )
    selector_ids = torch.from_numpy(batch_data.selector_ids).to(
      self.model.dev, torch.long
    )

    # TODO(github.com/ChrisCummins/ProGraML/issues/27): Consider performing
    # 1-hot expansion of node labels on device to save on data transfer.
    labels = torch.from_numpy(batch_data.node_labels).to(
      self.model.dev, torch.long
    )
    edge_lists = [
      torch.from_numpy(x).to(self.model.dev, torch.long)
      for x in graph_tuple.adjacencies
    ]
    edge_positions = [
      torch.from_numpy(x).to(self.model.dev, torch.long)
      for x in graph_tuple.edge_positions
    ]
    model_inputs = {
      "vocab_ids": vocab_ids,
      "selector_ids": selector_ids,
      "labels": labels,
      "edge_lists": edge_lists,
      "pos_lists": edge_positions,
    }

    # maybe fetch more inputs.
    # TODO:
    # if graph_tuple.has_graph_y:
    #   assert (
    #     epoch_type != epoch_pb2.TRAIN
    #     or graph_tuple.graph_tuple_count > 1
    #   ), f"graph_count is {graph_tuple.graph_tuple_count}"
    #   num_graphs = torch.tensor(graph_tuple.graph_tuple_count).to(
    #     self.model.dev, torch.long
    #   )
    #   graph_nodes_list = torch.from_numpy(
    #     graph_tuple.disjoint_nodes_list
    #   ).to(self.model.dev, torch.long)
    #
    #   aux_in = torch.from_numpy(graph_tuple.graph_x).to(
    #     self.model.dev, torch.get_default_dtype()
    #   )
    #   model_inputs.update(
    #     {
    #       "num_graphs": num_graphs,
    #       "graph_nodes_list": graph_nodes_list,
    #       "aux_in": aux_in,
    #     }
    #   )

    return model_inputs

  def RunBatch(
    self,
    epoch_type: epoch_pb2.EpochType,
    batch: BatchData,
    ctx: ProgressContext = NullContext,
  ) -> BatchResults:
    """Process a mini-batch of data through the GGNN.

    Args:
      epoch_type: The type of epoch being run.
      batch: The batch data returned by MakeBatch().
      ctx: A logging context.

    Returns:
      A batch results instance.
    """
    model_inputs = self.PrepareModelInputs(epoch_type, batch)
    unroll_steps = np.array(
      GetUnrollSteps(epoch_type, batch, FLAGS.unroll_strategy), dtype=np.int64,
    )

    # Set the model into the correct mode and feed through the batch data.
    if epoch_type == epoch_pb2.TRAIN:
      if not self.model.training:
        self.model.train()
      outputs = self.model(**model_inputs)
    else:
      if self.model.training:
        self.model.eval()
        self.model.opt.zero_grad()
      # Inference only, don't trace the computation graph.
      with torch.no_grad():
        outputs = self.model(**model_inputs)

    (targets, logits, graph_features, *unroll_stats,) = outputs

    loss = self.model.loss((logits, graph_features), targets)

    if epoch_type == epoch_pb2.TRAIN:
      loss.backward()
      # TODO(github.com/ChrisCummins/ProGraML/issues/27): NB, pytorch clips by
      # norm of the gradient of the model, while tf clips by norm of the grad
      # of each tensor separately. Therefore we change default from 1.0 to 6.0.
      # TODO(github.com/ChrisCummins/ProGraML/issues/27): Anyway: Gradients
      # shouldn't really be clipped if not necessary?
      if self.clip_gradient_norm > 0.0:
        nn.utils.clip_gradient_norm_(
          self.model.parameters(), self.clip_gradient_norm
        )
      self.model.opt.step()
      self.model.opt.zero_grad()
      # check for LR scheduler stepping
      if self.opt_step_count % FLAGS.lr_decay_steps == 0:
        # If scheduler exists, then step it after every epoch
        if self.model.scheduler is not None:
          old_learning_rate = self.model.learning_rate
          self.model.scheduler.step()
          app.Log(
            1,
            "LR Scheduler step. New learning rate is %s (was %s)",
            self.model.learning_rate,
            old_learning_rate,
          )

    model_converged = unroll_stats[1] if unroll_stats else False
    iteration_count = unroll_stats[0] if unroll_stats else unroll_steps

    return BatchResults.Create(
      targets=batch.model_data.node_labels,
      predictions=logits.detach().cpu().numpy(),
      model_converged=model_converged,
      learning_rate=self.model.learning_rate,
      iteration_count=iteration_count,
      loss=loss.item(),
    )

  def GetModelData(self) -> typing.Any:
    return {
      "model_state_dict": self.model.state_dict(),
      "optimizer_state_dict": self.model.opt.state_dict(),
      "scheduler_state_dict": self.model.scheduler.state_dict(),
    }

  def LoadModelData(self, data_to_load: typing.Any) -> None:
    self.model.load_state_dict(data_to_load["model_state_dict"])
    # only restore opt if needed. opt should be None o/w.
    if not self.test_only:
      self.model.opt.load_state_dict(data_to_load["optimizer_state_dict"])
      self.model.scheduler.load_state_dict(data_to_load["scheduler_state_dict"])


def GetUnrollSteps(
  epoch_type: epoch_pb2.EpochType, batch: BatchData, unroll_strategy: str
) -> int:
  """Determine the unroll factor using the --unroll_strategy flag."""
  if epoch_type == epoch_pb2.TRAIN:
    return 1
  elif unroll_strategy == "constant":
    # Unroll by a constant number of steps according to layer_timesteps.
    return 1
  elif unroll_strategy == "data_flow_max_steps":
    # TODO: Gather data_flow_steps during batch construction.
    max_data_flow_steps = max(
      graph.data_flow_steps for graph in batch.model_data.graphs
    )
    app.Log(3, "Determined max data flow steps to be %d", max_data_flow_steps)
    return max_data_flow_steps
  elif unroll_strategy == "edge_count":
    max_edge_count = max(graph.edge_count for graph in batch.model_data.graphs)
    app.Log(3, "Determined max edge count to be %d", max_edge_count)
    return max_edge_count
  elif unroll_strategy == "label_convergence":
    return 0
  else:
    raise ValueError(f"Unknown unroll strategy '{unroll_strategy}'")
