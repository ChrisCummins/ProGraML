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
"""An LSTM for instruction classification."""
import pathlib
import tempfile
from typing import Any, Dict, List

import numpy as np
import torch
from labm8.py import app
from labm8.py.progress import NullContext, ProgressContext
from torch import nn, optim

from programl.models.batch_data import BatchData
from programl.models.batch_results import BatchResults
from programl.models.ggnn.loss import Loss
from programl.models.ggnn.node_embeddings import NodeEmbeddings
from programl.models.lstm.lstm_batch import LstmBatchData
from programl.models.model import Model
from programl.proto import epoch_pb2

FLAGS = app.FLAGS

app.DEFINE_integer(
    "hidden_size",
    64,
    "The size of hidden layer(s).",
)
app.DEFINE_integer(
    "hidden_dense_layer_count",
    1,
    "The number of hidden dense layers between the final LSTM layer and the " "output.",
)
app.DEFINE_integer(
    "batch_size",
    64,
    "The number of padded sequences to concatenate into a batch.",
)
app.DEFINE_integer(
    "padded_sequence_length",
    5000,
    "For node-level models, the padded/truncated length of encoded node " "sequences.",
)
app.DEFINE_float(
    "selector_embedding_value",
    1,
    "The value used for the positive class in the 1-hot selector embedding "
    "vectors. Has no effect when selector embeddings are not used.",
)
app.DEFINE_float("learning_rate", 0.001, "The mode learning rate.")
app.DEFINE_boolean(
    "trainable_embeddings", True, "Whether the embeddings are trainable."
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


class Lstm(Model):
    """An LSTM model for node-level classification."""

    def __init__(
        self,
        vocabulary: Dict[str, int],
        node_y_dimensionality: int,
        graph_y_dimensionality: int,
        graph_x_dimensionality: int,
        use_selector_embeddings: bool,
        test_only: bool = False,
        name: str = "lstm",
    ):
        """Constructor."""
        super().__init__(test_only=test_only, vocabulary=vocabulary, name=name)

        self.vocabulary = vocabulary
        self.node_y_dimensionality = node_y_dimensionality
        self.graph_y_dimensionality = graph_y_dimensionality
        self.graph_x_dimensionality = graph_x_dimensionality
        self.node_selector_dimensionality = 2 if use_selector_embeddings else 0

        # Flag values.
        self.batch_size = FLAGS.batch_size
        self.padded_sequence_length = FLAGS.padded_sequence_length

        self.model = LstmModel(
            node_embeddings=NodeEmbeddings(
                node_embeddings_type=FLAGS.text_embedding_type,
                use_selector_embeddings=self.node_selector_dimensionality,
                selector_embedding_value=FLAGS.selector_embedding_value,
                embedding_shape=(
                    # Add one to the vocabulary size to account for the out-of-vocab token.
                    len(vocabulary) + 1,
                    FLAGS.text_embedding_dimensionality,
                ),
            ),
            loss=Loss(
                num_classes=self.node_y_dimensionality,
                has_aux_input=self.has_aux_input,
                intermediate_loss_weight=None,  # NOTE(cec): Intentionally broken.
                class_prevalence_weighting=False,
            ),
            padded_sequence_length=self.padded_sequence_length,
            learning_rate=FLAGS.learning_rate,
            test_only=test_only,
            hidden_size=FLAGS.hidden_size,
            hidden_dense_layer_count=FLAGS.hidden_dense_layer_count,
        )

    @property
    def num_classes(self) -> int:
        return self.node_y_dimensionality or self.graph_y_dimensionality

    @property
    def has_aux_input(self) -> bool:
        return self.graph_x_dimensionality > 0

    def CreateModelData(self, test_only: bool) -> None:
        """Initialize an LSTM model. This is called during Initialize()."""
        # Create the Tensorflow session and graph for the model.
        tf.get_logger().setLevel("ERROR")
        SetAllowedGrowthOnKerasSession()
        self.model = self.CreateKerasModel()

    def RunBatch(
        self,
        epoch_type: epoch_pb2.EpochType,
        batch_data: BatchData,
        ctx: ProgressContext = NullContext,
    ) -> BatchResults:
        """Run a batch of data through the model.

        Args:
          epoch_type: The type of the current epoch.
          batch: A batch of graphs and model data. This requires that batch data has
            'x' and 'y' properties that return lists of model inputs, a `targets`
            property that returns a flattened list of targets, a `GetPredictions()`
            method that recieves as input the data generated by model and returns
            a flattened array of the same shape as `targets`.
          ctx: A logging context.
        """
        model_data: LstmBatchData = batch_data.model_data

        assert model_data.encoded_sequences.shape == (
            self.batch_size,
            self.padded_sequence_length,
        ), model_data.encoded_sequences.shape
        assert model_data.selector_ids.shape == (
            self.batch_size,
            self.padded_sequence_length,
        ), model_data.selector_ids.shape

        x = [model_data.encoded_sequences, model_data.selector_vectors]
        y = [model_data.node_labels]

        if epoch_type == epoch_pb2.TRAIN:
            if not self.model.training:
                self.model.train()
            targets, logits = self.model(
                model_data.encoded_sequences,
                model_data.selector_ids,
                model_data.node_labels,
            )
        else:
            if self.model.training:
                self.model.eval()
                self.model.opt.zero_grad()
                # Inference only, don't trace the computation graph.
            with torch.no_grad():
                targets, logits = self.model(
                    model_data.encoded_sequences,
                    model_data.selector_ids,
                    model_data.node_labels,
                )

        loss = self.model.loss((logits, None), targets)

        if epoch_type == epoch_pb2.TRAIN:
            loss.backward()
            self.model.opt.step()
            self.model.opt.zero_grad()

        # Reshape the outputs.
        predictions = self.ReshapePaddedModelOutput(batch_data, outputs)

        # Flatten the targets and predictions lists so that we can compare them.
        # Shape (batch_node_count, node_y_dimensionality).
        targets = np.concatenate(model_data.targets)
        predictions = np.concatenate(predictions)

        return BatchResults.Create(
            targets=model_data.node_labels,
            predictions=logits.detach().cpu().numpy(),
            learning_rate=self.model.learning_rate,
            loss=loss.item(),
        )

    def ReshapePaddedModelOutput(
        self, batch_data: BatchData, padded_outputs: np.array
    ) -> List[np.array]:
        """Reshape the model outputs to an array of predictions of same shape as
        targets. Zeros are used as padding values when the model produces fewer
        outputs than there are nodes in a graph.

        Args:
          batch_data: The input batch data.
          padded_outputs: Model outputs with shape
            (batch_size, padded_node_count, node_y_dimensionality).

        Returns:
          A list of length batch_data.graph_count np arrays, where each array is
          of shape (?, node_y_dimensionality), where ? is the true number of nodes
          in that graph.
        """
        if padded_outputs.shape != (
            self.batch_size,
            self.padded_sequence_length,
            self.node_y_dimensionality,
        ):
            raise TypeError(
                f"Model produced output with shape {padded_outputs.shape} but "
                f"expected outputs with shape ({self.batch_size}, {self.padded_sequence_length})"
            )
        outputs = []
        for graph_node_size, padded_output in zip(
            batch_data.model_data.graph_node_sizes,
            padded_outputs[: batch_data.graph_count],
        ):
            active_nodes = padded_output[:graph_node_size]
            # Truncated graph input. Fill with random values since the model didn't see
            # this bit of the graph.
            padding = np.random.randn(
                max(graph_node_size - self.padded_sequence_length, 0),
                self.node_y_dimensionality,
            )
            outputs.append(np.concatenate((active_nodes, padding)))
        return outputs

    def GetModelData(self) -> Any:
        """Get the model state."""
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.model.opt.state_dict(),
            "scheduler_state_dict": self.model.scheduler.state_dict(),
        }

    def LoadModelData(self, data_to_load: Any) -> None:
        """Restore the model state."""
        self.model.load_state_dict(data_to_load["model_state_dict"])
        # only restore opt if needed. opt should be None o/w.
        if not self.test_only:
            self.model.opt.load_state_dict(data_to_load["optimizer_state_dict"])
            self.model.scheduler.load_state_dict(data_to_load["scheduler_state_dict"])


class LstmModel(nn.Module):
    def __init__(
        self,
        node_embeddings: NodeEmbeddings,
        loss: Loss,
        padded_sequence_length: int,
        test_only: bool,
        learning_rate: float,
        hidden_size: int,
        hidden_dense_layer_count: int,  # TODO(cec): Implement.
    ):
        super().__init__()
        self.node_embeddings = node_embeddings
        self.loss = loss
        self.padded_sequence_length = padded_sequence_length
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.lstm = nn.LSTM(
            self.node_embeddings.embedding_dimensionality + 2,
            self.hidden_size,
        )
        self.hidden2label = nn.Linear(self.hidden_size, 2)

        if test_only:
            self.opt = None
            self.eval()
        else:
            self.opt = optim.AdamW(self.parameters(), lr=self.learning_rate)

    def forward(
        self,
        encoded_sequences,
        selector_ids,
        node_labels,
    ):
        print("SHAPES", encoded_sequences.shape, selector_ids.shape, node_labels.shape)

        encoded_sequences = torch.tensor(encoded_sequences, dtype=torch.long)
        selector_ids = torch.tensor(selector_ids, dtype=torch.long)
        node_labels = torch.tensor(node_labels, dtype=torch.long)

        # Embed and concatenate sequences and selector vectors.
        embeddings = self.node_embeddings(encoded_sequences, selector_ids)

        lstm_out, _ = self.lstm(
            embeddings.view(self.padded_sequence_length, len(encoded_sequences), -1)
        )
        print(lstm_out.shape)

        label_space = self.hidden2label(lstm_out.view(self.padded_sequence_length, -1))
        logits = F.log_softmax(label_space, dim=2)

        targets = node_labels
        return logits, targets
