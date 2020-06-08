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
"""Node embeddings torch module."""
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from torch import nn


class NodeEmbeddings(nn.Module):
  """Construct node embeddings (content embeddings + selector embeddings)
  Args:
  pretrained_embeddings (Tensor, optional) – FloatTensor containing weights for
  the Embedding. First dimension is being passed to Embedding as
  num_embeddings, second as embedding_dim.

  Forward
  Args:
  vocab_ids: <N, 1>
  selector_ids: <N, 1>
  Returns:
  node_states: <N, config.hidden_size>
  """

  # TODO(github.com/ChrisCummins/ProGraML/issues/27):: Maybe LayerNorm and
  # Dropout on node_embeddings?
  #
  # TODO(github.com/ChrisCummins/ProGraML/issues/27):: Make selector embeddings
  # trainable?
  #
  # TODO(github.com/ChrisCummins/ml4pl/issues/12): In the future we may want
  # to be more flexible in supporting multiple types of embeddings tables, but
  # for now I have hardcoded this to always return a tuple
  # <node_embeddings, selector_embeddings>, where inst2vec_embeddings
  # is the augmented table of pre-trained statement embeddings (the
  # augmentation adds !MAGIC, !IMMEDIATE, and !IDENTIFIER vocabulary
  # elements). selector_embeddings is a 2x2 1-hot embedding table:
  # [[1, 0], [0, 1]. The selector_embeddings table is always constant, the
  # inst2vec_embeddings table can be made trainable or re-initialized with
  # random values using the --inst2vec_embeddings flag.

  def __init__(
    self,
    node_embeddings_type: str,
    use_selector_embeddings: bool,
    selector_embedding_value: float = 50,
    embedding_shape: Optional[Tuple[int, int]] = None,
    pretrained_embeddings: Optional[np.array] = None,
  ):
    super().__init__()

    if node_embeddings_type == "pretrained":
      assert embedding_shape is None
      assert pretrained_embeddings is not None
      self._node_embeddings = nn.Embedding.from_pretrained(
        pretrained_embeddings, freeze=True
      )
    elif node_embeddings_type == "finetune":
      assert embedding_shape is None
      assert pretrained_embeddings is not None
      self._node_embeddings = nn.Embedding.from_pretrained(
        pretrained_embeddings, freeze=False
      )
    elif node_embeddings_type == "constant_zero":
      assert embedding_shape is not None
      assert pretrained_embeddings is None
      init = torch.zeros(*embedding_shape)
      self._node_embeddings = nn.Embedding.from_pretrained(init, freeze=True)
    elif node_embeddings_type == "constant_random":
      assert embedding_shape is not None
      assert pretrained_embeddings is None
      init = torch.rand(*embedding_shape)
      self._node_embeddings = nn.Embedding.from_pretrained(init, freeze=True)
    elif node_embeddings_type == "random":
      assert embedding_shape is not None
      assert pretrained_embeddings is None
      self._node_embeddings = nn.Embedding(*embedding_shape)
    else:
      raise ValueError(
        f"Unsupported node embeddings type: {node_embeddings_type}"
      )

    if use_selector_embeddings:
      # The selector embeddings table is a 1-hot encoded table for <true,false>
      # binary selector values.
      selector_init = torch.tensor(
        [[selector_embedding_value, 0], [0, selector_embedding_value]],
        dtype=torch.get_default_dtype(),
      )
      self._selector_embeddings = nn.Embedding.from_pretrained(
        selector_init, freeze=True
      )
    else:
      self._selector_embeddings = None

    self.vocab_size = embedding_shape[0]
    self.text_embedding_dimensionality = embedding_shape[1]
    self.selector_embedding_dimensionality = (
      2 if self._selector_embeddings else 0
    )
    self.embedding_dimensionality = (
      self.text_embedding_dimensionality
      + self.selector_embedding_dimensionality
    )

  def forward(self, vocab_ids, selector_ids=None):
    embeddings = self._node_embeddings(vocab_ids)
    if self._selector_embeddings:
      selector_embeddings = self._selector_embeddings(selector_ids)
      embeddings = torch.cat((embeddings, selector_embeddings), dim=1)
    return embeddings
