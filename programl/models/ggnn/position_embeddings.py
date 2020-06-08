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
"""Position embeddings torch module."""
import torch
from torch import nn


class PositionEmbeddings(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, positions, demb, dpad: int = 0):
    """Transformer-like sinusoidal positional embeddings.

    Args:
      positions: 1d long Tensor of positions.
      demb: Size of embedding vector.
      dpad: Padding zeros to concatenate to embedding vectors.
    """
    inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))

    sinusoid_inp = torch.ger(positions, inv_freq)
    pos_emb = torch.cat(
      (torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1
    )

    if dpad > 0:
      in_length = positions.size()[0]
      pad = torch.zeros((in_length, dpad))
      pos_emb = torch.cat([pos_emb, pad], dim=1)
      assert torch.all(
        pos_emb[:, -1] == torch.zeros(in_length)
      ), f"test failed. pos_emb: \n{pos_emb}"

    return pos_emb
