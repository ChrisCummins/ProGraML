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
"""A linear net torch module."""
import torch
import torch.nn.functional as F
from torch import nn


class LinearNet(nn.Module):
  """Single Linear layer with WeightDropout, ReLU and Xavier Uniform
  initialization. Applies a linear transformation to the incoming data:
  :math:`y = xA^T + b`

  Args:
  in_features: size of each input sample
  out_features: size of each output sample
  bias: If set to ``False``, the layer will not learn an additive bias.
  Default: ``True``

  Shape:
  - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
  additional dimensions and :math:`H_{in} = \text{in\_features}`
  - Output: :math:`(N, *, H_{out})` where all but the last dimension
  are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
  """

  def __init__(self, in_features, out_features, bias=True, dropout=0.0):
    super().__init__()
    self.dropout = dropout
    self.in_features = in_features
    self.out_features = out_features
    self.test = nn.Parameter(torch.Tensor(out_features, in_features))
    if bias:
      self.bias = nn.Parameter(torch.Tensor(out_features))
    else:
      self.register_parameter("bias", None)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.xavier_uniform_(self.test)
    # TODO(github.com/ChrisCummins/ProGraML/issues/27): why use xavier_uniform,
    # not kaiming init? Seems old-school
    if self.bias is not None:
      #    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
      #    bound = 1 / math.sqrt(fan_in)
      #    nn.init.uniform_(self.bias, -bound, bound)
      nn.init.zeros_(self.bias)

  def forward(self, input):
    if self.dropout > 0.0:
      w = F.dropout(self.test, p=self.dropout, training=self.training)
    else:
      w = self.test
    return F.linear(input, w, self.bias)

  def extra_repr(self):
    return "in_features={}, out_features={}, bias={}, dropout={}".format(
      self.in_features, self.out_features, self.bias is not None, self.dropout,
    )
