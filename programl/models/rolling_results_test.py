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
"""Unit tests for //programl/ml/batch:rolling_results."""
import numpy as np

from labm8.py import test
from programl.models.batch_data import BatchData
from programl.models.batch_results import BatchResults
from programl.models.rolling_results import RollingResults


FLAGS = test.FLAGS


@test.Parametrize("weight", [1, 0.5, 10])
def test_RollingResults_iteration_count(weight: float):
  """Test aggreation of model iteration count and convergence."""
  rolling_results = RollingResults()

  data = BatchData(graph_count=1, model_data=None)
  results = BatchResults.Create(
    targets=np.array([[0, 1, 2]]),
    predictions=np.array([[0, 1, 2]]),
    iteration_count=1,
    model_converged=True,
  )

  for _ in range(10):
    rolling_results.Update(data, results, weight=weight)

  assert rolling_results.iteration_count == 1
  assert rolling_results.model_converged == 1


if __name__ == "__main__":
  test.Main()
