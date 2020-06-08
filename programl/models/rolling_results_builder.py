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
"""This module defines a builder for rolling results."""
from queue import Queue
from threading import Thread
from typing import Optional

from tqdm import tqdm

from programl.models.batch_data import BatchData
from programl.models.batch_results import BatchResults
from programl.models.rolling_results import RollingResults


class RollingResultsBuilder(object):
  """A threaded worker for updating rolling results.

  Using this class aims to decouple the (possible expensive) updating of rolling
  results and logging progress to stderr from the main model loop.

  Use this class as a context manager to build rolling results, then access the
  results once out of the "with" scope. For example:

    with RollingResultsBuilder("my epoch") as builder:
      for batch, batch_results in my_model.Train():
        # Feed the builder with batch results ...
        builder.AddBatch(batch, batch_results)

    # Now you can access builder.results attribute ...
    epoch_results = builder.results
  """

  def __init__(self, log_prefix: str, total_graph_count: Optional[int] = None):
    self._q = Queue(maxsize=100)
    self._results = RollingResults()
    self._bar = tqdm(
      desc=log_prefix,
      total=total_graph_count,
      leave=True,
      unit=" graphs",
      smoothing=0.05,
    )
    self._bar.set_postfix(
      loss="?", prec="?", rec="?", f1="?",
    )
    self._thread = Thread(target=self.worker)
    self._thread.start()

  def AddBatch(
    self, data: BatchData, results: BatchResults, weight: Optional[float] = None
  ) -> None:
    """Record a new batch of data.

    Arguments are forward to RollingResults.Update().
    """
    self._q.put((data, results, weight), block=True)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self._q.put(None, block=True)
    self._thread.join()
    self._bar.close()

  @property
  def results(self) -> RollingResults:
    """Access the rolling results.

    Results can only be accessed after exiting the "with" scope of an instance.
    """
    if self._thread.is_alive() is None:
      raise TypeError("Cannot access results yet")
    return self._results

  def worker(self):
    """Background thread worker which repeated updates rolling results and logs
    to stderr.
    """
    while True:
      item = self._q.get(block=True)
      # End of epoch results.
      if item is None:
        break

      data, results, weight = item
      self._results.Update(data, results, weight)
      self._bar.update(data.graph_count)
      self._bar.set_postfix(
        loss="-" if self._results.loss is None else f"{self._results.loss:.4f}",
        prec=f"{self._results.precision:.3f}",
        rec=f"{self._results.recall:.3f}",
        f1=f"{self._results.f1:.3f}",
      )

    self._bar.close()
