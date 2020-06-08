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
"""This module defines a class for asynchronously reading a batch builder."""
import time
from threading import Thread
from typing import List

from labm8.py import app
from labm8.py import humanize
from programl.models.base_batch_builder import BaseBatchBuilder
from programl.models.batch_data import BatchData


class AsyncBatchBuilder(object):
  """A class for running a batch builder in a background thread.

  An AsyncBatchBuilder runs a batch builder in the background until completion,
  storing the generated batches in an in-memory list. Use this class to
  construct a bunch of batches in the background while you are busy doing
  something else. For example:

      bb = AsyncBatchBuilder(MyBatchBuilder(my_graph_loader))
      # ... do some busywork
      for batch in bb.batches:
        # go nuts!

  Because the batches are loaded into a plain python list, this also provides
  a convenient means for reusing a set of batches. For example, to always
  use the same set of batches during validation runs of a model:

    val_batches = AsyncBatchBuilder(batch_builder)
    for epoch in range(10):
      # ... train model, val_batches are loading in the background
      model.RunBatch(val_batches.batches)
  """

  def __init__(self, batch_builder: BaseBatchBuilder):
    self._batches = []
    self._worker = Thread(target=lambda: self._Worker(batch_builder))
    self._worker.start()

  def _Worker(self, batch_builder: BaseBatchBuilder):
    start = time.time()
    self._batches = list(batch_builder)
    app.Log(
      2,
      "Async batch loader completed. %s batches loaded in %s",
      humanize.Commas(len(self._batches)),
      humanize.Duration(time.time() - start),
    )

  @property
  def batches(self) -> List[BatchData]:
    """Access the batches. Blocks until all batches are built."""
    self._worker.join()
    return self._batches
