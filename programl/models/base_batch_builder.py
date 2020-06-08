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
"""This module defines the base class for batch builders."""
from queue import Queue
from threading import Thread
from typing import Iterable
from typing import Optional

from programl.models.base_graph_loader import BaseGraphLoader
from programl.models.batch_data import BatchData


class BaseBatchBuilder(object):
  """Base class for building batches.

  A batch builder is a class which accepts as input a graph loader and produces
  an iterable sequence of batches.

  Instances of this class are used like any other iterator:

    batch_builder = MyBatchBuilder(my_graph_loader, max_queue_size=512)
    for batch in batch_builder:
      res = model.Train(batch)
      if FullyTrained(res):
        # If you want to stop early without consuming all batches, be sure to
        # call Stop() method.
        batch_builder.Stop()
        break

  Behind the scenes, this class runs the batch building loop in a background
  thread and buffers batches in a queue, ready to be used. This is to improve
  performance of a model loop by overlapping the construction of batches with
  the processing of them.

  This is an abstract base class, subclasses must implement the OnItem() and
  EndOfItem() methods to do the actual batch construction.
  """

  def __init__(
    self,
    graph_loader: BaseGraphLoader,
    max_batch_count: int = None,
    max_queue_size: int = 128,
  ):
    """Constructor.

    Args:
      graph_loader: The graph loader that will be used to construct batches.
      max_batch_count: A maximum number of batches to build. If not provided,
        the graph loader will be iterated over exhaustively. If the graph loader
        also has no termination criteria, then the batch builder will continue
        indefinitely.
      max_queue_size: The number of batches to buffer, waiting to be consumed.
        Once the queue is full, batch construction will halt until a batch is
        consumed (by iterating over the instance of this class).
    """
    self.graph_count = 0
    self.batch_count = 0
    self.max_batch_count = max_batch_count

    self._outq = Queue(maxsize=max_queue_size)
    self._stopped = False
    self._worker = Thread(target=lambda: self._Worker(graph_loader))
    self._worker.start()

  def __iter__(self) -> Iterable[BatchData]:
    """Iterator to construct and return batches."""
    value = self._outq.get(block=True)
    while value is not None:
      yield value
      value = self._outq.get(block=True)
    self._worker.join()

  def Stop(self) -> None:
    """Signal that no more batches are required, and any batch building
    resources can be freed.
    """
    self._stopped = True
    while self._worker.is_alive():
      if self._outq.get(block=True) is None:
        break
    self._worker.join()

  def OnItem(self, item) -> Optional[BatchData]:
    """Callback which fires when a new item is received from the graph loader.

    Subclasses must implement this method.

    Args:
      item: The iterable type of the graph loader.

    Returns:
      A batch, if one has been constructed, else None.
    """
    raise NotImplementedError("abstract class")

  def EndOfItems(self) -> Optional[BatchData]:
    """Callback which is fired at the end of batch construction.

    This is used to indicate that no more graphs will be loaded (i.e. no more
    calls to OnItem() will be fired). Subclasses can use this to construct one
    final batch.

    Subclasses must implement this method.

    Returns:
      A batch, if one has been constructed, else None.
    """
    raise NotImplementedError("abstract class")

  def _Worker(self, graph_loader: BaseGraphLoader):
    """Private worker method to construct batches and put them in a queue to
    be read.
    """
    for item in graph_loader:
      self.graph_count += 1
      # Check to see if we still have work to do.
      if self._stopped:
        break

      batch = self.OnItem(item)
      if batch:
        self._outq.put(batch)
        self.batch_count += 1
        if self.max_batch_count and self.batch_count >= self.max_batch_count:
          break

    # We've ran out of items, signal the worker.
    batch = self.EndOfItems()
    if batch:
      self._outq.put(batch)
      self.batch_count += 1

    graph_loader.Stop()
    self._outq.put(None)
