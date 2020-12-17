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
"""Base class for implementing classifier models."""
import pickle
from queue import Empty, Queue
from threading import Thread
from typing import Any, Dict, Iterable

from programl.models.batch_data import BatchData
from programl.models.batch_results import BatchResults
from programl.models.rolling_results_builder import RollingResultsBuilder
from programl.proto import checkpoint_pb2, epoch_pb2
from programl.util.py.progress import NullContext, ProgressContext


class BatchQueue(Thread):
    """A thread which reads batches onto a queue. That is all it does.
    c.f. https://youtu.be/X7HmltUWXgs
    """

    def __init__(self, batches: Iterable[BatchData], queue: Queue):
        super().__init__()
        self.batches = batches
        self.queue = queue

    def run(self) -> None:
        for i, batch_data in enumerate(self.batches, start=1):
            self.queue.put((i, batch_data))
        self.queue.put((None, None))


class Model(object):
    """Abstract base class for implementing classifiers.

    Before using the model, it must be initialized bu calling Initialize(), or
    restored from a checkpoint using RestoreFrom(checkpoint).

    Subclasses must implement the following methods:
      MakeBatch()        # construct a batch from input graphs.
      RunBatch()         # run the model on the batch.
      GetModelData()     # get model data to save.
      LoadModelData()    # load model data.

    And may optionally wish to implement these additional methods:
      CreateModelData()  # initialize an untrained model.
      Summary()          # return a string model summary.
      GraphReader()      # return a buffered graph reader.
      BatchIterator()    # return an iterator over batches.

    Example usage:

      model = MyModel()
      model.Initialize()
      results = model.RunBatches(epoch_pb2.TRAIN, batches)
      for result in result:
        ...

    Example restoring from checkpoint:

      model = MyModel()
      model.RestoreCheckpoint(checkpoint)
      model.RunBatches(...)
    """

    def __init__(self, name: str, vocabulary: Dict[str, int], test_only: bool = False):
        self._initialized = False
        self.name = name
        self.vocabulary = vocabulary
        self.test_only = test_only

    def Initialize(self) -> None:
        """Initialize an untrained model."""
        if self._initialized:
            raise TypeError("CreateModelData() called on already-initialized model")

        self._initialized = True
        self.CreateModelData(test_only=self.test_only)

    def CreateModelData(self, test_only: bool) -> None:
        """Initialize the starting state of a model.

        Use this method to perform any model-specific initialisation such as
        randomizing starting weights. When restoring a model from a checkpoint, this
        method is *not* called. Instead, LoadModelData() will be called.

        Note that subclasses must call this superclass method first.
        """
        pass

    def RunBatches(
        self,
        epoch_type: epoch_pb2.EpochType,
        batches: Iterable[BatchData],
        timeout: float = 600,
        **rolling_results_builder_opts,
    ) -> epoch_pb2.EpochResults:
        # Read batches into a queue so that we can use the blocking Queue.get()
        # to wait for a batch with a timeout. Using a timeout is useful for
        # catching cases where a dead iterator will lead to data starvation and
        # a non-terminating process.
        # See <https://github.com/ChrisCummins/ProGraML/issues/140>.
        queue = Queue(maxsize=128)
        batches = BatchQueue(batches, queue)
        batches.start()

        with RollingResultsBuilder(**rolling_results_builder_opts) as results_builder:
            while True:
                try:
                    i, batch_data = queue.get(timeout=timeout)
                except Empty as e:
                    raise ValueError(
                        f"Model received no batches within {timeout:.1f}s timeout, "
                        "did your batch generator die?"
                    ) from e
                # Done.
                if not batch_data:
                    break
                batch_results = self.RunBatch(epoch_type, batch_data)
                results_builder.AddBatch(batch_data, batch_results, weight=None)

        batches.join()
        return results_builder.results.ToEpochResults()

    def RestoreCheckpoint(self, checkpoint: checkpoint_pb2.Checkpoint):
        """Restore a model from a checkpoint."""
        self._initialized = True
        self.LoadModelData(pickle.loads(checkpoint.model_data))

    def SaveCheckpoint(self) -> checkpoint_pb2.Checkpoint:
        """Construct a checkpoint from the current model state.

        Returns:
          A checkpoint reference.
        """
        return checkpoint_pb2.Checkpoint(
            model_data=pickle.dumps(self.GetModelData()),
        )

    def Train(self, batch_data: BatchData) -> BatchResults:
        return self.RunBatch(epoch_pb2.TRAIN, batch_data)

    def Val(self, batch_data: BatchData) -> BatchResults:
        return self.RunBatch(epoch_pb2.VAL, batch_data)

    def Test(self, batch_data: BatchData) -> BatchResults:
        return self.RunBatch(epoch_pb2.TEST, batch_data)

    #############################################################################
    # Interface methods. Subclasses must implement these.
    #############################################################################

    def RunBatch(
        self,
        epoch_type: epoch_pb2.EpochType,
        batch_data: BatchData,
        ctx: ProgressContext = NullContext,
    ) -> BatchResults:
        raise NotImplementedError("abstract class")

    def LoadModelData(self, data_to_load: Any) -> None:
        """Set the model state from the given model data.

        Args:
          data_to_load: The return value of GetModelData().
        """
        raise NotImplementedError("abstract class")

    def GetModelData(self) -> Any:
        """Return the model state.

        Returns:
          A  model-defined blob of data that can later be passed to LoadModelData()
          to restore the current model state.
        """
        raise NotImplementedError("abstract class")
