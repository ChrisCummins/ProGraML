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
"""Benchmark for the dataflow GGNN pipeline."""
import contextlib
import os
import pathlib
import queue
import sys
import tempfile
import threading
import warnings
from typing import Any, NamedTuple

from absl import app, flags, logging
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm

from programl.models.ggnn.ggnn import Ggnn
from programl.proto import epoch_pb2
from programl.util.py import progress
from tasks.dataflow.ggnn_batch_builder import DataflowGgnnBatchBuilder
from tasks.dataflow.graph_loader import DataflowGraphLoader
from tests.plugins import llvm_program_graph, llvm_reachability_features

flags.DEFINE_integer("graph_count", None, "The number of graphs to load.")
flags.DEFINE_integer("batch_size", 10000, "The size of batches.")
flags.DEFINE_integer(
    "train_batch_count", 3, "The number of batches for testing model training"
)
flags.DEFINE_integer(
    "test_batch_count", 3, "The number of batches for testing model training"
)
FLAGS = flags.FLAGS


class ThreadedIterator:
    """An iterator that computes its elements in a parallel thread to be ready to
    be consumed.
    Exceptions raised by the threaded iterator are propagated to consumer.
    """

    def __init__(
        self,
        iterator,
        max_queue_size: int = 2,
        start: bool = True,
    ):
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._thread = threading.Thread(target=lambda: self.worker(iterator))
        if start:
            self.Start()

    def Start(self):
        self._thread.start()

    def worker(self, iterator):
        try:
            for element in iterator:
                self._queue.put(self._ValueOrError(value=element), block=True)
        except Exception as e:
            # Propagate an error in the iterator.
            self._queue.put(self._ValueOrError(error=e))
        # Mark that the iterator is done.
        self._queue.put(self._EndOfIterator(), block=True)

    def __iter__(self):
        next_element = self._queue.get(block=True)
        while not isinstance(next_element, self._EndOfIterator):
            value = next_element.GetOrRaise()
            yield value
            next_element = self._queue.get(block=True)
        self._thread.join()

    class _EndOfIterator(object):
        """Tombstone marker object for iterators."""

        pass

    class _ValueOrError(NamedTuple):
        """A tuple which represents the union of either a value or an error."""

        value: Any = None
        error: Exception = None

        def GetOrRaise(self) -> Any:
            """Return the value or raise the exception."""
            if self.error is None:
                return self.value
            else:
                raise self.error


@contextlib.contextmanager
def data_directory() -> pathlib.Path:
    """Create a dataset directory."""
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        (d / "labels").mkdir()
        os.symlink(llvm_program_graph.LLVM_IR_GRAPHS, d / "graphs")
        os.symlink(llvm_program_graph.LLVM_IR_GRAPHS, d / "train")
        os.symlink(llvm_program_graph.LLVM_IR_GRAPHS, d / "val")
        os.symlink(llvm_program_graph.LLVM_IR_GRAPHS, d / "test")
        os.symlink(
            llvm_reachability_features.LLVM_REACHABILITY_FEATURES,
            d / "labels" / "reachability",
        )
        yield d


def GraphLoader(path, use_cdfg: bool = False):
    return DataflowGraphLoader(
        path=path,
        epoch_type=epoch_pb2.TRAIN,
        analysis="reachability",
        min_graph_count=FLAGS.graph_count or 1,
        max_graph_count=FLAGS.graph_count,
        logfile=open(path / "graph_reader_log.txt", "w"),
        use_cdfg=use_cdfg,
    )


def BatchBuilder(graph_loader, vocab, max_batch_count=None, use_cdfg: bool = False):
    return DataflowGgnnBatchBuilder(
        graph_loader=graph_loader,
        vocabulary=vocab,
        max_node_size=FLAGS.batch_size,
        max_batch_count=max_batch_count,
        use_cdfg=use_cdfg,
    )


def Vocab():
    return {"": 0}


def Print(msg):
    print()
    print(msg)
    sys.stdout.flush()


def main(argv):
    if len(argv) != 1:
        raise app.UsageError(f"Unrecognized arguments: {argv[1:]}")
    # NOTE(github.com/ChrisCummins/ProGraML/issues/13): F1 score computation
    # warns that it is undefined when there are missing instances from a class,
    # which is fine for our usage.
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    with data_directory() as path:
        Print("=== BENCHMARK 1: Loading graphs from filesystem ===")
        graph_loader = GraphLoader(path)
        graphs = ThreadedIterator(graph_loader, max_queue_size=100)
        with progress.Profile("Benchmark graph loader"):
            for _ in tqdm(graphs, unit=" graphs"):
                pass
        logging.info("Skip count: %s", graph_loader.skip_count)

        Print(
            "=== BENCHMARK 1: Loading graphs from filesystem and converting to CDFG ==="
        )
        graph_loader = GraphLoader(path, use_cdfg=True)
        graphs = ThreadedIterator(graph_loader, max_queue_size=100)
        with progress.Profile("Benchmark CDFG graph loader"):
            for _ in tqdm(graphs, unit=" graphs"):
                pass
        logging.info("Skip count: %s", graph_loader.skip_count)

        Print("=== BENCHMARK 2: Batch construction ===")
        batches = BatchBuilder(GraphLoader(path), Vocab())
        batches = ThreadedIterator(batches, max_queue_size=100)
        cached_batches = []
        with progress.Profile("Benchmark batch construction"):
            for batch in tqdm(batches, unit=" batches"):
                cached_batches.append(batch)

        Print("=== BENCHMARK 2: CDFG batch construction ===")
        batches = BatchBuilder(GraphLoader(path, use_cdfg=True), Vocab(), use_cdfg=True)
        batches = ThreadedIterator(batches, max_queue_size=100)
        cached_batches = []
        with progress.Profile("Benchmark batch construction"):
            for batch in tqdm(batches, unit=" batches"):
                cached_batches.append(batch)

        Print("=== BENCHMARK 3: Model training ===")
        model = Ggnn(
            vocabulary=Vocab(),
            node_y_dimensionality=2,
            graph_y_dimensionality=0,
            graph_x_dimensionality=0,
            use_selector_embeddings=True,
        )

        with progress.Profile("Benchmark training (prebuilt batches)"):
            model.RunBatches(
                epoch_pb2.TRAIN,
                cached_batches[: FLAGS.train_batch_count],
                log_prefix="Train",
                total_graph_count=sum(
                    b.graph_count for b in cached_batches[: FLAGS.train_batch_count]
                ),
            )
        with progress.Profile("Benchmark training"):
            model.RunBatches(
                epoch_pb2.TRAIN,
                BatchBuilder(GraphLoader(path), Vocab(), FLAGS.train_batch_count),
                log_prefix="Train",
            )

        Print("=== BENCHMARK 4: Model inference ===")
        model = Ggnn(
            vocabulary=Vocab(),
            test_only=True,
            node_y_dimensionality=2,
            graph_y_dimensionality=0,
            graph_x_dimensionality=0,
            use_selector_embeddings=True,
        )

        with progress.Profile("Benchmark inference (prebuilt batches)"):
            model.RunBatches(
                epoch_pb2.TEST,
                cached_batches[: FLAGS.test_batch_count],
                log_prefix="Val",
                total_graph_count=sum(
                    b.graph_count for b in cached_batches[: FLAGS.test_batch_count]
                ),
            )
        with progress.Profile("Benchmark inference"):
            model.RunBatches(
                epoch_pb2.TEST,
                BatchBuilder(GraphLoader(path), Vocab(), FLAGS.test_batch_count),
                log_prefix="Val",
            )


if __name__ == "__main__":
    app.run(main)
