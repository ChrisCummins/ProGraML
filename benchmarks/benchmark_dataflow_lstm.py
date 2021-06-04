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
"""Benchmark for the dataflow LSTM pipeline."""
import contextlib
import os
import pathlib
import tempfile
import warnings

from absl import app, flags
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm

from models.lstm.lstm import Lstm
from programl.proto import epoch_pb2
from programl.third_party.ncc import vocabulary
from programl.util.py import progress
from programl.util.py.threaded_iterator import ThreadedIterator
from tasks.dataflow.graph_loader import DataflowGraphLoader
from tasks.dataflow.lstm_batch_builder import DataflowLstmBatchBuilder
from tests.plugins import llvm_program_graph, llvm_reachability_features

flags.DEFINE_integer("graph_count", None, "The number of graphs to load.")
flags.DEFINE_integer("batch_count", None, "The number of batches.")
flags.DEFINE_integer(
    "train_batch_count", 3, "The number of batches for testing model training"
)
flags.DEFINE_integer(
    "test_batch_count", 3, "The number of batches for testing model training"
)
FLAGS = flags.FLAGS


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


def GraphLoader(path):
    return DataflowGraphLoader(
        path=path,
        epoch_type=epoch_pb2.TRAIN,
        analysis="reachability",
        min_graph_count=FLAGS.graph_count or 1,
        max_graph_count=FLAGS.graph_count,
        logfile=open(path / "graph_reader_log.txt", "w"),
    )


def BatchBuilder(model: Lstm, graph_loader, vocab, max_batch_count=None):
    return DataflowLstmBatchBuilder(
        graph_loader=graph_loader,
        vocabulary=vocab,
        node_y_dimensionality=model.node_y_dimensionality,
        batch_size=model.batch_size,
        padded_sequence_length=model.padded_sequence_length,
        max_batch_count=max_batch_count or FLAGS.batch_count,
    )


def Vocab():
    with vocabulary.VocabularyZipFile.CreateFromPublishedResults() as inst2vec:
        return inst2vec.dictionary


def Print(msg):
    print()
    print(msg, flush=True)


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

        Print("=== BENCHMARK 2: Batch construction ===")
        model = Lstm(vocabulary=Vocab(), node_y_dimensionality=2)
        batches = BatchBuilder(model, GraphLoader(path), Vocab())
        batches = ThreadedIterator(batches, max_queue_size=100)
        cached_batches = []
        with progress.Profile("Benchmark batch construction"):
            for batch in tqdm(batches, unit=" batches"):
                cached_batches.append(batch)

        Print("=== BENCHMARK 3: Model training ===")
        model.Initialize()

        model.model.summary()

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
                BatchBuilder(
                    model, GraphLoader(path), Vocab(), FLAGS.train_batch_count
                ),
                log_prefix="Train",
            )

        Print("=== BENCHMARK 4: Model inference ===")
        model = Lstm(
            vocabulary=Vocab(),
            node_y_dimensionality=2,
            test_only=True,
        )
        model.Initialize()

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
                BatchBuilder(model, GraphLoader(path), Vocab(), FLAGS.test_batch_count),
                log_prefix="Val",
            )


if __name__ == "__main__":
    app.run(main)
