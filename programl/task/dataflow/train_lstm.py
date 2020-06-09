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
"""Train an LSTM to estimate solutions for classic data flow problems.

This script reads ProGraML graphs and uses an LSTM to predict binary
classification targets for data flow problems.
"""
import pathlib
import time
from typing import Dict

import numpy as np

from deeplearning.ncc import vocabulary
from labm8.py import app
from labm8.py import gpu_scheduler
from labm8.py import humanize
from labm8.py import pbutil
from programl.models.async_batch_builder import AsyncBatchBuilder
from programl.models.epoch_batch_iterator import EpochBatchIterator
from programl.models.lstm.lstm import Lstm
from programl.proto import epoch_pb2
from programl.task.dataflow import dataflow
from programl.task.dataflow.graph_loader import DataflowGraphLoader
from programl.task.dataflow.lstm_batch_builder import DataflowLstmBatchBuilder


app.DEFINE_integer(
  "max_data_flow_steps",
  30,
  "If > 0, limit the size of dataflow-annotated graphs used to only those "
  "with data_flow_steps <= --max_data_flow_steps",
)
FLAGS = app.FLAGS


def MakeBatchBuilder(
  dataset_root: pathlib.Path,
  log_dir: pathlib.Path,
  epoch_type: epoch_pb2.EpochType,
  model: Lstm,
  min_graph_count=None,
  max_graph_count=None,
  seed=None,
):
  logfile = (
    log_dir
    / "graph_loader"
    / f"{epoch_pb2.EpochType.Name(epoch_type).lower()}.txt"
  )
  return DataflowLstmBatchBuilder(
    graph_loader=DataflowGraphLoader(
      dataset_root,
      epoch_type=epoch_type,
      analysis=FLAGS.analysis,
      min_graph_count=min_graph_count,
      max_graph_count=max_graph_count,
      data_flow_step_max=FLAGS.max_data_flow_steps,
      require_inst2vec=True,
      # Append to logfile since we may be resuming a previous job.
      logfile=open(str(logfile), "a"),
      seed=seed,
    ),
    vocabulary=model.vocabulary,
    padded_sequence_length=model.padded_sequence_length,
    batch_size=model.batch_size,
    node_y_dimensionality=model.node_y_dimensionality,
  )


def TrainDataflowLSTM(
  path: pathlib.Path,
  vocab: Dict[str, int],
  val_seed: int,
  restore_from: pathlib.Path,
) -> pathlib.Path:
  if not path.is_dir():
    raise FileNotFoundError(path)

  if restore_from:
    log_dir = restore_from
  else:
    # Create the logging directories.
    log_dir = dataflow.CreateLoggingDirectories(
      dataset_root=path,
      model_name="inst2vec",
      analysis=FLAGS.analysis,
      run_id=FLAGS.run_id,
    )

  dataflow.PatchWarnings()
  dataflow.RecordExperimentalSetup(log_dir)

  # Cumulative totals for training graph counts at each "epoch".
  train_graph_counts = [int(x) for x in FLAGS.train_graph_counts]
  train_graph_cumsums = np.array(train_graph_counts, dtype=np.int32)
  # The number of training graphs in each "epoch".
  train_graph_counts = train_graph_cumsums - np.concatenate(
    ([0], train_graph_counts[:-1])
  )

  # Create the model, defining the shape of the graphs that it will process.
  #
  # For these data flow experiments, our graphs contain per-node binary
  # classification targets (e.g. reachable / not-reachable).
  model = Lstm(vocabulary=vocab, test_only=False, node_y_dimensionality=2,)

  if restore_from:
    # Pick up training where we left off.
    restored_epoch, checkpoint = dataflow.SelectTrainingCheckpoint(log_dir)
    # Skip the epochs that we have already done.
    # This requires that --train_graph_counts is the same as it was in the
    # run that we are resuming!
    start_epoch_step = restored_epoch.epoch_num
    start_graph_cumsum = sum(train_graph_counts[:start_epoch_step])
    train_graph_counts = train_graph_counts[start_epoch_step:]
    train_graph_cumsums = train_graph_cumsums[start_epoch_step:]
    model.RestoreCheckpoint(checkpoint)
  else:
    # Else initialize a new model.
    model.Initialize()
    start_epoch_step, start_graph_cumsum = 1, 0

  model.model.summary()

  # Create training batches and split into epochs.
  epochs = EpochBatchIterator(
    MakeBatchBuilder(
      dataset_root=path,
      log_dir=log_dir,
      epoch_type=epoch_pb2.TRAIN,
      model=model,
      seed=val_seed,
    ),
    train_graph_counts,
    start_graph_count=start_graph_cumsum,
  )

  # Read val batches asynchronously
  val_batches = AsyncBatchBuilder(
    batch_builder=MakeBatchBuilder(
      dataset_root=path,
      log_dir=log_dir,
      epoch_type=epoch_pb2.VAL,
      model=model,
      min_graph_count=FLAGS.val_graph_count,
      max_graph_count=FLAGS.val_graph_count,
      seed=val_seed,
    )
  )

  for (
    epoch_step,
    (train_graph_count, train_graph_cumsum, train_batches),
  ) in enumerate(epochs, start=start_epoch_step):
    start_time = time.time()
    hr_graph_cumsum = f"{humanize.Commas(train_graph_cumsum)} graphs"

    train_results = model.RunBatches(
      epoch_pb2.TRAIN,
      train_batches,
      log_prefix=f"Train to {hr_graph_cumsum}",
      total_graph_count=train_graph_count,
    )
    val_results = model.RunBatches(
      epoch_pb2.VAL,
      val_batches.batches,
      log_prefix=f"Val at {hr_graph_cumsum}",
      total_graph_count=FLAGS.val_graph_count,
    )

    # Write the epoch to file as an epoch list. This may seem redundant since
    # epoch list contains a single item, but it means that we can easily
    # concatenate a sequence of these epoch protos to produce a valid epoch
    # list using: `cat *.EpochList.pbtxt > epochs.pbtxt`
    epoch = epoch_pb2.EpochList(
      epoch=[
        epoch_pb2.Epoch(
          walltime_seconds=time.time() - start_time,
          epoch_num=epoch_step,
          train_results=train_results,
          val_results=val_results,
        )
      ]
    )
    print(epoch, end="")

    epoch_path = log_dir / "epochs" / f"{epoch_step:03d}.EpochList.pbtxt"
    pbutil.ToFile(epoch, epoch_path)
    app.Log(1, "Wrote %s", epoch_path)

    checkpoint_path = (
      log_dir / "checkpoints" / f"{epoch_step:03d}.Checkpoint.pb"
    )
    pbutil.ToFile(model.SaveCheckpoint(), checkpoint_path)

  return log_dir


def TestDataflowLSTM(
  path: pathlib.Path, log_dir: pathlib.Path, vocab: Dict[str, int],
):
  dataflow.PatchWarnings()
  dataflow.RecordExperimentalSetup(log_dir)

  # Create the logging directories.
  assert (log_dir / "epochs").is_dir()
  assert (log_dir / "checkpoints").is_dir()
  (log_dir / "graph_loader").mkdir(exist_ok=True)

  # Create the model, defining the shape of the graphs that it will process.
  #
  # For these data flow experiments, our graphs contain per-node binary
  # classification targets (e.g. reachable / not-reachable).
  model = Lstm(vocabulary=vocab, test_only=True, node_y_dimensionality=2,)
  restored_epoch, checkpoint = dataflow.SelectTestCheckpoint(log_dir)
  model.RestoreCheckpoint(checkpoint)

  batches = MakeBatchBuilder(
    dataset_root=path,
    log_dir=log_dir,
    epoch_type=epoch_pb2.TEST,
    model=model,
    min_graph_count=1,
  )

  start_time = time.time()
  test_results = model.RunBatches(epoch_pb2.TEST, batches, log_prefix="Test")
  epoch = epoch_pb2.EpochList(
    epoch=[
      epoch_pb2.Epoch(
        walltime_seconds=time.time() - start_time,
        epoch_num=restored_epoch.epoch_num,
        test_results=test_results,
      )
    ]
  )
  print(epoch, end="")

  epoch_path = log_dir / "epochs" / "TEST.EpochList.pbtxt"
  pbutil.ToFile(epoch, epoch_path)
  app.Log(1, "Wrote %s", epoch_path)


def Main():
  """Main entry point."""
  path = pathlib.Path(FLAGS.path)

  gpu_scheduler.LockExclusiveProcessGpuAccess()

  with vocabulary.VocabularyZipFile.CreateFromPublishedResults() as inst2vec:
    vocab = inst2vec.dictionary

  if FLAGS.test_only:
    log_dir = FLAGS.restore_from
  else:
    log_dir = TrainDataflowLSTM(
      path=path,
      vocab=vocab,
      val_seed=FLAGS.val_seed,
      restore_from=FLAGS.restore_from,
    )

  if FLAGS.test:
    TestDataflowLSTM(
      path=path, vocab=vocab, log_dir=log_dir,
    )


if __name__ == "__main__":
  app.Run(Main)
