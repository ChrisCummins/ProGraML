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
"""Logic for training and evaluating GGNNs."""
import pathlib
import time
from typing import Dict
from typing import List
from typing import Optional

import numpy as np

from labm8.py import app
from labm8.py import humanize
from labm8.py import pbutil
from programl.models.async_batch_builder import AsyncBatchBuilder
from programl.models.epoch_batch_iterator import EpochBatchIterator
from programl.models.ggnn.ggnn import Ggnn
from programl.proto import epoch_pb2
from programl.task.dataflow import dataflow
from programl.task.dataflow.ggnn_batch_builder import DataflowGgnnBatchBuilder
from programl.task.dataflow.graph_loader import DataflowGraphLoader


def MakeBatchBuilder(
  dataset_root: pathlib.Path,
  log_dir: pathlib.Path,
  analysis: str,
  epoch_type: epoch_pb2.EpochType,
  model: Ggnn,
  batch_size: int,
  use_cdfg: bool,
  limit_max_data_flow_steps: bool,
  min_graph_count=None,
  max_graph_count=None,
  seed=None,
):
  if limit_max_data_flow_steps:
    data_flow_step_max = model.message_passing_step_count
  else:
    data_flow_step_max = None
  logfile = (
    log_dir
    / "graph_loader"
    / f"{epoch_pb2.EpochType.Name(epoch_type).lower()}.txt"
  )
  return DataflowGgnnBatchBuilder(
    graph_loader=DataflowGraphLoader(
      dataset_root,
      epoch_type=epoch_type,
      analysis=analysis,
      min_graph_count=min_graph_count,
      max_graph_count=max_graph_count,
      data_flow_step_max=data_flow_step_max,
      # Append to logfile since we may be resuming a previous job.
      logfile=open(str(logfile), "a"),
      seed=seed,
      use_cdfg=use_cdfg,
    ),
    vocabulary=model.vocabulary,
    max_node_size=batch_size,
    use_cdfg=use_cdfg,
  )


def TrainDataflowGGNN(
  path: pathlib.Path,
  analysis: str,
  vocab: Dict[str, int],
  limit_max_data_flow_steps: bool,
  train_graph_counts: List[int],
  val_graph_count: int,
  val_seed: int,
  batch_size: int,
  use_cdfg: bool,
  run_id: Optional[str] = None,
  restore_from: pathlib.Path = None,
) -> pathlib.Path:
  if not path.is_dir():
    raise FileNotFoundError(path)

  if restore_from:
    log_dir = restore_from
  else:
    # Create the logging directories.
    log_dir = dataflow.CreateLoggingDirectories(
      dataset_root=path,
      model_name="cdfg" if use_cdfg else "programl",
      analysis=analysis,
      run_id=run_id,
    )

  dataflow.PatchWarnings()
  dataflow.RecordExperimentalSetup(log_dir)

  # Cumulative totals for training graph counts at each "epoch".
  train_graph_cumsums = np.array(train_graph_counts, dtype=np.int32)
  # The number of training graphs in each "epoch".
  train_graph_counts = train_graph_cumsums - np.concatenate(
    ([0], train_graph_counts[:-1])
  )

  # Create the model, defining the shape of the graphs that it will process.
  #
  # For these data flow experiments, our graphs contain per-node binary
  # classification targets (e.g. reachable / not-reachable).
  model = Ggnn(
    vocabulary=vocab,
    test_only=False,
    node_y_dimensionality=2,
    graph_y_dimensionality=0,
    graph_x_dimensionality=0,
    use_selector_embeddings=True,
  )

  if restore_from:
    # Pick up training where we left off.
    restored_epoch, checkpoint = dataflow.SelectTrainingCheckpoint(log_dir)
    # Skip the epochs that we have already done.
    # This requires that --train_graph_counts is the same as it was in the
    # run that we are resuming!
    start_epoch_step = restored_epoch.epoch_num
    start_graph_cumsum = sum(train_graph_counts[:start_epoch_step])
    train_graph_counts = train_graph_counts[start_epoch_step:]
    model.RestoreCheckpoint(checkpoint)
  else:
    # Else initialize a new model.
    model.Initialize()
    start_epoch_step, start_graph_cumsum = 1, 0

  app.Log(
    1,
    "GGNN has %s training params",
    humanize.Commas(model.trainable_parameter_count),
  )

  # Create training batches and split into epochs.
  epochs = EpochBatchIterator(
    MakeBatchBuilder(
      dataset_root=path,
      log_dir=log_dir,
      epoch_type=epoch_pb2.TRAIN,
      analysis=analysis,
      model=model,
      batch_size=batch_size,
      use_cdfg=use_cdfg,
      limit_max_data_flow_steps=limit_max_data_flow_steps,
    ),
    train_graph_counts,
    start_graph_count=start_graph_cumsum,
  )

  # Read val batches asynchronously.
  val_batches = AsyncBatchBuilder(
    MakeBatchBuilder(
      dataset_root=path,
      log_dir=log_dir,
      epoch_type=epoch_pb2.VAL,
      analysis=analysis,
      model=model,
      batch_size=batch_size,
      use_cdfg=use_cdfg,
      limit_max_data_flow_steps=limit_max_data_flow_steps,
      min_graph_count=val_graph_count,
      max_graph_count=val_graph_count,
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
      total_graph_count=val_graph_count,
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


def TestDataflowGGNN(
  path: pathlib.Path,
  log_dir: pathlib.Path,
  analysis: str,
  vocab: Dict[str, int],
  limit_max_data_flow_steps: bool,
  batch_size: int,
  use_cdfg: bool,
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
  model = Ggnn(
    vocabulary=vocab,
    test_only=True,
    node_y_dimensionality=2,
    graph_y_dimensionality=0,
    graph_x_dimensionality=0,
    use_selector_embeddings=True,
  )
  restored_epoch, checkpoint = dataflow.SelectTestCheckpoint(log_dir)
  model.RestoreCheckpoint(checkpoint)

  batches = MakeBatchBuilder(
    dataset_root=path,
    log_dir=log_dir,
    epoch_type=epoch_pb2.TEST,
    analysis=analysis,
    model=model,
    batch_size=batch_size,
    use_cdfg=use_cdfg,
    # Specify that we require at least one graph, as the default (no min) will
    # loop forever.
    min_graph_count=1,
    limit_max_data_flow_steps=limit_max_data_flow_steps,
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
