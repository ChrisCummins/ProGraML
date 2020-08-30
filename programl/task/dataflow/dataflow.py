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
"""This module defines functions for training and testing GGNN dataflow models.
"""
import json
import pathlib
import time
import warnings
from typing import Tuple

from labm8.py import app, humanize, pbutil
from sklearn.exceptions import UndefinedMetricWarning

from programl.models.base_batch_builder import BaseBatchBuilder
from programl.models.model import Model
from programl.proto import checkpoint_pb2, epoch_pb2

app.DEFINE_string(
    "path",
    str(pathlib.Path("~/programl/dataflow").expanduser()),
    "The path to read from",
)
app.DEFINE_string("analysis", "reachability", "The analysis type to use.")
app.DEFINE_integer(
    "val_graph_count", 10000, "The number of graphs to use in the validation set."
)
app.DEFINE_integer(
    "val_seed",
    0xCC,
    "The seed value for randomly sampling validation graphs.",
)
app.DEFINE_list(
    "train_graph_counts",
    [
        10000,
        20000,
        30000,
        40000,
        50000,
        100000,
        200000,
        300000,
        400000,
        500000,
        600000,
        700000,
        800000,
        900000,
        1000000,
    ],
    "The list of cumulative training graph counts to evaluate at.",
)
app.DEFINE_input_path(
    "restore_from",
    None,
    "The log directory of a previous model run to restore",
    is_dir=True,
)
app.DEFINE_boolean("test", True, "Whether to test the model after training.")
app.DEFINE_boolean(
    "test_only",
    False,
    "Whether to skip training and go straight to testing. "
    "Assumes that --restore_from is set.",
)
app.DEFINE_string(
    "run_id",
    None,
    "Optionally specify a name for the run. This must be unique. If not "
    "provided, a run ID is generated using the current time. If --restore_from "
    "is set, the ID of the restored run is used and this flag has no effect.",
)

FLAGS = app.FLAGS


def RecordExperimentalSetup(log_dir: pathlib.Path) -> None:
    """Create flags.txt and build_info.json files.

    These two files record a snapshot of the configuration and build information,
    useful for debugging and reproducibility.

    Args:
      log_dir: The path to write the files in.
    """
    with open(log_dir / "flags.txt", "w") as f:
        f.write(app.FlagsToString())
    with open(log_dir / "build_info.json", "w") as f:
        json.dump(app.ToJson(), f, sort_keys=True, indent=2, separators=(",", ": "))


def PatchWarnings():
    """Set global configuration options for dataflow experiments."""
    # Since we are dealing with binary classification we calculate
    # precesion / recall / F1 wrt only the positive class.
    FLAGS.batch_results_averaging_method = "binary"
    # NOTE(github.com/ChrisCummins/ProGraML/issues/13): F1 score computation
    # warns that it is undefined when there are missing instances from a class,
    # which is fine for our usage.
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def SelectTestCheckpoint(
    log_dir: pathlib.Path,
) -> Tuple[epoch_pb2.Epoch, checkpoint_pb2.Checkpoint]:
    """Select a checkpoint to load for testing.

    The training checkpoint with the highest validation F1 score is used for
    testing.

    Returns:
      A tuple of <Epoch, Checkpoint> messages.
    """
    best_f1 = -1
    best_epoch_num = None
    for path in (log_dir / "epochs").iterdir():
        if path.name.endswith(".EpochList.pbtxt"):
            epoch = pbutil.FromFile(path, epoch_pb2.EpochList())
            f1 = epoch.epoch[0].val_results.mean_f1
            epoch_num = epoch.epoch[0].epoch_num
            if f1 >= best_f1:
                best_f1 = f1
                best_epoch_num = epoch_num
    epoch = pbutil.FromFile(
        log_dir / "epochs" / f"{best_epoch_num:03d}.EpochList.pbtxt",
        epoch_pb2.EpochList(),
    )
    checkpoint = pbutil.FromFile(
        log_dir / "checkpoints" / f"{best_epoch_num:03d}.Checkpoint.pb",
        checkpoint_pb2.Checkpoint(),
    )
    app.Log(
        1,
        "Selected best checkpoint %d with val F1 score %.3f",
        epoch.epoch[0].epoch_num,
        epoch.epoch[0].val_results.mean_f1,
    )
    return epoch.epoch[0], checkpoint


def SelectTrainingCheckpoint(
    log_dir: pathlib.Path,
) -> Tuple[epoch_pb2.Epoch, checkpoint_pb2.Checkpoint]:
    """Select a checkpoint to load to resume training.

    Returns:
      A tuple of <Epoch, Checkpoint> messages.
    """
    epoch_num = -1
    for path in (log_dir / "epochs").iterdir():
        if path.name.endswith(".EpochList.pbtxt"):
            epoch = pbutil.FromFile(path, epoch_pb2.EpochList())
            if not epoch.epoch[0].train_results.graph_count:
                continue
            epoch_num = max(epoch_num, epoch.epoch[0].epoch_num)

    epoch = pbutil.FromFile(
        log_dir / "epochs" / f"{epoch_num:03d}.EpochList.pbtxt",
        epoch_pb2.EpochList(),
    )
    checkpoint = pbutil.FromFile(
        log_dir / "checkpoints" / f"{epoch_num:03d}.Checkpoint.pb",
        checkpoint_pb2.Checkpoint(),
    )
    app.Log(
        1,
        "Resuming training from checkpoint %d with val F1 score %.3f",
        epoch.epoch[0].epoch_num,
        epoch.epoch[0].val_results.mean_f1,
    )
    return epoch.epoch[0], checkpoint


def CreateLoggingDirectories(
    dataset_root: pathlib.Path, model_name: str, analysis: str, run_id: str = None
) -> pathlib.Path:
    """Create the logging directories for an ML model.

    Args:
      dataset_root: The root path of the dataset.
      model_name: The name of the model.
      analysis: The name of the analysis.
      run_id: An optional run ID to force. If not provided, a timestamp is used.

    Returns:
      The logging directory.
    """
    run_id = run_id or time.strftime("%y:%m:%dT%H:%M:%S")
    log_dir = dataset_root / "logs" / model_name / analysis / run_id
    if log_dir.is_dir():
        raise OSError(
            f"Logs directory already exists. Refusing to overwrite: {log_dir}"
        )
    app.Log(1, "Writing logs to %s", log_dir)
    log_dir.mkdir(parents=True)
    (log_dir / "epochs").mkdir()
    (log_dir / "checkpoints").mkdir()
    (log_dir / "graph_loader").mkdir()
    return log_dir


def run_training_loop(
    log_dir: pathlib.Path,
    epochs,
    val_batches: BaseBatchBuilder,
    start_epoch_step: int,
    model: Model,
    val_graph_count: int,
) -> pathlib.Path:
    """

    Args:
        log_dir: The logging directory.
        epochs: An epoch batch builder.
        val_batches: A batch builder for validation.
        start_epoch_step: The initial step count.
        model: The model to train.
        val_graph_count: The number of validation graphs.

    Returns:
        The log_dir first argument.
    """
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

        checkpoint_path = log_dir / "checkpoints" / f"{epoch_step:03d}.Checkpoint.pb"
        pbutil.ToFile(model.SaveCheckpoint(), checkpoint_path)

    return log_dir
