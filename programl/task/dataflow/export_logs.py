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
"""Export per-epoch statistics from machine learning logs.

Machine learning jobs write log files summarizing the performance of the model
after each training epoch. The log directory is printed at the start of
execution of a machine learning job, for example:

  $ bazel run //programl/task/dataflow:train_ggnn
  Writing logs to ~/programl/dataflow/logs/ggnn/reachability/foo@20:05:16T12:53:42
  ...

This script reads one of these log directories and prints a table of per-epoch
stats to stdout. For example:

  $ export-ml-logs --path=~/programl/dataflow/ml/logs/foo@20:05:16T12:53:42

CSV format can be exported using --fmt=csv:

  $ export-ml-logs --path=~/programl/dataflow/ml/logs/foo@20:05:16T12:53:42 \\
      --fmt=csv > stats.csv

Alternatively the summary can be uploaded to Google Sheets using:

  $ export-ml-logs --path=~/programl/dataflow/ml/logs/foo@20:05:16T12:53:42 \\
      --google_sheet='My Spreadsheet' \\
      --worksheet='reachability' \\
      --google_sheets_credentials=/tmp/credentials.json \\
      --google_sheets_default_share_with=joe@example.com
"""
import pathlib
import subprocess
import sys
from typing import Optional

import pandas as pd

from labm8.py import app
from labm8.py import google_sheets
from labm8.py import humanize
from labm8.py import pbutil
from labm8.py import pdutil
from labm8.py import prof
from programl.proto import epoch_pb2

app.DEFINE_input_path(
  "path",
  pathlib.Path("~/programl/dataflow").expanduser(),
  "The dataset directory root.",
  is_dir=True,
)
app.DEFINE_string(
  "google_sheet",
  None,
  "The name of a Google Sheets spreadsheet to export tables to. If it does "
  "not exist, the spreadsheet is created and shared with "
  "--google_sheets_default_share_with. See --google_sheets_credentials for "
  "setting the credentials required to use the Google Sheets API.",
)
app.DEFINE_string("fmt", "txt", "Stdout format.")
app.DEFINE_string(
  "worksheet", "Sheet1", "The name of the worksheet to export to"
)
FLAGS = app.FLAGS


def ReadEpochLogs(path: pathlib.Path) -> Optional[epoch_pb2.EpochList]:
  if not (path / "epochs").is_dir():
    return None
  epochs = []
  for path in (path / "epochs").iterdir():
    epoch = pbutil.FromFile(path, epoch_pb2.EpochList())
    # Skip files without data.
    if not len(epoch.epoch):
      continue
    epochs += list(epoch.epoch)
  return epoch_pb2.EpochList(epoch=sorted(epochs, key=lambda x: x.epoch_num))


def EpochsToDataFrame(epochs: epoch_pb2.EpochList) -> Optional[pd.DataFrame]:
  def V(results, field):
    if results.batch_count:
      return getattr(results, field)
    else:
      return None

  rows = []
  for e in epochs.epoch:
    rows.append(
      {
        "epoch_num": e.epoch_num,
        "walltime_seconds": e.walltime_seconds,
        "train_graph_count": V(e.train_results, "graph_count"),
        "train_batch_count": V(e.train_results, "batch_count"),
        "train_target_count": V(e.train_results, "target_count"),
        "train_learning_rate": V(e.train_results, "mean_learning_rate"),
        "train_loss": V(e.train_results, "mean_loss"),
        "train_accuracy": V(e.train_results, "mean_accuracy"),
        "train_precision": V(e.train_results, "mean_precision"),
        "train_recall": V(e.train_results, "mean_recall"),
        "train_f1": V(e.train_results, "mean_f1"),
        "train_walltime_seconds": V(e.train_results, "walltime_seconds"),
        "val_graph_count": V(e.val_results, "graph_count"),
        "val_batch_count": V(e.val_results, "batch_count"),
        "val_target_count": V(e.val_results, "target_count"),
        "val_loss": V(e.val_results, "mean_loss"),
        "val_accuracy": V(e.val_results, "mean_accuracy"),
        "val_precision": V(e.val_results, "mean_precision"),
        "val_recall": V(e.val_results, "mean_recall"),
        "val_f1": V(e.val_results, "mean_f1"),
        "val_walltime_seconds": V(e.val_results, "walltime_seconds"),
        "test_graph_count": V(e.test_results, "graph_count"),
        "test_batch_count": V(e.test_results, "batch_count"),
        "test_target_count": V(e.test_results, "target_count"),
        "test_loss": V(e.test_results, "mean_loss"),
        "test_accuracy": V(e.test_results, "mean_accuracy"),
        "test_precision": V(e.test_results, "mean_precision"),
        "test_recall": V(e.test_results, "mean_recall"),
        "test_f1": V(e.test_results, "mean_f1"),
        "test_walltime_seconds": V(e.test_results, "walltime_seconds"),
      }
    )
  if not len(rows):
    return
  df = pd.DataFrame(rows)

  # Add columns for cumulative totals.
  df["train_graph_count_cumsum"] = df.train_graph_count.cumsum()
  df["train_batch_count_cumsum"] = df.train_batch_count.cumsum()
  df["train_target_count_cumsum"] = df.train_target_count.cumsum()
  df["train_walltime_seconds_cumsum"] = df.train_walltime_seconds.cumsum()
  df["walltime_seconds_cumsum"] = df.walltime_seconds.cumsum()

  # Re-order columns.
  return df[
    [
      "epoch_num",
      "train_batch_count",
      "train_batch_count_cumsum",
      "train_graph_count",
      "train_graph_count_cumsum",
      "train_target_count",
      "train_target_count_cumsum",
      "train_learning_rate",
      "train_loss",
      "train_accuracy",
      "train_precision",
      "train_recall",
      "train_f1",
      "train_walltime_seconds",
      "train_walltime_seconds_cumsum",
      "val_batch_count",
      "val_graph_count",
      "val_target_count",
      "val_loss",
      "val_accuracy",
      "val_precision",
      "val_recall",
      "val_f1",
      "val_walltime_seconds",
      "test_batch_count",
      "test_graph_count",
      "test_target_count",
      "test_loss",
      "test_accuracy",
      "test_precision",
      "test_recall",
      "test_f1",
      "test_walltime_seconds",
    ]
  ]


def LogsToDataFrame(path: pathlib.Path) -> Optional[pd.DataFrame]:
  logdirs = (
    subprocess.check_output(
      [
        "find",
        "-L",
        str(path / "logs"),
        "-maxdepth",
        "3",
        "-mindepth",
        "3",
        "-type",
        "d",
      ],
      universal_newlines=True,
    )
    .rstrip()
    .split("\n")
  )

  dfs = []
  for logdir in logdirs:
    app.Log(2, "%s", logdir)
    logdir = pathlib.Path(logdir)
    epochs = ReadEpochLogs(logdir)
    if epochs is None:
      continue
    df = EpochsToDataFrame(epochs)
    if df is None:
      continue
    df.insert(0, "run_id", logdir.name)
    df.insert(0, "model", logdir.parent.parent.name)
    df.insert(0, "analysis", logdir.parent.name)
    dfs.append(df)

  if not dfs:
    return None
  df = pd.concat(dfs)
  df.sort_values(["analysis", "model", "run_id"], inplace=True)
  return df


def Main():
  path = FLAGS.path
  fmt = FLAGS.fmt
  spreadsheet_name = FLAGS.google_sheet
  worksheet_name = FLAGS.worksheet

  with prof.Profile("loading logs"):
    df = LogsToDataFrame(path)

  if df is None:
    print("No logs found", file=sys.stderr)
    sys.exit(1)

  if spreadsheet_name:
    gsheets = google_sheets.GoogleSheets.FromFlagsOrDie()
    spreadsheet = gsheets.GetOrCreateSpreadsheet(spreadsheet_name)
    worksheet = gsheets.GetOrCreateWorksheet(spreadsheet, worksheet_name)
    gsheets.ExportDataFrame(worksheet, df, index=False)
    app.Log(
      1,
      f"Exported %s rows to Google Sheet to %s:%s",
      humanize.Commas(len(df)),
      spreadsheet_name,
      worksheet_name,
    )
  elif fmt == "csv":
    df.to_csv(sys.stdout, header=True)
  elif fmt == "txt":
    print(pdutil.FormatDataFrameAsAsciiTable(df))
  else:
    raise app.UsageError(f"Unknown --fmt: {fmt}")


if __name__ == "__main__":
  app.Run(Main)
