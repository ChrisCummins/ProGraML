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
import subprocess
import shutil
import sys
import os
import tempfile
from pathlib import Path

from labm8.py import app
from labm8.py import bazelutil

TRAIN_LSTM = bazelutil.DataPath("programl/programl/task/dataflow/train_lstm")

LLVM_IR = bazelutil.DataPath(
    "programl/programl/test/data/llvm_ir"
)

LLVM_IR_GRAPHS = bazelutil.DataPath(
    "programl/programl/test/data/llvm_ir_graphs"
)

LLVM_IR_GRAPH_REACHABILITY_FEATURES = bazelutil.DataPath(
    "programl/programl/test/data/llvm_ir_reachability"
)

def make_test_reachability_dataflow_dataset(root: Path) -> Path:
  """Make a miniature dataset for reachability dataflow."""
  (root / "train").mkdir()
  (root / "val").mkdir()
  (root / "test").mkdir()
  (root / "labels").mkdir()

  shutil.copytree(LLVM_IR_GRAPHS, root / "graphs")
  shutil.copytree(LLVM_IR, root / "ir")
  shutil.copytree(
      LLVM_IR_GRAPH_REACHABILITY_FEATURES, root / "labels" / "reachability"
  )

  ngraphs = len(list(LLVM_IR_GRAPHS.iterdir()))
  ntrain = int(ngraphs * .6)
  nval = int(ngraphs * .8)

  for i, graph in enumerate(LLVM_IR_GRAPHS.iterdir()):
    if i < ntrain:
      dst = "train"
    elif i < nval:
      dst = "val"
    else:
      dst = "test"
    name = graph.name[:-len(".ProgramGraph.pb")]
    os.symlink(
        f"../graphs/{name}.ProgramGraph.pb",
        root / dst / f"{name}.ProgramGraph.pb",
        )

  return root


def main():
  with tempfile.TemporaryDirectory() as d:
    p = subprocess.Popen([
      TRAIN_LSTM,
      "--path", str(make_test_reachability_dataflow_dataset(Path(d))),
      "--analysis", "reachability",
      "--max_data_flow_steps", str(10),
      "--val_graph_count", str(10),
      "--val_seed", str(0xCC),
      "--train_graph_counts", "10,20"
    ])
    p.communicate()
    if p.returncode:
      sys.exit(1)


if __name__ == "__main__":
  app.Run(main)
