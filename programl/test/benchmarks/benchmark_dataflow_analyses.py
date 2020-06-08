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
"""Benchmark for analyze command."""
import subprocess
import sys
import time

import numpy as np

from labm8.py import app
from labm8.py import bazelutil
from labm8.py import viz

app.DEFINE_integer(
  "graph_count", 100, "If > 0, limit the number of graphs to benchmark."
)
FLAGS = app.FLAGS

LLVM_IR_GRAPHS = bazelutil.DataPath("phd/programl/test/data/llvm_ir_graphs")
ANALYZE = bazelutil.DataPath("phd/programl/cmd/analyze")

ANALYSES = [
  "reachability",
  "dominance",
  "datadep",
  "liveness",
  "subexpressions",
]


def BenchmarkAnalysis(analysis: str):
  runtimes = []
  paths = list(LLVM_IR_GRAPHS.iterdir())
  if FLAGS.graph_count:
    paths = paths[: FLAGS.graph_count]
  for path in paths:
    with open(str(path), "rb") as f:
      start_time = time.time()
      subprocess.call(
        [ANALYZE, analysis, "--stdin_fmt=pb", "--stdout_fmt=pb"],
        stdin=f,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
      )
      runtimes.append(time.time() - start_time)
  runtimes_ms = np.array(runtimes) * 1000
  print(f"Runtimes for {analysis} analysis on test LLVM-IR files:")
  print(viz.SummarizeFloats(runtimes_ms, unit="ms"))
  sys.stdout.flush()


def Main():
  for i, analysis in enumerate(ANALYSES):
    if i:
      print()
    BenchmarkAnalysis(analysis)


if __name__ == "__main__":
  app.Run(Main)
