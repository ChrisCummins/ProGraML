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
import time

import numpy as np
from absl import app, flags

from programl.util.py.runfiles_path import runfiles_path

flags.DEFINE_integer(
    "graph_count", 100, "If > 0, limit the number of graphs to benchmark."
)
FLAGS = flags.FLAGS

LLVM_IR_GRAPHS = runfiles_path("tests/data/llvm_ir_graphs")
ANALYZE = runfiles_path("programl/bin/analyze")

ANALYSES = [
    "reachability",
    "dominance",
    "datadep",
    "liveness",
    "subexpressions",
]


def SummarizeFloats(floats, nplaces: int = 2, unit: str = "") -> str:
    """Summarize a sequence of floats."""
    arr = np.array(list(floats), dtype=np.float32)
    percs = " ".join(
        [
            f"{p}%={np.percentile(arr, p):.{nplaces}f}{unit}"
            for p in [0, 50, 95, 99, 100]
        ]
    )
    return (
        f"n={len(arr)}, mean={arr.mean():.{nplaces}f}{unit}, stdev={arr.std():.{nplaces}f}{unit}, "
        f"percentiles=[{percs}]"
    )


def BenchmarkAnalysis(analysis: str):
    runtimes = []
    paths = list(LLVM_IR_GRAPHS.iterdir())
    if FLAGS.graph_count:
        paths = paths[: FLAGS.graph_count]
    print(f"Runtimes for {analysis} analysis on test LLVM-IR files:", flush=True)
    for path in paths:
        with open(str(path), "rb") as f:
            start_time = time.time()
            try:
                subprocess.call(
                    [ANALYZE, analysis, "--stdin_fmt=pb", "--stdout_fmt=pb"],
                    stdin=f,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
                runtimes.append(time.time() - start_time)
            except subprocess.TimeoutExpired:
                runtimes.append(float("inf"))
                break
    runtimes_ms = np.array(runtimes) * 1000
    print(SummarizeFloats(runtimes_ms, unit="ms"), flush=True)


def main(argv):
    if len(argv) != 1:
        raise app.UsageError(f"Unrecognized arguments: {argv[1:]}")
    for i, analysis in enumerate(ANALYSES):
        if i:
            print()
        BenchmarkAnalysis(analysis)


if __name__ == "__main__":
    app.run(main)
