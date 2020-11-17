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
"""Benchmark for llvm2graph."""
import subprocess
import time

import numpy as np
from absl import bazelutil, flags, viz

flags.DEFINE_float(
    "min_benchmark_time",
    30,
    "The minimum number of seconds to run the benchmark loop.",
)
FLAGS = flags.FLAGS

LLVM_IR = runfiles_path("programl/tests/data/llvm_ir")
LLVM2GRAPH = runfiles_path("programl/programl/cmd/llvm2graph")


def main(argv):
    if len(argv) != 1:
        raise app.UsageError(f"Unrecognized arguments: {argv[1:]}")
    runtimes = []
    start_time = time.time()
    while time.time() < start_time + FLAGS.min_benchmark_time:
        for path in LLVM_IR.iterdir():
            file_start_time = time.time()
            subprocess.check_call(
                [str(LLVM2GRAPH), str(path), "--stdout_fmt=pb"],
                stdout=subprocess.DEVNULL,
            )
            runtimes.append(time.time() - file_start_time)
    runtimes_ms = np.array(runtimes) * 1000
    print("Runtimes for llvm2graph on test LLVM-IR files:")
    print(viz.SummarizeFloats(runtimes_ms, unit="ms"))


if __name__ == "__main__":
    app.run(main)
