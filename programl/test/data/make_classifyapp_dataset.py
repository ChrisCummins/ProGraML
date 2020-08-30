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
"""Create a mini classifyapp dataset using test data.

Usage:

    $ bazel run //programl/test/data:make_classifyapp_dataset \
        --path /path/to/generated/dataset
"""
import os
import random
import shutil
import subprocess
from pathlib import Path

from labm8.py import app, bazelutil, pbutil

from programl.proto.program_graph_pb2 import ProgramGraph

app.DEFINE_string("path", None, "The path of to write the generated dataset to.")
FLAGS = app.FLAGS


LLVM_IR = bazelutil.DataPath("programl/programl/test/data/llvm_ir")

LLVM_IR_GRAPHS = bazelutil.DataPath("programl/programl/test/data/llvm_ir_graphs")

CREATE_VOCAB = bazelutil.DataPath(
    "programl/programl/task/dataflow/dataset/create_vocab"
)


def make_classifyapp_dataset(root: Path) -> Path:
    """Make a miniature dataset for classifyapp.

    Args:
        root: The root of the dataset.

    Returns:
        The root of the dataset.
    """
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir()
    (root / "test").mkdir()

    shutil.copytree(LLVM_IR_GRAPHS, root / "graphs")
    shutil.copytree(LLVM_IR, root / "ir")

    # Assign a random POJ-104 label to each graph.
    for path in (root / "graphs").iterdir():
        graph = pbutil.FromFile(path, ProgramGraph())
        graph.features.feature["poj104_label"].int64_list.value[:] = [
            random.randint(1, 104)
        ]
        pbutil.ToFile(graph, path)

    ngraphs = len(list(LLVM_IR_GRAPHS.iterdir()))
    ntrain = int(ngraphs * 0.6)
    nval = int(ngraphs * 0.8)

    for i, graph in enumerate(LLVM_IR_GRAPHS.iterdir()):
        if i < ntrain:
            dst = "train"
        elif i < nval:
            dst = "val"
        else:
            dst = "test"
        name = graph.name[: -len(".ProgramGraph.pb")]
        os.symlink(
            f"../graphs/{name}.ProgramGraph.pb",
            root / dst / f"{name}.ProgramGraph.pb",
        )

    subprocess.check_call(
        [str(CREATE_VOCAB), "--path", str(root)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return root


def main():
    """Main entry point."""
    assert FLAGS.path
    make_classifyapp_dataset(Path(FLAGS.path))


if __name__ == "__main__":
    app.Run(main)
