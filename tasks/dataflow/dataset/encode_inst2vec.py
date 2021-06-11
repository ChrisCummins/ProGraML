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
"""Run inst2vec encoder on ProGraML graphs.

This runs each of the graphs in the graphs/ directory through inst2vec encoder.
"""
import multiprocessing
import pathlib
import random
import time
from itertools import islice
from typing import List, Tuple

from absl import app, flags

from programl.ir.llvm.inst2vec_encoder import Inst2vecEncoder
from programl.proto import Ir, ProgramGraph
from programl.util.py import decorators, pbutil, progress
from tasks.dataflow.dataset import pathflag

FLAGS = flags.FLAGS


def chunkify(iterable, n):
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))


@decorators.timeout(seconds=60)
def Encode(encoder, graph, graph_path, ir_path):
    if ir_path.is_file():
        try:
            ir = pbutil.FromFile(ir_path, Ir()).text
        except pbutil.DecodeError:
            ir = None
    else:
        ir = None

    encoder.Encode(graph, ir=ir)
    pbutil.ToFile(graph, graph_path)


def _ProcessRows(job) -> Tuple[int, int, float]:
    start_time = time.time()
    encoded_count = 0

    encoder: Inst2vecEncoder = job[0]
    paths: List[Tuple[pathlib.Path, pathlib.Path]] = job[1]
    for graph_path, ir_path in paths:
        graph = pbutil.FromFile(graph_path, ProgramGraph())
        # Check to see if we have already processed this file.
        if len(graph.features.feature["inst2vec_annotated"].int64_list.value):
            continue

        encoded_count += 1
        try:
            Encode(encoder, graph, graph_path, ir_path)
        except AssertionError:
            # NCC codebase uses assertions to check for errors.
            pass
        except TimeoutError:
            pass
    return len(paths), encoded_count, time.time() - start_time


class Inst2vecEncodeGraphs(progress.Progress):
    """Run inst2vec encoder on all graphs in the dataset."""

    def __init__(self, path: pathlib.Path):
        self.path = path
        if not (path / "graphs").is_dir():
            raise FileNotFoundError(str(path / "graphs"))

        # Enumerate pairs of <program_graph, ir> paths.
        self.paths = [
            (
                graph_path,
                path / "ir" / f"{graph_path.name[:-len('.ProgramGraph.pb')]}.Ir.pb",
            )
            for graph_path in (path / "graphs").iterdir()
            if graph_path.name.endswith(".ProgramGraph.pb")
        ]

        # Load balance.
        random.shuffle(self.paths)

        super(Inst2vecEncodeGraphs, self).__init__(
            "inst2vec", i=0, n=len(self.paths), unit="graphs"
        )

    def Run(self):
        encoder = Inst2vecEncoder()
        jobs = [(encoder, chunk) for chunk in list(chunkify(self.paths, 128))]
        with open(self.path / "graphs.inst2vec_log.txt", "a") as f:
            with multiprocessing.Pool() as pool:
                for processed_count, encoded_count, runtime in pool.imap_unordered(
                    _ProcessRows, jobs
                ):
                    self.ctx.i += processed_count
                    f.write(
                        f"{processed_count}\t{encoded_count}\t{runtime:.4f}\t{runtime / processed_count:.4f}\n"
                    )
                    f.flush()
        self.ctx.i = self.ctx.n


def main(argv):
    if len(argv) != 1:
        raise app.UsageError(f"Unrecognized arguments: {argv[1:]}")
    path = pathlib.Path(pathflag.path())
    progress.Run(Inst2vecEncodeGraphs(path))


if __name__ == "__main__":
    app.run(main)
