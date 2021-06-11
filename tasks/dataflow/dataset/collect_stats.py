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
"""Aggregate stats describing the graphs and labels in the dataset."""
import csv
import pathlib

from absl import app, flags, logging

from programl.proto import ProgramGraph, ProgramGraphFeaturesList
from programl.util.py import pbutil, progress
from tasks.dataflow.dataset import pathflag

flags.DEFINE_boolean("graphs", True, "Collect stats on graphs")
flags.DEFINE_list(
    "analysis",
    ["reachability", "dominance", "datadep", "liveness", "subexpressions"],
    "The analyses labels to collect stats over.",
)
FLAGS = flags.FLAGS


class CollectGraphStats(progress.Progress):
    def __init__(
        self,
        dataset_root: pathlib.Path,
        split: str,
        writer: csv.writer,
        write_header: bool,
    ):
        self.dataset_root = dataset_root
        self.split = split
        self.writer = writer
        self.write_header = write_header

        self.files = [
            f
            for f in (dataset_root / split).iterdir()
            if f.name.endswith(".ProgramGraph.pb")
        ]
        super(CollectGraphStats, self).__init__(name=split, i=0, n=len(self.files))

    def Run(self):
        if self.write_header:
            self.writer.writerow(
                (
                    "split",
                    "graph_name",
                    "node_count",
                    "edge_count",
                    "function_count",
                    "module_count",
                )
            )

        for self.ctx.i, path in enumerate(self.files):
            graph_name = path.name[: -len(".ProgramGraph.pb")]
            graph = pbutil.FromFile(path, ProgramGraph())
            self.writer.writerow(
                (
                    self.split,
                    graph_name,
                    len(graph.node),
                    len(graph.edge),
                    len(graph.function),
                    len(graph.module),
                )
            )


class CollectAnalysisStats(progress.Progress):
    def __init__(
        self,
        dataset_root: pathlib.Path,
        analysis: str,
        writer: csv.writer,
        write_header: bool,
    ):
        self.dataset_root = dataset_root
        self.analysis = analysis
        self.writer = writer
        self.write_header = write_header

        self.files = [
            f
            for f in (dataset_root / "labels" / analysis).iterdir()
            if f.name.endswith(".ProgramGraphFeaturesList.pb")
        ]
        super(CollectAnalysisStats, self).__init__(
            name=analysis, i=0, n=len(self.files)
        )

    def Run(self):
        if self.write_header:
            self.writer.writerow(
                ("analysis", "graph_name", "i", "label_count", "step_count")
            )

        for self.ctx.i, path in enumerate(self.files):
            graph_name = path.name[: -len(".ProgramGraphFeaturesList.pb")]
            features = pbutil.FromFile(path, ProgramGraphFeaturesList())
            for i, graph in enumerate(features.graph):
                step_count = graph.features.feature[
                    "data_flow_step_count"
                ].int64_list.value
                if len(step_count):
                    step_count = step_count[0]
                else:
                    step_count = 0
                self.writer.writerow(
                    (
                        self.analysis,
                        graph_name,
                        i,
                        len(
                            graph.node_features.feature_list["data_flow_value"].feature
                        ),
                        step_count,
                    )
                )


def main(argv):
    if len(argv) != 1:
        raise app.UsageError(f"Unrecognized arguments: {argv[1:]}")
    path = pathlib.Path(pathflag.path())

    if FLAGS.graphs:
        with open(path / "graph_stats.csv", "w") as f:
            writer = csv.writer(f, delimiter=",")
            logging.info("Aggregating graph stats")
            progress.Run(CollectGraphStats(path, "test", writer, write_header=True))
            progress.Run(CollectGraphStats(path, "val", writer, write_header=True))
            progress.Run(CollectGraphStats(path, "train", writer, write_header=True))

    with open(path / "label_stats.csv", "w") as f:
        writer = csv.writer(f, delimiter=",")
        for i, analysis in enumerate(FLAGS.analysis):
            logging.info("Aggregating %s stats", analysis)
            progress.Run(
                CollectAnalysisStats(path, analysis, writer, write_header=not i)
            )


if __name__ == "__main__":
    app.run(main)
