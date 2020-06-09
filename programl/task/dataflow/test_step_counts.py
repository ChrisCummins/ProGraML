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
"""Enumerate all test graphs and print their data flow step count to stdout."""
import pathlib

from labm8.py import app
from labm8.py import pbutil
from programl.proto import program_graph_features_pb2
from programl.proto import program_graph_pb2


app.DEFINE_string(
  "path",
  str(pathlib.Path("~/programl/dataflow").expanduser()),
  "The path to read from",
)
app.DEFINE_string("analysis", "reachability", "The analysis type to use.")
app.DEFINE_integer("max_graph_node_count", 40000, "The max size of graph.")
FLAGS = app.FLAGS


def Main():
  """Main entry point."""
  path = pathlib.Path(FLAGS.path)

  graphs_path = path / "test"
  labels_path = path / "labels" / FLAGS.analysis

  for graph_path in graphs_path.iterdir():
    stem = graph_path.name[: -len("ProgramGraph.pb")]
    name = f"{stem}ProgramGraphFeaturesList.pb"
    features_path = labels_path / name
    # There is no guarantee that we have generated features for this
    # program graph, so we check for its existence. As a *very* defensive
    # measure, we also check for the existence of the graph file that we
    # enumerated at the start of this function. This check can be removed
    # later, it is only useful during development when you might be
    # modifying the dataset at the same time as having test jobs running.
    if not graph_path.is_file() or not features_path.is_file():
      continue

    graph = pbutil.FromFile(graph_path, program_graph_pb2.ProgramGraph())
    if not len(graph.node) or len(graph.node) > FLAGS.max_graph_node_count:
      continue

    features_list = pbutil.FromFile(
      features_path, program_graph_features_pb2.ProgramGraphFeaturesList()
    )

    for j, features in enumerate(features_list.graph):
      step_count_feature = features.features.feature[
        "data_flow_step_count"
      ].int64_list.value
      step_count = step_count_feature[0] if len(step_count_feature) else 0
      print(features_path.name, j, step_count)


if __name__ == "__main__":
  app.Run(Main)
