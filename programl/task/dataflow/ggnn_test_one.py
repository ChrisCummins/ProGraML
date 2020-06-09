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
"""Run inference of a trained GGNN model on a single graph input.
"""
import pathlib
from typing import Any
from typing import Iterable

import numpy as np

from labm8.py import app
from labm8.py import pbutil
from programl.models.base_graph_loader import BaseGraphLoader
from programl.models.batch_results import BatchResults
from programl.models.ggnn.ggnn import Ggnn
from programl.proto import checkpoint_pb2
from programl.proto import epoch_pb2
from programl.proto import program_graph_features_pb2
from programl.proto import program_graph_pb2
from programl.task.dataflow import dataflow
from programl.task.dataflow import vocabulary
from programl.task.dataflow.dataset import pathflag
from programl.task.dataflow.ggnn_batch_builder import DataflowGgnnBatchBuilder


app.DEFINE_boolean(
  "cdfg",
  False,
  "If set, use the CDFG representation for programs. Defaults to ProGraML "
  "representations.",
)
app.DEFINE_integer(
  "max_vocab_size",
  0,
  "If > 0, limit the size of the vocabulary to this number.",
)
app.DEFINE_float(
  "target_vocab_cumfreq", 1.0, "The target cumulative frequency that."
)
app.DEFINE_input_path("model", None, "The model checkpoint to restore")
app.DEFINE_string(
  "input",
  None,
  "Path of the input graph features list and index into it, "
  "e.g., /path/to/foo:1 to select the second graph from file /path/to/foo.",
)
FLAGS = app.FLAGS


class SingleGraphLoader(BaseGraphLoader):
  """`A graph loader which reads a single graph."""

  def __init__(
    self,
    graph: program_graph_pb2.ProgramGraph,
    features: program_graph_features_pb2.ProgramGraphFeatures,
  ):
    self.graph = graph
    self.features = features

  def IterableType(self) -> Any:
    return (
      program_graph_pb2.ProgramGraph,
      program_graph_features_pb2.ProgramGraphFeatures,
    )

  def __iter__(self) -> Iterable["IterableType"]:
    yield self.graph, self.features

  def Stop(self):
    pass


def TestOne(
  features_list_path: pathlib.Path,
  features_list_index: int,
  checkpoint_path: pathlib.Path,
) -> BatchResults:
  path = pathlib.Path(pathflag.path())

  features_list = pbutil.FromFile(
    features_list_path, program_graph_features_pb2.ProgramGraphFeaturesList(),
  )
  features = features_list.graph[features_list_index]

  graph_name = features_list_path.name[: -len(".ProgramGraphFeaturesList.pb")]
  graph = pbutil.FromFile(
    path / "graphs" / f"{graph_name}.ProgramGraph.pb",
    program_graph_pb2.ProgramGraph(),
  )

  # Instantiate and restore the model.
  vocab = vocabulary.LoadVocabulary(
    path,
    model_name="cdfg" if FLAGS.cdfg else "programl",
    max_items=FLAGS.max_vocab_size,
    target_cumfreq=FLAGS.target_vocab_cumfreq,
  )

  if FLAGS.cdfg:
    FLAGS.use_position_embeddings = False

  model = Ggnn(
    vocabulary=vocab,
    test_only=True,
    node_y_dimensionality=2,
    graph_y_dimensionality=0,
    graph_x_dimensionality=0,
    use_selector_embeddings=True,
  )
  checkpoint = pbutil.FromFile(checkpoint_path, checkpoint_pb2.Checkpoint())
  model.RestoreCheckpoint(checkpoint)

  batch = list(
    DataflowGgnnBatchBuilder(
      graph_loader=SingleGraphLoader(graph=graph, features=features),
      vocabulary=vocab,
      max_node_size=int(1e9),
      use_cdfg=FLAGS.cdfg,
      max_batch_count=1,
    )
  )[0]

  results = model.RunBatch(epoch_pb2.TEST, batch)

  return AnnotateGraphWithBatchResults(graph, features, results)


def AnnotateGraphWithBatchResults(
  graph: program_graph_pb2.ProgramGraph,
  features: program_graph_features_pb2.ProgramGraphFeatures,
  results: BatchResults,
):
  """Annotate graph with features describing the target labels and predicted outcomes."""
  assert len(graph.node) == len(
    features.node_features.feature_list["data_flow_value"].feature
  )
  assert len(graph.node) == len(
    features.node_features.feature_list["data_flow_root_node"].feature
  )
  assert len(graph.node) == results.targets.shape[0]

  true_y = np.argmax(results.targets, axis=1)
  pred_y = np.argmax(results.predictions, axis=1)

  for i, node in enumerate(graph.node):
    node.features.feature["data_flow_value"].CopyFrom(
      features.node_features.feature_list["data_flow_value"].feature[i]
    )
    node.features.feature["data_flow_root_node"].CopyFrom(
      features.node_features.feature_list["data_flow_root_node"].feature[i]
    )
    node.features.feature["target"].float_list.value[:] = results.targets[i]
    node.features.feature["prediction"].float_list.value[
      :
    ] = results.predictions[i]
    node.features.feature["true_y"].int64_list.value.append(true_y[i])
    node.features.feature["pred_y"].int64_list.value.append(pred_y[i])
    node.features.feature["correct"].int64_list.value.append(
      true_y[i] == pred_y[i]
    )

  graph.features.feature["loss"].float_list.value.append(results.loss)
  graph.features.feature["accuracy"].float_list.value.append(results.accuracy)
  graph.features.feature["precision"].float_list.value.append(results.precision)
  graph.features.feature["recall"].float_list.value.append(results.recall)
  graph.features.feature["f1"].float_list.value.append(results.f1)
  graph.features.feature["confusion_matrix"].int64_list.value[:] = np.hstack(
    results.confusion_matrix
  )
  return graph


def Main():
  """Main entry point."""
  dataflow.PatchWarnings()

  features_list_path, features_list_index = FLAGS.input.split(":")
  graph = TestOne(
    features_list_path=pathlib.Path(features_list_path),
    features_list_index=int(features_list_index),
    checkpoint_path=FLAGS.model,
  )
  print(graph)


if __name__ == "__main__":
  app.Run(Main)
