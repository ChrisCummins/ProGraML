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
"""Python bindings for running analyses."""
from programl.graph.analysis.py import analysis_pybind
from programl.proto import program_graph_features_pb2
from programl.proto import program_graph_pb2


def RunAnalysis(
  analysis: str, graph: program_graph_pb2.ProgramGraph
) -> program_graph_features_pb2.ProgramGraphFeaturesList:
  """Run the given analysis.

  Args:
    analysis: The name of the analysis to run.
    graph: The program graph to analyze.

  Returns:
    A program graph features list.

  Raises:
    ValueError: In case analysis fails.
  """
  graph_features = program_graph_features_pb2.ProgramGraphFeaturesList()
  serialized_graph_features = analysis_pybind.RunAnalysis(
    analysis, graph.SerializeToString()
  )
  graph_features.ParseFromString(serialized_graph_features)
  return graph_features
