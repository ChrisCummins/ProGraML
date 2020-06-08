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
"""Benchmark of the legacy ml4pl codebase pipeline."""
from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs.labelled import graph_batcher
from deeplearning.ml4pl.graphs.labelled import graph_tuple
from deeplearning.ml4pl.graphs.labelled.dataflow.reachability import (
  reachability,
)
from deeplearning.ml4pl.graphs.llvm2graph import llvm2graph
from deeplearning.ml4pl.graphs.llvm2graph import node_encoder
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import fs
from labm8.py import humanize
from labm8.py import prof

FLAGS = app.FLAGS

LLVM_IR = bazelutil.DataPath("phd/programl/test/data/llvm_ir")


def Main():
  irs = [fs.Read(path) for path in LLVM_IR.iterdir()]
  ir_count = len(irs)

  with prof.ProfileToStdout(
    lambda t: (
      f"STAGE 1: Construct unlabelled graphs (llvm2graph)         "
      f"({humanize.Duration(t / ir_count)} / IR)"
    )
  ):
    graphs = [llvm2graph.BuildProgramGraphNetworkX(ir) for ir in irs]

  encoder = node_encoder.GraphNodeEncoder()
  with prof.ProfileToStdout(
    lambda t: (
      f"STAGE 2: Encode graphs (inst2vec)                         "
      f"({humanize.Duration(t / ir_count)} / IR)"
    )
  ):
    for graph, ir in zip(graphs, irs):
      encoder.EncodeNodes(graph, ir)

  features_count = 0
  features_lists = []
  with prof.ProfileToStdout(
    lambda t: (
      f"STAGE 3: Produce labelled graphs (reachability analysis)  "
      f"({humanize.Duration(t / features_count)} / graph)"
    )
  ):
    for graph in graphs:
      analysis = reachability.ReachabilityAnnotator(
        programl.NetworkXToProgramGraph(graph)
      )
      features_list = analysis.MakeAnnotated(n=10).graphs
      features_count += len(features_list)
      features_lists.append(features_list)

  def iter():
    for features_list in features_lists:
      for graph in features_list:
        yield graph_tuple.GraphTuple.CreateFromNetworkX(graph)

  with prof.ProfileToStdout(
    lambda t: (
      f"STAGE 4: Construct graph tuples                           "
      f"({humanize.Duration(t / features_count)} / graph)"
    )
  ):
    batcher = graph_batcher.GraphBatcher(iter(), max_node_count=10000)
    graph_tuples = list(batcher)

  print("=================================")
  print(f"Unlabelled graphs count: {ir_count}")
  print(f"  Labelled graphs count: {features_count}")
  print(f"     Graph tuples count: {len(graph_tuples)}")
  print(f"       Total node count: {sum(gt.node_count for gt in graph_tuples)}")
  print(f"       Total edge count: {sum(gt.edge_count for gt in graph_tuples)}")


if __name__ == "__main__":
  app.Run(Main)
