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
"""A module for encoding node embeddings.

When executed as a binary, this program reads a single program graph from
stdin, encodes it, and writes a graph to stdout. Use --stdin_fmt and
--stdout_fmt to convert between different graph types, and --ir to read the
IR file that the graph was constructed from, required for resolving struct
definitions.

Example usage:

  Encode a program graph binary proto and write the result as text format:

    $ bazel run //deeplearning/ml4pl/graphs/llvm2graph:node_encoder -- \
        --stdin_fmt=pb \
        --stdout_fmt=pbtxt \
        --ir=/tmp/source.ll \
        < /tmp/proto.pb > /tmp/proto.pbtxt
"""
import pickle
from typing import List
from typing import Optional

import networkx as nx
import numpy as np

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ncc.inst2vec import inst2vec_preprocess
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import decorators
from labm8.py import fs


FLAGS = app.FLAGS
app.DEFINE_output_path(
  "ir",
  None,
  "The path of the IR file that was used to construct the graph. This is "
  "required to inline struct definitions. This argument may be omitted when "
  "struct definitions do not need to be inlined.",
)

DICTIONARY = bazelutil.DataPath(
  "phd/deeplearning/ml4pl/graphs/llvm2graph/node_embeddings/inst2vec_augmented_dictionary.pickle"
)
AUGMENTED_INST2VEC_EMBEDDINGS = bazelutil.DataPath(
  "phd/deeplearning/ml4pl/graphs/llvm2graph/node_embeddings/inst2vec_augmented_embeddings.pickle"
)


class GraphNodeEncoder(object):
  """An encoder for node 'x' features."""

  def __init__(self):
    with open(str(DICTIONARY), "rb") as f:
      self.dictionary = pickle.load(f)

    # TODO(github.com/ChrisCummins/ProGraML/issues/12): In the future we may
    # want to add support for different numbers of embeddings tables, or
    # embeddings tables with different types. This is hardcoded to support only
    # two embeddings tables: our augmented inst2vec statement embeddings, and
    # a binary 'selector' table which can be used to select one or more nodes
    # of interests in graphs, e.g. for setting the starting point for computing
    # iterative data flow analyses.
    with open(str(AUGMENTED_INST2VEC_EMBEDDINGS), "rb") as f:
      self.node_text_embeddings = pickle.load(f)

  def EncodeNodes(self, g: nx.DiGraph, ir: Optional[str] = None) -> None:
    """Pre-process the node text and set the text embedding index.

    For each node, this sets the 'preprocessed_text', 'x', and 'y' attributes.

    Args:
      g: The graph to encode the nodes of.
      ir: The LLVM IR that was used to construct the graph. This is required for
        struct inlining. If struct inlining is not required, this may be
        omitted.
    """
    # Pre-process the statements of the graph in a single pass.
    lines = [
      [data["text"]]
      for _, data in g.nodes(data=True)
      if data["type"] == programl_pb2.Node.STATEMENT
    ]

    if ir:
      # NOTE(github.com/ChrisCummins/ProGraML/issues/57): Extract the struct
      # definitions from the IR and inline their definitions in place of the
      # struct names. This is brittle string substitutions, in the future we
      # should do this inlining in llvm2graph where we have a parsed
      # llvm::Module.
      structs = inst2vec_preprocess.GetStructTypes(ir)
      for line in lines:
        for struct, definition in structs.items():
          line[0] = line[0].replace(struct, definition)

    preprocessed_lines, _ = inst2vec_preprocess.preprocess(lines)
    preprocessed_texts = [
      inst2vec_preprocess.PreprocessStatement(x[0] if len(x) else "")
      for x in preprocessed_lines
    ]
    for (_, data), text in zip(
      (
        (_, data)
        for _, data in g.nodes(data=True)
        if data["type"] == programl_pb2.Node.STATEMENT
      ),
      preprocessed_texts,
    ):
      data["preprocessed_text"] = text
      data["x"] = [self.dictionary.get(text, self.dictionary["!UNK"])]

    # Re-write the remaining graph nodes.
    for node, data in g.nodes(data=True):
      if data["type"] == programl_pb2.Node.IDENTIFIER:
        data["preprocessed_text"] = "!IDENTIFIER"
        data["x"] = [self.dictionary["!IDENTIFIER"]]
      elif data["type"] == programl_pb2.Node.IMMEDIATE:
        data["preprocessed_text"] = "!IMMEDIATE"
        data["x"] = [self.dictionary["!IMMEDIATE"]]

      data["y"] = []

  @decorators.memoized_property
  def embeddings_tables(self) -> List[np.array]:
    """Return the embeddings tables."""
    node_selector = np.vstack([[1, 0], [0, 1],]).astype(np.float64)
    return [self.node_text_embeddings, node_selector]


def Main():
  """Main entry point."""
  proto = programl.ReadStdin()
  g = programl.ProgramGraphToNetworkX(proto)
  encoder = GraphNodeEncoder()
  ir = fs.Read(FLAGS.ir) if FLAGS.ir else None
  encoder.EncodeNodes(g, ir=ir)
  programl.WriteStdout(programl.NetworkXToProgramGraph(g))


if __name__ == "__main__":
  app.Run(Main)
