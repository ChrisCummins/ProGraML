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
"""A module for encoding node embeddings."""
import pickle
from typing import List

import networkx as nx
import numpy as np

from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ncc.inst2vec import inst2vec_preprocess
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import decorators


FLAGS = app.FLAGS

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

  def EncodeNodes(self, g: nx.DiGraph) -> None:
    """Pre-process the node text and set the text embedding index.

    For each node, this sets the 'preprocessed_text', 'x', and 'y' attributes.

    Args:
      g: The graph to encode the nodes of.
    """
    # Pre-process the statements of the graph in a single pass.
    lines = [
      [data["text"]]
      for _, data in g.nodes(data=True)
      if data["type"] == programl_pb2.Node.STATEMENT
    ]
    preprocessed_lines, _ = inst2vec_preprocess.preprocess(lines)
    preprocessed_texts = [
      inst2vec_preprocess.PreprocessStatement(x[0] if len(x) else "")
      for x in preprocessed_lines
    ]
    for (node, data), text in zip(g.nodes(data=True), preprocessed_texts):
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
