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
"""A module for encoding LLVM-IR program graphs using inst2vec."""
import multiprocessing
import pathlib
import pickle
import random
import time
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from deeplearning.ncc.inst2vec import inst2vec_preprocess
from labm8.py import bazelutil
from labm8.py import decorators
from labm8.py import labtypes
from labm8.py import pbutil
from labm8.py import progress
from programl.proto import node_pb2
from programl.proto import program_graph_pb2


DICTIONARY = bazelutil.DataPath(
  "phd/programl/ir/llvm/internal/inst2vec_augmented_dictionary.pickle"
)
AUGMENTED_INST2VEC_EMBEDDINGS = bazelutil.DataPath(
  "phd/programl/ir/llvm/internal/inst2vec_augmented_embeddings.pickle"
)


def NodeFullText(node: node_pb2.Node) -> str:
  """Get the full text of a node, or an empty string if not set."""
  if len(node.features.feature["full_text"].bytes_list.value):
    return (
      node.features.feature["full_text"].bytes_list.value[0].decode("utf-8")
    )
  return ""


class Inst2vecEncoder(object):
  """An encoder for LLVM program graphs using inst2vec."""

  def __init__(self):
    with open(str(DICTIONARY), "rb") as f:
      self.dictionary = pickle.load(f)

    with open(str(AUGMENTED_INST2VEC_EMBEDDINGS), "rb") as f:
      self.node_text_embeddings = pickle.load(f)

  def RunOnDataset(self, dataset_root: pathlib.Path) -> None:
    progress.Run(_Inst2vecEncodeDataset(self, dataset_root))

  def RunOnDirectory(self, directory: pathlib.Path) -> None:
    progress.Run(_Inst2vecEncodeDirectory(self, directory))

  def Encode(
    self, proto: program_graph_pb2.ProgramGraph, ir: Optional[str] = None
  ) -> program_graph_pb2.ProgramGraph:
    """Pre-process the node text and set the text embedding index.

    For each node, this sets 'inst2vec_preprocessed' and 'inst2vec_embedding'
    features.

    Args:
      proto: The ProgramGraph to encode.
      ir: The LLVM IR that was used to construct the graph. This is required for
        struct inlining. If struct inlining is not required, this may be
        omitted.

    Returns:
      The input proto.
    """
    # Gather the instruction texts to pre-process.
    lines = [
      [NodeFullText(node)]
      for node in proto.node
      if node.type == node_pb2.Node.INSTRUCTION
    ]

    if ir:
      # NOTE(github.com/ChrisCummins/ProGraML/issues/57): Extract the struct
      # definitions from the IR and inline their definitions in place of the
      # struct names. These is brittle string substitutions, in the future we
      # should do this inlining in llvm2graph where we have a parsed
      # llvm::Module.
      try:
        structs = inst2vec_preprocess.GetStructTypes(ir)
        for line in lines:
          for struct, definition in structs.items():
            line[0] = line[0].replace(struct, definition)
      except ValueError:
        pass

    preprocessed_lines, _ = inst2vec_preprocess.preprocess(lines)
    preprocessed_texts = [
      inst2vec_preprocess.PreprocessStatement(x[0] if len(x) else "")
      for x in preprocessed_lines
    ]

    # Add the node features.
    var_embedding = self.dictionary["!IDENTIFIER"]
    const_embedding = self.dictionary["!IMMEDIATE"]

    text_index = 0
    for node in proto.node:
      if node.type == node_pb2.Node.INSTRUCTION:
        text = preprocessed_texts[text_index].encode("utf-8")
        text_index += 1
        embedding = self.dictionary.get(text, self.dictionary["!UNK"])
        node.features.feature["inst2vec_preprocessed"].bytes_list.value.append(
          text
        )
        node.features.feature["inst2vec_embedding"].int64_list.value.append(
          embedding
        )
      elif node.type == node_pb2.Node.VARIABLE:
        node.features.feature["inst2vec_embedding"].int64_list.value.append(
          var_embedding
        )
      elif node.type == node_pb2.Node.CONSTANT:
        node.features.feature["inst2vec_embedding"].int64_list.value.append(
          const_embedding
        )

    proto.features.feature["inst2vec_annotated"].int64_list.value.append(1)
    return proto

  @decorators.memoized_property
  def embeddings_tables(self) -> List[np.array]:
    """Return the embeddings tables."""
    node_selector = np.vstack([[1, 0], [0, 1],]).astype(np.float64)
    return [self.node_text_embeddings, node_selector]


@decorators.timeout(seconds=60)
def _EncodeOne(
  encoder: Inst2vecEncoder,
  graph: program_graph_pb2.ProgramGraph,
  graph_path: pathlib.Path,
  ir_path: Optional[pathlib.Path],
):
  if ir_path:
    with open(str(ir_path)) as f:
      ir = f.read()
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
    graph = pbutil.FromFile(graph_path, program_graph_pb2.ProgramGraph())
    # Check to see if we have already processed this file.
    if len(graph.features.feature["inst2vec_annotated"].int64_list.value):
      continue

    encoded_count += 1
    try:
      _EncodeOne(encoder, graph, graph_path, ir_path)
    except AssertionError:
      # NCC codebase uses assertions in place of regular exceptions.
      pass
    except TimeoutError:
      pass
  return len(paths), encoded_count, time.time() - start_time


class _Inst2vecEncodeJob(progress.Progress):
  """Run inst2vec encoder on all graphs in the dataset."""

  def __init__(
    self,
    encoder: Inst2vecEncoder,
    graph_ir_paths: List[Tuple[pathlib.Path, pathlib.Path]],
    logfile: Optional[pathlib.Path] = None,
  ):
    self.encoder = encoder
    self.graph_ir_paths = graph_ir_paths
    self.logfile = logfile

    # Load balance.
    random.shuffle(self.graph_ir_paths)

    super(_Inst2vecEncodeJob, self).__init__(
      "inst2vec", i=0, n=len(self.graph_ir_paths), unit="graphs"
    )

  def Run(self):
    jobs = [
      (self.encoder, chunk)
      for chunk in list(labtypes.Chunkify(self.graph_ir_paths, 128))
    ]
    logfile = None
    if self.logfile:
      logfile = open(str(self.logfile), "a")

    with multiprocessing.Pool() as pool:
      for processed_count, encoded_count, runtime in pool.imap_unordered(
        _ProcessRows, jobs
      ):
        self.ctx.i += processed_count
        if self.logfile:
          logfile.write(
            f"{processed_count}\t{encoded_count}\t{runtime:.4f}\t{runtime / processed_count:.4f}\n"
          )
          logfile.flush()

    if self.logfile:
      logfile.close()
    self.ctx.i = self.ctx.n


class _Inst2vecEncodeDataset(_Inst2vecEncodeJob):
  """Run inst2vec encoder on all graphs in the dataset."""

  def __init__(self, encoder: Inst2vecEncoder, path: pathlib.Path):
    self.encoder = encoder
    if not (path / "graphs").is_dir():
      raise FileNotFoundError(str(path / "graphs"))
    if not (path / "ir").is_dir():
      raise FileNotFoundError(str(path / "ir"))

    graph_paths = [
      p
      for p in (path / "graphs").iterdir()
      if p.name.endswith(".ProgramGraph.pb")
    ]
    ir_paths = [
      path / "ir" / f"{p.name[:-len('.ProgramGraph.pb')]}.ll"
      for p in graph_paths
    ]
    ir_paths = [p if p.is_file() else None for p in ir_paths]
    graph_ir_paths = list(zip(graph_paths, ir_paths))

    super(_Inst2vecEncodeDataset, self).__init__(
      encoder=encoder,
      graph_ir_paths=graph_ir_paths,
      logfile=path / "inst2vec_log.txt",
    )


class _Inst2vecEncodeDirectory(_Inst2vecEncodeJob):
  """Run inst2vec encoder on all graphs in the dataset."""

  def __init__(self, encoder: Inst2vecEncoder, path: pathlib.Path):
    self.encoder = encoder
    if not path.is_dir():
      raise FileNotFoundError(str(path))

    graph_paths = [
      p for p in path.iterdir() if p.name.endswith(".ProgramGraph.pb")
    ]
    ir_paths = [
      path / f"{p.name[:-len('.ProgramGraph.pb')]}.ll" for p in graph_paths
    ]
    ir_paths = [p if p.is_file() else None for p in ir_paths]
    graph_ir_paths = list(zip(graph_paths, ir_paths))

    super(_Inst2vecEncodeDirectory, self).__init__(
      encoder=encoder, graph_ir_paths=graph_ir_paths,
    )
