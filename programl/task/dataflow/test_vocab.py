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
"""Check coverage of the vocabularies on the test set."""
import csv
import pathlib
from collections import defaultdict

from deeplearning.ncc.vocabulary import VocabularyZipFile
from labm8.py import app
from labm8.py import pbutil
from labm8.py import progress
from programl.proto import node_pb2
from programl.proto import program_graph_pb2
from programl.task.dataflow.dataset import pathflag
from programl.task.dataflow.vocabulary import LoadVocabulary


app.DEFINE_integer("limit", 0, "Limit the number of files read.")
FLAGS = app.FLAGS


class TestVocab(progress.Progress):
  def __init__(self, path: pathlib.Path):
    self.path = path
    self.graphs = list((path / "test").iterdir())
    if FLAGS.limit:
      self.graphs = self.graphs[: FLAGS.limit]

    with VocabularyZipFile.CreateFromPublishedResults() as inst2vec_vocab:
      self.inst2vec = inst2vec_vocab.dictionary

    self.cdfg = LoadVocabulary(path, "cdfg")
    self.programl = LoadVocabulary(path, "programl")

    super(TestVocab, self).__init__(name="vocab", i=0, n=len(self.graphs))

  def Run(self):
    inst2vec = defaultdict(int)
    cdfg = defaultdict(int)
    programl = defaultdict(int)
    node_count = 0

    for self.ctx.i, path in enumerate(self.graphs, start=1):
      graph = pbutil.FromFile(path, program_graph_pb2.ProgramGraph())

      for node in graph.node:
        node_count += 1

        try:
          n = (
            node.features.feature["inst2vec_preprocessed"]
            .bytes_list.value[0]
            .decode("utf-8")
          )
          if n in self.inst2vec:
            inst2vec[n] += 1
        except IndexError:
          pass

        if node.text in self.cdfg:
          cdfg[node.text] += 1

        if node.text in self.programl:
          programl[node.text] += 1

    ToCsv(
      self.path / "vocab" / "inst2vec_test_coverage.csv", inst2vec, node_count,
    )
    ToCsv(
      self.path / "vocab" / "cdfg_test_coverage.csv", cdfg, node_count,
    )
    ToCsv(
      self.path / "vocab" / "programl_test_coverage.csv", programl, node_count,
    )


def ToCsv(
  path: pathlib.Path, vocab_counts: defaultdict, node_count: int,
):
  vocab_entries = sorted(vocab_counts.items(), key=lambda x: -x[1])
  total_count = sum(vocab_counts.values())

  cumfreq = 0
  node_cumfreq = 0
  with open(str(path), "w") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(
      ("cumulative_frequency", "cumulative_node_frequency", "count", "text",)
    )
    for text, count in vocab_entries:
      cumfreq += count / total_count
      node_cumfreq += count / node_count
      writer.writerow((cumfreq, node_cumfreq, count, text))


def Main():
  progress.Run(TestVocab(pathlib.Path(pathflag.path())))


if __name__ == "__main__":
  app.Run(Main)
