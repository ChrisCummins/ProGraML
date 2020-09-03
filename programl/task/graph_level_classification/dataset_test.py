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
"""Unit tests for //programl/task/graph_level_classification:dataset."""
from pathlib import Path

from labm8.py import test
import torch
from programl.task.graph_level_classification import dataset


FLAGS = test.FLAGS

pytest_plugins = [
    "programl.test.py.plugins.llvm_program_graph",
    "programl.test.py.plugins.reachability_dataflow_dataset",
]


@test.Fixture(scope="function")
def programl_vocabulary(reachability_dataflow_dataset: Path):
    return dataset.load_vocabulary(reachability_dataflow_dataset / "vocab" / "programl.csv")


@test.Fixture(scope="function")
def cdfg_vocabulary(reachability_dataflow_dataset: Path):
  return dataset.load_vocabulary(reachability_dataflow_dataset / "vocab" / "cdfg.csv")


@test.Fixture(
  scope="function",
  params=["programl_vocab", "cdfg_vocab"],
)
def vocab(request, reachability_dataflow_dataset):
    if request.param == "programl_vocab":
      return dataset.load_vocabulary(reachability_dataflow_dataset / "vocab" / "programl.csv")
    elif request.param == "cdfg_vocab":
      return dataset.load_vocabulary(reachability_dataflow_dataset / "vocab" / "cdfg.csv")
    else:
      raise TypeError("unreachable")


def test_RollingResults_iteration_count(reachability_dataflow_dataset: Path):
    """Test aggreation of model iteration count and convergence."""
    pass

def test_load(llvm_program_graph_path: Path):
    graph = dataset.load(llvm_program_graph_path)
    assert len(graph.node)
    assert len(graph.edge)


def test_load_vocab(vocab):
    assert len(programl_vocabulary)


def test_nx2data(llvm_program_graph, vocab):
  dataset.nx2data(graph, vocab)


def test_nx2data_no_vocab(llvm_program_graph, vocab):
  dataset.nx2data(graph, vocab, ablate_vocab=dataset.AblationVocab.NO_VOCAB)


def test_nx2data_node_type_only(llvm_program_graph, vocab):
  dataset.nx2data(graph, vocab, ablate_vocab=dataset.AblationVocab.NODE_TYPE_ONLY)


# data = dataset.nx2data(graph, vocab, "poj104_label")
# print("Created POJ-104 labelled graph data", data)


# graph = dataset.load(TEST_PROTO, cdfg=True)
# print("Loaded CDFG graph with", len(graph.node), "nodes and", len(graph.edge), "edges")

# if os.path.isdir(DEVMAP):
#     dataset.DevmapDataset(root=DEVMAP, split="amd")
#     dataset.DevmapDataset(root=DEVMAP, split="amd", cdfg=True)

# assert dataset.filename("foo", False, dataset.AblationVocab.NONE) == "foo_data.pt"
# assert dataset.filename("foo", True, dataset.AblationVocab.NONE) == "foo_cdfg_data.pt"
# assert dataset.filename("foo", True, dataset.AblationVocab.NO_VOCAB) == "foo_cdfg_no_vocab_data.pt"


if __name__ == "__main__":
    test.Main()
