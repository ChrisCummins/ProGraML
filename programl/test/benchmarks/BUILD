# This package contains performance benchmarks.
#
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

py_test(
    name = "benchmark_dataflow_analyses",
    timeout = "long",
    srcs = ["benchmark_dataflow_analyses.py"],
    data = [
        "//programl/cmd:analyze",
        "//programl/test/data:llvm_ir_graphs",
    ],
    tags = ["notravis"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:viz",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "benchmark_dataflow_ggnn",
    timeout = "long",
    srcs = ["benchmark_dataflow_ggnn.py"],
    tags = ["notravis"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:ppar",
        "//labm8/py:prof",
        "//programl/models/ggnn",
        "//programl/proto:epoch_py",
        "//programl/task/dataflow:ggnn_batch_builder",
        "//programl/task/dataflow:graph_loader",
        "//programl/test/py/plugins:llvm_program_graph",
        "//programl/test/py/plugins:llvm_reachability_features",
        "//third_party/py/tqdm",
    ],
)

py_test(
    name = "benchmark_dataflow_lstm",
    timeout = "long",
    srcs = ["benchmark_dataflow_lstm.py"],
    tags = ["notravis"],
    deps = [
        "//deeplearning/ncc:vocabulary",
        "//labm8/py:app",
        "//labm8/py:ppar",
        "//labm8/py:prof",
        "//programl/models/lstm",
        "//programl/proto:epoch_py",
        "//programl/task/dataflow:graph_loader",
        "//programl/task/dataflow:lstm_batch_builder",
        "//programl/test/py/plugins:llvm_program_graph",
        "//programl/test/py/plugins:llvm_reachability_features",
        "//third_party/py/tqdm",
    ],
)

py_test(
    name = "benchmark_llvm2graph",
    timeout = "long",
    srcs = ["benchmark_llvm2graph.py"],
    data = [
        "//programl/cmd:llvm2graph",
        "//programl/test/data:llvm_ir",
    ],
    tags = ["notravis"],
    deps = [
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:viz",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "legacy_pipeline",
    timeout = "long",
    srcs = ["legacy_pipeline.py"],
    data = [
        "//programl/test/data:llvm_ir",
    ],
    tags = ["notravis"],
    deps = [
        "//deeplearning/ml4pl/graphs/labelled:graph_batcher",
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple",
        "//deeplearning/ml4pl/graphs/labelled/dataflow/reachability",
        "//deeplearning/ml4pl/graphs/llvm2graph:llvm2graph_py",
        "//deeplearning/ml4pl/graphs/llvm2graph:node_encoder",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:fs",
        "//labm8/py:humanize",
        "//labm8/py:prof",
    ],
)
