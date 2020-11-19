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
        "//tests/data:llvm_ir_graphs",
    ],
    deps = [
        "//programl/util/py:runfiles_path",
        "//third_party/py/absl",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "benchmark_dataflow_ggnn",
    timeout = "long",
    srcs = ["benchmark_dataflow_ggnn.py"],
    deps = [
        "//programl/models/ggnn",
        "//programl/proto:programl_py",
        "//programl/util/py:progress",
        "//programl/util/py:threaded_iterator",
        "//tasks/dataflow:ggnn_batch_builder",
        "//tasks/dataflow:graph_loader",
        "//tests/plugins:llvm_program_graph",
        "//tests/plugins:llvm_reachability_features",
        "//third_party/py/absl",
        "//third_party/py/tqdm",
    ],
)

#py_test(
#    name = "benchmark_dataflow_lstm",
#    timeout = "long",
#    srcs = ["benchmark_dataflow_lstm.py"],
#    deps = [
#        "//programl/models/lstm",
#        "//programl/proto:programl_py",
#        "//programl/util/py:progress",
#        "//programl/util/py:threaded_iterator",
#        "//tests/plugins:llvm_program_graph",
#        "//tests/plugins:llvm_reachability_features",
#        "//tasks/dataflow:graph_loader",
#        "//tasks/dataflow:lstm_batch_builder",
#        "//third_party/py/absl",
#        "//third_party/py/ncc:vocabulary",
#        "//third_party/py/tqdm",
#    ],
#)

py_test(
    name = "benchmark_llvm2graph",
    timeout = "long",
    srcs = ["benchmark_llvm2graph.py"],
    data = [
        "//programl/cmd:llvm2graph",
        "//tests/data:llvm_ir",
    ],
    deps = [
        "//programl/util/py:runfiles_path",
        "//third_party/py/absl",
        "//third_party/py/numpy",
    ],
)
