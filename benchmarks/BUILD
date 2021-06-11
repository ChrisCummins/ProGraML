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
    timeout = "eternal",
    srcs = ["benchmark_dataflow_analyses.py"],
    data = [
        "//programl/bin:analyze",
        "//tests/data:llvm_ir_graphs",
    ],
    deps = [
        "//programl/util/py:runfiles_path",
    ],
)

py_test(
    name = "benchmark_llvm2graph",
    timeout = "eternal",
    srcs = ["benchmark_llvm2graph.py"],
    data = [
        "//programl/bin:llvm2graph-10",
        "//tests/data:llvm_ir",
    ],
    deps = [
        "//programl/util/py:runfiles_path",
    ],
)
