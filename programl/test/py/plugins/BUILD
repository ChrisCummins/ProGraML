# This package defines pytest plugins for testing.
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

package(default_visibility = ["//programl:__subpackages__"])

py_library(
    name = "llvm_ir",
    testonly = 1,
    srcs = ["llvm_ir.py"],
    data = ["//programl/test/data:llvm_ir"],
    deps = [
        "//labm8/py:bazelutil",
        "//labm8/py:test",
    ],
)

py_library(
    name = "llvm_program_graph",
    testonly = 1,
    srcs = ["llvm_program_graph.py"],
    data = ["//programl/test/data:llvm_ir_graphs"],
    deps = [
        "//labm8/py:bazelutil",
        "//labm8/py:pbutil",
        "//labm8/py:test",
        "//programl/proto:program_graph_py",
    ],
)

py_library(
    name = "llvm_reachability_features",
    testonly = 1,
    srcs = ["llvm_reachability_features.py"],
    data = ["//programl/test/data:llvm_ir_reachability_features"],
    deps = [
        "//labm8/py:bazelutil",
    ],
)
