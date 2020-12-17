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

package(default_visibility = ["//visibility:public"])

py_library(
    name = "llvm_ir",
    testonly = 1,
    srcs = ["llvm_ir.py"],
    data = ["//tests/data:llvm_ir"],
    deps = [
        "//programl/util/py:runfiles_path",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "llvm_program_graph",
    testonly = 1,
    srcs = ["llvm_program_graph.py"],
    data = ["//tests/data:llvm_ir_graphs"],
    deps = [
        "//programl/proto:programl_py",
        "//programl/util/py:pbutil",
        "//programl/util/py:runfiles_path",
        "//third_party/py/pytest",
    ],
)

py_library(
    name = "llvm_reachability_features",
    testonly = 1,
    srcs = ["llvm_reachability_features.py"],
    data = ["//tests/data:llvm_ir_reachability_features"],
    deps = [
        "//programl/util/py:runfiles_path",
    ],
)

py_library(
    name = "tempdir",
    testonly = 1,
    srcs = ["tempdir.py"],
    deps = [
        "//third_party/py/pytest",
    ],
)
