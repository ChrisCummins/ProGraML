# This package python APIs for working with graphs.
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

load(
    "//third_party/pybind11_bazel:build_defs.bzl",
    "pybind_extension",
)

pybind_extension(
    "program_graph_builder_pybind",
    srcs = ["program_graph_builder_pybind.cc"],
    deps = [
        "//programl/graph:program_graph_builder",
    ],
)

py_library(
    name = "program_graph_builder",
    srcs = ["program_graph_builder.py"],
    data = [":program_graph_builder_pybind.so"],
    visibility = ["//visibility:public"],
    deps = [
        "//programl/proto:program_graph_py",
    ],
)

py_test(
    name = "program_graph_builder_test",
    srcs = ["program_graph_builder_test.py"],
    deps = [
        ":program_graph_builder",
        "//labm8/py:test",
        "//programl/proto:edge_py",
        "//programl/proto:node_py",
        "//programl/proto:program_graph_py",
    ],
)
